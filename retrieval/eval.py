import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from retrieval.lookup_gen import DATA_PATH, INDEX_NAME, encode
from retrieval.main import QuoteMatcher

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VAL_PATH = "data/processed/pairs_val.parquet"
TRAIN_PATH = "data/processed/pairs_train.parquet"
MODE_ORDER = ["question", "advice", "statement", "nonsense"]


def encode_all(texts, model, tokenizer, batch_size=256):
    chunks = [encode(texts[i:i + batch_size], model, tokenizer) for i in range(0, len(texts), batch_size)]
    return torch.cat(chunks)


def quote_embeddings(model_path, model, tokenizer, quotes):
    index_path = os.path.join(model_path, INDEX_NAME)
    if os.path.isdir(model_path) and QuoteMatcher._stale_reason(index_path, model_path, quotes) is None:
        return torch.load(index_path, weights_only=True)["embeddings"].to(model.device)
    return encode_all(quotes["text"].tolist(), model, tokenizer)


def ranks(queries, candidates, targets, chunk=1024):
    if not torch.is_tensor(targets):
        targets = np.array(targets)  # copy: arrays from pandas are read-only, which torch warns about
    targets = torch.as_tensor(targets, device=queries.device)
    out = []
    for i in range(0, len(queries), chunk):
        scores = queries[i:i + chunk] @ candidates.T
        own = scores.gather(1, targets[i:i + chunk, None])
        out.append((scores > own).sum(dim=1) + 1)
    return torch.cat(out).cpu().numpy()


def cluster_bootstrap(values, groups, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    _, group = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(group, weights=values)
    counts = np.bincount(group)
    pick = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return values.mean(), lo, hi


def evaluate(name, quotes, val, train, absurd_ids, device):
    path = BASE_MODEL if name == "base" else name
    print(f"evaluating {path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).to(device).eval()

    col = {qid: i for i, qid in enumerate(quotes["quote_id"])}
    corpus = quote_embeddings(path, model, tokenizer, quotes)

    val_emb = encode_all(val["input"].tolist(), model, tokenizer)
    result = {"val": ranks(val_emb, corpus, val["quote_id"].map(col).to_numpy())}

    held_ids = val["quote_id"].unique()
    held_pos = {qid: i for i, qid in enumerate(held_ids)}
    held_cols = torch.tensor([col[qid] for qid in held_ids], device=corpus.device)
    result["held"] = ranks(val_emb, corpus[held_cols], val["quote_id"].map(held_pos).to_numpy())

    nonsense = torch.tensor((val["mode"] == "nonsense").to_numpy(), device=val_emb.device)
    is_absurd = torch.tensor(quotes["quote_id"].isin(absurd_ids).to_numpy(), device=corpus.device)
    top1 = (val_emb[nonsense] @ corpus.T).argmax(dim=1)
    result["absurd_top1"] = is_absurd[top1].float().mean().item() if len(top1) else float("nan")

    result["train"] = None
    if train is not None:
        train_emb = encode_all(train["input"].tolist(), model, tokenizer)
        result["train"] = ranks(train_emb, corpus, train["quote_id"].map(col).to_numpy())

    del model, corpus
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def pct(x):
    return f"{100 * x:.1f}%"


def print_table(names, results, val):
    labels = [os.path.basename(n.rstrip("/")) for n in names]
    modes = [m for m in MODE_ORDER if m in set(val["mode"])] + sorted(set(val["mode"]) - set(MODE_ORDER))

    rows = []
    for k in (1, 5, 10):
        rows.append((f"R@{k}", [pct((r["val"] <= k).mean()) for r in results]))
    rows.append(("MRR", [f"{(1 / r['val']).mean():.3f}" for r in results]))
    rows.append(("median rank", [f"{int(np.median(r['val'])):,}" for r in results]))
    rows.append(("R@5 by mode", [""] * len(results)))
    for mode in modes:
        mask = (val["mode"] == mode).to_numpy()
        rows.append((f"  {mode} ({mask.sum():,})", [pct((r["val"][mask] <= 5).mean()) for r in results]))
    rows.append(("nonsense -> absurd #1", [pct(r["absurd_top1"]) for r in results]))
    rows.append(("R@5, held-out only", [pct((r["held"] <= 5).mean()) for r in results]))
    if results[0]["train"] is not None:
        rows.append(("R@5, train sample", [pct((r["train"] <= 5).mean()) for r in results]))
    if len(results) > 1:
        first = (results[0]["val"] <= 5).astype(float)
        cells = ["-"]
        for r in results[1:]:
            diff, lo, hi = cluster_bootstrap((r["val"] <= 5) - first, val["quote_id"])
            cells.append(f"{100 * diff:+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f})")
        rows.append((f"R@5 vs {labels[0]}, pts", cells))

    label_w = max(len(label) for label, _ in rows) + 2
    col_w = [max(len(labels[i]), *(len(cells[i]) for _, cells in rows)) + 2 for i in range(len(labels))]
    print("".ljust(label_w) + "".join(lab.rjust(w) for lab, w in zip(labels, col_w)))
    for label, cells in rows:
        print(label.ljust(label_w) + "".join(c.rjust(w) for c, w in zip(cells, col_w)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", help="model dir or HF name, or 'base'; repeat to compare")
    parser.add_argument("--train-sample", type=int, default=4000, help="training pairs to rank (0 to skip)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    names = args.model or ["models/v1"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quotes = pd.read_parquet(DATA_PATH)
    val = pd.read_parquet(VAL_PATH).reset_index(drop=True)
    train_all = pd.read_parquet(TRAIN_PATH)
    absurd_ids = set(pd.concat([train_all, val]).query("mode == 'nonsense'")["quote_id"])

    missing = ~val["quote_id"].isin(quotes["quote_id"])
    if missing.any():
        print(f"dropping {missing.sum()} val pairs whose quote is not in {DATA_PATH}")
        val = val[~missing].reset_index(drop=True)

    train = None
    if args.train_sample > 0:
        n = min(args.train_sample, len(train_all))
        train = train_all.sample(n=n, random_state=args.seed).reset_index(drop=True)
        train = train[train["quote_id"].isin(quotes["quote_id"])].reset_index(drop=True)

    results = [evaluate(name, quotes, val, train, absurd_ids, device) for name in names]

    print()
    print(f"val: {len(val):,} messages from {val['quote_id'].nunique():,} held-out quotes, "
          f"ranked against {len(quotes):,} quotes")
    if train is not None:
        print(f"train sample: {len(train):,} of {len(train_all):,} pairs (seed {args.seed})")
    print(f"known absurd quotes: {len(absurd_ids):,} ({pct(len(absurd_ids) / len(quotes))} of the corpus)")
    print()
    print_table(names, results, val)


if __name__ == "__main__":
    main()
