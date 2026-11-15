"""Step 5: turn the saved responses into clean training pairs, then split train/val.

Run:  python -m gen.filter

Combines every generation round from 1 to config.ROUND. Each round is split by quote on
its own, so adding a round never moves an earlier round's quote between train and val:
models trained before the new round can still be evaluated fairly on the new val set.

Writes data/processed/pairs.parquet, pairs_train.parquet and pairs_val.parquet,
each with columns: input, quote_id, mode, philosopher, round.
"""

import random
import re
from collections import Counter

import pandas as pd

from gen import config
from gen.collect import load_generations

# Common English words that carry little meaning. They are ignored when we measure
# how many of the quote's words an input reuses. Contractions are split on the
# apostrophe by the tokenizer, so their pieces ("don", "t", "ll") are listed too.
STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself no nor not now of off on once only or other ought our ours
ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours yourself yourselves
also get got like one really thing things much many even ever let us may might must shall
s t d ll m re ve don doesn didn isn wasn aren weren won wouldn couldn shouldn hasn haven hadn
""".split())


def content_words(text):
    """The set of lowercase words in text, minus stopwords and single letters."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def overlap_ratio(input_text, quote_text):
    """Fraction of the quote's content words that also appear in the input (0.0 to 1.0)."""
    quote_words = content_words(quote_text)
    if not quote_words:
        return 0.0
    return len(quote_words & content_words(input_text)) / len(quote_words)


def normalize_for_dedup(text):
    """Lowercase and collapse whitespace, so "Hi  there" and "hi there" count as the same."""
    return " ".join(text.lower().split())


def pairs_from_entries(entries):
    """One row per generated input, for quotes the model said fit the mode."""
    return [
        {"input": text, "quote_id": entry["quote_id"], "mode": entry["mode"], "round": entry.get("round", 1)}
        for entry in entries
        if entry["fits"]
        for text in entry["inputs"]
    ]


def apply_filters(rows, quote_text_by_id):
    """Drop bad inputs. Returns (kept_rows, drop_counts).

    Each dropped input is counted under the first filter that caught it, in this order:
    empty -> too_long -> overlap -> duplicate. For duplicates the first copy is kept.
    """
    dropped = Counter({"empty": 0, "too_long": 0, "overlap": 0, "duplicate": 0})
    seen = set()  # (mode, normalized input) pairs we have already kept
    kept = []
    for row in rows:
        text = row["input"].strip()
        if not text:
            dropped["empty"] += 1
        elif len(text) > config.MAX_INPUT_CHARS:
            dropped["too_long"] += 1
        elif overlap_ratio(text, quote_text_by_id[row["quote_id"]]) > config.MAX_OVERLAP:
            dropped["overlap"] += 1
        elif (row["mode"], normalize_for_dedup(text)) in seen:
            dropped["duplicate"] += 1
        else:
            seen.add((row["mode"], normalize_for_dedup(text)))
            kept.append({**row, "input": text})
    return kept, dropped


def split_by_quote(pairs, val_fraction, seed):
    """Split pairs into (train, val) so that all pairs of a quote land in the same split."""
    quote_ids = sorted(pairs["quote_id"].unique())  # sort first so the shuffle is reproducible
    random.Random(seed).shuffle(quote_ids)
    n_val = round(len(quote_ids) * val_fraction)
    val_ids = set(quote_ids[:n_val])
    is_val = pairs["quote_id"].isin(val_ids)
    return pairs[~is_val].reset_index(drop=True), pairs[is_val].reset_index(drop=True)


def split_each_round(pairs, val_fraction, seed):
    """split_by_quote applied to each round separately, so a new round can't reshuffle old ones."""
    splits = [split_by_quote(pairs[pairs["round"] == n], val_fraction, seed) for n in sorted(pairs["round"].unique())]
    train = pd.concat([t for t, _ in splits], ignore_index=True)
    val = pd.concat([v for _, v in splits], ignore_index=True)
    return train, val


def load_all_rounds(last_round):
    """Entries from rounds 1 .. last_round, in round order, each tagged with its round."""
    entries, n_unparseable = [], 0
    for n in range(1, last_round + 1):
        raw_dir = config.round_dir(n) / "raw"
        round_entries, bad = load_generations(raw_dir)
        if not round_entries:
            print(f"Round {n}: no results in {raw_dir} yet, skipping it")
        entries += [{**entry, "round": n} for entry in round_entries]
        n_unparseable += bad
    return entries, n_unparseable


def main():
    quotes = pd.read_parquet(config.QUOTES_PATH)
    quote_text_by_id = dict(zip(quotes["quote_id"], quotes["text"]))
    author_by_id = dict(zip(quotes["quote_id"], quotes["author"]))

    # Round order matters: when a later round repeats an input, the earlier copy is the one kept.
    entries, n_unparseable = load_all_rounds(config.ROUND)
    if not entries:
        raise SystemExit("No results in any round. Run gen.submit and gen.collect first.")
    kept, dropped = apply_filters(pairs_from_entries(entries), quote_text_by_id)

    pairs = pd.DataFrame(kept, columns=["input", "quote_id", "mode", "round"])
    pairs["philosopher"] = pairs["quote_id"].map(author_by_id)
    pairs = pairs[["input", "quote_id", "mode", "philosopher", "round"]]
    train, val = split_each_round(pairs, config.VAL_FRACTION, config.SEED)

    pairs.to_parquet(config.PAIRS_PATH, index=False)
    train.to_parquet(config.TRAIN_PATH, index=False)
    val.to_parquet(config.VAL_PATH, index=False)

    # --- Summary -----------------------------------------------------------------
    fit_table = pd.DataFrame(entries).groupby("mode")["fits"].agg(quotes="size", fit="sum")
    fit_table["fit_rate"] = (fit_table["fit"] / fit_table["quotes"]).round(3)
    print("Quotes judged per mode:")
    print(fit_table.to_string(), "\n")
    if n_unparseable:
        print(f"Skipped {n_unparseable} unparseable saved responses\n")

    print(f"Inputs generated: {sum(dropped.values()) + len(kept)}")
    print("Dropped by filter:")
    for name, count in dropped.items():
        print(f"  {name:<10} {count}")
    print(f"Kept: {len(kept)}\n")

    by_split = pd.concat([train.assign(split="train"), val.assign(split="val")])
    table = pd.crosstab(by_split["mode"], by_split["split"], margins=True, margins_name="total")
    print("Pairs per mode and split:")
    print(table.to_string(), "\n")
    per_round = pd.crosstab(by_split["round"], by_split["split"], margins=True, margins_name="total")
    print("Pairs per round and split:")
    print(per_round.to_string(), "\n")
    print(f"Quotes: {train['quote_id'].nunique()} train, {val['quote_id'].nunique()} val")
    print(f"Wrote {config.PAIRS_PATH}, {config.TRAIN_PATH.name}, {config.VAL_PATH.name}")


if __name__ == "__main__":
    main()
