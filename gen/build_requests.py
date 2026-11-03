"""Step 1: sample quotes and write one API request per (mode, group of 8 quotes).

Run:  python -m gen.build_requests

Writes:
  data/gen/sample_ids.txt  - the 4,000 sampled quote_ids (reused on later runs)
  data/gen/requests.jsonl  - 2,000 requests, one JSON object per line
"""

import json
import random

import pandas as pd

from gen import config
from gen.prompts import MODES, NONSENSE_THEMES, SYSTEM_PROMPT, build_user_message


def normalize_text(text):
    """Lowercase and collapse whitespace, so small formatting differences don't matter."""
    return " ".join(str(text).lower().split())


def eligible_quotes(quotes, min_words, max_words):
    """Keep quotes with between min_words and max_words words (inclusive)."""
    word_counts = quotes["text"].map(lambda text: len(text.split()))
    return quotes[(word_counts >= min_words) & (word_counts <= max_words)]


def load_eval_quotes(eval_dir):
    """Collect the quote_ids and (normalized) quote texts of every quote in eval_dir.

    The eval folder's format isn't fixed yet, so this reads any .parquet/.csv/.jsonl
    file and uses a "quote_id" column and/or a "text" or "quote" column.
    """
    eval_ids, eval_texts = set(), set()
    data_files = [p for p in sorted(eval_dir.rglob("*")) if p.suffix in (".parquet", ".csv", ".jsonl")]
    for path in data_files:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_json(path, lines=True)

        found_column = False
        if "quote_id" in df.columns:
            eval_ids.update(df["quote_id"].astype(str))
            found_column = True
        for column in ("text", "quote"):
            if column in df.columns:
                eval_texts.update(df[column].map(normalize_text))
                found_column = True
        if not found_column:
            # Stop instead of guessing: a silent miss would leak eval quotes into training.
            raise ValueError(f"{path} has no quote_id, text or quote column; can't tell which quotes it holds")

    if not data_files:
        print(f"Warning: {eval_dir} exists but holds no .parquet/.csv/.jsonl files; nothing excluded.")
    return eval_ids, eval_texts


def exclude_eval(quotes, eval_ids, eval_texts):
    """Drop quotes whose id or text appears in the eval set."""
    in_eval = quotes["quote_id"].isin(eval_ids) | quotes["text"].map(normalize_text).isin(eval_texts)
    return quotes[~in_eval]


def sample_quote_ids(quotes, n, seed):
    """Randomly pick n quote_ids with a fixed seed."""
    # Sort first, so the sample depends only on which quotes exist, not on their row order.
    ordered = quotes.sort_values("quote_id")
    return ordered.sample(n=n, random_state=seed)["quote_id"].tolist()


def chunk(items, size):
    """Split a list into consecutive groups of `size` (the last group may be smaller)."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def nonsense_themes(batch_index):
    """The themes for one nonsense request. Seeded per request, so rebuilding gives the same ones."""
    rng = random.Random(f"{config.SEED}-{batch_index}")
    return rng.sample(NONSENSE_THEMES, config.NONSENSE_THEMES_PER_REQUEST)


def make_params(mode, quotes, themes=None):
    """The Messages API parameters for one request. quotes: list of (quote_id, text)."""
    return {
        "model": config.MODEL,
        "max_tokens": config.MAX_TOKENS,
        "output_config": {"effort": config.EFFORT},
        # The system prompt is identical in every request, so mark it for caching:
        # later requests read it from the cache at a fraction of the price.
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral", "ttl": config.CACHE_TTL},
            }
        ],
        "messages": [{"role": "user", "content": build_user_message(mode, quotes, themes)}],
    }


def build_requests(quote_ids, text_by_id):
    """One request per (mode, group of QUOTES_PER_REQUEST quotes).

    custom_id is "{mode}-{batch_index}", e.g. "advice-12". Group 12 holds the same
    8 quotes in every mode. We also store the quote_ids so results can be matched back.
    """
    groups = chunk(quote_ids, config.QUOTES_PER_REQUEST)
    requests = []
    for mode in MODES:
        for batch_index, group in enumerate(groups):
            quotes = [(quote_id, text_by_id[quote_id]) for quote_id in group]
            themes = nonsense_themes(batch_index) if mode == "nonsense" else None
            requests.append(
                {
                    "custom_id": f"{mode}-{batch_index}",
                    "quote_ids": group,
                    "params": make_params(mode, quotes, themes),
                }
            )
    return requests


def load_or_create_sample(quotes):
    """Return the sampled quote_ids, reusing data/gen/sample_ids.txt if it already exists."""
    if config.SAMPLE_IDS_PATH.exists():
        quote_ids = config.SAMPLE_IDS_PATH.read_text().split()
        missing = set(quote_ids) - set(quotes["quote_id"])
        if missing:
            raise ValueError(f"{len(missing)} ids in {config.SAMPLE_IDS_PATH} are no longer in quotes.parquet")
        if config.EVAL_DIR.exists():
            # data/eval may have been created after the sample was saved: make sure it doesn't overlap.
            sampled = quotes[quotes["quote_id"].isin(quote_ids)]
            leaked = len(sampled) - len(exclude_eval(sampled, *load_eval_quotes(config.EVAL_DIR)))
            if leaked:
                raise ValueError(f"{leaked} sampled quotes are in {config.EVAL_DIR}. "
                                 f"Delete {config.SAMPLE_IDS_PATH} to resample.")
        print(f"Reusing {len(quote_ids)} sampled ids from {config.SAMPLE_IDS_PATH}")
        return quote_ids

    print(f"Quotes in corpus:            {len(quotes)}")
    pool = eligible_quotes(quotes, config.MIN_WORDS, config.MAX_WORDS)
    print(f"With {config.MIN_WORDS}-{config.MAX_WORDS} words:           {len(pool)}")
    if config.EVAL_DIR.exists():
        eval_ids, eval_texts = load_eval_quotes(config.EVAL_DIR)
        before = len(pool)
        pool = exclude_eval(pool, eval_ids, eval_texts)
        print(f"After excluding eval quotes: {len(pool)} ({before - len(pool)} removed)")
    else:
        print(f"No {config.EVAL_DIR} folder, so no eval quotes to exclude")

    quote_ids = sample_quote_ids(pool, config.N_QUOTES, config.SEED)
    config.GEN_DIR.mkdir(parents=True, exist_ok=True)
    config.SAMPLE_IDS_PATH.write_text("\n".join(quote_ids) + "\n")
    print(f"Sampled {len(quote_ids)} ids (seed {config.SEED}) -> {config.SAMPLE_IDS_PATH}")
    return quote_ids


def main():
    if config.BATCHES_PATH.exists():
        # Retries re-send lines from requests.jsonl, so it must match what was submitted.
        raise SystemExit(f"{config.BATCHES_PATH} exists: a batch was already submitted. Not overwriting requests.")

    quotes = pd.read_parquet(config.QUOTES_PATH)
    quote_ids = load_or_create_sample(quotes)
    text_by_id = dict(zip(quotes["quote_id"], quotes["text"]))

    requests = build_requests(quote_ids, text_by_id)
    with open(config.REQUESTS_PATH, "w") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    print(f"Wrote {len(requests)} requests ({len(MODES)} modes x {len(requests) // len(MODES)}) -> {config.REQUESTS_PATH}")


if __name__ == "__main__":
    main()
