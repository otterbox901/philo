"""Step 2: a small pilot with normal (non-batch) API calls, to check quality and cost.

Run:  python -m gen.pilot     (after `python -m gen.build_requests`)

Uses the first PILOT_QUOTES sampled quotes in all 4 modes, built exactly like
the full run's requests. Responses are saved to data/gen/pilot/raw/, so
re-running only calls the API for requests that haven't been done yet.
Pilot results are NOT used as training data.
"""

import json
import math
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd

from gen import config
from gen.build_requests import build_requests
from gen.collect import load_generations
from gen.filter import apply_filters, pairs_from_entries
from gen.prompts import MODES

USAGE_FIELDS = ["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"]


def call_one(client, request):
    """Send one request and save the full response. Skips requests already saved."""
    path = config.PILOT_RAW_DIR / f"{request['custom_id']}.json"
    if path.exists():
        return
    try:
        message = client.messages.create(**request["params"])
    except anthropic.APIError as error:  # the SDK already retried rate limits and server errors twice
        print(f"  {request['custom_id']}: API error, not saved: {error}")
        return
    record = {"custom_id": request["custom_id"], "quote_ids": request["quote_ids"], "message": message.to_dict()}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"  {request['custom_id']}: {message.stop_reason}, {message.usage.output_tokens} output tokens")


def run_calls(requests):
    config.PILOT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    todo = [r for r in requests if not (config.PILOT_RAW_DIR / f"{r['custom_id']}.json").exists()]
    print(f"Pilot: {len(requests)} requests, {len(todo)} still to call ({config.MODEL})")
    if not todo:
        return
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    # Send one request alone first so the system prompt gets cached, then the rest 4 at a time.
    call_one(client, todo[0])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda r: call_one(client, r), todo[1:]))


def word_count(text):
    return len(text.split())


def report(requests, quotes):
    quote_text_by_id = dict(zip(quotes["quote_id"], quotes["text"]))
    author_by_id = dict(zip(quotes["quote_id"], quotes["author"]))
    records = [json.loads(p.read_text()) for p in sorted(config.PILOT_RAW_DIR.glob("*.json"))]
    entries, n_unparseable = load_generations(config.PILOT_RAW_DIR)
    print(f"\n{len(records)}/{len(requests)} responses saved, {n_unparseable} unparseable")
    stop_reasons = pd.Series([r["message"]["stop_reason"] for r in records]).value_counts()
    print("stop_reason:", stop_reasons.to_dict())

    # --- Fit rate per mode -------------------------------------------------------
    print("\n=== Fit rate per mode ===")
    entries_df = pd.DataFrame(entries)
    fit = entries_df.groupby("mode")["fits"].agg(quotes="size", fit="sum")
    fit["fit_rate"] = (fit["fit"] / fit["quotes"]).round(2)
    # How often the model followed the length rule (one input < 15 words, one with 40+).
    fitting = entries_df[entries_df["fits"]]
    length_ok = fitting["inputs"].map(
        lambda inputs: any(word_count(s) < 15 for s in inputs) and any(word_count(s) >= 40 for s in inputs)
    )
    fit["length_rule_met"] = length_ok.groupby(fitting["mode"]).mean().round(2)
    fit["inputs_per_fit"] = fitting["inputs"].map(len).groupby(fitting["mode"]).mean().round(1)
    print(fit.reindex(MODES).to_string())

    # --- Token usage ---------------------------------------------------------------
    print("\n=== Actual token usage (from response.usage) ===")
    usage = pd.DataFrame(
        [{"mode": r["custom_id"].rsplit("-", 1)[0], "n_quotes": len(r["quote_ids"]),
          **{f: r["message"]["usage"].get(f) or 0 for f in USAGE_FIELDS}} for r in records]
    )
    by_mode = usage.groupby("mode")[USAGE_FIELDS].sum().reindex(MODES)
    by_mode.loc["total"] = by_mode.sum()
    print(by_mode.to_string())

    # --- Projection ----------------------------------------------------------------
    # Average over the full 8-quote requests only (the pilot's last group per mode is smaller).
    full = usage[usage["n_quotes"] == config.QUOTES_PER_REQUEST]
    n_full_run = len(MODES) * math.ceil(config.N_QUOTES / config.QUOTES_PER_REQUEST)
    per_request = full[USAGE_FIELDS].mean()
    print(f"\n=== Projection for the full run ({n_full_run} requests), "
          f"from the {len(full)} full {config.QUOTES_PER_REQUEST}-quote pilot requests ===")
    for field in USAGE_FIELDS:
        print(f"  {field:<30} {per_request[field]:>8.0f} per request -> {per_request[field] * n_full_run:>12,.0f}")
    total_in = per_request[USAGE_FIELDS[:3]].sum() * n_full_run
    print(f"  {'all input tokens':<30} {'':>8}                {total_in:>12,.0f}")
    print("  (output tokens include the model's thinking. In batches, cache hits are best-effort,")
    print("   so some cache reads may turn into cache writes or plain input.)")

    # --- Filters ---------------------------------------------------------------------
    kept, dropped = apply_filters(pairs_from_entries(entries), quote_text_by_id)
    print(f"\n=== Filters on pilot inputs: kept {len(kept)}, dropped {dict(dropped)} ===")

    # --- Samples to read -------------------------------------------------------------
    rng = random.Random(config.SEED)
    kept_by_mode = defaultdict(list)
    for row in kept:
        kept_by_mode[row["mode"]].append(row)
    for mode in MODES:
        print(f"\n{'=' * 30} {mode}: 10 random pairs {'=' * 30}")
        for row in rng.sample(kept_by_mode[mode], min(10, len(kept_by_mode[mode]))):
            print(f"\nINPUT: {row['input']}")
            print(f"QUOTE: {quote_text_by_id[row['quote_id']]}  ({author_by_id[row['quote_id']]})")
        rejected = [e for e in entries if e["mode"] == mode and not e["fits"]]
        print(f"\n--- {mode}: 3 random rejections (of {len(rejected)}) ---")
        for entry in rng.sample(rejected, min(3, len(rejected))):
            print(f"QUOTE: {quote_text_by_id[entry['quote_id']]}\n  reason: {entry['reason']}")


def main():
    if not config.SAMPLE_IDS_PATH.exists():
        raise SystemExit("Run `python -m gen.build_requests` first.")
    quotes = pd.read_parquet(config.QUOTES_PATH)
    pilot_ids = config.SAMPLE_IDS_PATH.read_text().split()[: config.PILOT_QUOTES]
    requests = build_requests(pilot_ids, dict(zip(quotes["quote_id"], quotes["text"])))
    run_calls(requests)
    report(requests, quotes)


if __name__ == "__main__":
    main()
