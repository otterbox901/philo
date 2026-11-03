"""Step 4: wait for the batch, download results, retry failures once.

Run:  python -m gen.collect

Safe to stop (Ctrl+C) and re-run: it picks up where it left off and never
re-sends a request that already has a good result.

Writes:
  data/gen/raw/<custom_id>.json            - each successful, parseable response
  data/gen/raw/failed/<custom_id>.attemptN.json - raw responses of failed attempts
  data/gen/failed.jsonl                    - requests that still failed after the retry
"""

import json
import time
from collections import Counter

import anthropic

from gen import config
from gen.submit import has_result, load_batches, load_requests, save_batches, submit_batch


# --- Parsing the model's answer -----------------------------------------------


class ParseError(ValueError):
    """The model's text couldn't be turned into usable results."""


def mode_of(custom_id):
    """"advice-12" -> "advice"."""
    return custom_id.rsplit("-", 1)[0]


def message_text(message):
    """Join the text blocks of a saved message dict (thinking blocks are skipped)."""
    return "".join(block.get("text", "") for block in message.get("content", []) if block.get("type") == "text")


def extract_json(text):
    """Find the JSON object in the model's text, even if it is wrapped in ```json fences."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ParseError("no JSON object found")
    try:
        # strict=False allows raw newlines inside strings, which models sometimes write.
        return json.loads(text[start : end + 1], strict=False)
    except json.JSONDecodeError as error:
        raise ParseError(f"invalid JSON: {error}") from error


def parse_response(text, expected_quote_ids):
    """Turn the model's text into a list of clean result entries.

    Each entry is {"quote_id", "fits", "reason", "inputs"}. Entries for quote_ids we
    didn't send, repeated entries, and malformed entries are skipped. Raises ParseError
    if nothing usable is left.
    """
    data = extract_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ParseError('missing "results" list')

    expected = set(expected_quote_ids)
    results = {}
    for entry in data["results"]:
        if not isinstance(entry, dict):
            continue
        quote_id = str(entry.get("quote_id", ""))
        if quote_id not in expected or quote_id in results:
            continue
        fits = entry.get("fits")
        if not isinstance(fits, bool):
            continue
        inputs = entry.get("inputs", []) if fits else []
        if not isinstance(inputs, list):
            continue
        results[quote_id] = {
            "quote_id": quote_id,
            "fits": fits,
            "reason": str(entry.get("reason", "")),
            "inputs": [s.strip() for s in inputs if isinstance(s, str)],
        }

    if not results:
        raise ParseError("no valid entries for the quote_ids we sent")
    return [results[q] for q in expected_quote_ids if q in results]  # keep the order we sent


def load_generations(raw_dir):
    """Read every saved response in raw_dir.

    Returns (entries, n_unparseable). entries has one dict per (request, quote):
    custom_id, mode, quote_id, fits, reason, inputs. Files are read in sorted order
    so later steps (like dedup) always see the same order.
    """
    entries, n_unparseable = [], 0
    for path in sorted(raw_dir.glob("*.json")):
        record = json.loads(path.read_text())
        try:
            parsed = parse_response(message_text(record["message"]), record["quote_ids"])
        except ParseError:
            n_unparseable += 1
            continue
        for entry in parsed:
            entries.append({"custom_id": record["custom_id"], "mode": mode_of(record["custom_id"]), **entry})
    return entries, n_unparseable


# --- Talking to the Batches API --------------------------------------------------


def wait_until_ended(client, batch_id):
    """Poll the batch every POLL_SECONDS until it has finished processing."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        c = batch.request_counts
        print(f"  {batch_id}: {batch.processing_status} (processing {c.processing}, "
              f"succeeded {c.succeeded}, errored {c.errored}) - checking again in {config.POLL_SECONDS}s")
        time.sleep(config.POLL_SECONDS)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def download_results(client, batch_info, requests_by_id):
    """Download one finished batch. Saves good results; returns {custom_id: failure reason}."""
    failures = {}
    attempt = batch_info["attempt"]
    for result in client.messages.batches.results(batch_info["batch_id"]):
        custom_id = result.custom_id
        failed_path = config.FAILED_RAW_DIR / f"{custom_id}.attempt{attempt}.json"

        if result.result.type != "succeeded":  # "errored", "canceled" or "expired"
            failures[custom_id] = result.result.type
            save_json(failed_path, result.to_dict())
            continue

        message = result.result.message.to_dict()
        record = {"custom_id": custom_id, "quote_ids": requests_by_id[custom_id]["quote_ids"],
                  "attempt": attempt, "message": message}
        try:
            parse_response(message_text(message), record["quote_ids"])
        except ParseError as error:
            failures[custom_id] = f"unparseable ({error}; stop_reason={message.get('stop_reason')})"
            save_json(failed_path, record)
            continue
        save_json(config.RAW_DIR / f"{custom_id}.json", record)
    return failures


def main():
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    requests_by_id = {r["custom_id"]: r for r in load_requests()}
    if not load_batches():
        raise SystemExit("No batches submitted yet. Run `python -m gen.submit --confirm` first.")

    while True:
        # 1. Download every batch we haven't collected yet.
        batches = load_batches()
        for batch_info in batches:
            if batch_info["collected"]:
                continue
            print(f"Waiting for batch {batch_info['batch_id']} (attempt {batch_info['attempt']}, "
                  f"{len(batch_info['custom_ids'])} requests)")
            wait_until_ended(client, batch_info["batch_id"])
            failures = download_results(client, batch_info, requests_by_id)
            batch_info["collected"] = True
            batch_info["failures"] = failures
            save_batches(batches)
            print(f"  downloaded: {len(batch_info['custom_ids']) - len(failures)} ok, {len(failures)} failed")

        # 2. Retry, once, every request that failed and has only been tried once.
        times_sent = Counter(custom_id for b in batches for custom_id in b["custom_ids"])
        to_retry = [c for c, n in times_sent.items() if n == 1 and not has_result(c)]
        if not to_retry:
            break
        print(f"Retrying {len(to_retry)} failed requests once")
        submit_batch(client, [requests_by_id[c] for c in sorted(to_retry)], attempt=2)

    # 3. Log whatever still failed after the retry.
    last_reason = {}
    for batch_info in batches:
        last_reason.update(batch_info.get("failures", {}))
    still_failed = sorted(c for c in times_sent if not has_result(c))
    with open(config.FAILED_LOG_PATH, "w") as f:
        for custom_id in still_failed:
            f.write(json.dumps({"custom_id": custom_id, "reason": last_reason.get(custom_id, "unknown")}) + "\n")

    n_done = sum(has_result(c) for c in requests_by_id)
    print(f"\nDone: {n_done}/{len(requests_by_id)} requests have results in {config.RAW_DIR}")
    if still_failed:
        print(f"{len(still_failed)} still failed after the retry; see {config.FAILED_LOG_PATH}")
    print("Next: python -m gen.filter")


if __name__ == "__main__":
    main()
