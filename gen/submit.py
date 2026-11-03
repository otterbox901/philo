"""Step 3 (only after the pilot is approved): submit the full run as one Message Batch.

Run:  python -m gen.submit            # dry run: only shows what would be sent
      python -m gen.submit --confirm  # really submits (this costs money)

Only requests that have never been sent are submitted. Failed requests are
retried by collect.py, not here.
"""

import argparse
import json
from datetime import datetime, timezone

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from gen import config


def load_requests():
    with open(config.REQUESTS_PATH) as f:
        return [json.loads(line) for line in f]


def load_batches():
    """The list of batches submitted so far (empty if none)."""
    if not config.BATCHES_PATH.exists():
        return []
    return json.loads(config.BATCHES_PATH.read_text())


def save_batches(batches):
    config.BATCHES_PATH.write_text(json.dumps(batches, indent=2))


def has_result(custom_id):
    """True if we already saved a successful, parseable response for this request."""
    return (config.RAW_DIR / f"{custom_id}.json").exists()


def submit_batch(client, requests, attempt):
    """Send requests as one batch and record it in batches.json. Returns the batch id."""
    batch = client.messages.batches.create(
        requests=[
            Request(custom_id=r["custom_id"], params=MessageCreateParamsNonStreaming(**r["params"]))
            for r in requests
        ]
    )
    # Record the batch right away, so a crash after this point can't make us submit it twice.
    batches = load_batches()
    batches.append(
        {
            "batch_id": batch.id,
            "attempt": attempt,  # 1 = first try, 2 = the retry
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collected": False,
            "custom_ids": [r["custom_id"] for r in requests],
        }
    )
    save_batches(batches)
    return batch.id


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm", action="store_true", help="actually submit the batch")
    args = parser.parse_args()

    requests = load_requests()
    already_sent = {custom_id for batch in load_batches() for custom_id in batch["custom_ids"]}
    to_send = [r for r in requests if r["custom_id"] not in already_sent and not has_result(r["custom_id"])]

    print(f"{len(requests)} requests in {config.REQUESTS_PATH}")
    print(f"{len(requests) - len(to_send)} already sent or done, {len(to_send)} to send, model {config.MODEL}")
    if not to_send:
        print("Nothing to submit. Run `python -m gen.collect` to download results.")
        return
    if not args.confirm:
        print("Dry run. Re-run with --confirm to submit.")
        return

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    batch_id = submit_batch(client, to_send, attempt=1)
    print(f"Submitted batch {batch_id}. Now run `python -m gen.collect`.")


if __name__ == "__main__":
    main()
