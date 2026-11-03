"""Data generation pipeline: (user input -> quote) training pairs.

Run the steps in this order, from the project root:

    python -m gen.build_requests   # 1. sample 4,000 quotes, write data/gen/requests.jsonl
    python -m gen.pilot            # 2. 50-quote pilot with normal API calls; read the report
    python -m gen.submit --confirm # 3. submit the full run as a Message Batch (costs money)
    python -m gen.collect          # 4. wait for the batch, download results, retry failures once
    python -m gen.filter           # 5. filter + split into data/processed/pairs*.parquet

All settings (model, seed, sizes, paths) live in gen/config.py.
"""
