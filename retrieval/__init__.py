"""Quote retrieval: the fine-tuned MiniLM bi-encoder.

Run from the project root:

    python -m retrieval.train        # fine-tune MiniLM, one checkpoint per epoch in models/v2-e*
    python -m retrieval.eval --model base --model models/v2-e2   # score checkpoints on held-out pairs
    python -m retrieval.main --model models/v2-e2   # terminal version, without intent routing

lookup_gen.py holds the pooling, encoding and quote index; main.py holds QuoteMatcher,
which server.py uses. `python -m retrieval.lookup_gen --model DIR` rebuilds a model's
index by hand, though main.py also rebuilds it automatically when it is out of date.
"""
