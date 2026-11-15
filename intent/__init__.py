"""Intent routing: a frozen Qwen reads each message, a small trained head predicts its mode.

Run from the project root, in this order:

    python -m intent.build_intent_features   # 1. encode every training message with Qwen (about an hour)
    python -m intent.train_intent_head       # 2. train the head -> models/intent_head.pt

qwen_intent.py loads Qwen and pools its last token; router.py turns the head's mode
probabilities into a route (answer, resonate or absurd) for server.py.
"""
