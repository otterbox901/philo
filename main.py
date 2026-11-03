import argparse
import os
import random

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from lookup_gen import DATA_PATH, INDEX_NAME, build_index, encode, model_hash

MAX_CHARS = 1000

class QuoteMatcher:
    def __init__(self, model_dir="models/v1", data_path=DATA_PATH, device="cpu"):
        if torch.cuda.is_available(): device="cuda"

        self.model = AutoModel.from_pretrained(model_dir).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        data = pd.read_parquet(data_path)

        index = self._load_or_rebuild(model_dir, data)

        self.embeddings = index["embeddings"].to(self.model.device)
        self.quote_ids = index["quote_ids"]

        self.quotes = data.set_index("quote_id")
        assert self.quotes.index.is_unique, "quote_id must be unique"

    def _load_or_rebuild(self, model_dir, data):
        index_path = os.path.join(model_dir, INDEX_NAME)
        reason = self._stale_reason(index_path, model_dir, data)
        if reason:
            print(f"rebuilding quote table ({reason})...")
            build_index(self.model, self.tokenizer, data, index_path)
        return torch.load(index_path, weights_only=True)

    @staticmethod
    def _stale_reason(index_path, model_dir, data):
        if not os.path.exists(index_path):
            return "no table yet"
        index = torch.load(index_path, weights_only=True)
        if index.get("model_hash") != model_hash(model_dir):
            return "model was retrained"
        if index["quote_ids"] != data["quote_id"].tolist():
            return "quotes changed"
        return None

    def match(self, text, k=1, pool=5):
        text = text.strip()
        if not text:
            return []
        if len(text) > MAX_CHARS:
            raise ValueError(f"input is {len(text)} characters; the limit is 1000")

        q = encode([text], self.model, self.tokenizer)[0]
        k = min(k, len(self.quote_ids))
        scores, rows = torch.topk(self.embeddings @ q, pool)
        random_index=random.sample(range(pool), k)
        scores=scores[random_index]
        rows= rows[random_index]

        results = []
        for score, row in zip(scores.tolist(), rows.tolist()):
            quote = self.quotes.loc[self.quote_ids[row]]
            results.append((quote["text"], quote["author"], score))
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/v1")
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    matcher = QuoteMatcher(args.model)
    print("Type something (quit to exit).")

    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() == "quit":
            break

        try:
            results = matcher.match(text, k=args.k)
        except ValueError as e:
            print(e)
            continue

        for quote, author, score in results:
            print(f'"{quote}" - {author}  ({score:.3f})')


if __name__ == "__main__":
    main()