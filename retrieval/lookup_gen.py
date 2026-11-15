import argparse
import hashlib
import os

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

DATA_PATH = "data/processed/quotes.parquet"
INDEX_NAME = "quote_index.pt"


def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)

    counts = mask.sum(dim=1)
    counts = torch.clamp(counts, min=1e-9)

    mean = summed / counts
    return mean / mean.norm(dim=1, keepdim=True)


def encode(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return mean_pool(outputs.last_hidden_state, inputs["attention_mask"])


def model_hash(model_dir):
    h = hashlib.sha256()
    with open(os.path.join(model_dir, "model.safetensors"), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_index(model, tokenizer, data, out_path, batch_size=256):
    model.eval()
    chunks = []
    for i in range(0, len(data), batch_size):
        batch = data["text"].iloc[i:i + batch_size].tolist()
        chunks.append(encode(batch, model, tokenizer).cpu())

    embeddings = torch.cat(chunks)
    quote_ids = data["quote_id"].tolist()
    assert embeddings.shape[0] == len(quote_ids)

    torch.save(
        {
            "embeddings": embeddings,
            "quote_ids": quote_ids,
            "model_hash": model_hash(model.name_or_path),
        },
        out_path,
    )
    print(f"saved {tuple(embeddings.shape)} to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/v1")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = pd.read_parquet(DATA_PATH)
    model = AutoModel.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    build_index(model, tokenizer, data, os.path.join(args.model, INDEX_NAME))