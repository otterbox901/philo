import hashlib
import torch
import time
import pandas as pd

from qwen_intent import intent_model
from qwen_intent import encode_intent, load_intent_model, template

train_path= "data/processed/pairs_train.parquet"
val_path= "data/processed/pairs_val.parquet"
out_path= "models/intent_features.pt"

def hash_text(text):
    hash = hashlib.sha256("\n".join(text).encode()).hexdigest()
    return hash

def encode_sorted(model, tokenizer, texts, batch_size= 32):
    order = sorted(range(len(texts)), key=lambda k: len(texts[k]))
    out= torch.empty(len(texts), model.config.hidden_size, dtype=torch.float16)

    start_time = time.time()

    for start in range(0, len(texts), batch_size):
        idx= order[start:start+batch_size]
        out[idx]  =encode_intent(model, tokenizer, [texts[i] for i in idx]).half().cpu()

        if start% (batch_size*200)==0:
            print(f"{start:,}/{len(texts):,} {time.time()-start_time:.0f}s", flush=True)

    return out

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_intent_model(device)

    train = pd.read_parquet(train_path)["input"].tolist()
    val = pd.read_parquet(val_path)["input"].tolist()

    print(f"encoding: {len(train)} train messages")

    train_x = encode_sorted(model, tokenizer, train)
    val_x = encode_sorted(model, tokenizer, val)

    mean= train_x.float().mean(dim=0)
    std= train_x.float().std(dim=0).clamp_min(1e-6)

    torch.save({
        "model": intent_model, "template": template,
        "train": train_x, "val": val_x, "mean": mean, "std": std,
        "train_hash": hash_text(train), "val_hash": hash_text(val),
    }, out_path)

    print(f"saved -> {out_path}")

if "__main__" == __name__:
    main()