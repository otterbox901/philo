import torch
import pandas as pd
from collections import deque
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

device = "cuda" if torch.cuda.is_available() else "cpu"
name= "sentence-transformers/all-MiniLM-L6-v2"
tokenizer= AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name).to(device)

def mean_pool(hidden, mask):
    mask= mask.unsqueeze(-1)
    summed= (hidden*mask).sum(dim=1)

    counts= mask.sum(dim=1)
    counts= torch.clamp(counts, min=1e-9)

    mean = summed/counts
    return mean/mean.norm(dim=1, keepdim=True)

def encode_train(texts):
    inputs= tokenizer(texts, return_tensors= "pt", padding= True, truncation= True, max_length= 256).to(device)
    outputs= model(**inputs)
    return mean_pool(outputs.last_hidden_state, inputs['attention_mask'])

def make_batches(df, batch_size= 64):
    queue = deque(df.sample(frac=1).itertuples(index=False))
    while len(queue) >= batch_size:
        batch, seen, skipped = [], set(), []
        while queue and len(batch)<batch_size:
            r=queue.popleft()
            if r.quote_id in seen:
                skipped.append(r)

            else:
                batch.append(r)
                seen.add(r.quote_id)
        queue.extendleft(reversed(skipped))
        if len(batch) < batch_size:
            break
        yield [r.input for r in batch], [r.text for r in batch]



quotes= pd.read_parquet("data/processed/quotes.parquet")
train = pd.read_parquet("data/processed/pairs_train.parquet")
train= train.merge(quotes[["quote_id", "text"]], on = "quote_id")

batch_size, temperature = 64, 0.05
total_steps= len(train) // batch_size
loss_function= torch.nn.CrossEntropyLoss()
optimizer= torch.optim.AdamW(model.parameters(), lr = 2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1*total_steps), total_steps)

#start training

model.train()
for step, (inputs, quote_texts) in enumerate(make_batches(train, batch_size)):
    optimizer.zero_grad()
    q= encode_train(inputs)
    d= encode_train(quote_texts)
    scores = q @ d.T / temperature

    labels = torch.arange( len(inputs), device= device)
    loss= loss_function(scores, labels)

    loss.backward()
    optimizer.step()
    scheduler.step()

    if step%50==0:
        print(step, round(loss.item(),3))

model.save_pretrained("models/v1")
tokenizer.save_pretrained("models/v1")
















