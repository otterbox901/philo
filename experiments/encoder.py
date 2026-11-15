import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

tokenizer= AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model= AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
ref= SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


model.eval()

def cos(a,b):
    a= a.squeeze()
    b= b.squeeze()
    res= torch.dot(a, b) / (a.norm() * b.norm())
    return res


def mean_pool(hidden, mask):
    mask= mask.unsqueeze(-1)
    summed= (hidden*mask).sum(dim=1)

    counts= mask.sum(dim=1)
    counts= torch.clamp(counts, min=1e-9)

    mean = summed/counts
    return mean/mean.norm(dim=1, keepdim=True)

def encode(texts):
    inputs= tokenizer(texts, return_tensors= "pt", padding= True, truncation= True)
    with torch.no_grad():
        outputs= model(**inputs)
    return mean_pool(outputs.last_hidden_state, inputs['attention_mask'])

data = pd.read_csv("data/processed/quotes_sample.csv")

def match_func():
    user_text = input("Enter Text:")
    user_tensor = encode(user_text)

    best_score=-1.0
    res=""
    for texts in data['text']:
        current_tensor = encode(texts)
        current_score = cos(user_tensor, current_tensor)

        if current_score > best_score:
            best_score = current_score
            res = texts
    print(f"Closest Match (Score: {best_score:.4f}):\n{res}")

match_func()



