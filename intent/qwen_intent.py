import torch
from transformers import AutoTokenizer, AutoModel

intent_model = "Qwen/Qwen2.5-0.5B-Instruct"
template= 'Message:"{}"\n The intent of this message is:'

def load_intent_model(device):

    tokenizer = AutoTokenizer.from_pretrained(intent_model)
    tokenizer.padding_side= "right"

    model = AutoModel.from_pretrained(intent_model, dtype = torch.float32).to(device).eval()
    return model, tokenizer

def pool_last_token(hidden, mask):
    last = mask.sum(dim=1)-1
    rows= torch.arange(hidden.shape[0], device=hidden.device)

    return hidden[rows,last]

def encode_intent(model, tokenizer, texts):
    prompts= [template.format(t) for t in texts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        output = model(**inputs)

    return pool_last_token(output.last_hidden_state, inputs["attention_mask"]).float()

