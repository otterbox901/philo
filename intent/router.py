import torch
import torch.nn.functional as F

from intent.qwen_intent import encode_intent, intent_model, load_intent_model, template
from intent.train_intent_head import IntentHead

HEAD_PATH = "models/intent_head.pt"
ABSURD_AT = 0.8


class IntentRouter:
    def __init__(self, head_path=HEAD_PATH, device="cpu"):
        saved = torch.load(head_path, weights_only=True)
        if saved["model"] != intent_model or saved["template"] != template:
            raise ValueError(f"{head_path} was trained with a different Qwen model or template; retrain it")

        self.model, self.tokenizer = load_intent_model(device)
        self.head = IntentHead(saved["mean"].numel(), len(saved["modes"]))
        self.head.load_state_dict(saved["state_dict"])
        self.head.eval()
        self.mean, self.std = saved["mean"], saved["std"]
        self.routes, self.mode_to_route = saved["routes"], saved["mode_to_route"]

    def route_probs(self, text):
        x = encode_intent(self.model, self.tokenizer, [text]).cpu()
        x = (x - self.mean) / self.std
        with torch.no_grad():
            mode_probs = F.softmax(self.head(x), dim=1)[0]
        probs = torch.zeros(len(self.routes)).index_add_(0, self.mode_to_route, mode_probs)
        return dict(zip(self.routes, probs.tolist()))

    def route(self, text):
        probs = self.route_probs(text)
        if probs["absurd"] >= ABSURD_AT:
            return "absurd", probs
        return max(["answer", "resonate"], key=probs.get), probs