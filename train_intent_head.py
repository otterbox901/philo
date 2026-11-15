import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from build_intent_features import hash_text

train_path = "data/processed/pairs_train.parquet"
val_path = "data/processed/pairs_val.parquet"
features_path = "models/intent_features.pt"
out_path = "models/intent_head.pt"
modes = ["question", "advice", "statement", "nonsense"]
routes = ["answer", "resonate", "absurd"]
mode_to_route = torch.tensor([0, 0, 1, 2])  # question, advice -> answer; statement -> resonate; nonsense -> absurd


class IntentHead(nn.Module):
    def __init__(self, dim, n_modes, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_modes))

    def forward(self, x):
        return self.net(x)


def load_split(path, features, split):
    pairs = pd.read_parquet(path)
    if hash_text(pairs["input"].tolist()) != features[f"{split}_hash"]:
        raise SystemExit(f"{path} changed after {features_path} was built: rerun build_intent_features.py")
    x = (features[split].float() - features["mean"]) / features["std"]
    y = torch.tensor(pairs["mode"].map(modes.index).to_numpy())
    return x, y


def predict(head, x, batch_size=4096):
    head.eval()
    with torch.no_grad():
        return torch.cat([F.softmax(head(x[i:i + batch_size]), dim=1) for i in range(0, len(x), batch_size)])


def train_head(train_x, train_y, val_x, val_y, epochs=4, batch_size=256, lr=1e-3, weight_decay=0.01):
    torch.manual_seed(0)
    head = IntentHead(train_x.shape[1], len(modes))
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        head.train()
        order = torch.randperm(len(train_x))
        total = 0.0
        for i in range(0, len(train_x), batch_size):
            idx = order[i:i + batch_size]
            loss = F.cross_entropy(head(train_x[idx]), train_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        val_acc = (predict(head, val_x).argmax(dim=1) == val_y).float().mean().item()
        print(f"epoch {epoch + 1}: train loss {total / len(train_x):.3f}, val accuracy {val_acc:.1%}")
    return head


def report(probs, y):
    pred = probs.argmax(dim=1)
    print(f"\nval accuracy, 4 modes: {(pred == y).float().mean():.1%}\n")

    true = pd.Series([modes[i] for i in y], name="true")
    guess = pd.Series([modes[i] for i in pred], name="predicted")
    table = pd.crosstab(true, guess, normalize="index").reindex(index=modes, columns=modes, fill_value=0)
    print("share of each true mode predicted as (rows sum to 100%):")
    print((100 * table).round(1).to_string(), "\n")

    route_probs = torch.zeros(len(probs), len(routes)).index_add_(1, mode_to_route, probs)
    confidence, route_pred = route_probs.max(dim=1)
    correct = route_pred == mode_to_route[y]
    print(f"val accuracy, 3 routes: {correct.float().mean():.1%}\n")
    print("route confidence   share of messages   accuracy")
    for threshold in (0.7, 0.8, 0.9, 0.95):
        sure = confidence >= threshold
        print(f"  >= {threshold:<4}  {sure.float().mean():>22.1%}  {correct[sure].float().mean():>9.1%}")


def main():
    features = torch.load(features_path, weights_only=True)
    train_x, train_y = load_split(train_path, features, "train")
    val_x, val_y = load_split(val_path, features, "val")

    head = train_head(train_x, train_y, val_x, val_y)
    report(predict(head, val_x), val_y)

    torch.save({
        "state_dict": head.state_dict(), "modes": modes, "routes": routes,
        "mode_to_route": mode_to_route,
        "mean": features["mean"], "std": features["std"],
        "model": features["model"], "template": features["template"],
    }, out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
