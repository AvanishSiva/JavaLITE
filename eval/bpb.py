import math, torch

@torch.no_grad()
def bits_per_byte(score_fn, encode, texts, block = 1024):
    total_nats, total_bytes = 0.0, 0 
    for t in texts:
        ids = encode(t)
        for start in range(0, len(ids)-1, block):
            chunk = ids[start:start+block+1]
            if len(chunk) > 1:
                total_nats += score_fn(chunk)
        total_bytes += len(t.encode("utf-8"))
    return total_nats / total_bytes / math.log(2)

def hf_scorer(model):
    def score(chunk):
        x = torch.tensor([chunk], device=model.device)
        logits = model(x).logits[0, :-1].float()
        return torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum").item()
    return score

def javalite_scorer(model):
    def score(chunk):
        x = torch.tensor([chunk])
        logits, _ = model(x[:, :-1])
        return torch.nn.functional.cross_entropy(logits[0], x[0, 1:], reduction="sum").item()
    return score

 