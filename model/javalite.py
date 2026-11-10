from dataclasses import dataclass

import torch
from model.layers import Block
import torch.nn as nn
import math
from torch.nn import functional as F

@dataclass
class Config:
    vocab_size: int = 16384
    block_size: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.0


class JavaLite(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer  = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe  = nn.Embedding(config.block_size, config.n_embd), 
                drop = nn.Dropout(config.dropout),
                hidden = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size , bias=False)
        self.transformer.wte.weight =  self.lm_head.weight 
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(parameter, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std =0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std =0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, idx, targets = None):
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.hidden:
            x = block(x)
        
        logits = self.lm_head(self.transformer.ln_f(x))

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, stop_id=None):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.config.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim = -1), 1)
            idx = torch.cat([idx, nxt], dim = 1)
            if stop_id is not None and nxt.item() == stop_id:
                break
        return idx
