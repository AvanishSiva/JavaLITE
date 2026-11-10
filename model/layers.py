import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.n_heads = config.n_heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = config.dropout
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k , v = self.c_attn(x).split(C, dim=-1)
        head_size = C // self.n_heads
        q = q.view(B, T, self.n_heads, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_heads, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, head_size).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )
        out = attn.transpose(1,2).contiguous().view(B, T, C)

        return self.resid_drop(self.c_proj(out))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.expand = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(
            self.c_proj(
                self.gelu(
                    self.expand(x)
                )
            )
        )

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


