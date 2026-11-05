from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 16384
    block_size: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.0
