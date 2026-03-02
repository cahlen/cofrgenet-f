"""CoFrGeNet-F model configuration."""

from dataclasses import dataclass


@dataclass
class CoFrGeNetConfig:
    # Transformer
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50257
    dropout: float = 0.0
    bias: bool = False
    # Cffn
    num_ladders: int = 3
    cf_depth: int = 5
    epsilon: float = 0.01
