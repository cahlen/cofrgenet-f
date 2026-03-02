"""Baseline GPT-2 Small model configuration."""

from dataclasses import dataclass


@dataclass
class BaselineConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50257
    dropout: float = 0.0
    bias: bool = False
    ffn_expansion: int = 4
