"""CoFrGeNet-F Transformer — standard Transformer with Cffn replacing FFN.

Identical to the baseline except each block uses a Continued Fraction FFN (Cffn)
instead of the standard 2-layer FFN. Includes dyadic training schedule support.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CoFrGeNetConfig
from .cffn import Cffn
from src.baseline.model import CausalSelfAttention, TransformerBlock


def get_unfrozen_depth(current_step, total_steps, max_depth):
    """Return the maximum depth that should be unfrozen at current_step.

    Dyadic schedule: depth i unfreezes at step (1 - 1/2^i) * total_steps.
    """
    for d in range(max_depth, 0, -1):
        unfreeze_at = total_steps * (1 - 1 / (2 ** d))
        if current_step >= unfreeze_at:
            return d
    return 0


class CoFrGeNetTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                config,
                Cffn(dim=config.n_embd, num_ladders=config.num_ladders,
                     depth=config.cf_depth, epsilon=config.epsilon)
            )
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_active_depth(self, depth):
        """Set the active depth for all Cffn layers (dyadic schedule)."""
        for block in self.blocks:
            block.ffn.set_active_depth(depth)

    def forward(self, idx, targets=None):
        B, S = idx.shape
        pos = torch.arange(0, S, dtype=torch.long, device=idx.device)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
