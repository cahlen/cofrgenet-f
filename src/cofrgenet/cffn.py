"""Continued Fraction FFN (Cffn) — replaces standard Transformer FFN.

Architecture:
    y = U·x + V·z
    z_j = a_0^(j) + f̃(a_1^(j), ..., a_d^(j))  for j = 1, ..., L

where each a^(j) = W^(j) · x, and f̃ is the continued fraction computed
via continuant polynomials.

Parameter count: L*p*(d+1) + 2p² + L*p  (vs standard FFN: 2*α*p² where α=4)
"""

import torch
import torch.nn as nn
from .continuant import continued_fraction


class Cffn(nn.Module):
    """Continued Fraction FFN — replaces standard Transformer FFN.

    Args:
        dim: Hidden dimension (p).
        num_ladders: Number of CF ladders (L).
        depth: Continued fraction depth (d).
        epsilon: Pole avoidance threshold.
    """

    def __init__(self, dim, num_ladders=3, depth=5, epsilon=0.01):
        super().__init__()
        self.dim = dim
        self.num_ladders = num_ladders
        self.depth = depth
        self.epsilon = epsilon

        # Direct linear path: U (p -> p)
        self.U = nn.Linear(dim, dim, bias=False)

        # Gating projection: G (p -> p)
        # Paper: "input to the ladders is a gated non-expanded representation"
        # gated_x = sigmoid(G·x) ⊙ x
        self.gate_proj = nn.Linear(dim, dim, bias=False)

        # Ladder weight matrices: W^(j) maps input to (d+1) partial denominators
        # Row 0 is a_0 (linear term), rows 1..d are a_1..a_d (fraction terms)
        self.ladder_weights = nn.ModuleList([
            nn.Linear(dim, depth + 1, bias=False)
            for _ in range(num_ladders)
        ])

        # Combination layer: V (L -> p)
        self.V = nn.Linear(num_ladders, dim, bias=False)

        # Track active depth for dyadic schedule
        self._active_depth = depth
        # Register hooks for depth-based gradient masking
        self._grad_hooks = []

    def set_active_depth(self, active_depth):
        """Set the active depth for dyadic training schedule.

        Args:
            active_depth: Maximum depth level with active gradients.
                0 = only linear components (U, V, a_0 row of each ladder)
                1..d = include fraction depths up to this level
        """
        self._active_depth = active_depth
        # Remove old hooks
        for hook in self._grad_hooks:
            hook.remove()
        self._grad_hooks = []

        if active_depth >= self.depth:
            return  # All depths active, no masking needed

        # Install gradient hooks to zero out frozen depth rows
        for lw in self.ladder_weights:
            def make_hook(layer, max_active):
                def hook(grad):
                    mask = torch.zeros_like(grad)
                    # Row 0 (a_0) is always active
                    mask[0, :] = 1.0
                    # Rows 1..max_active are active (a_1..a_{max_active})
                    if max_active > 0:
                        mask[1:max_active + 1, :] = 1.0
                    return grad * mask
                return hook
            h = lw.weight.register_hook(make_hook(lw, active_depth))
            self._grad_hooks.append(h)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, seq_len, dim)

        Returns:
            (batch, seq_len, dim)
        """
        # Direct linear path (uses raw x)
        linear_out = self.U(x)  # (batch, seq_len, dim)

        # Gated input for ladders: sigmoid(G·x) ⊙ x
        gated_x = torch.sigmoid(self.gate_proj(x)) * x  # (batch, seq_len, dim)

        # Continued fraction ladders (use gated input)
        ladder_outputs = []
        for j in range(self.num_ladders):
            a = self.ladder_weights[j](gated_x)  # (batch, seq_len, d+1)
            # a_0 is the linear term, a_1..a_d form the continued fraction
            a_0 = a[..., 0]                # (batch, seq_len)
            a_cf = a[..., 1:]              # (batch, seq_len, d)
            z_j = a_0 + continued_fraction(a_cf, self.epsilon)  # (batch, seq_len)
            ladder_outputs.append(z_j)

        z = torch.stack(ladder_outputs, dim=-1)  # (batch, seq_len, L)
        combined = self.V(z)                       # (batch, seq_len, dim)

        return linear_out + combined
