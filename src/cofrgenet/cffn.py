"""Continued Fraction FFN (Cffn) — replaces standard Transformer FFN.

Architecture (paper eq. 8):
    y = U·x + V·z
    z_j = f̃(W^(j) ⊙ x)  for j = 1, ..., L

Each ladder is p-variate: W^(j) has shape (p, d) and multiplies element-wise
with the p-dim input, producing p independent continued fractions per ladder.

Parameter count: L*p*d + 2p² + p*L  (= L*p*(d+1) + 2p² per paper formula)
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

        # P-variate ladder weights: W^(j) shape (p, d) — element-wise multiply
        # Each W[i, k] scales x[i] for depth k of the continued fraction
        self.ladder_weights = nn.ParameterList([
            nn.Parameter(torch.empty(dim, depth))
            for _ in range(num_ladders)
        ])
        for w in self.ladder_weights:
            nn.init.normal_(w, std=0.02)

        # Combination weights: V shape (p, L) — per-dimension weighting of ladders
        self.V = nn.Parameter(torch.empty(dim, num_ladders))
        nn.init.normal_(self.V, std=0.02)

        # Track active depth for dyadic schedule
        self._active_depth = depth
        # Register hooks for depth-based gradient masking
        self._grad_hooks = []

    def set_active_depth(self, active_depth):
        """Set the active depth for dyadic training schedule.

        Args:
            active_depth: Maximum depth level with active gradients.
                0 = only linear components (U, V, gate_proj); all ladder columns masked
                1..d = unmask ladder columns 0..active_depth-1
        """
        self._active_depth = active_depth
        # Remove old hooks
        for hook in self._grad_hooks:
            hook.remove()
        self._grad_hooks = []

        if active_depth >= self.depth:
            return  # All depths active, no masking needed

        # Install gradient hooks to zero out frozen depth columns
        for w in self.ladder_weights:
            def make_hook(max_active):
                def hook(grad):
                    mask = torch.zeros_like(grad)
                    # Columns 0..max_active-1 are active (depth 1..max_active)
                    if max_active > 0:
                        mask[:, :max_active] = 1.0
                    return grad * mask
                return hook
            h = w.register_hook(make_hook(active_depth))
            self._grad_hooks.append(h)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, seq_len, dim)

        Returns:
            (batch, seq_len, dim)
        """
        # Direct linear path (uses raw x)
        linear_out = self.U(x)  # (B, S, p)

        # Gated input for ladders: sigmoid(G·x) ⊙ x
        gated_x = torch.sigmoid(self.gate_proj(x)) * x  # (B, S, p)

        # P-variate continued fraction ladders
        ladder_outputs = []
        for j in range(self.num_ladders):
            # Element-wise: a[b,s,i,k] = gated_x[b,s,i] * W[i,k]
            a = gated_x.unsqueeze(-1) * self.ladder_weights[j]  # (B,S,p,1)*(p,d) → (B,S,p,d)
            z_j = continued_fraction(a, self.epsilon)  # (B,S,p,d) → (B,S,p)
            ladder_outputs.append(z_j)

        z = torch.stack(ladder_outputs, dim=-1)  # (B, S, p, L)
        combined = (z * self.V).sum(dim=-1)  # (B, S, p)

        return linear_out + combined
