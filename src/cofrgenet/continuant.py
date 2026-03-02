"""Continuant-based continued fraction computation with custom backward.

The continued fraction f̃(a_1, ..., a_d) is computed via continuant polynomials K:
    K_0 = 1
    K_1(a_d) = a_d
    K_k(a_{d-k+1}, ..., a_d) = a_{d-k+1} * K_{k-1} + K_{k-2}

    f̃ = K_{d-1} / K_d

Custom backward uses Proposition 1 from the CoFrGeNet paper:
    ∂f̃/∂a_k = (-1)^k * [K_{d-k} / K_d]²

This reduces divisions from d (naive) to 1 (compute 1/K_d once).
"""

import torch
from torch.autograd import Function


class ContinuedFractionFunction(Function):
    """Custom autograd for continued fraction via continuants."""

    @staticmethod
    def forward(ctx, a, epsilon=0.01):
        # a: (..., d) where d is the continued fraction depth
        # Returns: (...,) scalar output per element
        d = a.shape[-1]

        # Compute all continuants K_0 through K_d
        # K[i] = K_i(a_{d-i+1}, ..., a_d)
        K = [None] * (d + 1)
        K[0] = torch.ones(a.shape[:-1], dtype=a.dtype, device=a.device)
        K[1] = a[..., -1]  # K_1 = a_d
        for i in range(2, d + 1):
            # K_i = a_{d-i+1} * K_{i-1} + K_{i-2}
            # a_{d-i+1} is at index (d - i) in 0-indexed a
            K[i] = a[..., d - i] * K[i - 1] + K[i - 2]

        # Pole avoidance on K_d
        K_d = K[d]
        K_d_safe = torch.sign(K_d) * torch.clamp(K_d.abs(), min=epsilon)
        # Handle exact zero: sign(0)=0 would give 0, so clamp the result
        K_d_safe = torch.where(K_d_safe == 0, torch.full_like(K_d_safe, epsilon), K_d_safe)

        # f̃ = K_{d-1} / K_d
        result = K[d - 1] / K_d_safe

        # Save for backward
        ctx.save_for_backward(a, K_d_safe)
        ctx.K = K
        ctx.epsilon = epsilon

        return result

    @staticmethod
    def backward(ctx, grad_output):
        a, K_d_safe = ctx.saved_tensors
        K = ctx.K
        d = a.shape[-1]

        # ∂f̃/∂a_k = (-1)^k * [K_{d-k} / K_d]² for k=1,...,d
        inv_K_d_sq = (1.0 / K_d_safe) ** 2

        grad_a = torch.zeros_like(a)
        for k in range(1, d + 1):
            sign = 1.0 if k % 2 == 0 else -1.0
            K_d_minus_k = K[d - k]
            grad_a[..., k - 1] = sign * K_d_minus_k ** 2 * inv_K_d_sq

        grad_a = grad_a * grad_output.unsqueeze(-1)
        return grad_a, None  # None for epsilon


def continued_fraction(a, epsilon=0.01):
    """Compute continued fraction f̃(a) with custom backward.

    Args:
        a: Tensor of shape (..., d) containing partial denominators a_1, ..., a_d.
        epsilon: Minimum absolute value for K_d (pole avoidance).

    Returns:
        Tensor of shape (...,) containing f̃(a_1, ..., a_d).
    """
    return ContinuedFractionFunction.apply(a, epsilon)
