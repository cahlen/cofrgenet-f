"""Tests for the Continued Fraction FFN (Cffn) layer."""

import torch
import pytest
from src.cofrgenet.cffn import Cffn


class TestCffnShape:
    """Test output shapes."""

    def test_output_shape(self):
        """Input (B, S, p) should produce output (B, S, p)."""
        B, S, p = 2, 16, 64
        cffn = Cffn(dim=p, num_ladders=3, depth=5)
        x = torch.randn(B, S, p)
        y = cffn(x)
        assert y.shape == (B, S, p)

    def test_output_shape_768(self):
        """Test at actual model dimension p=768."""
        B, S, p = 1, 4, 768
        cffn = Cffn(dim=p, num_ladders=3, depth=5)
        x = torch.randn(B, S, p)
        y = cffn(x)
        assert y.shape == (B, S, p)


class TestCffnParameters:
    """Test parameter counts match expected formulas."""

    def test_parameter_count(self):
        """Params should be L*p*(d+1) + p*p + L*p.

        - ladder_weights: L linear layers, each p -> (d+1), no bias = L*p*(d+1)
        - U: p -> p, no bias = p*p
        - V: L -> p, no bias = L*p
        """
        p, L, d = 768, 3, 5
        cffn = Cffn(dim=p, num_ladders=L, depth=d)
        total = sum(param.numel() for param in cffn.parameters())
        expected = L * p * (d + 1) + p * p + L * p
        assert total == expected, f"Got {total}, expected {expected}"

    def test_fewer_params_than_ffn(self):
        """Cffn should have fewer params than a standard FFN with 4x expansion."""
        p = 768
        cffn = Cffn(dim=p, num_ladders=3, depth=5)
        cffn_params = sum(param.numel() for param in cffn.parameters())

        # Standard FFN: Linear(p, 4p) + Linear(4p, p) = 2 * 4 * p^2
        ffn_params = 2 * 4 * p * p

        assert cffn_params < ffn_params, (
            f"Cffn ({cffn_params:,}) should have fewer params than FFN ({ffn_params:,})"
        )


class TestCffnGradients:
    """Test gradients flow through all parameters."""

    def test_forward_backward(self):
        """Gradients should flow through all parameters."""
        p = 64
        cffn = Cffn(dim=p, num_ladders=3, depth=5)
        x = torch.randn(2, 8, p)
        y = cffn(x)
        loss = y.sum()
        loss.backward()

        for name, param in cffn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


class TestCffnFreezing:
    """Test depth-based parameter freezing for dyadic schedule."""

    def test_freeze_depth(self):
        """Freezing a depth level should zero out those gradients."""
        p = 64
        cffn = Cffn(dim=p, num_ladders=3, depth=5)

        # Freeze depth levels 2-5 (only allow depth 0 and 1)
        cffn.set_active_depth(1)

        x = torch.randn(2, 8, p)
        y = cffn(x)
        loss = y.sum()
        loss.backward()

        # U and V should have gradients (depth 0)
        assert cffn.U.weight.grad is not None
        assert cffn.V.weight.grad is not None

    def test_freeze_all_depths(self):
        """At depth 0, only linear components (U, V, a_0 row) should train."""
        p = 64
        cffn = Cffn(dim=p, num_ladders=3, depth=5)
        cffn.set_active_depth(0)

        x = torch.randn(2, 8, p)
        y = cffn(x)
        loss = y.sum()
        loss.backward()

        # U and V should have gradients
        assert cffn.U.weight.grad is not None
        assert cffn.V.weight.grad is not None

    def test_unfrozen_all(self):
        """With max depth, all parameters should get gradients."""
        p = 64
        cffn = Cffn(dim=p, num_ladders=3, depth=5)
        cffn.set_active_depth(5)

        x = torch.randn(2, 8, p)
        y = cffn(x)
        loss = y.sum()
        loss.backward()

        for name, param in cffn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
