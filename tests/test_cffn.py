"""Tests for the Continued Fraction FFN (Cffn) layer."""

import torch
import torch.nn as nn
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
        """Params should be L*p*d + 2*p*p + p*L.

        - ladder_weights: L parameters, each (p, d) = L*p*d
        - U: p -> p, no bias = p*p
        - gate_proj: p -> p, no bias = p*p
        - V: (p, L) = p*L
        """
        p, L, d = 768, 3, 5
        cffn = Cffn(dim=p, num_ladders=L, depth=d)
        total = sum(param.numel() for param in cffn.parameters())
        expected = L * p * d + 2 * p * p + p * L
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

    def test_paper_formula_equivalence(self):
        """L*p*d + 2*p*p + p*L should equal L*p*(d+1) + 2*p*p (paper formula)."""
        p, L, d = 768, 3, 5
        our_formula = L * p * d + 2 * p * p + p * L
        paper_formula = L * p * (d + 1) + 2 * p * p
        assert our_formula == paper_formula


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
        """Freezing to depth 1 should zero out columns 1..d-1 gradients."""
        p = 64
        cffn = Cffn(dim=p, num_ladders=3, depth=5)

        # Allow depth 0 (linear) and depth 1 (column 0 of ladder weights)
        cffn.set_active_depth(1)

        x = torch.randn(2, 8, p)
        y = cffn(x)
        loss = y.sum()
        loss.backward()
        cffn.zero_frozen_grads()

        # U, V, and gate_proj should have gradients
        assert cffn.U.weight.grad is not None
        assert cffn.V.grad is not None
        assert cffn.gate_proj.weight.grad is not None

        # Ladder weights: column 0 should have grad, columns 1..4 should be zero
        for w in cffn.ladder_weights:
            assert w.grad[:, 0].abs().sum() > 0, "Column 0 should have gradients"
            assert w.grad[:, 1:].abs().sum() == 0, "Columns 1..4 should be zero"

    def test_freeze_all_depths(self):
        """At depth 0, all ladder columns should be masked (only U, V, gate_proj train)."""
        p = 64
        cffn = Cffn(dim=p, num_ladders=3, depth=5)
        cffn.set_active_depth(0)

        x = torch.randn(2, 8, p)
        y = cffn(x)
        loss = y.sum()
        loss.backward()
        cffn.zero_frozen_grads()

        # U, V, and gate_proj should have gradients
        assert cffn.U.weight.grad is not None
        assert cffn.V.grad is not None
        assert cffn.gate_proj.weight.grad is not None

        # All ladder columns should be zeroed
        for w in cffn.ladder_weights:
            assert w.grad.abs().sum() == 0, "All ladder grads should be zero at depth 0"

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


class TestCffnGating:
    """Test the gating mechanism."""

    def test_gate_proj_exists(self):
        """gate_proj should exist with correct shape (p, p)."""
        p = 64
        cffn = Cffn(dim=p, num_ladders=3, depth=5)
        assert hasattr(cffn, 'gate_proj')
        assert cffn.gate_proj.weight.shape == (p, p)

    def test_gated_output_is_nonlinear(self):
        """At depth 0, cffn(2x) != 2*cffn(x) due to sigmoid gating.

        Without gating, depth-0 Cffn is purely linear. The gate introduces
        nonlinearity even before fraction depths are unfrozen.
        """
        p = 64
        cffn = Cffn(dim=p, num_ladders=3, depth=5)
        cffn.set_active_depth(0)

        torch.manual_seed(123)
        x = torch.randn(1, 4, p)

        with torch.no_grad():
            y1 = cffn(x)
            y2 = cffn(2.0 * x)

        # If linear: y2 == 2*y1. Gating makes this false.
        assert not torch.allclose(y2, 2.0 * y1, atol=1e-5), (
            "Cffn output is linear — gating is not working"
        )


class TestCffnPVariate:
    """Test p-variate ladder behavior."""

    def test_ladder_output_is_p_dimensional(self):
        """Each ladder should produce a p-dim output, not a scalar."""
        p = 64
        L = 3
        d = 5
        cffn = Cffn(dim=p, num_ladders=L, depth=d)

        x = torch.randn(2, 8, p)
        gated_x = torch.sigmoid(cffn.gate_proj(x)) * x

        # Manually compute one ladder to verify shape
        a = gated_x.unsqueeze(-1) * cffn.ladder_weights[0]  # (B,S,p,d)
        assert a.shape == (2, 8, p, d)

        from src.cofrgenet.continuant import continued_fraction
        z = continued_fraction(a, cffn.epsilon)
        assert z.shape == (2, 8, p), f"Expected (2, 8, {p}), got {z.shape}"

    def test_ladder_weights_are_elementwise(self):
        """Ladder weights should be (p, d) parameters, not linear layers."""
        p = 64
        cffn = Cffn(dim=p, num_ladders=3, depth=5)
        for w in cffn.ladder_weights:
            assert isinstance(w, nn.Parameter)
            assert w.shape == (p, 5)

    def test_V_is_parameter(self):
        """V should be a (p, L) parameter, not a linear layer."""
        p = 64
        L = 3
        cffn = Cffn(dim=p, num_ladders=L, depth=5)
        assert isinstance(cffn.V, nn.Parameter)
        assert cffn.V.shape == (p, L)


def test_zero_frozen_grads():
    """zero_frozen_grads should zero out gradient columns beyond active depth."""
    from src.cofrgenet.cffn import Cffn
    cffn = Cffn(dim=16, num_ladders=2, depth=5)
    x = torch.randn(1, 4, 16)
    y = cffn(x)
    y.sum().backward()
    cffn.zero_frozen_grads(active_depth=2)
    for w in cffn.ladder_weights:
        assert (w.grad[:, 2:] == 0).all(), "Frozen depth columns should have zero gradients"
