"""Tests for continuant-based continued fraction computation."""

import torch
import pytest
from src.cofrgenet.continuant import continued_fraction


class TestContinuantBaseCases:
    """Test fundamental continuant identities."""

    def test_continuant_depth_1(self):
        """f̃([a_1]) = 1/a_1 = K_0/K_1."""
        a = torch.tensor([3.0])
        result = continued_fraction(a.unsqueeze(0)).squeeze(0)
        expected = 1.0 / 3.0
        assert torch.allclose(result, torch.tensor(expected), atol=1e-6)

    def test_continuant_depth_2(self):
        """f̃([a_1, a_2]) = a_2 / (a_1*a_2 + 1).

        K_0 = 1, K_1 = a_2, K_2 = a_1*a_2 + 1
        f̃ = K_1 / K_2 = a_2 / (a_1*a_2 + 1)
        """
        a1, a2 = 2.0, 5.0
        a = torch.tensor([[a1, a2]])
        result = continued_fraction(a).squeeze()
        expected = a2 / (a1 * a2 + 1)
        assert torch.allclose(result, torch.tensor(expected), atol=1e-6)

    def test_continuant_depth_3(self):
        """f̃([a_1, a_2, a_3]) = (a_2*a_3 + 1) / (a_1*a_2*a_3 + a_1 + a_3).

        K_0 = 1, K_1 = a_3, K_2 = a_2*a_3 + 1, K_3 = a_1*(a_2*a_3+1) + a_3
        f̃ = K_2 / K_3
        """
        a1, a2, a3 = 2.0, 3.0, 4.0
        a = torch.tensor([[a1, a2, a3]])
        result = continued_fraction(a).squeeze()
        K2 = a2 * a3 + 1  # 13
        K3 = a1 * K2 + a3  # 30
        expected = K2 / K3
        assert torch.allclose(result, torch.tensor(expected), atol=1e-6)


class TestContinuedFractionValues:
    """Test continued fraction evaluates correctly against naive recursion."""

    def _naive_cf(self, a):
        """Compute continued fraction naively via recursion (no continuants).

        f̃(a_1, ..., a_d) = 1 / (a_1 + 1 / (a_2 + ... + 1/a_d))
        """
        d = len(a)
        if d == 1:
            return 1.0 / a[0]
        # Build from the bottom up
        result = a[-1]
        for i in range(d - 2, 0, -1):
            result = a[i] + 1.0 / result
        result = 1.0 / (a[0] + 1.0 / result) if d > 1 else 1.0 / result
        return result

    def test_depth_4_against_naive(self):
        a_vals = [1.5, 2.5, 3.5, 4.5]
        a = torch.tensor([a_vals])
        result = continued_fraction(a).squeeze()
        expected = self._naive_cf(a_vals)
        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-5)

    def test_depth_7_against_naive(self):
        a_vals = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]
        a = torch.tensor([a_vals])
        result = continued_fraction(a).squeeze()
        expected = self._naive_cf(a_vals)
        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-5)

    def test_random_depths(self):
        """Test multiple random inputs at various depths."""
        torch.manual_seed(42)
        for d in range(1, 8):
            a_vals = (torch.randn(d) * 2 + 3).tolist()  # bias away from 0
            a = torch.tensor([a_vals])
            result = continued_fraction(a).squeeze()
            expected = self._naive_cf(a_vals)
            assert torch.allclose(
                result, torch.tensor(expected, dtype=torch.float32), atol=1e-4
            ), f"Failed at depth {d} with a={a_vals}"


class TestGradients:
    """Test custom backward matches autograd and finite differences."""

    def _naive_cf_torch(self, a):
        """Naive continued fraction in pure PyTorch (for autograd comparison)."""
        d = a.shape[-1]
        if d == 1:
            return 1.0 / a[..., 0]
        result = a[..., -1]
        for i in range(d - 2, 0, -1):
            result = a[..., i] + 1.0 / result
        result = 1.0 / (a[..., 0] + 1.0 / result)
        return result

    def test_gradient_matches_autograd_naive(self):
        """Compare custom gradient against naive torch.autograd."""
        torch.manual_seed(123)
        for d in range(1, 8):
            a = torch.randn(1, d) * 2 + 3  # bias away from 0
            a.requires_grad_(True)

            # Custom backward
            a_custom = a.detach().clone().requires_grad_(True)
            out_custom = continued_fraction(a_custom)
            out_custom.sum().backward()
            grad_custom = a_custom.grad.clone()

            # Naive autograd
            a_naive = a.detach().clone().requires_grad_(True)
            out_naive = self._naive_cf_torch(a_naive)
            out_naive.sum().backward()
            grad_naive = a_naive.grad.clone()

            assert torch.allclose(grad_custom, grad_naive, atol=1e-4), (
                f"Gradient mismatch at depth {d}:\n"
                f"  custom:  {grad_custom}\n"
                f"  naive:   {grad_naive}\n"
                f"  diff:    {(grad_custom - grad_naive).abs()}"
            )

    def test_gradient_numerical(self):
        """Finite-difference gradient check."""
        torch.manual_seed(456)
        for d in [2, 4, 6]:
            a = torch.randn(1, d, dtype=torch.float64) * 2 + 3
            a.requires_grad_(True)
            assert torch.autograd.gradcheck(
                lambda x: continued_fraction(x, epsilon=1e-8),
                (a,),
                eps=1e-6,
                atol=1e-4,
            ), f"gradcheck failed at depth {d}"


class TestPoleAvoidance:
    """Test behavior near poles (K_d ≈ 0)."""

    def test_pole_avoidance_no_nan(self):
        """When inputs would cause K_d ≈ 0, output should be finite (clamped)."""
        # Construct input that would make K_d small
        # For d=2: K_2 = a_1*a_2 + 1. If a_1*a_2 = -1, pole.
        a = torch.tensor([[1.0, -1.0]])  # K_2 = 1*(-1) + 1 = 0
        result = continued_fraction(a)
        assert torch.isfinite(result).all(), f"Got non-finite result: {result}"

    def test_pole_avoidance_gradient_finite(self):
        """Gradients should be finite near poles."""
        a = torch.tensor([[1.0, -1.0]], requires_grad=True)
        result = continued_fraction(a)
        result.sum().backward()
        assert torch.isfinite(a.grad).all(), f"Got non-finite gradient: {a.grad}"


class TestBatched:
    """Test batched computation across various shapes."""

    def test_batch_dimension(self):
        """Batch of inputs should produce batch of outputs."""
        B = 8
        d = 5
        torch.manual_seed(42)
        a = torch.randn(B, d) * 2 + 3
        result = continued_fraction(a)
        assert result.shape == (B,)

    def test_batch_seq_dimension(self):
        """(B, S, d) input should produce (B, S) output."""
        B, S, d = 4, 16, 5
        torch.manual_seed(42)
        a = torch.randn(B, S, d) * 2 + 3
        result = continued_fraction(a)
        assert result.shape == (B, S)

    def test_batched_matches_individual(self):
        """Batched computation should match individual computations."""
        torch.manual_seed(42)
        B, d = 8, 5
        a = torch.randn(B, d) * 2 + 3
        batched = continued_fraction(a)
        individual = torch.stack([continued_fraction(a[i:i+1]).squeeze(0) for i in range(B)])
        assert torch.allclose(batched, individual, atol=1e-6)

    def test_batched_gradient_matches_individual(self):
        """Batched gradients should match individual gradients."""
        torch.manual_seed(42)
        B, d = 4, 4
        a_base = torch.randn(B, d) * 2 + 3

        # Batched
        a_batch = a_base.clone().requires_grad_(True)
        continued_fraction(a_batch).sum().backward()
        grad_batch = a_batch.grad.clone()

        # Individual
        grads = []
        for i in range(B):
            a_i = a_base[i:i+1].clone().requires_grad_(True)
            continued_fraction(a_i).sum().backward()
            grads.append(a_i.grad.clone())
        grad_individual = torch.cat(grads, dim=0)

        assert torch.allclose(grad_batch, grad_individual, atol=1e-6)
