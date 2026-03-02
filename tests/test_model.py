"""Tests for baseline and CoFrGeNet-F models."""

import torch
import pytest
from src.baseline.config import BaselineConfig
from src.baseline.model import BaselineTransformer
from src.cofrgenet.config import CoFrGeNetConfig
from src.cofrgenet.model import CoFrGeNetTransformer


# ---------- Baseline Transformer ----------

class TestBaselineForward:
    """Test forward pass shapes."""

    def test_forward_shape(self):
        """(B, S) input -> (B, S, vocab_size) logits."""
        cfg = BaselineConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256)
        model = BaselineTransformer(cfg)
        x = torch.randint(0, 256, (2, 16))
        logits, loss = model(x)
        assert logits.shape == (2, 16, 256)

    def test_forward_with_targets(self):
        """When targets provided, should return loss."""
        cfg = BaselineConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256)
        model = BaselineTransformer(cfg)
        x = torch.randint(0, 256, (2, 16))
        targets = torch.randint(0, 256, (2, 16))
        logits, loss = model(x, targets)
        assert loss is not None
        assert loss.ndim == 0  # scalar


class TestBaselineParameters:
    """Test parameter counts."""

    def test_parameter_count_approx(self):
        """Full-scale baseline should be ~124M params."""
        cfg = BaselineConfig()
        model = BaselineTransformer(cfg)
        total = sum(p.numel() for p in model.parameters())
        # GPT-2 Small is ~124M. Allow some variance from implementation details.
        assert 100_000_000 < total < 150_000_000, f"Got {total:,} params"

    def test_weight_tying(self):
        """Token embedding and output projection should share weights."""
        cfg = BaselineConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256)
        model = BaselineTransformer(cfg)
        assert model.tok_emb.weight is model.lm_head.weight


class TestBaselineCausalMasking:
    """Test causal attention masking."""

    def test_causal_masking(self):
        """Changing future tokens should not affect earlier logits."""
        cfg = BaselineConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256)
        model = BaselineTransformer(cfg)
        model.eval()

        x1 = torch.tensor([[1, 2, 3, 4, 5]])
        x2 = torch.tensor([[1, 2, 3, 99, 99]])

        with torch.no_grad():
            logits1, _ = model(x1)
            logits2, _ = model(x2)

        # First 3 positions should be identical
        assert torch.allclose(logits1[:, :3, :], logits2[:, :3, :], atol=1e-5)


class TestBaselineGeneration:
    """Test autoregressive generation."""

    def test_generate(self):
        """Model should generate tokens autoregressively."""
        cfg = BaselineConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256)
        model = BaselineTransformer(cfg)
        model.eval()

        prompt = torch.tensor([[1, 2, 3]])
        generated = model.generate(prompt, max_new_tokens=10)
        assert generated.shape == (1, 13)  # 3 prompt + 10 generated
        assert (generated[:, :3] == prompt).all()


# ---------- CoFrGeNet-F Transformer ----------

class TestCoFrGeNetForward:
    """Test forward pass shapes."""

    def test_forward_shape(self):
        """(B, S) input -> (B, S, vocab_size) logits."""
        cfg = CoFrGeNetConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256)
        model = CoFrGeNetTransformer(cfg)
        x = torch.randint(0, 256, (2, 16))
        logits, loss = model(x)
        assert logits.shape == (2, 16, 256)

    def test_forward_with_targets(self):
        """When targets provided, should return loss."""
        cfg = CoFrGeNetConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256)
        model = CoFrGeNetTransformer(cfg)
        x = torch.randint(0, 256, (2, 16))
        targets = torch.randint(0, 256, (2, 16))
        logits, loss = model(x, targets)
        assert loss is not None
        assert loss.ndim == 0


class TestCoFrGeNetParameters:
    """Test parameter counts."""

    def test_fewer_params_than_baseline(self):
        """CoFrGeNet-F should have fewer params than baseline at same config."""
        baseline_cfg = BaselineConfig()
        cofrgenet_cfg = CoFrGeNetConfig()

        baseline = BaselineTransformer(baseline_cfg)
        cofrgenet = CoFrGeNetTransformer(cofrgenet_cfg)

        baseline_params = sum(p.numel() for p in baseline.parameters())
        cofrgenet_params = sum(p.numel() for p in cofrgenet.parameters())

        assert cofrgenet_params < baseline_params, (
            f"CoFrGeNet-F ({cofrgenet_params:,}) should have fewer params "
            f"than baseline ({baseline_params:,})"
        )

    def test_weight_tying(self):
        """Token embedding and output projection should share weights."""
        cfg = CoFrGeNetConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256)
        model = CoFrGeNetTransformer(cfg)
        assert model.tok_emb.weight is model.lm_head.weight


class TestCoFrGeNetCausalMasking:
    """Test causal attention masking."""

    def test_causal_masking(self):
        """Changing future tokens should not affect earlier logits."""
        cfg = CoFrGeNetConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256)
        model = CoFrGeNetTransformer(cfg)
        model.eval()

        x1 = torch.tensor([[1, 2, 3, 4, 5]])
        x2 = torch.tensor([[1, 2, 3, 99, 99]])

        with torch.no_grad():
            logits1, _ = model(x1)
            logits2, _ = model(x2)

        assert torch.allclose(logits1[:, :3, :], logits2[:, :3, :], atol=1e-5)


class TestCoFrGeNetGeneration:
    """Test autoregressive generation."""

    def test_generate(self):
        """Model should generate tokens autoregressively."""
        cfg = CoFrGeNetConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256)
        model = CoFrGeNetTransformer(cfg)
        model.eval()

        prompt = torch.tensor([[1, 2, 3]])
        generated = model.generate(prompt, max_new_tokens=10)
        assert generated.shape == (1, 13)
        assert (generated[:, :3] == prompt).all()


class TestCoFrGeNetDyadicSchedule:
    """Test dyadic training schedule for Cffn layers."""

    def test_set_active_depth(self):
        """Setting active depth should propagate to all Cffn layers."""
        cfg = CoFrGeNetConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256,
                              num_ladders=2, cf_depth=3)
        model = CoFrGeNetTransformer(cfg)
        model.set_active_depth(1)

        for block in model.blocks:
            assert block.ffn._active_depth == 1

    def test_get_unfrozen_depth(self):
        """Verify dyadic schedule unfreezes at correct steps."""
        from src.cofrgenet.model import get_unfrozen_depth
        total = 19000

        assert get_unfrozen_depth(0, total, 5) == 0
        assert get_unfrozen_depth(9500, total, 5) == 1
        assert get_unfrozen_depth(14250, total, 5) == 2
        assert get_unfrozen_depth(16625, total, 5) == 3
        assert get_unfrozen_depth(17812, total, 5) == 3  # not quite at depth 4
        assert get_unfrozen_depth(17813, total, 5) == 4
        assert get_unfrozen_depth(18500, total, 5) == 5
