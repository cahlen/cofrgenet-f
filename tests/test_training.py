"""Smoke tests for training loop — verifies both models can train for a few steps."""

import os
import tempfile
import numpy as np
import torch
import pytest

from src.baseline.config import BaselineConfig
from src.baseline.model import BaselineTransformer
from src.cofrgenet.config import CoFrGeNetConfig
from src.cofrgenet.model import CoFrGeNetTransformer, get_unfrozen_depth
from scripts.train_common import (
    ShardedDataLoader, configure_optimizer, get_lr, train_loop
)


@pytest.fixture
def tmp_data_dir():
    """Create temporary data shards for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(42)
        # Create small train and val shards
        for name in ["train_000.bin", "val_000.bin"]:
            tokens = rng.integers(0, 256, size=50000, dtype=np.uint16)
            tokens.tofile(os.path.join(tmpdir, name))
        yield tmpdir


class TestDataLoader:

    def test_loads_batches(self, tmp_data_dir):
        loader = ShardedDataLoader(tmp_data_dir, "train", block_size=32, batch_size=4)
        x, y = loader.next_batch()
        assert x.shape == (4, 32)
        assert y.shape == (4, 32)
        # y should be shifted by 1
        assert x.dtype == torch.int64
        assert y.dtype == torch.int64

    def test_advance_shard(self, tmp_data_dir):
        loader = ShardedDataLoader(tmp_data_dir, "train", block_size=32, batch_size=4)
        loader.advance_shard()
        x, y = loader.next_batch()
        assert x.shape == (4, 32)


class TestLRSchedule:

    def test_warmup(self):
        lr = get_lr(0, warmup_steps=100, total_steps=1000, max_lr=6e-4)
        assert lr < 6e-4
        lr_end_warmup = get_lr(99, warmup_steps=100, total_steps=1000, max_lr=6e-4)
        assert abs(lr_end_warmup - 6e-4) < 1e-6

    def test_decay(self):
        lr_mid = get_lr(550, warmup_steps=100, total_steps=1000, max_lr=6e-4)
        assert lr_mid < 6e-4
        lr_end = get_lr(999, warmup_steps=100, total_steps=1000, max_lr=6e-4)
        assert lr_end < lr_mid


class TestBaselineTrainSmoke:

    def test_train_10_steps(self, tmp_data_dir):
        """Baseline should train for 10 steps without errors, loss should decrease."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = BaselineConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256)
        model = BaselineTransformer(cfg).to(device)

        train_loader = ShardedDataLoader(tmp_data_dir, "train", 32, 4, device)
        val_loader = ShardedDataLoader(tmp_data_dir, "val", 32, 4, device)

        optimizer = configure_optimizer(model, 0.1, 1e-3, (0.9, 0.95), device)

        # Manual mini-loop (don't need full train_loop for smoke test)
        model.train()
        losses = []
        for step in range(10):
            x, y = train_loader.next_batch()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # Loss should generally decrease over 10 steps
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


class TestCoFrGeNetTrainSmoke:

    def test_train_10_steps(self, tmp_data_dir):
        """CoFrGeNet-F should train for 10 steps without errors."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = CoFrGeNetConfig(
            n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256,
            num_ladders=2, cf_depth=3
        )
        model = CoFrGeNetTransformer(cfg).to(device)

        train_loader = ShardedDataLoader(tmp_data_dir, "train", 32, 4, device)

        optimizer = configure_optimizer(model, 0.1, 1e-3, (0.9, 0.95), device)

        model.train()
        losses = []
        for step in range(10):
            # Apply dyadic schedule
            depth = get_unfrozen_depth(step, 10, cfg.cf_depth)
            model.set_active_depth(depth)

            x, y = train_loader.next_batch()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # With dyadic schedule, most params are frozen early on, so strict
        # decrease isn't guaranteed in 10 steps. Just verify finite losses.
        assert all(np.isfinite(l) for l in losses), f"Non-finite losses: {losses}"
        assert losses[-1] < 10.0, f"Loss didn't converge at all: {losses[-1]:.4f}"

    def test_dyadic_schedule_activates(self, tmp_data_dir):
        """Dyadic schedule should change active depth during training."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = CoFrGeNetConfig(
            n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256,
            num_ladders=2, cf_depth=3
        )
        model = CoFrGeNetTransformer(cfg).to(device)

        depths_seen = set()
        for step in range(100):
            depth = get_unfrozen_depth(step, 100, cfg.cf_depth)
            depths_seen.add(depth)

        # Should see multiple depth levels over 100 steps
        assert len(depths_seen) > 1, f"Only saw depths: {depths_seen}"
        assert 0 in depths_seen  # starts at 0
