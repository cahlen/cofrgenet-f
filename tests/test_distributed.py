"""Tests for distributed training utilities (CPU-only, no actual multi-GPU needed)."""

import os
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn

from scripts.train_common import setup_distributed, cleanup_distributed, is_distributed, ShardedDataLoader


class TestDistributedSetup:

    def test_is_distributed_false_by_default(self):
        """Without RANK env var, should not be in distributed mode."""
        os.environ.pop("RANK", None)
        os.environ.pop("LOCAL_RANK", None)
        assert is_distributed() is False

    def test_setup_returns_rank_info(self):
        """setup_distributed should return (rank, local_rank, world_size)."""
        os.environ.pop("RANK", None)
        rank, local_rank, world_size = setup_distributed()
        assert rank == 0
        assert local_rank == 0
        assert world_size == 1


class TestDistributedDataLoader:

    @pytest.fixture
    def tmp_data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.default_rng(42)
            for i in range(8):
                tokens = rng.integers(0, 256, size=50000, dtype=np.uint16)
                tokens.tofile(os.path.join(tmpdir, f"train_{i:03d}.bin"))
            val = rng.integers(0, 256, size=50000, dtype=np.uint16)
            val.tofile(os.path.join(tmpdir, "val_000.bin"))
            yield tmpdir

    def test_distributed_shards_are_different(self, tmp_data_dir):
        loader0 = ShardedDataLoader(tmp_data_dir, "train", block_size=32, batch_size=4,
                                     device="cpu", rank=0, world_size=4)
        loader1 = ShardedDataLoader(tmp_data_dir, "train", block_size=32, batch_size=4,
                                     device="cpu", rank=1, world_size=4)
        assert loader0.current_shard_idx != loader1.current_shard_idx

    def test_single_gpu_backward_compatible(self, tmp_data_dir):
        loader = ShardedDataLoader(tmp_data_dir, "train", block_size=32, batch_size=4, device="cpu")
        x, y = loader.next_batch()
        assert x.shape == (4, 32)


import tempfile
from scripts.train_common import save_checkpoint_fsdp, load_checkpoint_fsdp


class TestFSDPCheckpointing:

    def test_save_load_non_fsdp_model(self):
        """For non-FSDP models (single GPU), should fall back to normal save/load."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "step_000005.safetensors")
            save_checkpoint_fsdp(model, optimizer, step=5, loss=1.0, path=path, rank=0)
            assert os.path.exists(path)

            model2 = nn.Linear(10, 10)
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            step = load_checkpoint_fsdp(model2, optimizer2, tmpdir, device="cpu")
            assert step == 5


from src.cofrgenet.config import CoFrGeNetConfig
from src.cofrgenet.model import CoFrGeNetTransformer
from src.baseline.config import BaselineConfig
from src.baseline.model import BaselineTransformer
from scripts.train_common import configure_optimizer, train_loop


class TestSingleGPUTrainLoop:

    @pytest.fixture
    def tmp_data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.default_rng(42)
            for name in ["train_000.bin", "val_000.bin"]:
                tokens = rng.integers(0, 256, size=50000, dtype=np.uint16)
                tokens.tofile(os.path.join(tmpdir, name))
            yield tmpdir

    def test_cofrgenet_train_loop_with_distributed_params(self, tmp_data_dir):
        """Full train_loop with rank/world_size params should work single-GPU."""
        device = "cpu"
        cfg = CoFrGeNetConfig(
            n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256,
            num_ladders=2, cf_depth=3
        )
        model = CoFrGeNetTransformer(cfg).to(device)
        optimizer = configure_optimizer(model, 0.1, 1e-3, (0.9, 0.95), device)
        train_loader = ShardedDataLoader(tmp_data_dir, "train", 32, 4, device)
        val_loader = ShardedDataLoader(tmp_data_dir, "val", 32, 4, device)

        def grad_zero_cb():
            model.zero_frozen_grads()

        with tempfile.TemporaryDirectory() as ckpt_dir:
            train_loop(
                model=model, train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, total_steps=5, warmup_steps=2, max_lr=1e-3,
                grad_accum_steps=1, grad_clip=1.0, save_interval=5, eval_interval=5,
                checkpoint_dir=ckpt_dir, model_name="test", device=device,
                rank=0, world_size=1, grad_zero_callback=grad_zero_cb,
            )
            assert os.path.exists(os.path.join(ckpt_dir, "final.safetensors"))

    def test_baseline_train_loop_with_distributed_params(self, tmp_data_dir):
        """Baseline train_loop with new params should work."""
        device = "cpu"
        cfg = BaselineConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256)
        model = BaselineTransformer(cfg).to(device)
        optimizer = configure_optimizer(model, 0.1, 1e-3, (0.9, 0.95), device)
        train_loader = ShardedDataLoader(tmp_data_dir, "train", 32, 4, device)
        val_loader = ShardedDataLoader(tmp_data_dir, "val", 32, 4, device)

        with tempfile.TemporaryDirectory() as ckpt_dir:
            train_loop(
                model=model, train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, total_steps=5, warmup_steps=2, max_lr=1e-3,
                grad_accum_steps=1, grad_clip=1.0, save_interval=5, eval_interval=5,
                checkpoint_dir=ckpt_dir, model_name="test", device=device,
                rank=0, world_size=1,
            )
            assert os.path.exists(os.path.join(ckpt_dir, "final.safetensors"))
