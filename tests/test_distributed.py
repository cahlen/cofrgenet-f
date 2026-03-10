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
