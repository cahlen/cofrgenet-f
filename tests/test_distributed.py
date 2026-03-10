"""Tests for distributed training utilities (CPU-only, no actual multi-GPU needed)."""

import os
import pytest
import torch
import torch.nn as nn

from scripts.train_common import setup_distributed, cleanup_distributed, is_distributed


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
