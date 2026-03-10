# B200 Scale-Up: Largest Public CoFrGeNet-F — Design Spec

## Goal

Train the largest and best publicly available CoFrGeNet-F model, leveraging 8x NVIDIA B200 GPUs (DGX B200). Reproduce the original paper's key results at ~1B scale, then scale beyond anything previously tested.

## Context

- **Hardware**: DGX B200 — 8x B200 (179 GB VRAM each), 2 TiB RAM, 224 CPU threads
- **Time budget**: ~18 days of GPU training time within a 30-day cluster access window
- **Current state**: Single-GPU training code, 10B tokens of FineWeb-Edu, largest public CoFrGeNet-F is 128M
- **Paper gap**: IBM tested at 985M (GPT-2 XL) and 2.1B (Llama) but released no weights. No one has tested beyond 3.2B.

## Research Questions

1. **Can we reproduce the paper's key claim?** — CoFrGeNet-F (985M) matching/beating a 1.5B baseline at GPT-2 XL scale
2. **Does CoFrGeNet-F continue to improve at scales beyond the paper?** — first test at 10-20B
3. **What is the optimal L/d configuration?** — paper tested {1,3,5,7} but never published isolated ablations

## Architecture: Multi-GPU Training via FSDP

### Why FSDP over DDP

- DDP replicates the full model on each GPU — works for ~1B models but breaks at 10B+ (won't fit in 179 GB with optimizer states)
- FSDP (Fully Sharded Data Parallel) shards parameters, gradients, and optimizer states across GPUs
- FSDP works for ALL model sizes (small models just don't benefit from sharding), so we build once and use for everything
- Compatible with `torch.compile`, `torch.autograd.Function` (custom continuant backward), and gradient checkpointing

### Sharding Strategy

- Wrap each `TransformerBlock` as an FSDP unit via `size_based_auto_wrap_policy`
- Use `FULL_SHARD` strategy (maximum memory savings)
- Mixed precision: `bfloat16` compute, `float32` reduce
- Gradient checkpointing: enabled for models > 5B parameters

### Data Pipeline

- Extend `01_download_data.py` to download arbitrary amounts of FineWeb-Edu (not just sample-10BT)
- Distributed data loading: each rank reads different shards (shard_id = rank + n*world_size)
- Store data on `/raid` (27.9 TB NVMe RAID0) for maximum I/O throughput

### Training Changes

- Launch via `torchrun --nproc_per_node=8`
- Auto-detect distributed mode (check `RANK` env var)
- Logging/checkpointing only on rank 0
- FSDP-aware checkpointing: `FULL_STATE_DICT` for final model, `SHARDED_STATE_DICT` for resumption

## Experiment Plan

### Phase 1: Reproduce Paper Results (~4 days)

| Model | Layers | Dim | Heads | L | d | Params | Data | Est. Time |
|-------|--------|-----|-------|---|---|--------|------|-----------|
| Baseline 1.5B | 48 | 1600 | 25 | - | - | ~1.5B | 50B tok | ~1 day |
| CoFrGeNet-F 1B | 48 | 1600 | 25 | 3 | 5 | ~1B | 50B tok | ~1 day |
| Eval + comparison | - | - | - | - | - | - | - | hours |

This replicates the paper's GPT-2 XL experiment on our hardware. If CoFrGeNet-F matches or beats the baseline with 34% fewer params, the architecture is validated.

### Phase 2: Scale Beyond the Paper (~12 days)

| Model | Layers | Dim | Heads | L | d | Params | Data | Est. Time |
|-------|--------|-----|-------|---|---|--------|------|-----------|
| Baseline 10B | 48 | 4096 | 32 | - | - | ~10B | 100B tok | ~5 days |
| CoFrGeNet-F 10B | 48 | 5120 | 40 | 8 | 5 | ~7.8B | 100B tok | ~5 days |
| Eval + comparison | - | - | - | - | - | - | - | hours |

The CoFrGeNet-F is configured to be ~78% of the baseline's params (similar ratio as the paper). If the architecture's advantages scale, it should match or beat the 10B baseline.

### Phase 3: Upload & Document

- Upload all models to `cahlen/cofrgenet-f` on HuggingFace
- Update model card with scaling results
- Full comparison table across all scales

## File Changes

### Modified
- `scripts/train_common.py` — add FSDP init, distributed data loading, FSDP-aware checkpointing
- `scripts/02_train_baseline.py` — add FSDP wrapping, configurable model size
- `scripts/03_train_cofrgenet.py` — add FSDP wrapping, configurable model size
- `scripts/01_download_data.py` — support downloading arbitrary FineWeb-Edu subsets
- `src/baseline/config.py` — add `ffn_expansion` scaling for larger models
- `src/cofrgenet/config.py` — no changes needed (already configurable)
- `Dockerfile` — update base image for B200 compatibility
- `requirements.txt` — pin versions for B200 environment

### New
- `scripts/launch_training.sh` — torchrun launcher with common settings
- `configs/experiments/phase1_baseline_1.5b.yaml` — Phase 1 baseline config
- `configs/experiments/phase1_cofrgenet_1b.yaml` — Phase 1 CoFrGeNet-F config
- `configs/experiments/phase2_baseline_10b.yaml` — Phase 2 baseline config
- `configs/experiments/phase2_cofrgenet_10b.yaml` — Phase 2 CoFrGeNet-F config
- `tests/test_distributed.py` — FSDP smoke tests (CPU-based)
