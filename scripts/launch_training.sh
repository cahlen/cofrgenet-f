#!/bin/bash
# Launch distributed training on B200 GPUs via torchrun
#
# Usage:
#   ./scripts/launch_training.sh cofrgenet --config configs/experiments/phase1_cofrgenet_1b.yaml
#   ./scripts/launch_training.sh baseline --config configs/experiments/phase1_baseline_1.5b.yaml
#
# Environment variables:
#   NPROC     - Number of GPUs (default: 8)
#   DATA_DIR  - Path to tokenized data (default: data/tokenized)

set -euo pipefail

MODEL_TYPE="${1:?Usage: $0 <cofrgenet|baseline> [extra args...]}"
shift

NPROC="${NPROC:-8}"
DATA_DIR="${DATA_DIR:-data/tokenized}"

if [ "$MODEL_TYPE" = "cofrgenet" ]; then
    SCRIPT="scripts/03_train_cofrgenet.py"
elif [ "$MODEL_TYPE" = "baseline" ]; then
    SCRIPT="scripts/02_train_baseline.py"
else
    echo "Unknown model type: $MODEL_TYPE (expected 'cofrgenet' or 'baseline')"
    exit 1
fi

# ── B200/Blackwell performance settings ──
# NCCL auto-tunes on Blackwell NVLink topology — no manual algo override needed
# Async error handling for faster collective ops
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Use CUDA memory pool for NCCL allocations (zero-copy over NVLink)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# torch.compile cache — persist compiled kernels across restarts
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torch_inductor"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1

echo "════════════════════════════════════════════════════════════"
echo "  Launching $MODEL_TYPE training on $NPROC GPUs"
echo "  Data dir: $DATA_DIR"
echo "  Extra args: $@"
echo "════════════════════════════════════════════════════════════"

PYTHONPATH=. torchrun \
    --standalone \
    --nproc_per_node="$NPROC" \
    "$SCRIPT" \
    --data_dir "$DATA_DIR" \
    --compile \
    "$@"
