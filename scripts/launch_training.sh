#!/bin/bash
# Launch distributed training on B200 GPUs via torchrun
#
# Usage:
#   ./scripts/launch_training.sh cofrgenet [--config configs/experiments/phase1_cofrgenet_1b.yaml]
#   ./scripts/launch_training.sh baseline [--n_embd 1600 --n_head 25 --n_layer 48]
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

echo "Launching $MODEL_TYPE training on $NPROC GPUs..."
echo "Data dir: $DATA_DIR"
echo "Extra args: $@"

torchrun \
    --standalone \
    --nproc_per_node="$NPROC" \
    "$SCRIPT" \
    --data_dir "$DATA_DIR" \
    --compile \
    --no_wandb \
    "$@"
