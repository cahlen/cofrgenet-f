#!/bin/bash
# Auto-restarting training wrapper.
# If training crashes (NCCL timeout, SIGABRT, OOM, etc.), waits 30s
# for GPUs to clear, then resumes from the latest checkpoint.
#
# Usage:
#   ./scripts/train_with_autorestart.sh baseline --config configs/experiments/pair1_baseline_450m.yaml
#   ./scripts/train_with_autorestart.sh cofrgenet --config configs/experiments/pair1_cofrgenet_410m.yaml
#
# Set MAX_RESTARTS to limit retries (default: 20).
# Set RESTART_DELAY to change cooldown between restarts (default: 30s).

set -uo pipefail

MODEL_TYPE="${1:?Usage: $0 <cofrgenet|baseline> [extra args...]}"
shift

MAX_RESTARTS="${MAX_RESTARTS:-20}"
RESTART_DELAY="${RESTART_DELAY:-30}"
DATA_DIR="${DATA_DIR:-data/tokenized}"

restart_count=0

while [ "$restart_count" -lt "$MAX_RESTARTS" ]; do
    echo ""
    echo "════════════════════════════════════════════════════════════"
    if [ "$restart_count" -eq 0 ]; then
        echo "  Starting training: $MODEL_TYPE"
    else
        echo "  Auto-restart #${restart_count}/${MAX_RESTARTS}: $MODEL_TYPE"
    fi
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════"

    # Always pass --resume so it picks up from latest checkpoint if available
    DATA_DIR="$DATA_DIR" CUDA_DEVICES="${CUDA_DEVICES:-}" ./scripts/launch_training.sh "$MODEL_TYPE" --resume "$@"
    exit_code=$?

    # Exit code 0 = training completed normally
    if [ "$exit_code" -eq 0 ]; then
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "  Training completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
        echo "════════════════════════════════════════════════════════════"
        exit 0
    fi

    restart_count=$((restart_count + 1))
    echo ""
    echo "[CRASH] Exit code $exit_code at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[CRASH] Restart $restart_count/$MAX_RESTARTS in ${RESTART_DELAY}s..."

    # Wait for GPUs to fully release memory
    sleep "$RESTART_DELAY"

    # Verify our GPUs are free before restarting
    if [ -n "${CUDA_DEVICES:-}" ]; then
        # Only check our assigned GPUs
        for gpu_id in $(echo "$CUDA_DEVICES" | tr ',' ' '); do
            gpu_procs=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
            if [ "$gpu_procs" -gt 0 ]; then
                echo "[WARN] GPU $gpu_id still has $gpu_procs processes, waiting 30s more..."
                sleep 30
                break
            fi
        done
    else
        gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        if [ "$gpu_procs" -gt 0 ]; then
            echo "[WARN] $gpu_procs GPU processes still running, waiting 30s more..."
            sleep 30
        fi
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  FAILED: Exhausted $MAX_RESTARTS restart attempts"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"
exit 1
