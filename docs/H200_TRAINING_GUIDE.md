# H200 Training Guide — CoFrGeNet-F 10B Token Run

Step-by-step instructions for a Claude agent (or human) to set up and run CoFrGeNet-F training on an H200 server.

## Prerequisites

- NVIDIA H200 GPU with CUDA drivers installed
- Python 3.10+ with pip
- ~25 GB free disk for tokenized data
- ~5 GB free disk for checkpoints
- Internet access (to download dataset from HuggingFace)

## Step 1: Clone the repo

```bash
git clone https://github.com/cahlen/cofrgenet-f.git
cd cofrgenet-f
```

## Step 2: Install dependencies

```bash
pip install -r requirements.txt
pip install pytest
```

Verify CUDA is available:

```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

## Step 3: Set PYTHONPATH

The project uses `src/` layout imports. Set PYTHONPATH so scripts can find modules:

```bash
export PYTHONPATH=$(pwd)
```

Add this to your shell session or prefix all python commands with it.

## Step 4: Run tests (quick sanity check)

```bash
pytest tests/ -v
```

All tests should pass except possibly `TestBaselineTrainSmoke::test_train_10_steps` which is flaky (loss doesn't always decrease in 10 random steps). The important ones are all `test_cffn.py` and `test_continuant.py` tests.

## Step 5: Download and tokenize the dataset

This downloads FineWeb-Edu sample-10BT (~10B tokens) from HuggingFace, tokenizes with GPT-2 tokenizer, and writes binary shards to `data/tokenized/`.

```bash
python3 scripts/01_download_data.py
```

**This takes 2-4 hours** depending on network speed. It produces:
- `data/tokenized/val_000.bin` — 100M tokens for validation
- `data/tokenized/train_000.bin` through `train_099.bin` — ~100M tokens each

Total: ~19 GB on disk, ~10.05B tokens.

**If you already have the tokenized data** from another machine, just copy the `data/tokenized/` directory (all `.bin` files) instead. This is much faster.

## Step 6: Smoke test (10 steps)

Verify the model trains without errors before committing to a full run:

```bash
python3 scripts/03_train_cofrgenet.py \
    --max_steps 10 \
    --eval_interval 5 \
    --save_interval 10 \
    --no_wandb \
    --lr 3e-4
```

Expected output:
- Model: ~82M parameters
- Should complete 10 steps in under a minute
- Loss should be ~10-11 (random init, cross-entropy over 50K vocab)
- No OOM errors (H200 has 80 GB HBM3, model needs ~10-12 GB)

## Step 7: Full training run

```bash
python3 scripts/03_train_cofrgenet.py --no_wandb --lr 3e-4
```

Or to run detached with logging:

```bash
nohup python3 scripts/03_train_cofrgenet.py --no_wandb --lr 3e-4 > cofrgenet_train.log 2>&1 &
echo $! > cofrgenet_pid.txt
```

### Training parameters (all defaults except LR)

| Parameter | Value |
|-----------|-------|
| Learning rate | **3e-4** (must override — default is 6e-4 for baseline) |
| Total steps | 19,073 (one epoch over 10B tokens) |
| Warmup | 700 steps |
| Batch size | 524,288 tokens per update |
| Micro batch | 16 sequences x 1024 tokens |
| Grad accumulation | 32 steps |
| Precision | bfloat16 (automatic) |
| Gradient clip | 1.0 max norm |

### Optional: Enable torch.compile for faster training

```bash
python3 scripts/03_train_cofrgenet.py --no_wandb --lr 3e-4 --compile
```

This adds a ~2 minute compilation warmup but should give 10-30% faster throughput.

### Optional: Enable wandb logging

Remove `--no_wandb` and optionally set a run name:

```bash
python3 scripts/03_train_cofrgenet.py --lr 3e-4 --wandb_run_name "cofrgenet-f-82m-h200"
```

## Step 8: Monitor training

```bash
tail -f cofrgenet_train.log
```

What to expect:
- **Step 0**: Only linear components train (dyadic depth 0)
- **Step 9,536**: Depth 1 unfreezes (first fraction level) — loss may jump briefly
- **Step 14,305**: Depth 2 unfreezes
- **Step 16,689**: Depth 3 unfreezes
- **Step 17,881**: Depth 4 unfreezes
- **Step 18,477**: Depth 5 unfreezes (full depth)
- Training loss should decrease from ~10-11 down to ~3-4 range
- Validation evals print every 500 steps

## Step 9: Retrieve results

Checkpoints are saved to `checkpoints/cofrgenet/`:
- `step_XXXXX.safetensors` — periodic checkpoints (every 1000 steps)
- `final.safetensors` — final model

## Expected timeline on H200

The H200 should be significantly faster than RTX 5090 (~142K tok/s measured):
- **Estimated throughput**: ~400-600K tok/s (H200 has ~2x memory bandwidth, ~4x BF16 TFLOPS)
- **Estimated total time**: ~3-6 hours for full 10B tokens

## Troubleshooting

**ImportError / ModuleNotFoundError**: Make sure `PYTHONPATH=$(pwd)` is set.

**CUDA OOM**: Unlikely on H200 (80 GB). If it happens, reduce micro batch: `--micro_batch_size 8`.

**Dataset download stalls**: The HuggingFace streaming download can be slow. If interrupted, delete `data/tokenized/` and restart — the script does not support resuming partial downloads.

**Loss explodes / NaN**: This shouldn't happen at LR 3e-4. If it does, try `--lr 1.5e-4`. Do NOT use the baseline LR of 6e-4 — the paper explicitly uses half LR for CoFrGeNet.
