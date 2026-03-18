# CLAUDE.md — CoFrGeNet-F: Continued Fraction Language Model

## What This Project Is

An open-source implementation of CoFrGeNet-F, a continued fraction architecture that replaces Transformer FFN layers with continued fraction networks. Based on IBM Research's paper [arXiv:2601.21766](https://arxiv.org/abs/2601.21766) (January 2026). Implemented from the paper's mathematics.

**Goal:** Train CoFrGeNet-F models at multiple scales and compare head-to-head against standard Transformer baselines on identical data. Release both with benchmarks, Gradio demo, and technical write-up.

## Current Status (2026-03-18)

### Completed
- **All core architecture**: continuant.py, cffn.py, both models, configs, tests
- **Data pipeline**: FineWeb-Edu 10BT (in `data/tokenized/`) and 50BT (in `data/tokenized_50b/`)
- **Training infrastructure**: shared training loop, dyadic schedule, checkpointing, DDP/FSDP, auto-restart, YAML configs
- **Prior single-GPU experiments 1–3**: 82M, 128M, 128M-L8 on RTX 5090 / H200 (10B tokens each)
- **Evaluation script**: `scripts/04_evaluate.py` — WikiText-2, WikiText-103, LAMBADA, throughput, generation speed
- **HuggingFace repos**: [`cahlen/cofrgenet-f`](https://huggingface.co/cahlen/cofrgenet-f) (prior experiments), [`cahlen/pair3-baseline-7b`](https://huggingface.co/cahlen/pair3-baseline-7b) (Pair 3 baseline)
- **GitHub Wiki**: 7 pages of detailed architecture + math documentation
- **DGX B200 Pair 1** (450M baseline vs 410M CoFrGeNet-F, 50B tokens): Both complete. Baseline PPL 23.69, CoFrGeNet-F PPL 56.61.
- **DGX B200 Pair 3 Baseline** (7.5B, 50B tokens): Training complete (95,367 steps, ~5.5 days on 8x B200). Evaluated. Uploaded to HuggingFace.
- **Code audit**: Full audit of continuant math, Cffn architecture, dyadic schedule, and training loop against the paper. All correct.

### In Progress
- **DGX B200 Pair 3 CoFrGeNet-F** (4.8B, 50B tokens): Queued to launch on 8x B200 (FSDP). Config at `configs/experiments/pair3_cofrgenet_5b.yaml`. This is the key experiment — does CoFrGeNet-F's advantage emerge at 7B+ scale?

### Remaining
- Evaluate Pair 3 CoFrGeNet-F and compare against baseline at matching steps
- Pairs 4–5 (contingent on cluster time)
- Gradient stabilization experiments (post-pair training)
- `scripts/05_generate_examples.py` — text generation comparison
- `demo/app.py` — Gradio demo (side-by-side generation)
- Blog post / technical write-up

## Prior Single-GPU Experiments (10B tokens)

These early experiments were run on single GPUs (RTX 5090 / H200) before the DGX B200 paired experiments. They motivated the larger-scale runs. The IBM paper showed CoFrGeNet-F's advantage only emerging at GPT-2 XL scale (~1B params).

### Experiment 1: Parameter-Efficient (82M vs 124M)

**Question:** Can CoFrGeNet-F match baseline quality with 34% fewer parameters?

| Model | Config | Params |
|-------|--------|--------|
| Baseline | 12L, 768d, 12h, standard FFN | 124,337,664 |
| CoFrGeNet-F | 12L, 768d, 12h, L=3 ladders, d=5 depth | 82,036,224 |

**Result:** No. CoFrGeNet-F significantly underperforms at this scale.

| Metric | Baseline (124M) | CoFrGeNet-F (82M) |
|--------|-----------------|-------------------|
| WikiText-2 PPL | **40.79** | 110.32 |
| WikiText-103 PPL | **40.79** | 110.32 |
| LAMBADA PPL | **37.45** | 166.57 |
| LAMBADA Acc | **19.06%** | 8.77% |
| Throughput | **277,827** tok/s | 103,455 tok/s |
| Gen Speed | **5.53** ms/tok | 10.92 ms/tok |

Evaluated on same H200 with identical code (`scripts/04_evaluate.py`).

### Experiment 2: Iso-Parameter (128M vs 124M) — COMPLETE

**Question:** With equal parameter budget, does CoFrGeNet-F match or beat the baseline?

| Model | Config | Params |
|-------|--------|--------|
| Baseline | 12L, 768d, 12h, standard FFN | 124,337,664 |
| CoFrGeNet-F | 12L, 1024d, 16h, L=3 ladders, d=5 depth | 128,256,000 |

**Result:** CoFrGeNet-F improves significantly over Experiment 1 but still underperforms the baseline.

| Metric | Baseline (124M) | CoFrGeNet-F (128M) |
|--------|-----------------|-------------------|
| WikiText-2 PPL | **40.79** | 82.46 |
| WikiText-103 PPL | **40.79** | 82.46 |
| LAMBADA PPL | **37.45** | 111.26 |
| LAMBADA Acc | **19.06%** | 11.41% |
| Throughput | **452,622** tok/s | 128,206 tok/s |
| Gen Speed | **3.68** ms/tok | 10.50 ms/tok |

Trained on H200 GPU 2, ~114K tok/s with `torch.compile`, 24.3 hours.
**Checkpoints:** `checkpoints/cofrgenet-128m/`

### Experiment 3: More Ladders (128M, L=8 vs L=3) — IN PROGRESS

**Question:** Does increasing the number of continued fraction ladders from 3 to 8 improve quality? Each ladder is an independent rational approximation — more ladders = richer function space, at minimal parameter cost (+0.3%).

| Model | Config | Params |
|-------|--------|--------|
| CoFrGeNet-F (Exp 2) | 12L, 1024d, 16h, L=3 ladders, d=5 depth | 128,256,000 |
| CoFrGeNet-F (Exp 3) | 12L, 1024d, 16h, L=8 ladders, d=5 depth | 128,624,640 |

**Checkpoints:** `checkpoints/cofrgenet-128m-L8/`

## DGX B200 Paired Experiments (50B–100B tokens)

The main experiments train matched pairs of baseline Transformers and CoFrGeNet-F models on 8x NVIDIA B200 GPUs. Each pair runs sequentially (baseline first, then CoFrGeNet-F) on identical data with identical hyperparameters. Full details in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

### Pair 1: 450M Baseline vs 410M CoFrGeNet-F (50B tokens) — COMPLETE

| Metric | Baseline (450M) | CoFrGeNet-F (410M) |
|--------|-----------------|-------------------|
| WikiText-2 PPL | **23.69** | 56.61 |
| LAMBADA Acc | **26.88%** | 15.51% |
| Throughput | **2.12M** tok/s | 800K tok/s |

CoFrGeNet-F 2.4x worse on perplexity. Consistent with IBM paper — advantage only emerges at larger scale.

### Pair 3: 7.5B Baseline vs 4.8B CoFrGeNet-F (50B tokens)

| | Baseline | CoFrGeNet-F |
|-|----------|-------------|
| **Params** | ~7.5B | ~4.8B (35% fewer) |
| **Architecture** | 36L, 4096d, 32h, standard FFN | 36L, 4608d, 36h, L=3, d=5 |
| **Parallelism** | FSDP FULL_SHARD (8 GPUs) | FSDP FULL_SHARD (8 GPUs) |
| **compile** | false | false |
| **micro_batch_size** | 64 | 16 |
| **save_interval** | 2000 | 5000 |
| **data_dir** | `data/tokenized_50b` | `data/tokenized_50b` |
| **Status** | **COMPLETE** | **QUEUED** |

Both models use FSDP FULL_SHARD (>2B params triggers FSDP in `train_common.py`). Baseline runs at mbs=64 (grad_accum=1), CoFrGeNet-F at mbs=16 (grad_accum=4) due to Cffn activation memory (ladder broadcasts to `(B,S,p,d)` per ladder per layer). Effective batch size is identical (524,288 tokens/step). `torch.compile` disabled due to dtype mismatch crash with 7B+ models. Dyadic schedule uses detach-in-forward (not gradient zeroing) for FSDP compatibility.

#### Pair 3 Baseline Results (COMPLETE)

Trained ~5.5 days on 8x B200 at ~132,800 tok/s. Model is massively overparameterized for 50B tokens (Chinchilla optimal would be ~150B), so it overfits heavily. Best generalization at step 10K, final checkpoint memorized the training set (train loss 0.008).

| Metric | Step 10K (Best LLM) | Step 20K | Step 95K (Final) |
|--------|---------------------|----------|-----------------|
| WikiText-2 PPL | **39.52** | 52.21 | 2,952,579 |
| LAMBADA Acc | **15.89%** | 13.12% | 6.31% |
| Throughput | 29,561 tok/s | 26,799 tok/s | 55,693 tok/s |

**Checkpoints kept:** step 10K (best generalization), step 20K (reference), step 95K (final for comparison). All others deleted to save disk. HuggingFace: [`cahlen/pair3-baseline-7b`](https://huggingface.co/cahlen/pair3-baseline-7b).

**Baseline val loss curve** logged at every 1,000 steps in `logs/pair3_baseline_7b.log` for direct comparison with CoFrGeNet-F at matching steps.

#### Pair 3 CoFrGeNet-F (QUEUED)

Key questions:
1. Does CoFrGeNet-F's val loss ever beat the baseline's best (2.94 at step 8K)?
2. At what step does the crossover happen, if at all?
3. How does the learning curve compare at matching steps throughout training?

CoFrGeNet-F saves checkpoints every 5K steps and evals every 1K steps, matching the baseline's eval resolution for head-to-head comparison.

## Architecture Overview

### What Is CoFrGeNet-F?

A standard Transformer block has two components: **Multi-Head Attention** and a **Feed-Forward Network (FFN)**. CoFrGeNet-F keeps standard attention but replaces the FFN with a **Continued Fraction FFN (Cffn)**.

A continued fraction computes:

```
f̃(a₁, ..., a_d) = 1/(a₁ + 1/(a₂ + ... + 1/a_d))
```

where each `a_k = w_k · x` (learnable weight times input). This is computed via **continuant polynomials** `K`:

```
K₀ = 1
K₁(a_d) = a_d
K_k = a_{d-k+1} · K_{k-1} + K_{k-2}
```

The continued fraction equals `K_{d-1} / K_d`, and gradients (Proposition 1) are:

```
∂f̃/∂a_k = (-1)^k · [K_{d-k} / K_d]²
```

This reduces divisions from d (naive) to **1** (compute `1/K_d` once).

### Cffn Architecture

The Cffn replaces a standard 2-layer FFN (`Linear(p, 4p) → GELU → Linear(4p, p)`) with:

1. **A direct linear path** `U` (p×p) — skip connection through the fraction
2. **A gating projection** `G` (p×p) — `gated_x = sigmoid(G·x) ⊙ x`
3. **L p-variate continued fraction ladders**, each of depth d, operating element-wise per hidden dimension
4. **Combination weights** `V` (p×L) — per-dimension weighting of ladder outputs

```
y = U·x + V·z
where z_j = f̃(gated_x ⊙ W^(j))   for j = 1, ..., L
```

**Parameter count per Cffn layer:** `2p² + L·p·(d+1)`

### Dyadic Training Schedule (CRITICAL)

Without this, performance degrades 10-80%. Progressively unfreezes continued fraction depth by detaching frozen columns in the forward pass (FSDP-compatible):

```
Depth i parameters: unfrozen at step (1 - 1/2^i) × total_steps
```

For 19,073 steps: depth 1 at step 9,537, depth 2 at 14,305, depth 3 at 16,689, depth 4 at 17,881, depth 5 at 18,477.

## Model Parameter Counts

### Baseline Transformer: 124,337,664
- Embeddings: tok (38.6M) + pos (786K)
- Per block (×12): LN (1.5K) + Attn QKV (1.77M) + Attn out (590K) + FFN fc1 (2.36M) + FFN fc2 (2.36M) = 7.08M
- 12 blocks: 84.95M + final LN (768)

### CoFrGeNet-F 82M: 82,036,224 (Experiment 1)
- Config: 12L, 768d, 12h, L=3, d=5
- Per block Cffn: U (590K) + G (590K) + 3 ladders (11.5K) + V (2.3K) = 1.19M

### CoFrGeNet-F 128M: 128,256,000 (Experiment 2)
- Config: 12L, 1024d, 16h, L=3, d=5
- Wider hidden dim compensates for Cffn's smaller FFN

## Training Recipe (Prior Single-GPU Experiments)

The prior single-GPU experiments (Exp 1–3) used these settings. DGX B200 paired experiments have per-pair configs in `configs/experiments/` — see `docs/EXPERIMENTS.md`.

| Hyperparameter | Value |
|---------------|-------|
| **Dataset** | FineWeb-Edu sample-10BT (~10B tokens) |
| **Tokenizer** | GPT-2 (`tiktoken.get_encoding("gpt2")`) |
| **Optimizer** | AdamW (fused) |
| **Learning rate** | 6e-4 peak, cosine decay to 0 |
| **Warmup** | 700 steps (~350M tokens) |
| **Weight decay** | 0.1 (on 2D weight tensors only) |
| **Beta1 / Beta2** | 0.9 / 0.95 |
| **Gradient clipping** | 1.0 (max norm) |
| **Batch size** | 524,288 tokens per update |
| **Total steps** | 19,073 (one epoch over 10B tokens) |
| **Precision** | bfloat16 |
| **torch.compile** | Used for 128M experiment (~2.3x throughput boost) |
| **Seed** | 42 |

### Hardware
- **Baseline training:** NVIDIA RTX 5090 (32 GB), ~141K tok/s, ~19.7 hours
- **CoFrGeNet-F 82M:** H200 GPU 0, container `cofrgenet-train`, ~74K tok/s, ~37.3 hours
- **CoFrGeNet-F 128M:** H200 GPU 2, container `cofrgenet-128m`, ~105K tok/s (with `torch.compile`), ~26 hours (est.)
- **Docker image:** `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`

### Running the 128M Experiment

```bash
docker run -d --name cofrgenet-128m \
  --gpus '"device=2"' \
  -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 \
  -v "$(pwd)":/workspace -v "$(pwd)"/data:/workspace/data -v "$(pwd)"/checkpoints:/workspace/checkpoints \
  pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  bash -c "pip install tiktoken safetensors datasets > /dev/null 2>&1 && python3 scripts/03_train_cofrgenet.py --n_embd 1024 --n_head 16 --checkpoint_dir checkpoints/cofrgenet-128m --no_wandb --compile"
```

Check progress: `docker logs cofrgenet-128m --tail 10` (new checkpoint every ~2.6 hours)

## Project Structure

```
cofrgenet-f/
├── CLAUDE.md                  # This file
├── README.md                  # Public-facing project README
├── Dockerfile                 # PyTorch base image for training
├── docker-compose.yml         # Docker compose for training
├── pyproject.toml             # Package config
├── requirements.txt           # Pinned dependencies
├── src/
│   ├── cofrgenet/
│   │   ├── __init__.py
│   │   ├── model.py           # CoFrGeNet-F model (Cffn + standard attention)
│   │   ├── cffn.py            # Continued Fraction FFN layer (p-variate, gated)
│   │   ├── continuant.py      # Continuant computation + custom backward
│   │   └── config.py          # CoFrGeNetConfig dataclass
│   └── baseline/
│       ├── __init__.py
│       ├── model.py           # Standard GPT-2 (also houses shared TransformerBlock, CausalSelfAttention)
│       └── config.py          # BaselineConfig dataclass
├── scripts/
│   ├── 01_download_data.py    # Download & tokenize FineWeb-Edu
│   ├── 02_train_baseline.py   # Train standard transformer
│   ├── 03_train_cofrgenet.py  # Train CoFrGeNet-F (with dyadic schedule, configurable dimensions)
│   ├── train_common.py        # Shared: DataLoader, LR schedule, training loop, DDP/FSDP, checkpointing
│   ├── 04_evaluate.py         # Benchmark evaluation (WikiText-2/103, LAMBADA, throughput, gen speed)
│   ├── launch_training.sh     # torchrun wrapper for multi-GPU training
│   ├── train_with_autorestart.sh  # Auto-restart wrapper (crash recovery)
│   └── 05_generate_examples.py # NOT YET IMPLEMENTED
├── configs/
│   ├── baseline.yaml
│   ├── cofrgenet_f.yaml
│   └── experiments/           # Per-pair YAML configs (pair1_*, pair3_*, etc.)
├── tests/
│   ├── test_continuant.py     # Continuant math tests
│   ├── test_cffn.py           # Cffn layer tests
│   ├── test_model.py          # Full model tests
│   └── test_training.py       # Smoke test
├── demo/
│   └── app.py                 # NOT YET IMPLEMENTED
├── docs/
│   ├── plans/
│   │   └── 2026-03-01-cofrgenet-f-implementation.md
│   └── H200_TRAINING_GUIDE.md
├── checkpoints/               # .gitignored
│   ├── baseline/              # Prior exp: 124M baseline + eval_results.json
│   ├── cofrgenet/             # Prior exp 1: 82M model + eval_results.json
│   ├── cofrgenet-128m/        # Prior exp 2: 128M model
│   ├── pair1-baseline-450m/   # DGX Pair 1 baseline
│   ├── pair1-cofrgenet-410m/  # DGX Pair 1 CoFrGeNet-F
│   ├── pair3-baseline-7b/     # DGX Pair 3 baseline (COMPLETE: step 10K, 20K, 95K kept)
│   └── pair3-cofrgenet-5b/    # DGX Pair 3 CoFrGeNet-F (queued)
└── data/                      # .gitignored
    ├── tokenized/             # 10B tokens (100 train shards + 1 val shard)
    └── tokenized_50b/         # 50B tokens (for Pair 3+)
```

### Key Code Architecture Decisions

- **`TransformerBlock` takes FFN as a parameter**: Both models reuse `TransformerBlock(config, ffn_module)` from `src/baseline/model.py`.
- **Custom autograd**: `ContinuedFractionFunction` in `continuant.py` uses `torch.autograd.Function`. Proposition 1 gradients use only 1 division.
- **Dyadic schedule via forward detach**: `Cffn.set_active_depth()` controls which depth columns are active. Frozen columns are **detached in the forward pass** — they contribute to the output but receive no gradients. This replaced an earlier gradient-zeroing approach that failed under FSDP (params flattened to 1D shards). A legacy `zero_frozen_grads()` method exists as a safety fallback for non-FSDP training.
- **Checkpointing**: `safetensors.torch.save_model` for weights, separate `.pt` for optimizer state.
- **Weight tying**: Both models tie `lm_head.weight = tok_emb.weight`.
- **Configurable dimensions**: `03_train_cofrgenet.py` accepts `--n_embd`, `--n_head`, `--n_layer`, `--num_ladders`, `--cf_depth`, `--checkpoint_dir` for training different model sizes.

## HuggingFace

- **Prior experiments:** [`cahlen/cofrgenet-f`](https://huggingface.co/cahlen/cofrgenet-f) — 82M, 128M models + eval results
- **Pair 3 baseline:** [`cahlen/pair3-baseline-7b`](https://huggingface.co/cahlen/pair3-baseline-7b) — step 10K (best LLM) + step 95K (final), eval results, detailed model card
- **Pair 3 CoFrGeNet-F:** `cahlen/pair3-cofrgenet-5b` (to be created after training)
- **Trackio dashboard:** [`cahlen/cofrgenet-f-trackio`](https://huggingface.co/spaces/cahlen/cofrgenet-f-trackio) — live training metrics
- **Note:** HuggingFace does NOT support inline `$...$` math — only `$$` display blocks. GitHub supports both.

## Reference Links

- **CoFrGeNet paper:** https://arxiv.org/abs/2601.21766
- **CoFrGeNet HTML (full tables):** https://arxiv.org/html/2601.21766v2
- **CoFrNet (predecessor, NeurIPS 2021):** https://arxiv.org/abs/2506.05586
- **FineWeb-Edu dataset:** https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **nanoGPT (reference implementation):** https://github.com/karpathy/nanoGPT
- **Project Wiki:** https://github.com/cahlen/cofrgenet-f/wiki

## IBM Patent Note

IBM has US Patent Application #20230401438 on CoFrNets. The paper itself is CC BY 4.0. Our implementation is from-scratch based on published math. This is an academic reproduction / educational project.
