# CLAUDE.md — CoFrGeNet-F: Continued Fraction Language Model

## What This Project Is

An open-source implementation of CoFrGeNet-F, a continued fraction architecture that replaces Transformer FFN layers with continued fraction networks. Based on IBM Research's paper [arXiv:2601.21766](https://arxiv.org/abs/2601.21766) (January 2026). Implemented from the paper's mathematics.

**Goal:** Train CoFrGeNet-F models at multiple scales and compare head-to-head against standard Transformer baselines on identical data. Release both with benchmarks, Gradio demo, and technical write-up.

## Current Status (2026-03-06)

### Completed
- **All core architecture**: continuant.py, cffn.py, both models, configs, tests
- **Data pipeline**: FineWeb-Edu 10BT downloaded and tokenized (100 train shards + 1 val shard in `data/tokenized/`)
- **Training infrastructure**: shared training loop, dyadic schedule, checkpointing, CLI args for model dimensions
- **Baseline model (124M)**: Fully trained (19,073 steps on RTX 5090, ~19.7 hours, ~141K tok/s)
- **CoFrGeNet-F 82M (Experiment 1)**: Fully trained (19,073 steps on H200, ~37.3 hours, ~74K tok/s)
- **Evaluation script**: `scripts/04_evaluate.py` — WikiText-2, WikiText-103, LAMBADA, throughput, generation speed
- **All models evaluated**: Head-to-head on same H200, same code (see Results below)
- **CoFrGeNet-F 128M (Experiment 2)**: Fully trained (19,073 steps on H200, ~24.3 hours, ~114K tok/s with `torch.compile`)
- **HuggingFace repo**: Public at [`cahlen/cofrgenet-f`](https://huggingface.co/cahlen/cofrgenet-f) with both model weights + eval results
- **GitHub Wiki**: 7 pages of detailed architecture + math documentation

### Remaining
- Update HuggingFace repo with 128M results
- `scripts/05_generate_examples.py` — text generation comparison
- `demo/app.py` — Gradio demo (side-by-side generation)
- Blog post / technical write-up

## Experiments

### Why Multiple Experiments?

The paper's strongest results were at GPT-2 XL scale (985M CoFrGeNet-F vs 1.5B baseline). Our Experiment 1 at 82M (34% fewer params than baseline) showed CoFrGeNet-F underperforming. Experiment 2 tests whether matching the baseline's parameter count closes the gap — an **iso-parameter comparison** that isolates the architectural difference.

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

Without this, performance degrades 10-80%. Progressively unfreezes continued fraction depth via gradient hooks:

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

## Training Recipe

All models trained with identical hyperparameters on the same data.

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
│   ├── 01_download_data.py    # Download & tokenize FineWeb-Edu 10BT
│   ├── 02_train_baseline.py   # Train standard transformer
│   ├── 03_train_cofrgenet.py  # Train CoFrGeNet-F (with dyadic schedule, configurable dimensions)
│   ├── train_common.py        # Shared: DataLoader, LR schedule, training loop, checkpointing
│   ├── 04_evaluate.py         # Benchmark evaluation (WikiText-2/103, LAMBADA, throughput, gen speed)
│   └── 05_generate_examples.py # NOT YET IMPLEMENTED
├── configs/
│   ├── baseline.yaml
│   └── cofrgenet_f.yaml
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
│   ├── baseline/              # Final baseline model + eval_results.json
│   ├── cofrgenet/             # Experiment 1: 82M model + eval_results.json
│   └── cofrgenet-128m/        # Experiment 2: 128M model (training in progress)
└── data/                      # .gitignored
    └── tokenized/             # 100 train shards + 1 val shard (uint16 .bin)
```

### Key Code Architecture Decisions

- **`TransformerBlock` takes FFN as a parameter**: Both models reuse `TransformerBlock(config, ffn_module)` from `src/baseline/model.py`.
- **Custom autograd**: `ContinuedFractionFunction` in `continuant.py` uses `torch.autograd.Function`. Proposition 1 gradients use only 1 division.
- **Dyadic schedule via gradient hooks**: `Cffn.set_active_depth()` installs `register_hook` callbacks that zero out gradients for frozen depth columns.
- **Checkpointing**: `safetensors.torch.save_model` for weights, separate `.pt` for optimizer state.
- **Weight tying**: Both models tie `lm_head.weight = tok_emb.weight`.
- **Configurable dimensions**: `03_train_cofrgenet.py` accepts `--n_embd`, `--n_head`, `--n_layer`, `--num_ladders`, `--cf_depth`, `--checkpoint_dir` for training different model sizes.

## HuggingFace

- **Repo:** [`cahlen/cofrgenet-f`](https://huggingface.co/cahlen/cofrgenet-f) (public)
- **Structure:** `cofrgenet/model.safetensors`, `baseline/model.safetensors`, `src/`, eval results
- **Model card:** LaTeX math, head-to-head comparison table, full architecture docs
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
