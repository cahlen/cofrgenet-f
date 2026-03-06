# CLAUDE.md — CoFrGeNet-F: First Open-Source Continued Fraction Language Model

## What This Project Is

This is the **first open-source implementation** of CoFrGeNet-F, a continued fraction architecture that replaces Transformer FFN layers with continued fraction networks. Based on IBM Research's paper [arXiv:2601.21766](https://arxiv.org/abs/2601.21766) (January 2026). No public code exists — we implemented from the paper's math.

**Goal:** Train a 125M-parameter CoFrGeNet-F model and a standard 125M Transformer baseline on identical data, then release both with a head-to-head comparison blog post and interactive Gradio demo.

## Current Status (2026-03-05)

### Completed
- **All core architecture**: continuant.py, cffn.py, both models, configs, tests
- **Data pipeline**: FineWeb-Edu 10BT downloaded and tokenized (100 train shards + 1 val shard in `data/tokenized/`)
- **Training infrastructure**: shared training loop, dyadic schedule, checkpointing
- **Baseline model**: Fully trained (19,073 steps on RTX 5090, ~19.7 hours, ~141K tok/s)
- **Baseline evaluation**: WikiText-2 PPL 34.13, LAMBADA PPL 37.47, LAMBADA acc 19.15%
- **Baseline released on HuggingFace** at `checkpoints/baseline/`
- **GitHub Wiki**: 7 pages of detailed architecture + math documentation

### In Progress
- **CoFrGeNet-F training**: Running on H200 via Docker container `cofrgenet-train`. ~74K tok/s. Dyadic schedule depth 1 unfroze at step 9,537 (big loss spike, recovered), depth 2 unfroze at step 14,305 (smooth). Depths 3–5 still to come.

### Remaining
- `scripts/04_evaluate.py` — benchmark evaluation script
- `scripts/05_generate_examples.py` — text generation comparison
- `demo/app.py` — Gradio demo (side-by-side generation)
- CoFrGeNet-F HuggingFace release
- Blog post / technical write-up

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

### Why CoFrGeNet-F Specifically?

The paper tested three variants. CoFrGeNet-F (FFN-only replacement) is the best:

| Variant | Params (GPT2-xl scale) | What's Replaced | Result |
|---------|----------------------|-----------------|--------|
| CoFrGeNet-F | 985M (34% fewer) | FFN only | **Best on most benchmarks** |
| CoFrGeNet-A | 1.21B (19% fewer) | Attention only | Modest gains |
| CoFrGeNet | 798M (47% fewer) | Both | Competitive but trails -F |

CoFrGeNet-F beat the full 1.5B GPT2-xl on all 8 GLUE tasks and all 6 perplexity benchmarks.

### Cffn Architecture (What We Implemented)

The Cffn replaces a standard 2-layer FFN (`Linear(p, 4p) → GELU → Linear(4p, p)`) with:

1. **A direct linear path** `U` (p×p) — skip connection through the fraction
2. **A gating projection** `G` (p×p) — `gated_x = sigmoid(G·x) ⊙ x`
3. **L p-variate continued fraction ladders**, each of depth d, operating element-wise per hidden dimension
4. **Combination weights** `V` (p×L) — per-dimension weighting of ladder outputs

```
y = U·x + V·z
where z_j = f̃(gated_x ⊙ W^(j))   for j = 1, ..., L
```

Each ladder is **p-variate**: `W^(j)` has shape `(p, d)` and multiplies element-wise with the p-dim gated input, producing p independent continued fractions per ladder.

- `U` ∈ R^{p×p} — direct linear path (589,824 params)
- `G` ∈ R^{p×p} — gate projection (589,824 params)
- `W^(j)` ∈ R^{p×d} — weights for each ladder (3,840 params each)
- `V` ∈ R^{p×L} — combination weights (2,304 params)

**Validated parameter count per Cffn layer:** `2p² + L·p·(d+1) = 2×768² + 3×768×6 = 1,193,472`

### Pole Avoidance

Continued fractions can have poles (division by zero). The fix:

```python
K_d_safe = torch.sign(K_d) * torch.clamp(K_d.abs(), min=0.01)
K_d_safe = torch.where(K_d_safe == 0, torch.full_like(K_d_safe, epsilon), K_d_safe)
```

### Dyadic Training Schedule (CRITICAL)

Without this, performance degrades 10-80%. The schedule progressively unfreezes continued fraction depth via gradient hooks that mask frozen ladder weight columns:

```
Depth 0 (linear components U, G, V): trained from step 0
Depth i parameters: unfrozen at step (1 - 1/2^i) × total_steps
```

**Validated unfreeze steps** for our 19,073-step run:

| Depth | Step | % of training | Status |
|-------|------|---------------|--------|
| 0 (linear only) | 0 | 0% | ✅ Done |
| 1 | 9,537 | 50.0% | ✅ Unfroze — loss spike 3.96→6.82, recovered in ~500 steps |
| 2 | 14,305 | 75.0% | ✅ Unfroze — smooth, no visible spike |
| 3 | 16,689 | 87.5% | Pending |
| 4 | 17,881 | 93.75% | Pending |
| 5 | 18,477 | 96.875% | Pending |

## Validated Model Parameter Counts

### Baseline Transformer: 124,337,664 parameters

| Component | Shape | Params |
|-----------|-------|--------|
| Token embedding | 50,257 × 768 | 38,597,376 |
| Position embedding | 1,024 × 768 | 786,432 |
| **Per block (×12):** | | |
| └ LayerNorm 1 | 768 | 768 |
| └ Attention QKV | 2,304 × 768 | 1,769,472 |
| └ Attention out_proj | 768 × 768 | 589,824 |
| └ LayerNorm 2 | 768 | 768 |
| └ FFN fc1 | 3,072 × 768 | 2,359,296 |
| └ FFN fc2 | 768 × 3,072 | 2,359,296 |
| Block total | | 7,079,424 |
| 12 blocks | | 84,953,088 |
| Final LayerNorm | 768 | 768 |
| LM head | tied with tok_emb | 0 |
| **Total** | | **124,337,664** |

### CoFrGeNet-F: 82,036,224 parameters

| Component | Shape | Params |
|-----------|-------|--------|
| Token embedding | 50,257 × 768 | 38,597,376 |
| Position embedding | 1,024 × 768 | 786,432 |
| **Per block (×12):** | | |
| └ LayerNorm 1 | 768 | 768 |
| └ Attention QKV | 2,304 × 768 | 1,769,472 |
| └ Attention out_proj | 768 × 768 | 589,824 |
| └ LayerNorm 2 | 768 | 768 |
| └ Cffn U | 768 × 768 | 589,824 |
| └ Cffn gate_proj | 768 × 768 | 589,824 |
| └ Cffn ladder_weights (×3) | 768 × 5 | 3,840 each (11,520) |
| └ Cffn V | 768 × 3 | 2,304 |
| Block total | | 3,554,304 |
| 12 blocks | | 42,651,648 |
| Final LayerNorm | 768 | 768 |
| LM head | tied with tok_emb | 0 |
| **Total** | | **82,036,224** |

**Reduction**: 42,301,440 fewer params (34.0%). FFN→Cffn ratio: 3.95×.

## Baseline Results

| Benchmark | Metric | Value |
|-----------|--------|-------|
| WikiText-2 | Perplexity | **34.13** |
| WikiText-103 | Perplexity | **34.13** |
| LAMBADA | Perplexity | **37.47** |
| LAMBADA | Accuracy | **19.15%** |
| Throughput | tok/s | 442,851 |
| Generation speed | ms/tok | 1.41 |

Trained on RTX 5090 (32 GB), 19,073 steps, ~19.7 hours, ~141K tok/s.

## Training Recipe

Both models trained identically except for the FFN architecture.

| Hyperparameter | Value |
|---------------|-------|
| **Dataset** | FineWeb-Edu sample-10BT (~10B tokens, ~28.5 GB) |
| **Tokenizer** | GPT-2 (`tiktoken.get_encoding("gpt2")`) |
| **Optimizer** | AdamW (fused) |
| **Learning rate** | 6e-4 peak, cosine decay to 0 |
| **Warmup** | 700 steps (~350M tokens) |
| **Weight decay** | 0.1 (on 2D weight tensors only) |
| **Beta1 / Beta2** | 0.9 / 0.95 |
| **Gradient clipping** | 1.0 (max norm) |
| **Batch size** | 524,288 tokens per update (micro-batch × grad accum × seq_len) |
| **Total steps** | 19,073 (one epoch over 10B tokens) |
| **Precision** | bfloat16 |
| **Seed** | 42 |

### Dataset

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
```

HuggingFace dataset ID: `HuggingFaceFW/fineweb-edu`, subset `sample-10BT`. Fields: `text`, `id`, `score` (educational quality 0-5).

Data is tokenized into 100 binary shards (~100M tokens each) + 1 validation shard, stored in `data/tokenized/` as uint16 `.bin` files.

### Hardware

- **Baseline training:** NVIDIA RTX 5090 (32 GB GDDR7), ~141K tok/s, ~19.7 hours
- **CoFrGeNet-F training:** NVIDIA H200 (80 GB HBM3e) via Docker, ~74K tok/s
- **Docker image:** `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` (see `Dockerfile`)
- **Container:** `cofrgenet-train` — currently running CoFrGeNet-F training

## Project Structure

```
cofrgenet-f/
├── CLAUDE.md                  # This file — all context for implementation
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
│       ├── model.py           # Standard GPT-2 Small (also houses shared TransformerBlock, CausalSelfAttention)
│       └── config.py          # BaselineConfig dataclass
├── scripts/
│   ├── 01_download_data.py    # Download & tokenize FineWeb-Edu 10BT
│   ├── 02_train_baseline.py   # Train standard transformer
│   ├── 03_train_cofrgenet.py  # Train CoFrGeNet-F (with dyadic schedule)
│   ├── train_common.py        # Shared: DataLoader, LR schedule, training loop, checkpointing
│   ├── 04_evaluate.py         # NOT YET IMPLEMENTED — eval both models on benchmarks
│   └── 05_generate_examples.py # NOT YET IMPLEMENTED — text generation comparison
├── configs/
│   ├── baseline.yaml
│   └── cofrgenet_f.yaml
├── tests/
│   ├── __init__.py
│   ├── test_continuant.py     # Continuant math: base cases, naive recursion, gradients, finite-diff, pole avoidance
│   ├── test_cffn.py           # Cffn: shapes, param counts, gradients, depth freezing, gating
│   ├── test_model.py          # Full model: forward shapes, param counts, causal masking, generation, dyadic schedule
│   └── test_training.py       # Smoke test: DataLoader, LR schedule, 10-step training
├── demo/
│   └── app.py                 # NOT YET IMPLEMENTED — Gradio demo
├── docs/
│   ├── plans/
│   │   └── 2026-03-01-cofrgenet-f-implementation.md
│   └── H200_TRAINING_GUIDE.md # Guide for remote H200 training setup
├── checkpoints/               # .gitignored except HuggingFace release files
│   ├── baseline/              # Final baseline model + eval_results.json + HF README
│   └── cofrgenet/             # Step checkpoints (every 1K steps) + optimizer state
└── data/                      # .gitignored — created at runtime
    └── tokenized/             # 100 train shards + 1 val shard (uint16 .bin)
```

### Key Code Architecture Decisions

- **`TransformerBlock` takes FFN as a parameter**: Both models reuse the same `TransformerBlock(config, ffn_module)` class from `src/baseline/model.py`. The CoFrGeNet-F model imports `TransformerBlock` and `CausalSelfAttention` from the baseline module.
- **Custom autograd**: `ContinuedFractionFunction` in `continuant.py` uses `torch.autograd.Function` — do NOT rely on PyTorch autograd for the continued fraction. Proposition 1 gradients use only 1 division.
- **Dyadic schedule via gradient hooks**: `Cffn.set_active_depth()` installs `register_hook` callbacks on ladder weights that zero out gradients for frozen depth columns. The `make_hook(max_active)` closure captures the active depth correctly.
- **Checkpointing**: Uses `safetensors.torch.save_model` (not `save_file`) for model weights, separate `.pt` files for optimizer state.
- **Weight tying**: Both models tie `lm_head.weight = tok_emb.weight`.

## Evaluation Benchmarks

After training, evaluate both models on:

| Benchmark | Metric | How | Baseline Result |
|-----------|--------|-----|-----------------|
| WikiText-2 | Perplexity | Stride-512 evaluation | 34.13 |
| WikiText-103 | Perplexity | Stride-512 evaluation | 34.13 |
| LAMBADA | Perplexity + Accuracy | Last-word prediction | 37.47 / 19.15% |
| HellaSwag | Accuracy | Zero-shot, multiple choice | Not yet evaluated |
| Parameter count | Total params | `sum(p.numel() for p in model.parameters())` | 124,337,664 |
| Throughput | Tokens/sec | Measure during training | 141K (5090) |
| Inference speed | ms/token | Measure during generation | 1.41 |

## Reference Links

- **CoFrGeNet paper:** https://arxiv.org/abs/2601.21766
- **CoFrGeNet HTML (full tables):** https://arxiv.org/html/2601.21766v2
- **CoFrNet (predecessor, NeurIPS 2021):** https://arxiv.org/abs/2506.05586
- **CoFrNet in AIX360:** https://github.com/Trusted-AI/AIX360/tree/master/examples/cofrnet
- **FineWeb-Edu dataset:** https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **nanoGPT (reference implementation):** https://github.com/karpathy/nanoGPT
- **llm.c GPT-2 reproduction:** https://github.com/karpathy/llm.c/discussions/481
- **Project Wiki:** https://github.com/cahlen/cofrgenet-f/wiki

## IBM Patent Note

IBM has US Patent Application #20230401438 on CoFrNets. The paper itself is CC BY 4.0. Our implementation is from-scratch based on published math. This is an academic reproduction / educational project.
