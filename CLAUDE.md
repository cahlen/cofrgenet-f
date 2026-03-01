# CLAUDE.md — CoFrGeNet-F: First Open-Source Continued Fraction Language Model

## What This Project Is

This is the **first open-source implementation** of CoFrGeNet-F, a continued fraction architecture that replaces Transformer FFN layers with continued fraction networks. Based on IBM Research's paper [arXiv:2601.21766](https://arxiv.org/abs/2601.21766) (January 2026). No public code exists — we are implementing from the paper's math.

**Goal:** Train a 125M-parameter CoFrGeNet-F model and a standard 125M Transformer baseline on identical data, then release both with a head-to-head comparison blog post and interactive Gradio demo.

## Architecture Overview

### What Is CoFrGeNet-F?

A standard Transformer block has two components: **Multi-Head Attention** and a **Feed-Forward Network (FFN)**. CoFrGeNet-F keeps standard attention but replaces the FFN with a **Continued Fraction FFN (Cffn)**.

A continued fraction computes:

```
f(a₀, a) = a₀ + 1/(a₁ + 1/(a₂ + ... + 1/a_d))
```

where each `a_k = w_k · x` (learnable weight times input). The key insight: this can be expressed as a ratio of **continuant polynomials** `K`, computed via a simple recursion:

```
K₀ = 1
K₁(a_d) = a_d
K_k = a_{d-k+1} · K_{k-1} + K_{k-2}
```

The continued fraction then equals `K_{d-1} / K_d`, and gradients are:

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

### Cffn Architecture (What We Implement)

The Cffn replaces a standard 2-layer FFN (`Linear(p, 4p) → GELU → Linear(4p, p)`) with:

1. **L continued fraction ladders**, each of depth d, operating on the full p-dimensional input
2. A **direct linear path** (skip connection through the fraction)
3. A **linear combination layer** that merges ladder outputs

```
y = U·x + V·z
where z_j = f̃(W^(j) · x)   for j = 1, ..., L
```

- `U` ∈ R^{p×p} — direct linear path
- `V` ∈ R^{p×L} — combination weights for ladder outputs
- `W^(j)` ∈ R^{(d+1)×p} — weights for each ladder

**Parameter count:** `L·p·(d+1) + 2p²` (vs standard FFN: `2·α·p²` where α=4)

For our 125M model (p=768): standard FFN per layer = 2×4×768² ≈ 4.7M params. With Cffn (L=3, d=5): L·p·(d+1) + 2p² = 3·768·6 + 2·768² ≈ 1.2M params — roughly **4x fewer per layer**.

### Pole Avoidance

Continued fractions can have poles (division by zero). The fix:

```python
K_d = torch.sign(K_d) * torch.clamp(K_d.abs(), min=0.01)
```

### Dyadic Training Schedule (CRITICAL)

Without this, performance degrades 10-80%. The schedule progressively unfreezes continued fraction depth:

```
Depth 0 (linear component U): trained from step 0
Depth i parameters: unfrozen at step (1 - 1/2^i) × total_steps
```

For a 19,000-step training run:
- Step 0: Only linear components train
- Step 9,500: Depth 1 (first fraction level) unfreezes
- Step 14,250: Depth 2 unfreezes
- Step 16,625: Depth 3 unfreezes
- Step 17,813: Depth 4 unfreezes
- Step 18,406: Depth 5 unfreezes

## Model Configurations

### CoFrGeNet-F (125M target)

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Attention heads | 12 |
| Hidden dim (p) | 768 |
| Sequence length | 1024 |
| Vocab size | 50,257 (GPT-2 tokenizer) |
| Attention | Standard multi-head (unchanged) |
| FFN replacement | Cffn with L=3 ladders, d=5 depth |
| Dropout | 0.0 |
| Bias | False |

The exact parameter count will be lower than 125M due to the Cffn savings. This is expected and is the whole point — we compare against a standard 125M Transformer to show competitive quality at fewer parameters.

### Baseline Transformer (125M)

Standard GPT-2 Small architecture:

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Attention heads | 12 |
| Hidden dim | 768 |
| FFN inner dim | 3072 (4× expansion) |
| Sequence length | 1024 |
| Vocab size | 50,257 |
| Dropout | 0.0 |
| Bias | False |

## Training Recipe

Both models trained identically except for the FFN architecture.

| Hyperparameter | Value |
|---------------|-------|
| **Dataset** | FineWeb-Edu sample-10BT (~10B tokens, ~28.5 GB) |
| **Tokenizer** | GPT-2 (`tiktoken.get_encoding("gpt2")`) |
| **Optimizer** | AdamW |
| **Learning rate** | 6e-4 peak, cosine decay to 0 |
| **Warmup** | 700 steps (~350M tokens) |
| **Weight decay** | 0.1 (on 2D weight tensors only) |
| **Beta1 / Beta2** | 0.9 / 0.95 |
| **Gradient clipping** | 1.0 (max norm) |
| **Batch size** | 524,288 tokens per update (micro-batch × grad accum × seq_len) |
| **Total steps** | ~19,073 (one epoch over 10B tokens) |
| **Precision** | bfloat16 |
| **Seed** | 42 |

### Dataset

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
```

HuggingFace dataset ID: `HuggingFaceFW/fineweb-edu`, subset `sample-10BT`. Fields: `text`, `id`, `score` (educational quality 0-5).

### Hardware

- **Training GPU:** NVIDIA RTX 5090 (32 GB GDDR7, ~209 BF16 TFLOPS, 1,792 GB/s bandwidth)
- **Expected throughput:** ~150,000-180,000 tokens/sec for 125M model
- **Expected training time:** ~4 hours per model for 2.5B tokens, ~15-16 hours for full 10B tokens
- **Note:** Use `torch.compile` for optimal performance. Flash Attention should work on RTX 5090 (SM 10.0 / Blackwell consumer).

## Project Structure

```
cofrgenet-f/
├── CLAUDE.md              # This file — all context for implementation
├── README.md              # Public-facing project README
├── pyproject.toml         # Package config
├── requirements.txt       # Pinned dependencies
├── src/
│   ├── cofrgenet/
│   │   ├── __init__.py
│   │   ├── model.py       # CoFrGeNet-F model (Cffn + standard attention)
│   │   ├── cffn.py        # Continued Fraction FFN layer
│   │   ├── continuant.py  # Continuant computation + custom backward
│   │   └── config.py      # Model config dataclass
│   └── baseline/
│       ├── __init__.py
│       ├── model.py       # Standard GPT-2 Small transformer
│       └── config.py      # Baseline config dataclass
├── scripts/
│   ├── 01_download_data.py    # Download & tokenize FineWeb-Edu 10BT
│   ├── 02_train_baseline.py   # Train standard transformer
│   ├── 03_train_cofrgenet.py  # Train CoFrGeNet-F (with dyadic schedule)
│   ├── 04_evaluate.py         # Eval both models on benchmarks
│   └── 05_generate_examples.py # Generate text samples for comparison
├── configs/
│   ├── baseline.yaml
│   └── cofrgenet_f.yaml
├── tests/
│   ├── __init__.py
│   ├── test_continuant.py     # Unit tests for continuant math
│   ├── test_cffn.py           # Unit tests for Cffn layer
│   ├── test_model.py          # Integration tests for full model
│   └── test_training.py       # Smoke test for training loop
├── demo/
│   └── app.py                 # Gradio demo (side-by-side generation)
├── docs/
│   └── plans/
│       └── 2026-03-01-cofrgenet-f-implementation.md
└── data/                      # .gitignored — created at runtime
    ├── raw/                   # Downloaded FineWeb-Edu parquet
    └── tokenized/             # Tokenized binary shards
```

## Implementation Priorities

1. **Continuant computation first** (`src/cofrgenet/continuant.py`) — this is the mathematical core. Implement forward + custom backward with the gradient formula from Proposition 1. Write thorough unit tests comparing against naive recursive computation.

2. **Cffn layer** (`src/cofrgenet/cffn.py`) — ensemble of L ladders with direct linear path. Test parameter counts match expected formulas.

3. **Full CoFrGeNet-F model** (`src/cofrgenet/model.py`) — standard Transformer with Cffn replacing FFN. Should be a drop-in replacement — the model is identical to baseline except for the FFN.

4. **Baseline Transformer** (`src/baseline/model.py`) — clean GPT-2 Small implementation. Keep it simple, nanoGPT-style.

5. **Data pipeline** — download FineWeb-Edu 10BT, tokenize with GPT-2 tokenizer, shard into binary files.

6. **Training scripts** — train both models with identical hyperparameters. The only difference: CoFrGeNet-F uses the dyadic schedule for Cffn parameters.

7. **Evaluation** — perplexity on WikiText-2, WikiText-103, LAMBADA. Parameter count comparison. Throughput comparison (tokens/sec).

8. **Demo + release** — Gradio app, HuggingFace model upload, blog post.

## Key Implementation Details

### Custom Autograd for Continuants

Do NOT rely on PyTorch autograd for the continued fraction computation. The paper explicitly derives custom gradients because:
- Naive autograd requires d divisions (numerically unstable)
- Custom backward uses only 1 division (the `1/K_d` term)

Use `torch.autograd.Function` with a custom `backward()` method.

### The Cffn Forward Pass

```python
# Pseudocode for Cffn forward
def forward(self, x):
    # x: (batch, seq_len, p)

    # Direct linear path
    linear_out = self.U(x)  # (batch, seq_len, p)

    # Continued fraction ladders
    ladder_outputs = []
    for j in range(self.L):
        a = self.W[j](x)  # (batch, seq_len, d+1)
        z_j = continued_fraction(a)  # (batch, seq_len, 1) — the f̃ value
        ladder_outputs.append(z_j)

    z = torch.cat(ladder_outputs, dim=-1)  # (batch, seq_len, L)
    combined = self.V(z)  # (batch, seq_len, p)

    return linear_out + combined
```

### Dyadic Schedule Implementation

```python
def get_unfrozen_depth(current_step, total_steps, max_depth):
    """Return the maximum depth that should be unfrozen at current_step."""
    for d in range(max_depth, 0, -1):
        unfreeze_at = total_steps * (1 - 1 / (2 ** d))
        if current_step >= unfreeze_at:
            return d
    return 0  # only linear component
```

Freeze/unfreeze Cffn parameters each step based on this schedule.

### Shared Components

Both models should share:
- Token + positional embeddings
- Layer norm
- Attention implementation
- Training loop
- Data loading

Only the FFN differs. Factor the code so the Transformer block takes the FFN as a parameter.

## Dependencies

Core:
- `torch` (2.5+, with CUDA support for RTX 5090)
- `tiktoken` (GPT-2 tokenizer)
- `datasets` (HuggingFace, for FineWeb-Edu streaming)
- `safetensors` (model saving)
- `wandb` (training logging)
- `gradio` (demo)

Testing:
- `pytest`

## Evaluation Benchmarks

After training, evaluate both models on:

| Benchmark | Metric | How |
|-----------|--------|-----|
| WikiText-2 | Perplexity | Stride-512 evaluation |
| WikiText-103 | Perplexity | Stride-512 evaluation |
| LAMBADA | Perplexity + Accuracy | Last-word prediction |
| HellaSwag | Accuracy | Zero-shot, multiple choice |
| Parameter count | Total params | `sum(p.numel() for p in model.parameters())` |
| Throughput | Tokens/sec | Measure during training |
| Inference speed | ms/token | Measure during generation |

## Reference Links

- **CoFrGeNet paper:** https://arxiv.org/abs/2601.21766
- **CoFrGeNet HTML (full tables):** https://arxiv.org/html/2601.21766v2
- **CoFrNet (predecessor, NeurIPS 2021):** https://arxiv.org/abs/2506.05586
- **CoFrNet in AIX360:** https://github.com/Trusted-AI/AIX360/tree/master/examples/cofrnet
- **FineWeb-Edu dataset:** https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **nanoGPT (reference implementation):** https://github.com/karpathy/nanoGPT
- **llm.c GPT-2 reproduction:** https://github.com/karpathy/llm.c/discussions/481

## IBM Patent Note

IBM has US Patent Application #20230401438 on CoFrNets. The paper itself is CC BY 4.0. Our implementation is from-scratch based on published math. This is an academic reproduction / educational project.
