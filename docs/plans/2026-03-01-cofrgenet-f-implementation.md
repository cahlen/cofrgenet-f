# CoFrGeNet-F Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the first open-source CoFrGeNet-F (continued fraction FFN) language model, train it alongside a standard Transformer baseline on FineWeb-Edu, and release both with benchmarks, a Gradio demo, and a technical write-up.

**Architecture:** CoFrGeNet-F replaces standard Transformer FFN layers with continued fraction networks (Cffn) while keeping multi-head attention unchanged. The Cffn uses ensembles of continued fraction "ladders" with a direct linear path, computed efficiently via continuant polynomials with custom gradients. A dyadic training schedule progressively unfreezes deeper fraction levels.

**Tech Stack:** PyTorch, tiktoken, HuggingFace datasets, safetensors, wandb, gradio, pytest

---

## Task 1: Environment Setup

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`

**Step 1: Create requirements.txt**

```txt
torch>=2.5.0
tiktoken>=0.8.0
datasets>=3.0.0
safetensors>=0.4.0
wandb>=0.18.0
gradio>=5.0.0
numpy>=1.26.0
pyyaml>=6.0
tqdm>=4.66.0
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "cofrgenet-f"
version = "0.1.0"
description = "First open-source CoFrGeNet-F: Continued Fraction FFN for Language Models"
requires-python = ">=3.10"
license = {text = "MIT"}

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 3: Create and activate virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

**Step 4: Verify CUDA is available**

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected: `CUDA: True, Device: NVIDIA GeForce RTX 5090`

**Step 5: Commit**

```bash
git add requirements.txt pyproject.toml
git commit -m "feat: add project dependencies and config"
```

---

## Task 2: Continuant Computation Core

**Files:**
- Create: `src/cofrgenet/continuant.py`
- Create: `tests/test_continuant.py`

This is the mathematical heart of the project. The continuant polynomial K is defined recursively:

```
K_0 = 1
K_1(a_d) = a_d
K_k(a_{d-k+1}, ..., a_d) = a_{d-k+1} * K_{k-1} + K_{k-2}
```

The continued fraction value is `f̃(a) = K_{d-1}(a_2,...,a_d) / K_d(a_1,...,a_d)`.

The gradient (Proposition 1 from the paper) is:
```
∂f̃/∂a_k = (-1)^k * [K_{d-k}(a_{k+1},...,a_d) / K_d(a_1,...,a_d)]²
```

**Step 1: Write tests for continuant computation**

Test cases:
- `test_continuant_base_cases`: K_0=1, K_1(a)=a
- `test_continuant_depth_2`: K_2(a,b) = a*b + 1 (verify by hand)
- `test_continuant_depth_3`: K_3(a,b,c) = a*b*c + a + c (verify by hand)
- `test_continued_fraction_depth_1`: f̃([a]) = 1/a
- `test_continued_fraction_depth_2`: f̃([a,b]) = b/(a*b+1)
- `test_gradient_matches_autograd_naive`: Compare custom gradient against naive torch.autograd through recursive fraction computation. Use several random inputs, depths 1-7.
- `test_pole_avoidance`: When a value would cause K_d ≈ 0, verify clamping activates
- `test_batched_computation`: Verify works with batch dimensions (B, S, d+1)
- `test_gradient_numerical`: Finite-difference gradient check against custom backward

Run: `pytest tests/test_continuant.py -v`
Expected: All FAIL (not implemented yet)

**Step 2: Implement ContinuedFractionFunction (torch.autograd.Function)**

In `src/cofrgenet/continuant.py`:

```python
import torch
from torch.autograd import Function

class ContinuedFractionFunction(Function):
    """Custom autograd for continued fraction via continuants.

    Forward: compute f̃(a) = K_{d-1}(a_2,...,a_d) / K_d(a_1,...,a_d)
    Backward: ∂f̃/∂a_k = (-1)^k * [K_{d-k} / K_d]² (Proposition 1)
    """

    @staticmethod
    def forward(ctx, a, epsilon=0.01):
        # a: (..., d) where d is the continued fraction depth
        # Returns: (...,) scalar output per element
        d = a.shape[-1]

        # Compute all continuants K_0 through K_d
        # K[i] = K_i(a_{d-i+1}, ..., a_d)
        K = [None] * (d + 1)
        K[0] = torch.ones_like(a[..., 0])
        K[1] = a[..., -1]  # K_1 = a_d
        for i in range(2, d + 1):
            # K_i = a_{d-i+1} * K_{i-1} + K_{i-2}
            K[i] = a[..., d - i] * K[i - 1] + K[i - 2]

        # Pole avoidance on K_d
        K_d = K[d]
        K_d_safe = torch.sign(K_d) * torch.clamp(K_d.abs(), min=epsilon)

        # f̃ = K_{d-1} / K_d
        if d >= 2:
            result = K[d - 1] / K_d_safe
        else:
            # d=1: f̃([a_1]) = 1/a_1 = K_0/K_1
            result = K[0] / K_d_safe

        # Save for backward
        ctx.save_for_backward(a)
        ctx.K = K
        ctx.K_d_safe = K_d_safe
        ctx.epsilon = epsilon

        return result

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        K = ctx.K
        K_d_safe = ctx.K_d_safe
        d = a.shape[-1]

        # ∂f̃/∂a_k = (-1)^k * [K_{d-k} / K_d]² for k=1,...,d
        inv_K_d_sq = (1.0 / K_d_safe) ** 2

        grad_a = torch.zeros_like(a)
        for k in range(1, d + 1):
            sign = (-1.0) ** k
            K_d_minus_k = K[d - k]  # K_{d-k}(a_{k+1},...,a_d)
            grad_a[..., k - 1] = sign * K_d_minus_k ** 2 * inv_K_d_sq

        grad_a = grad_a * grad_output.unsqueeze(-1)
        return grad_a, None  # None for epsilon


def continued_fraction(a, epsilon=0.01):
    """Compute continued fraction f̃(a) with custom backward."""
    return ContinuedFractionFunction.apply(a, epsilon)
```

**Step 3: Run tests**

```bash
pytest tests/test_continuant.py -v
```

Expected: All PASS

**Step 4: Commit**

```bash
git add src/cofrgenet/continuant.py tests/test_continuant.py
git commit -m "feat: continuant-based continued fraction with custom backward"
```

---

## Task 3: Cffn Layer (Continued Fraction FFN)

**Files:**
- Create: `src/cofrgenet/cffn.py`
- Create: `tests/test_cffn.py`

The Cffn replaces the standard FFN. It consists of:
- L continued fraction ladders, each of depth d, operating on p-dim input
- A direct linear path U (p → p)
- A combination layer V (L → p) that merges ladder outputs

Output: `y = U·x + V·z` where `z_j = f̃(W^(j) · x)` for j=1,...,L

**Step 1: Write tests**

Test cases:
- `test_cffn_output_shape`: Input (B, S, p) → output (B, S, p)
- `test_cffn_parameter_count`: Verify params = L*p*(d+1) + 2*p² (approximately, accounting for biases)
- `test_cffn_forward_backward`: Verify gradients flow through all parameters
- `test_cffn_fewer_params_than_ffn`: Compare against standard FFN with 4x expansion
- `test_cffn_freezing`: Verify depth-based parameter freezing works for dyadic schedule

Run: `pytest tests/test_cffn.py -v`
Expected: All FAIL

**Step 2: Implement Cffn**

```python
import torch
import torch.nn as nn
from .continuant import continued_fraction

class Cffn(nn.Module):
    """Continued Fraction FFN — replaces standard Transformer FFN.

    Architecture:
        y = U·x + V·z
        z_j = f̃(W_j · x)  for j = 1, ..., L

    Args:
        dim: Hidden dimension (p)
        num_ladders: Number of CF ladders (L)
        depth: Continued fraction depth (d)
        epsilon: Pole avoidance threshold
    """

    def __init__(self, dim, num_ladders=3, depth=5, epsilon=0.01):
        super().__init__()
        self.dim = dim
        self.num_ladders = num_ladders
        self.depth = depth
        self.epsilon = epsilon

        # Direct linear path: U (p → p)
        self.U = nn.Linear(dim, dim, bias=False)

        # Ladder weight matrices: W^(j) maps input to (d+1) partial denominators
        # Each ladder: input (p) → (d+1) values that form the CF
        self.ladder_weights = nn.ModuleList([
            nn.Linear(dim, depth + 1, bias=False)
            for _ in range(num_ladders)
        ])

        # Combination layer: V (L → p)
        self.V = nn.Linear(num_ladders, dim, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        linear_out = self.U(x)

        ladder_outputs = []
        for j in range(self.num_ladders):
            a = self.ladder_weights[j](x)  # (batch, seq_len, d+1)
            # Split: a_0 is the linear term, a_1..a_d form the fraction
            a_0 = a[..., 0]                # (batch, seq_len)
            a_cf = a[..., 1:]              # (batch, seq_len, d)
            z_j = a_0 + continued_fraction(a_cf, self.epsilon)  # (batch, seq_len)
            ladder_outputs.append(z_j)

        z = torch.stack(ladder_outputs, dim=-1)  # (batch, seq_len, L)
        combined = self.V(z)                       # (batch, seq_len, dim)

        return linear_out + combined

    def get_depth_parameters(self, depth_level):
        """Return parameters at a specific depth level for dyadic scheduling."""
        # depth_level 0: U and V (linear components)
        # depth_level 1-d: corresponding rows in ladder_weights
        if depth_level == 0:
            params = list(self.U.parameters()) + list(self.V.parameters())
            # Also include a_0 row from each ladder
            for lw in self.ladder_weights:
                # First row (index 0) is the a_0 linear term
                # We'll handle this via masking in the training loop
                pass
            return params
        else:
            # Return ladder weight parameters (the training loop will
            # handle which rows to mask based on depth)
            return [lw.weight for lw in self.ladder_weights]
```

Note: The exact depth-based freezing mechanism will be refined during implementation. The key idea is that at each training step, we check `get_unfrozen_depth(step, total_steps, max_depth)` and only allow gradients for the appropriate rows of the ladder weights.

**Step 3: Run tests**

```bash
pytest tests/test_cffn.py -v
```

Expected: All PASS

**Step 4: Commit**

```bash
git add src/cofrgenet/cffn.py tests/test_cffn.py
git commit -m "feat: Cffn layer with continued fraction ladders"
```

---

## Task 4: Model Config Dataclasses

**Files:**
- Create: `src/cofrgenet/config.py`
- Create: `src/baseline/config.py`
- Create: `configs/baseline.yaml`
- Create: `configs/cofrgenet_f.yaml`

**Step 1: Write CoFrGeNet-F config**

```python
from dataclasses import dataclass

@dataclass
class CoFrGeNetConfig:
    # Transformer
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50257
    dropout: float = 0.0
    bias: bool = False
    # Cffn
    num_ladders: int = 3
    cf_depth: int = 5
    epsilon: float = 0.01
```

**Step 2: Write baseline config**

```python
from dataclasses import dataclass

@dataclass
class BaselineConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50257
    dropout: float = 0.0
    bias: bool = False
    ffn_expansion: int = 4
```

**Step 3: Write YAML configs** for easy CLI override

**Step 4: Commit**

```bash
git add src/cofrgenet/config.py src/baseline/config.py configs/
git commit -m "feat: model configuration dataclasses and YAML configs"
```

---

## Task 5: Baseline Transformer Model

**Files:**
- Create: `src/baseline/model.py`
- Create: `tests/test_model.py`

Standard GPT-2 Small implementation (nanoGPT-style). This is the control group.

**Step 1: Write tests**

- `test_baseline_forward_shape`: (B, S) → (B, S, vocab_size) logits
- `test_baseline_parameter_count`: Should be ~124M
- `test_baseline_causal_masking`: Future tokens cannot attend to past
- `test_baseline_generation`: Can generate tokens autoregressively

**Step 2: Implement baseline model**

Standard components:
- Token + learned positional embeddings
- N transformer blocks, each with:
  - LayerNorm → Multi-Head Self-Attention → residual
  - LayerNorm → FFN (Linear 768→3072 → GELU → Linear 3072→768) → residual
- Final LayerNorm → Linear (768 → vocab_size)

Use pre-norm (GPT-2 style). No bias. No dropout.

**Step 3: Run tests, verify**

```bash
pytest tests/test_model.py -v
```

**Step 4: Commit**

```bash
git add src/baseline/model.py tests/test_model.py
git commit -m "feat: baseline GPT-2 Small transformer"
```

---

## Task 6: CoFrGeNet-F Model

**Files:**
- Create: `src/cofrgenet/model.py`
- Modify: `tests/test_model.py` (add CoFrGeNet-F tests)

This is identical to the baseline except the FFN is replaced with Cffn.

**Step 1: Write tests**

- `test_cofrgenet_forward_shape`: Same as baseline
- `test_cofrgenet_fewer_params`: Must have fewer params than baseline
- `test_cofrgenet_causal_masking`: Same as baseline
- `test_cofrgenet_generation`: Same as baseline
- `test_cofrgenet_dyadic_schedule`: Verify parameter freezing at different steps

**Step 2: Implement CoFrGeNet-F model**

The implementation should share as much code as possible with the baseline. Factor out a `TransformerBlock` that takes the FFN module as a parameter. The CoFrGeNet-F model passes `Cffn(dim=768, num_ladders=3, depth=5)` where the baseline passes `FFN(dim=768, expansion=4)`.

**Step 3: Print parameter comparison**

```python
baseline_params = sum(p.numel() for p in baseline.parameters())
cofrgenet_params = sum(p.numel() for p in cofrgenet.parameters())
print(f"Baseline: {baseline_params:,}")
print(f"CoFrGeNet-F: {cofrgenet_params:,}")
print(f"Reduction: {(1 - cofrgenet_params/baseline_params)*100:.1f}%")
```

**Step 4: Run tests**

```bash
pytest tests/test_model.py -v
```

**Step 5: Commit**

```bash
git add src/cofrgenet/model.py tests/test_model.py
git commit -m "feat: CoFrGeNet-F model with Cffn replacing FFN"
```

---

## Task 7: Data Pipeline

**Files:**
- Create: `scripts/01_download_data.py`

Download FineWeb-Edu 10BT, tokenize with GPT-2 tokenizer, shard into binary files for efficient training.

**Step 1: Implement data download and tokenization**

```python
# Pseudocode
from datasets import load_dataset
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

# Tokenize in parallel, shard into ~100M token binary files
# Save as np.uint16 (GPT-2 vocab fits in uint16)
# Split: 99% train, 1% val
```

Output:
```
data/tokenized/train_000.bin
data/tokenized/train_001.bin
...
data/tokenized/val_000.bin
```

**Step 2: Test by loading first shard**

```bash
python scripts/01_download_data.py
python -c "import numpy as np; d = np.memmap('data/tokenized/train_000.bin', dtype=np.uint16); print(f'Tokens: {len(d):,}')"
```

**Step 3: Commit**

```bash
git add scripts/01_download_data.py
git commit -m "feat: data download and tokenization pipeline"
```

---

## Task 8: Training Loop (Shared)

**Files:**
- Create: `scripts/02_train_baseline.py`
- Create: `scripts/03_train_cofrgenet.py`

Both scripts share the same training loop, differing only in model construction and the dyadic schedule (CoFrGeNet-F only).

**Step 1: Implement shared training infrastructure**

Key components:
- Data loader: memory-mapped binary shards, random sampling of (block_size,) sequences
- Learning rate schedule: linear warmup (700 steps) → cosine decay to 0
- Gradient accumulation: micro_batch_size × grad_accum_steps × seq_len = 524,288 tokens/update
- Logging: wandb (loss, lr, throughput, grad norm)
- Checkpointing: save every 1,000 steps as safetensors
- Mixed precision: `torch.autocast("cuda", dtype=torch.bfloat16)`

**Step 2: Implement baseline training script**

```bash
python scripts/02_train_baseline.py
```

Saves checkpoints to `checkpoints/baseline/step_XXXXX.safetensors`

**Step 3: Implement CoFrGeNet-F training script**

Same as baseline but:
1. Uses CoFrGeNet-F model
2. Implements dyadic schedule: at each step, compute unfrozen depth and freeze/unfreeze Cffn parameters accordingly

```python
def get_unfrozen_depth(step, total_steps, max_depth):
    for d in range(max_depth, 0, -1):
        if step >= total_steps * (1 - 1 / (2 ** d)):
            return d
    return 0
```

**Step 4: Smoke test (10 steps)**

```bash
python scripts/02_train_baseline.py --max_steps 10 --eval_interval 5
python scripts/03_train_cofrgenet.py --max_steps 10 --eval_interval 5
```

Verify both scripts run, loss decreases, checkpoints save.

**Step 5: Commit**

```bash
git add scripts/02_train_baseline.py scripts/03_train_cofrgenet.py
git commit -m "feat: training scripts for baseline and CoFrGeNet-F"
```

---

## Task 9: Full Training Run

**No new files — execution only.**

**Step 1: Train baseline**

```bash
nohup python scripts/02_train_baseline.py \
    --total_steps 19073 \
    --warmup_steps 700 \
    --lr 6e-4 \
    --batch_tokens 524288 \
    --eval_interval 500 \
    --save_interval 1000 \
    > logs/baseline_train.log 2>&1 &
```

Expected: ~4 hours for 2.5B tokens, ~15 hours for full 10B tokens on RTX 5090.

**Step 2: Train CoFrGeNet-F**

```bash
nohup python scripts/03_train_cofrgenet.py \
    --total_steps 19073 \
    --warmup_steps 700 \
    --lr 6e-4 \
    --batch_tokens 524288 \
    --eval_interval 500 \
    --save_interval 1000 \
    > logs/cofrgenet_train.log 2>&1 &
```

**Step 3: Monitor training**

Watch wandb dashboards for:
- Loss curves (should be comparable between models)
- Learning rate schedule
- Throughput (tokens/sec)
- Gradient norms

**Step 4: Commit training configs and logs**

```bash
git add logs/
git commit -m "feat: training run logs and final checkpoints"
```

---

## Task 10: Evaluation

**Files:**
- Create: `scripts/04_evaluate.py`

**Step 1: Implement evaluation script**

Evaluate both models on:
- WikiText-2 perplexity (stride 512)
- WikiText-103 perplexity (stride 512)
- LAMBADA perplexity + last-word accuracy
- HellaSwag zero-shot accuracy (if time permits)

Also measure:
- Total parameter count
- Inference throughput (tokens/sec)
- Memory usage

**Step 2: Run evaluation**

```bash
python scripts/04_evaluate.py --model baseline --checkpoint checkpoints/baseline/final.safetensors
python scripts/04_evaluate.py --model cofrgenet --checkpoint checkpoints/cofrgenet/final.safetensors
```

**Step 3: Generate comparison table**

Output a markdown table comparing both models across all metrics.

**Step 4: Commit**

```bash
git add scripts/04_evaluate.py
git commit -m "feat: evaluation script with perplexity and benchmark results"
```

---

## Task 11: Text Generation Examples

**Files:**
- Create: `scripts/05_generate_examples.py`

**Step 1: Implement generation script**

Generate text samples from both models using the same prompts:
- "The meaning of life is"
- "In a dark forest, the"
- "The theory of continued fractions"
- "Once upon a time"
- Several more diverse prompts

Use top-k (50) and top-p (0.95) sampling with temperature 0.8.

**Step 2: Run generation**

```bash
python scripts/05_generate_examples.py
```

Save outputs to `output/examples/`.

**Step 3: Commit**

```bash
git add scripts/05_generate_examples.py
git commit -m "feat: text generation comparison script"
```

---

## Task 12: Gradio Demo

**Files:**
- Create: `demo/app.py`

**Step 1: Implement Gradio demo**

Side-by-side generation from both models:
- Text input box for prompt
- Sliders for temperature, top-k, top-p, max tokens
- Two output boxes: "Baseline Transformer" and "CoFrGeNet-F"
- Parameter count display
- Generation time display

**Step 2: Test locally**

```bash
python demo/app.py
```

**Step 3: Commit**

```bash
git add demo/app.py
git commit -m "feat: Gradio demo for side-by-side generation"
```

---

## Task 13: README and Documentation

**Files:**
- Create: `README.md`

Write a comprehensive README with:
- Project overview and motivation
- Key results table (parameter counts, perplexities, throughput)
- Architecture diagram (text-based)
- Installation instructions
- Training reproduction instructions
- Citation info
- Links to paper, HuggingFace models, Gradio demo

**Commit:**

```bash
git add README.md
git commit -m "docs: comprehensive README with results and instructions"
```

---

## Task 14: HuggingFace Upload

**No new files — execution only.**

**Step 1: Create HuggingFace repos**

```bash
# Model repos
huggingface-cli repo create cahlen/cofrgenet-f-125m --private
huggingface-cli repo create cahlen/gpt2-small-baseline-125m --private

# Upload models
huggingface-cli upload cahlen/cofrgenet-f-125m checkpoints/cofrgenet/final.safetensors
huggingface-cli upload cahlen/gpt2-small-baseline-125m checkpoints/baseline/final.safetensors
```

**Step 2: Add model cards to each repo**

---

## Execution Order Summary

| Task | Description | Depends On | Est. Time |
|------|-------------|------------|-----------|
| 1 | Environment setup | — | 10 min |
| 2 | Continuant core | 1 | 1-2 hr |
| 3 | Cffn layer | 2 | 1 hr |
| 4 | Config dataclasses | 1 | 15 min |
| 5 | Baseline model | 4 | 1 hr |
| 6 | CoFrGeNet-F model | 3, 5 | 1 hr |
| 7 | Data pipeline | 1 | 30 min + download |
| 8 | Training loop | 5, 6, 7 | 2 hr |
| 9 | Full training run | 8 | 8-30 hr (GPU time) |
| 10 | Evaluation | 9 | 1 hr |
| 11 | Generation examples | 9 | 30 min |
| 12 | Gradio demo | 9 | 1 hr |
| 13 | README | 10 | 30 min |
| 14 | HuggingFace upload | 10, 13 | 30 min |
