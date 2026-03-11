# CoFrGeNet-F: Continued Fraction Language Model

The first independent, large-scale replication of **CoFrGeNet-F** — a continued fraction architecture that replaces Transformer FFN layers with continued fraction networks. Training matched pairs from **450M to 13B parameters** on **50–100B tokens** to test IBM Research's claim that the architecture's advantage grows with scale.

Based on: [CoFrGeNet: Continued Fraction Architectures for Language Generation](https://arxiv.org/abs/2601.21766) (arXiv:2601.21766, IBM Research, January 2026). Implemented from the paper's mathematics.

## Why This Matters

IBM's paper showed CoFrGeNet-F achieving comparable perplexity with **35% fewer parameters** at GPT-2 XL scale (~985M vs ~1.5B), claiming the advantage grows with model size. But their largest experiment was ~1.5B parameters on ~26B tokens. **Nobody has publicly validated whether this scaling claim holds at larger scales.**

We're running that experiment — training matched pairs of standard Transformers and CoFrGeNet-F models from 450M to 13B parameters, on identical data, with identical hyperparameters. The only difference: the FFN layer.

## What Is CoFrGeNet-F?

Standard Transformers use Multi-Head Attention + Feed-Forward Networks (FFN). CoFrGeNet-F keeps standard attention but replaces the FFN with a **Continued Fraction FFN (Cffn)** — an ensemble of continued fraction "ladders" that compute rational functions via continuant polynomials:

$$
\tilde{f}(a_1, a_2, \ldots, a_d) \;=\; \cfrac{1}{a_1 + \cfrac{1}{a_2 + \cfrac{1}{\ddots + \cfrac{1}{a_d}}}}
$$

where each coefficient is a learned linear function of the input. The continued fraction is computed via **continuant polynomials**:

$$
K_0 = 1, \qquad K_1(a_d) = a_d, \qquad K_k = a_{d-k+1} \cdot K_{k-1} + K_{k-2}
$$

yielding the ratio:

$$
\tilde{f} = \frac{K_{d-1}}{K_d}
$$

This replaces the standard 2-layer FFN (Linear → GELU → Linear) with a **rational function** that is strictly more expressive per parameter — achieving **~4× fewer parameters per FFN layer**.

## Paired Experiments (DGX B200)

All experiments run on **8× NVIDIA B200 GPUs** (179 GB VRAM each) using DDP. Each pair trains a standard Transformer baseline and a CoFrGeNet-F model on **identical data, identical hyperparameters, identical hardware**. The only variable is the FFN layer.

### Pair 1: 450M vs 410M (50B tokens) — In Progress

Sanity check at small scale. Validates the multi-GPU pipeline.

| | Baseline | CoFrGeNet-F |
|-|----------|-------------|
| **Parameters** | ~450M | ~410M |
| **Architecture** | 12L, 1600d, 25h, standard FFN | 12L, 2048d, 16h, Cffn (L=3, d=5) |
| **Data** | 50B tokens (95,367 steps) | 50B tokens (95,367 steps) |

**Baseline:** Complete. Final loss 2.67, val loss 2.68, ~2.12M tok/s, ~3.5 hours.
**CoFrGeNet-F:** Training. ~27% complete, loss 3.58, ~800K tok/s.

### Pair 3: 7.5B vs 4.8B (100B tokens) — Planned

Key experiment — **7× beyond the IBM paper's largest scale**. Tests the central claim.

| | Baseline | CoFrGeNet-F |
|-|----------|-------------|
| **Parameters** | ~7.5B | ~4.8B (35% fewer) |
| **Architecture** | 36L, 4096d, 32h, standard FFN | 36L, 4608d, 36h, Cffn (L=3, d=5) |

### Pair 4: 9.9B vs 7.8B (100B tokens) — Planned

Deep + wide. Tests 48 layers and more ladders (L=5).

| | Baseline | CoFrGeNet-F |
|-|----------|-------------|
| **Parameters** | ~9.9B | ~7.8B |
| **Architecture** | 48L, 4096d, 32h, standard FFN | 48L, 5120d, 40h, Cffn (L=5, d=5) |

### Pair 5: 12.9B vs 7.9B (100B tokens) — Planned

Push scale. 38% fewer parameters. Deeper continued fractions (d=7).

| | Baseline | CoFrGeNet-F |
|-|----------|-------------|
| **Parameters** | ~12.9B | ~7.9B (38% fewer) |
| **Architecture** | 40L, 5120d, 40h, standard FFN | 40L, 5632d, 44h, Cffn (L=5, d=7) |

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the full experiment plan, gradient stability research, and evaluation methodology.

## Prior Experiments (Single-GPU)

Before the DGX B200 paired runs, three smaller experiments were conducted on single GPUs (RTX 5090 / H200) training on 10B tokens:

| Experiment | Baseline | CoFrGeNet-F | WikiText-2 PPL | Result |
|-----------|----------|-------------|:-:|--------|
| **Exp 1** (param-efficient) | 124M | 82M (34% fewer) | 40.79 vs 110.32 | CoFrGeNet-F significantly worse |
| **Exp 2** (iso-parameter) | 124M | 128M | 40.79 vs 82.46 | Better, but baseline still wins |
| **Exp 3** (more ladders, L=8) | — | 128M | — | Completed, evaluating |

These results at small scale are consistent with the IBM paper, which showed CoFrGeNet-F's advantage only emerging at GPT-2 XL scale (~1B params). The paired experiments above test whether that trend continues.

Model weights for Exp 1 and 2 are on HuggingFace: [cahlen/cofrgenet-f](https://huggingface.co/cahlen/cofrgenet-f).

## Gradient Stability Research

During training, CoFrGeNet-F exhibits **elevated gradient norms** (22–28× the baseline) due to poles in the continued fraction's rational function. The gradient formula:

$$
\frac{\partial \tilde{f}}{\partial a_k} = (-1)^{k} \left( \frac{K_{d-k}}{K_d} \right)^{2}
$$

explodes when the denominator continuant $K_d$ approaches zero. We've identified several stabilization approaches for post-experiment testing, including **log-space continuant computation** and **adaptive gradient clipping (AGC)**.

This work connects to prior research on the structural properties of continued fraction convergents:

> **Humphreys, C.** "Prime Numbers and the Convergents of a Continued Fraction." NCUR, 2013. ([PDF](docs/Humphreys_2013_Convergents_of_Continued_Fractions.pdf))

The continuant recurrence in that paper ($A_n = a_n A_{n-1} + A_{n-2}$) is identical to CoFrGeNet-F's computation. Theorem 3.5 (reciprocal symmetry of convergents) provides a theoretical basis for principled pole-aware gradient damping. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) Section 7 for the full analysis.

## Quick Start

```bash
git clone https://github.com/cahlen/cofrgenet-f.git
cd cofrgenet-f
pip install -r requirements.txt

# Download and tokenize data (default: 10B tokens)
python scripts/01_download_data.py

# Train baseline (single GPU)
python scripts/02_train_baseline.py

# Train CoFrGeNet-F (single GPU, with dyadic schedule)
python scripts/03_train_cofrgenet.py

# Evaluate both models
python scripts/04_evaluate.py --model both
```

### Multi-GPU Training (DDP)

```bash
# Using experiment configs on 8 GPUs
bash scripts/launch_training.sh scripts/02_train_baseline.py \
  --config configs/experiments/pair1_baseline_450m.yaml

bash scripts/launch_training.sh scripts/03_train_cofrgenet.py \
  --config configs/experiments/pair1_cofrgenet_410m.yaml
```

### Unattended Training with Auto-Restart

```bash
# Automatically resumes from latest checkpoint on crash (up to 20 retries)
bash scripts/train_with_autorestart.sh scripts/03_train_cofrgenet.py \
  --config configs/experiments/pair1_cofrgenet_410m.yaml
```

### Docker

```bash
docker build -t cofrgenet-f .
docker run --gpus all \
  -v "$(pwd)/data:/workspace/data" \
  -v "$(pwd)/checkpoints:/workspace/checkpoints" \
  cofrgenet-f python scripts/03_train_cofrgenet.py
```

## Training Details

### Paired Experiments (DGX B200)

| Hyperparameter | Pair 1 (50B tok) | Pairs 3–5 (100B tok) |
|---------------|:-:|:-:|
| Dataset | FineWeb-Edu | FineWeb-Edu |
| Tokenizer | GPT-2 (tiktoken, vocab 50,257) | GPT-2 |
| Optimizer | AdamW (fused), β₁=0.9, β₂=0.95 | AdamW (fused) |
| Learning rate | 6e-4, cosine → 0 | 1.5–3e-4, cosine → 0 |
| Warmup | 700 steps | 2,000 steps |
| Weight decay | 0.1 | 0.1 |
| Gradient clipping | 1.0 max norm | 1.0 max norm |
| Batch size | 524,288 tokens/update | 524,288 tokens/update |
| Precision | bfloat16 | bfloat16 |
| Parallelism | DDP (8× B200) | DDP / FSDP (8× B200) |
| torch.compile | max-autotune | max-autotune |

CoFrGeNet-F additionally uses a **dyadic training schedule** that progressively unfreezes continued fraction depth levels — without this, the paper reports 10–80% performance degradation:

$$
s_i = \left(1 - \frac{1}{2^i}\right) \times S_{\text{total}}
$$

## The Cffn Layer

Each **Continued Fraction FFN (Cffn)** replaces the standard two-layer FFN. Where a standard Transformer block uses Linear → GELU → Linear with a 4× expansion, the Cffn computes:

$$
y = U x + \sum_{j=1}^{L} V_j \cdot z_j
$$

**Direct linear path** — skip connection: $U \in \mathbb{R}^{p \times p}$

**Gating projection** — modulates input before the ladders: $\hat{x} = \sigma(G x) \odot x$

**Continued fraction ladders** — each computes $p$ independent continued fractions element-wise:

$$
z_j = \tilde{f}\!\left( \hat{x} \odot W^{(j)} \right), \qquad W^{(j)} \in \mathbb{R}^{p \times d}, \quad j = 1, \ldots, L
$$

**Combination weights** — per-dimension weighting: $V \in \mathbb{R}^{p \times L}$

## Architecture Details

See the **[Wiki](https://github.com/cahlen/cofrgenet-f/wiki)** for detailed documentation:

- [Architecture Overview](https://github.com/cahlen/cofrgenet-f/wiki/Architecture-Overview) — Side-by-side comparison of standard Transformer vs CoFrGeNet-F
- [Continued Fractions and Continuants](https://github.com/cahlen/cofrgenet-f/wiki/Continued-Fractions-and-Continuants) — The mathematics from first principles
- [The Cffn Layer](https://github.com/cahlen/cofrgenet-f/wiki/The-Cffn-Layer) — Full code walkthrough mapped to math
- [Custom Autograd](https://github.com/cahlen/cofrgenet-f/wiki/Custom-Autograd) — Gradient derivation and implementation
- [Dyadic Training Schedule](https://github.com/cahlen/cofrgenet-f/wiki/Dyadic-Training-Schedule) — The critical progressive unfreezing schedule
- [Parameter Efficiency](https://github.com/cahlen/cofrgenet-f/wiki/Parameter-Efficiency) — Layer-by-layer parameter breakdown

## Project Status

- [x] Core architecture (continuant, Cffn, both models, configs, tests)
- [x] Data pipeline (FineWeb-Edu tokenized to binary shards)
- [x] Training infrastructure (DDP, auto-restart, dyadic schedule, YAML configs)
- [x] Prior experiments 1–3 (single-GPU, 10B tokens)
- [x] Evaluation script (WikiText-2/103, LAMBADA, throughput)
- [x] HuggingFace model weights ([cahlen/cofrgenet-f](https://huggingface.co/cahlen/cofrgenet-f))
- [x] GitHub Wiki (7 pages of architecture + math documentation)
- [x] Pair 1 baseline (450M) trained
- [ ] **Pair 1 CoFrGeNet-F (410M) — training in progress**
- [ ] Pairs 3–5 (7B–13B scale, 100B tokens)
- [ ] Gradient stabilization experiments (post-pair training)
- [ ] Interactive Gradio demo (side-by-side generation)
- [ ] Technical write-up / blog post

## References

1. [CoFrGeNet paper (arXiv:2601.21766)](https://arxiv.org/abs/2601.21766) — IBM Research, January 2026
2. [CoFrNet (NeurIPS 2021)](https://arxiv.org/abs/2506.05586) — Predecessor for tabular/classification tasks
3. [Humphreys, C. "Prime Numbers and the Convergents of a Continued Fraction." NCUR, 2013](docs/Humphreys_2013_Convergents_of_Continued_Fractions.pdf) — Continuant theory applied to gradient stability
4. [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — Training dataset
5. [nanoGPT](https://github.com/karpathy/nanoGPT) — Architecture reference

## IBM Patent Note

IBM has US Patent Application #20230401438 on CoFrNets. The paper itself is CC BY 4.0. This implementation is from-scratch based on published mathematics. This is an academic reproduction / educational project.

## License

MIT
