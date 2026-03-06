# CoFrGeNet-F: Continued Fraction Language Model

An open-source implementation of **CoFrGeNet-F**, a continued fraction architecture that replaces Transformer FFN layers with continued fraction networks, achieving 34% parameter reduction.

Based on IBM Research's paper: [CoFrGeNet: Continued Fraction Architectures for Language Generation](https://arxiv.org/abs/2601.21766) (arXiv:2601.21766, January 2026). Implemented from the paper's mathematics.

## What Is This?

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

This replaces the standard 2-layer FFN (Linear → GELU → Linear) with a **rational function** that is strictly more expressive per parameter. Rational functions approximate a broader class of functions than polynomials with the same parameter count — achieving **~4× fewer parameters per FFN layer**.

## Models

Two models trained on [FineWeb-Edu 10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (~10 billion tokens) with identical hyperparameters. The only difference is the FFN architecture:

| Model | Parameters | FFN Type | FFN Params/Layer |
|-------|-----------|----------|-----------------|
| **Baseline** (GPT-2 Small) | 124.3M | Standard (Linear → GELU → Linear) | 4,718,592 |
| **CoFrGeNet-F** | 82.0M (34% fewer) | Continued Fraction FFN (Cffn) | 1,193,472 |

Both models: 12 layers, 12 attention heads, 768 hidden dim, 1024 context length, GPT-2 tokenizer (50,257 vocab).

### Head-to-Head Results

Both models evaluated on the same NVIDIA H200 with identical evaluation code.

| Metric | Baseline (124M) | CoFrGeNet-F (82M) | Note |
|--------|-----------------|-------------------|------|
| Parameters | 124,337,664 | **82,036,224** | 34.0% fewer |
| WikiText-2 PPL | **40.79** | 110.32 | lower is better |
| WikiText-103 PPL | **40.79** | 110.32 | lower is better |
| LAMBADA PPL | **37.45** | 166.57 | lower is better |
| LAMBADA Accuracy | **19.06%** | 8.77% | higher is better |
| Throughput | **277,827** tok/s | 103,455 tok/s | |
| Generation Speed | **5.53** ms/tok | 10.92 ms/tok | |

At GPT-2 Small scale, CoFrGeNet-F does not yet match the standard Transformer baseline. The paper's strongest results were at GPT-2 XL scale (985M params), where CoFrGeNet-F beat the full 1.5B GPT2-xl on all benchmarks. The architecture likely requires larger scale or longer training to realize its advantages.

### HuggingFace

Both model weights are available on HuggingFace: [cahlen/cofrgenet-f-82m](https://huggingface.co/cahlen/cofrgenet-f-82m)

## Project Status

- [x] Core architecture: continuant computation, Cffn layer, full CoFrGeNet-F model
- [x] Baseline Transformer (GPT-2 Small)
- [x] Data pipeline (FineWeb-Edu 10BT tokenized to binary shards)
- [x] Training infrastructure (shared training loop, dyadic schedule)
- [x] Baseline model trained and evaluated
- [x] CoFrGeNet-F model trained and evaluated
- [x] Head-to-head benchmark comparison
- [x] Both models released on HuggingFace
- [ ] Interactive Gradio demo (side-by-side generation)
- [ ] Technical write-up / blog post

## Architecture Details

See the **[Wiki](https://github.com/cahlen/cofrgenet-f/wiki)** for detailed documentation:

- [Architecture Overview](https://github.com/cahlen/cofrgenet-f/wiki/Architecture-Overview) — Side-by-side comparison of standard Transformer vs CoFrGeNet-F
- [Continued Fractions and Continuants](https://github.com/cahlen/cofrgenet-f/wiki/Continued-Fractions-and-Continuants) — The mathematics from first principles
- [The Cffn Layer](https://github.com/cahlen/cofrgenet-f/wiki/The-Cffn-Layer) — Full code walkthrough mapped to math
- [Custom Autograd](https://github.com/cahlen/cofrgenet-f/wiki/Custom-Autograd) — Gradient derivation and implementation
- [Dyadic Training Schedule](https://github.com/cahlen/cofrgenet-f/wiki/Dyadic-Training-Schedule) — The critical progressive unfreezing schedule
- [Parameter Efficiency](https://github.com/cahlen/cofrgenet-f/wiki/Parameter-Efficiency) — Layer-by-layer parameter breakdown

## The Cffn Layer

Each **Continued Fraction FFN (Cffn)** replaces the standard two-layer FFN. Where a standard Transformer block uses Linear → GELU → Linear with a 4× expansion, the Cffn instead computes:

$$
y = U x + \sum_{j=1}^{L} V_j \cdot z_j
$$

The components are:

**Direct linear path** — a skip connection through the fraction block:

$$
U \in \mathbb{R}^{p \times p}
$$

**Gating projection** — modulates the input before it enters the ladders:

$$
\hat{x} = \sigma(G x) \odot x, \qquad G \in \mathbb{R}^{p \times p}
$$

**Continued fraction ladders** — each ladder computes p independent continued fractions element-wise:

$$
z_j = \tilde{f}\!\left( \hat{x} \odot W^{(j)} \right), \qquad W^{(j)} \in \mathbb{R}^{p \times d}, \quad j = 1, \ldots, L
$$

**Combination weights** — per-dimension weighting of ladder outputs:

$$
V \in \mathbb{R}^{p \times L}
$$

### Efficient Gradients (Proposition 1)

The continuant formulation yields gradients that require only **one division** instead of d:

$$
\frac{\partial \tilde{f}}{\partial a_k} = (-1)^{k} \left( \frac{K_{d-k}}{K_d} \right)^{2}
$$

By caching the reciprocal of the final continuant and reusing it across all depth levels, the cost drops from O(d) divisions to O(1).

### Pole Avoidance

Continued fractions can encounter poles when the denominator vanishes. We apply safe clamping:

$$
K_d^{\text{safe}} = \text{sign}(K_d) \cdot \max\!\left( |K_d|, \; \epsilon \right), \qquad \epsilon = 0.01
$$

### Parameter Count per Cffn Layer

Each Cffn layer uses:

$$
2p^2 + L \cdot p \cdot d + L \cdot p = 2 \times 768^2 + 3 \times 768 \times 5 + 3 \times 768 = 1{,}193{,}472 \text{ params}
$$

A standard FFN uses 4,718,592 params — a **3.95× reduction** per layer.

## Quick Start

```bash
git clone https://github.com/cahlen/cofrgenet-f.git
cd cofrgenet-f
pip install -r requirements.txt

# Download and tokenize data
python scripts/01_download_data.py

# Train baseline
python scripts/02_train_baseline.py

# Train CoFrGeNet-F (with dyadic schedule)
python scripts/03_train_cofrgenet.py

# Evaluate both models
python scripts/04_evaluate.py --model both
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

| Hyperparameter | Value |
|---------------|-------|
| Dataset | FineWeb-Edu sample-10BT (~10B tokens) |
| Tokenizer | GPT-2 (tiktoken) |
| Optimizer | AdamW |
| Learning rate | 6e-4 peak, cosine decay to 0 |
| Warmup | 700 steps |
| Weight decay | 0.1 |
| Betas | (0.9, 0.95) |
| Gradient clipping | 1.0 max norm |
| Batch size | 524,288 tokens per update |
| Total steps | 19,073 (one epoch) |
| Precision | bfloat16 |

CoFrGeNet-F additionally uses a **dyadic training schedule** that progressively unfreezes continued fraction depth levels — without this, the paper reports 10–80% performance degradation:

$$
s_i = \left(1 - \frac{1}{2^i}\right) \times S_{\text{total}}
$$

| Depth | Unfreeze Step | % of Training |
|-------|--------------|---------------|
| 0 (linear components U, G, V) | 0 | 0% |
| 1 (ladder column 0) | 9,537 | 50% |
| 2 (ladder column 1) | 14,305 | 75% |
| 3 (ladder column 2) | 16,689 | 87.5% |
| 4 (ladder column 3) | 17,881 | 93.75% |
| 5 (ladder column 4) | 18,477 | 96.875% |

## References

- [CoFrGeNet paper (arXiv:2601.21766)](https://arxiv.org/abs/2601.21766) — IBM Research, January 2026
- [CoFrNet (NeurIPS 2021)](https://arxiv.org/abs/2506.05586) — Predecessor for tabular/classification tasks
- [CoFrNet in AIX360](https://github.com/Trusted-AI/AIX360/tree/master/examples/cofrnet) — IBM's reference implementation of CoFrNet
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — Training dataset
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Architecture reference

## License

MIT
