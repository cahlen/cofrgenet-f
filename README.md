# CoFrGeNet-F: First Open-Source Continued Fraction Language Model

The first open-source implementation of **CoFrGeNet-F**, a continued fraction architecture that replaces Transformer FFN layers with continued fraction networks while achieving competitive performance at ~34% fewer parameters.

Based on IBM Research's paper: [CoFrGeNet: Continued Fraction Architectures for Language Generation](https://arxiv.org/abs/2601.21766) (arXiv:2601.21766, January 2026). No public code exists — this is implemented from the paper's math.

## What Is This?

Standard Transformers use Multi-Head Attention + Feed-Forward Networks (FFN). CoFrGeNet-F keeps standard attention but replaces the FFN with a **Continued Fraction FFN (Cffn)** — an ensemble of continued fraction "ladders" that compute rational functions via continuant polynomials:

```
f̃(a₁, ..., a_d) = K_{d-1} / K_d
```

where each `a_k = w_k · x` with learnable weights, and `K` are continuant polynomials computed via a simple recursion. This replaces the standard 2-layer FFN (Linear → GELU → Linear) with a **rational function** that is strictly more expressive per parameter.

Key insight: rational functions can approximate a broader class of functions than polynomials with the same number of parameters — achieving **~4× fewer parameters per FFN layer** without sacrificing quality.

## Models

We train two models on [FineWeb-Edu 10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (~10 billion tokens) with identical hyperparameters. The only difference is the FFN:

| Model | Parameters | FFN Type | FFN Params/Layer |
|-------|-----------|----------|-----------------|
| **Baseline** (GPT-2 Small) | 124.3M | Standard (Linear → GELU → Linear) | 4,718,592 |
| **CoFrGeNet-F** | 82.0M (34% fewer) | Continued Fraction FFN (Cffn) | 1,193,472 |

Both models: 12 layers, 12 attention heads, 768 hidden dim, 1024 context length, GPT-2 tokenizer (50,257 vocab).

### Baseline Results

| Benchmark | Metric | Value |
|-----------|--------|-------|
| WikiText-2 | Perplexity | **34.13** |
| WikiText-103 | Perplexity | **34.13** |
| LAMBADA | Perplexity | **37.47** |
| LAMBADA | Accuracy | **19.15%** |

CoFrGeNet-F results will be added once training completes.

## Project Status

- [x] Core architecture: continuant computation, Cffn layer, full CoFrGeNet-F model
- [x] Baseline Transformer (GPT-2 Small)
- [x] Data pipeline (FineWeb-Edu 10BT tokenized to binary shards)
- [x] Training infrastructure (shared training loop, dyadic schedule)
- [x] Baseline model fully trained and evaluated
- [x] Baseline model released on HuggingFace
- [ ] **CoFrGeNet-F training (in progress — ~80% complete)**
- [ ] CoFrGeNet-F evaluation and HuggingFace release
- [ ] Head-to-head benchmark comparison
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
```

### Docker

```bash
docker build -t cofrgenet-f .
docker run --gpus all -v $(pwd)/data:/workspace/data -v $(pwd)/checkpoints:/workspace/checkpoints cofrgenet-f python scripts/03_train_cofrgenet.py
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

CoFrGeNet-F additionally uses a **dyadic training schedule** that progressively unfreezes continued fraction depth levels on a powers-of-2 schedule — without this, the paper reports 10–80% performance degradation.

## References

- [CoFrGeNet paper (arXiv:2601.21766)](https://arxiv.org/abs/2601.21766) — IBM Research, January 2026
- [CoFrNet (NeurIPS 2021)](https://arxiv.org/abs/2506.05586) — Predecessor for tabular/classification tasks
- [CoFrNet in AIX360](https://github.com/Trusted-AI/AIX360/tree/master/examples/cofrnet) — IBM's reference implementation of CoFrNet
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — Training dataset
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Architecture reference

## License

MIT
