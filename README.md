# CoFrGeNet-F: First Open-Source Continued Fraction Language Model

The first open-source implementation of **CoFrGeNet-F**, a continued fraction architecture that replaces Transformer FFN layers with continued fraction networks while achieving competitive performance at ~34% fewer parameters.

Based on IBM Research's paper: [CoFrGeNet: Continued Fraction Architectures for Language Generation](https://arxiv.org/abs/2601.21766) (arXiv:2601.21766, January 2026).

## What Is This?

Standard Transformers use Multi-Head Attention + Feed-Forward Networks (FFN). CoFrGeNet-F keeps standard attention but replaces the FFN with a **Continued Fraction FFN (Cffn)** — an ensemble of continued fraction "ladders" that compute:

```
f(x) = a₀ + 1/(a₁ + 1/(a₂ + ... + 1/a_d))
```

where each `a_k = w_k · x` with learnable weights. This is computed efficiently via **continuant polynomials** with custom gradients (1 division instead of d).

## Project Status

**Work in progress.** This repo contains the implementation plan and project scaffold. Training will be done on an RTX 5090.

## Goals

- Train a **125M CoFrGeNet-F** and a **125M standard Transformer baseline** on identical data (FineWeb-Edu 10B tokens)
- Head-to-head benchmark comparison (perplexity, throughput, parameter count)
- Release both models on HuggingFace
- Interactive Gradio demo for side-by-side text generation
- Technical write-up explaining the architecture

## Quick Start

```bash
git clone https://github.com/cahlen/cofrgenet-f.git
cd cofrgenet-f
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

See `docs/plans/2026-03-01-cofrgenet-f-implementation.md` for the full implementation plan.

## References

- [CoFrGeNet paper (arXiv:2601.21766)](https://arxiv.org/abs/2601.21766)
- [CoFrNet (NeurIPS 2021)](https://arxiv.org/abs/2506.05586)
- [CoFrNet in AIX360](https://github.com/Trusted-AI/AIX360/tree/master/examples/cofrnet)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

## License

MIT
