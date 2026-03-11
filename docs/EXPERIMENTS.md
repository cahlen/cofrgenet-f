# CoFrGeNet-F Experiment Plan & Research Log

**Principal Investigator:** Cahlen Humphreys
**Hardware:** DGX B200 (8x NVIDIA B200 GPUs, 179 GB VRAM each)
**Repository:** [github.com/cahlen/cofrgenet-f](https://github.com/cahlen/cofrgenet-f)
**Paper under test:** [arXiv:2601.21766](https://arxiv.org/abs/2601.21766) — IBM Research, January 2026

---

## 1. Motivation

IBM's CoFrGeNet paper replaces standard Transformer FFN layers with continued fraction networks (Cffn). Their key claim: CoFrGeNet-F achieves **comparable perplexity with 35% fewer parameters**, and the advantage grows with scale. However, their largest experiment was GPT-2 XL scale (~1.5B baseline vs ~985M CoFrGeNet-F on ~26B tokens). Nobody has publicly validated whether this scaling claim holds at larger scales.

This project trains matched pairs of baseline Transformers and CoFrGeNet-F models from 450M to 13B parameters on 50-100B tokens of FineWeb-Edu data, providing the first independent, large-scale replication of CoFrGeNet-F.

### Prior Work (Single-GPU Experiments, RTX 5090 / H200)

Before the DGX B200 paired experiments, three smaller experiments were run:

| Experiment | Baseline | CoFrGeNet-F | Data | Result |
|-----------|----------|-------------|------|--------|
| **Exp 1** (param-efficient) | 124M (12L, 768d, 12h) | 82M (12L, 768d, 12h, L=3, d=5) | 10B tok | CoFrGeNet-F significantly worse (PPL 110 vs 41) |
| **Exp 2** (iso-parameter) | 124M (12L, 768d, 12h) | 128M (12L, 1024d, 16h, L=3, d=5) | 10B tok | Better than Exp 1 but still behind (PPL 82 vs 41) |
| **Exp 3** (more ladders) | — | 128M (12L, 1024d, 16h, L=8, d=5) | 10B tok | Testing if L=8 > L=3 |

Full results in [CLAUDE.md](../CLAUDE.md). These experiments motivated the larger-scale paired training on the DGX B200.

---

## 2. Paired Experiment Design

### Principles
- **Identical data**: Both models in a pair see the same tokens in the same order
- **Identical hyperparameters**: Same optimizer, LR schedule, warmup, weight decay, batch size
- **Same hardware**: Both trained on the same 8x B200 GPUs with DDP
- **Only the FFN differs**: Standard 2-layer FFN (GELU) vs Cffn (continued fraction)

### Training Recipe (All Pairs)

| Hyperparameter | Value |
|---------------|-------|
| Dataset | FineWeb-Edu ([HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)) |
| Tokenizer | GPT-2 (`tiktoken.get_encoding("gpt2")`, vocab 50,257) |
| Optimizer | AdamW (fused), beta1=0.9, beta2=0.95 |
| Weight decay | 0.1 (on 2D weight tensors only) |
| Gradient clipping | 1.0 (max norm) |
| Batch size | 524,288 tokens per update |
| Precision | bfloat16 via `torch.autocast` |
| Parallelism | DDP across 8 GPUs (FSDP reserved for models >2B params) |
| `torch.compile` | `mode='max-autotune'` with inductor cache |
| Dyadic schedule | CoFrGeNet-F only (see Section 4) |

---

## 3. Experiment Pairs

### Pair 1: 450M Baseline vs 410M CoFrGeNet-F (50B tokens)

**Purpose:** Sanity check at small scale. Validates the 8-GPU pipeline and establishes whether the parameter-efficiency claim holds at 450M scale.

**Status:** Baseline COMPLETE. CoFrGeNet-F IN PROGRESS (step ~25K/95K).

| | Baseline | CoFrGeNet-F |
|-|----------|-------------|
| **Config** | `configs/experiments/pair1_baseline_450m.yaml` | `configs/experiments/pair1_cofrgenet_410m.yaml` |
| **Params** | ~450M | ~410M |
| **Architecture** | 12L, 1600d, 25h, standard FFN | 12L, 2048d, 16h, L=3, d=5 |
| **Data** | 50B tokens (95,367 steps) | 50B tokens (95,367 steps) |
| **LR** | 6e-4 (cosine decay) | 6e-4 (cosine decay) |
| **Warmup** | 700 steps | 700 steps |
| **micro_batch_size** | 64 | 64 |
| **Checkpoints** | `checkpoints/pair1-baseline-450m/` | `checkpoints/pair1-cofrgenet-410m/` |

**Baseline results:** Completed 95,367 steps. Final train loss 2.67, val loss 2.68. Throughput ~2.12M tok/s. Total time ~3.5 hours.

**CoFrGeNet-F observations (in progress):**
- Throughput ~800K tok/s (lower due to Cffn's sequential continuant computation)
- Gradient norms elevated: 22-28 (vs baseline ~0.25 at same stage), clipped to 1.0
- Loss decreasing steadily despite gradient warnings (3.58 at step 25K)

### Pair 3: 7.5B Baseline vs 4.8B CoFrGeNet-F (100B tokens)

**Purpose:** Key experiment. 7x beyond the IBM paper's largest scale. Tests the central claim that CoFrGeNet-F's advantage grows with model size.

| | Baseline | CoFrGeNet-F |
|-|----------|-------------|
| **Config** | `configs/experiments/pair3_baseline_7b.yaml` | `configs/experiments/pair3_cofrgenet_5b.yaml` |
| **Params** | ~7.5B | ~4.8B (35% fewer) |
| **Architecture** | 36L, 4096d, 32h, standard FFN | 36L, 4608d, 36h, L=3, d=5 |
| **Data** | 100B tokens (190,734 steps) | 100B tokens (190,734 steps) |
| **LR** | 3e-4 (cosine decay) | 3e-4 (cosine decay) |
| **Warmup** | 2,000 steps | 2,000 steps |
| **micro_batch_size** | 4 | 4 |

### Pair 4: 9.9B Baseline vs 7.8B CoFrGeNet-F (100B tokens)

**Purpose:** Deep + wide scaling point. Tests whether adding more layers (48L) and more ladders (L=5) changes the dynamics.

| | Baseline | CoFrGeNet-F |
|-|----------|-------------|
| **Config** | `configs/experiments/pair4_baseline_10b.yaml` | `configs/experiments/pair4_cofrgenet_8b.yaml` |
| **Params** | ~9.9B | ~7.8B |
| **Architecture** | 48L, 4096d, 32h, standard FFN | 48L, 5120d, 40h, L=5, d=5 |
| **Data** | 100B tokens (190,734 steps) | 100B tokens (190,734 steps) |
| **LR** | 2e-4 | 2e-4 |
| **Warmup** | 2,000 steps | 2,000 steps |
| **micro_batch_size** | 2 | 2 |

### Pair 5: 12.9B Baseline vs 7.9B CoFrGeNet-F (100B tokens)

**Purpose:** Push scale. 38% parameter reduction. CoFrGeNet-F uses deeper continued fractions (d=7 vs d=5) and more ladders (L=5). Tests whether deeper CF ladders unlock richer function approximation at scale.

| | Baseline | CoFrGeNet-F |
|-|----------|-------------|
| **Config** | `configs/experiments/pair5_baseline_13b.yaml` | `configs/experiments/pair5_cofrgenet_8b.yaml` |
| **Params** | ~12.9B | ~7.9B (38% fewer) |
| **Architecture** | 40L, 5120d, 40h, standard FFN | 40L, 5632d, 44h, L=5, d=7 |
| **Data** | 100B tokens (190,734 steps) | 100B tokens (190,734 steps) |
| **LR** | 1.5e-4 | 1.5e-4 |
| **Warmup** | 2,000 steps | 2,000 steps |
| **micro_batch_size** | 2 | 2 |

---

## 4. The Dyadic Training Schedule

All CoFrGeNet-F models use the dyadic progressive unfreezing schedule from the IBM paper. Without it, performance degrades 10-80%.

**How it works:** Continued fraction depth parameters are frozen at initialization and progressively unfrozen on an exponential schedule. Depth layer `i` is unfrozen at step `(1 - 1/2^i) * total_steps`.

For Pair 1 (95,367 total steps):

| Depth | Unfreezes at step | % through training |
|-------|------------------|--------------------|
| 0 | 0 | 0% |
| 1 | 47,683 | 50% |
| 2 | 71,525 | 75% |
| 3 | 83,446 | 87.5% |
| 4 | 89,407 | 93.75% |

Implementation: `Cffn.set_active_depth()` installs `register_hook` callbacks that zero out gradients for frozen depth columns. See `src/cofrgenet/cffn.py`.

**Source:** Section 3.3 of [arXiv:2601.21766](https://arxiv.org/abs/2601.21766).

---

## 5. Observed: Gradient Explosion in Continued Fractions

### The Problem

During Pair 1 CoFrGeNet-F (410M) training, gradient norms climbed from ~0.58 (step 2K) to 26+ (step 25K). The baseline at the same stage had gradient norms ~0.25. Training is stable because `max_norm=1.0` gradient clipping caps the actual parameter updates, but the raw gradient magnitudes are 100x higher than the baseline.

### Root Cause

The continued fraction gradient formula (Proposition 1 of the IBM paper):

```
df/da_k = (-1)^k * [K_{d-k} / K_d]^2
```

This is the ratio of two continuant polynomials, squared. When `K_d` (the denominator continuant) approaches zero — i.e., the learned weights place the continued fraction near a **pole** of the rational function — the gradient explodes.

Current mitigation in `src/cofrgenet/continuant.py` (line 41): epsilon-clamping `K_d` to `min=0.01` in the forward pass. This prevents division by zero but doesn't prevent `K_d` from being *small*, which still produces large gradients.

### Why This Matters

Aggressive gradient clipping discards directional information. When the raw gradient norm is 26 but clipped to 1, 96% of the gradient signal is lost. This may explain why CoFrGeNet-F converges slower and to worse loss than the baseline — it's effectively training with a heavily distorted loss landscape.

---

## 6. Post-Experiment: Gradient Stabilization Research

**IMPORTANT:** These ideas are to be tested AFTER the current paired experiments complete. No modifications during active training.

### 6a. Engineering Approaches

**Adaptive Gradient Clipping (AGC)** — from the NFNet paper ([Brock et al., 2021](https://arxiv.org/abs/2102.06171)). Per-layer clipping based on `grad_norm / weight_norm` ratio. Lets attention layers train normally while specifically taming CF layers.

**Per-layer gradient scaling** — Scale the Cffn backward pass gradients by `1/||grad_cf||` before joining the rest of the network. Normalizes CF gradient magnitudes without affecting attention gradients.

### 6b. Mathematical Approaches

**Log-space continuant computation** — Work in log space where the `K_{d-1}/K_d` ratio becomes a difference `log(K_{d-1}) - log(K_d)`. Analogous to log-sum-exp stabilizing softmax. Gradients are bounded by construction because you never compute a ratio directly.

**Normalized continuant recurrence** — At each recurrence step, divide by `max(|K_{k-1}|, |K_{k-2}|)` and carry the normalization factor separately. The final ratio is unaffected (factors cancel), but intermediates stay numerically stable. Similar to techniques used in RNN stabilization.

**Denominator clamping in backward pass only** — Use `max(|K_d|, epsilon)` in the gradient computation only. Forward pass remains exact, backward is bounded. Analogous to epsilon in batch normalization.

**Learnable gradient scaling** — Per-ladder learnable scalar `alpha` that scales the backward pass. The network learns to attenuate its own gradient explosions.

### 6c. Recommended Combination

**Log-space continuants + AGC** is the most promising combination. Log-space fixes the root cause mathematically (the ratio never explodes because it's computed as a difference in log space). AGC handles any residual per-layer instability.

---

## 7. Connection to Prior Research on Continued Fraction Convergents

### Cahlen Humphreys, "Prime Numbers and the Convergents of a Continued Fraction" (NCUR 2013)

**Full text:** [Included in this repository](Humphreys_2013_Convergents_of_Continued_Fractions.pdf) | [Original hosted PDF](https://libjournals.unca.edu/ncur/wp-content/uploads/2021/09/579-Humphreys.pdf)
**Faculty Advisor:** Dr. Liljana Babinkostova, Boise State University

This paper studies the growth rate and prime factorization of numerators and denominators of continued fraction convergents. Its mathematical framework is directly relevant to the gradient explosion problem in CoFrGeNet-F.

### Key Results

**Theorem 1.4 (Continuant Recurrence):**
The numerator `A_n` and denominator `B_n` of the nth convergent of a continued fraction `[a_0; a_1, a_2, ...]` satisfy:

```
A_n = a_n * A_{n-1} + A_{n-2}
B_n = a_n * B_{n-1} + B_{n-2}
```

This is **identical** to the continuant recurrence in CoFrGeNet-F's `continuant.py`: `K_k = a_k * K_{k-1} + K_{k-2}`. The `K` values in our code are exactly the `A_n` and `B_n` from classical continued fraction theory.

**Theorem 3.5 (Reciprocal Symmetry — original result by Humphreys):**
The nth convergent of an irrational `x` (with `0 <= x <= 1`) is the reciprocal of the nth convergent of `1/x`. This means the numerator of one continued fraction's convergent is structurally the denominator of the reciprocal's convergent: `A_n(x) = B_n(1/x)`.

**Corollary 3.6 (Numerator Growth — extends Erdos-Mahler 1939):**
For almost all irrationals `zeta`, the greatest prime factor of the numerator `A_n` of the nth convergent satisfies:

```
G(A_n) >= e^(n / (50 * ln(n)))
```

The original Erdos-Mahler result (1939) only proved this for denominators `B_n`. Humphreys extended it to numerators via the reciprocal symmetry of Theorem 3.5.

### Application to CoFrGeNet-F Gradient Stability

The gradient formula `df/da_k = (-1)^k * [K_{d-k} / K_d]^2` is a ratio of continuants. From Theorem 3.5, the numerator continuant `K_{d-k}` and denominator continuant `K_d` are structurally linked through the reciprocal relationship.

**Proposed diagnostic:** Monitor the convergent ratio `K_{d-1}/K_d` during the forward pass. When this ratio indicates proximity to a pole — i.e., `K_d` is shrinking relative to `K_{d-k}`, meaning the reciprocal symmetry from Theorem 3.5 is breaking down — apply **adaptive damping** proportional to the pole proximity.

This is more mathematically principled than blind gradient clipping because it uses the *structural properties* of continued fractions (the same properties characterized in the Humphreys paper) rather than treating the continued fraction as a black box.

**Connection to log-space approach:** Working in log space naturally prevents the `K_{d-k}/K_d` ratio from exploding because you compute `log(K_{d-k}) - log(K_d)` instead of a direct division. The growth bounds from Corollary 3.6 suggest that in log space, the continuants grow approximately linearly with depth, making the difference well-behaved.

### Planned Experiments (Post-Pair Training)

1. **Implement convergent-ratio monitoring** — Log `K_{d-1}/K_d` per ladder, per layer, per step. Characterize the distribution and identify how often poles are approached during training.
2. **Correlate poles with gradient spikes** — Test whether the observed gradient norm spikes (22-28 at step 25K) correspond to poles in specific layers/ladders.
3. **Adaptive pole-aware damping** — When the convergent ratio exceeds a threshold, scale the backward pass for that specific ladder. Compare against AGC and log-space continuants.
4. **Log-space continuants with growth bounds** — Implement log-space computation and verify that Corollary 3.6's growth bounds hold empirically during training.

---

## 8. References

1. **CoFrGeNet paper:** Alonso, Fuentes, Perez-Cerrolaza, Paoletti, Perez-Bernabeu. "CoFrGeNet: A Generative Network Based on Continued Fractions." [arXiv:2601.21766](https://arxiv.org/abs/2601.21766), January 2026.
2. **CoFrNet (predecessor):** Peebles, Peharz, Shah. "Learning with Continued Fractions." NeurIPS 2021. [arXiv:2506.05586](https://arxiv.org/abs/2506.05586).
3. **Humphreys, C.** "Prime Numbers and the Convergents of a Continued Fraction." National Conference on Undergraduate Research (NCUR), 2013. Faculty Advisor: Dr. Liljana Babinkostova, Boise State University. [PDF (in repo)](Humphreys_2013_Convergents_of_Continued_Fractions.pdf) | [PDF (original)](https://libjournals.unca.edu/ncur/wp-content/uploads/2021/09/579-Humphreys.pdf).
4. **Erdos, P. and Mahler, K.** "Some Arithmetical Properties of the Convergents of a Continued Fraction." *Journal of the London Mathematical Society*, 1939. (Extended by Humphreys [3] to numerators.)
5. **NFNet / AGC:** Brock, De, Smith, Simonyan. "High-Performance Large-Scale Image Recognition Without Normalization." [arXiv:2102.06171](https://arxiv.org/abs/2102.06171), 2021.
6. **FineWeb-Edu:** HuggingFace. [huggingface.co/datasets/HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
7. **nanoGPT (reference implementation):** Karpathy. [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).

---

## 9. Evaluation Plan

All pairs evaluated head-to-head on the same hardware with `scripts/04_evaluate.py`:

- **WikiText-2 perplexity** — standard LM benchmark
- **WikiText-103 perplexity** — longer-context LM benchmark
- **LAMBADA perplexity & accuracy** — tests long-range dependency modeling
- **Throughput** (tokens/second) — measures training/inference efficiency
- **Generation speed** (ms/token) — measures practical generation latency

Results will be added to this document as each pair completes.

---

*Last updated: 2026-03-11*
