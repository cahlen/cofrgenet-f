"""CoFrGeNet-F Training Dashboard — Live monitoring with educational explanations.

A Gradio app that displays live training metrics from Trackio with
plain-language explanations of what each metric means.

Queries the Trackio Space API server-side using HF_TOKEN for authentication.
Deploy to HuggingFace Spaces: cahlen/cofrgenet-f-training
"""

import gradio as gr
import json
import os
import math
import pandas as pd

# ── Trackio API client ──────────────────────────────────────

TRACKIO_SPACE = "cahlen/cofrgenet-f-trackio"
PROJECT = "cofrgenet-f"

def _get_client():
    """Get authenticated Gradio client for Trackio Space."""
    try:
        from gradio_client import Client
        token = os.environ.get("HF_TOKEN")
        return Client(TRACKIO_SPACE, hf_token=token, verbose=False)
    except Exception:
        return None


def _api_call(api_name, **kwargs):
    """Make an API call to the Trackio Space. Returns None on failure."""
    try:
        client = _get_client()
        if client is None:
            return None
        return client.predict(api_name=api_name, **kwargs)
    except Exception:
        return None


# ── Theme & CSS ──────────────────────────────────────────────

CUSTOM_CSS = """
.metric-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
    background: #fafbfc;
}
.explanation {
    color: #555;
    font-size: 0.95em;
    line-height: 1.6;
    border-left: 3px solid #4a9eff;
    padding-left: 12px;
    margin: 10px 0;
}
.math-box {
    background: #f0f4ff;
    border: 1px solid #c0d0ff;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
    font-size: 1.05em;
}
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 0.85em;
}
.scaling-table th {
    background: #f0f4ff;
}
"""

# ── Content Sections ─────────────────────────────────────────

HEADER_MD = """
# CoFrGeNet-F: Scaling Continued Fractions in Language Models

### The Mission

Modern AI language models are powerful — but expensive. A single model can have **billions**
of tiny adjustable numbers (called "parameters") that it learns from text. Training these
models costs millions of dollars in electricity and specialized hardware.

**What if we could build models that are just as smart, but 30-40% smaller?**

That's the promise of **CoFrGeNet-F** — a new architecture from IBM Research that replaces
a core component of standard AI (the "feed-forward network") with **continued fractions**,
a mathematical concept dating back to the 1600s. Think of it as swapping out the engine
of a car for one that gets the same horsepower with fewer cylinders.

### Why We're Running This Experiment

IBM proved the idea works at small scale (~1 billion parameters). But no one has tested
it at the sizes where modern AI actually operates — **7 to 13 billion parameters**. That's
what we're doing here, on 8 NVIDIA B200 GPUs (some of the most powerful AI hardware in
the world).

**The big question:**

> *As models get bigger, does the continued fraction advantage **grow**, **shrink**,
> or **stay the same**?*

If it grows, continued fractions could fundamentally change how efficient AI models are
built. If it shrinks, that's still valuable to know — it tells the research community
where to focus instead.

**This dashboard is your front-row seat to that experiment.** Browse the tabs to understand
the architecture, the math, the hardware, and the live training metrics — all explained
for anyone from curious beginners to ML researchers.

---
"""

ARCHITECTURE_MD = """
## How CoFrGeNet-F Works

### The Standard Approach (Transformer)

Every modern language model processes text in repeated "blocks." Each block does two things:

1. **Attention** — figures out which words in a sentence relate to each other
   *(e.g., in "The cat sat on its mat", connecting "its" back to "cat")*

2. **Feed-Forward Network (FFN)** — transforms the information, like a filter that
   refines the model's understanding of each word

The FFN is simple but large: it takes each word's representation, expands it to 4x the size,
applies a mathematical curve (activation function), and compresses it back down. This single
component uses **~2/3 of the model's total parameters**.

### The CoFrGeNet-F Approach

CoFrGeNet-F keeps the attention step identical but **replaces the FFN** with a
**Continued Fraction FFN (Cffn)**. Instead of the simple expand-compress pattern,
it uses continued fractions — nested divisions that look like this:

$$\\frac{1}{a_1 + \\frac{1}{a_2 + \\frac{1}{a_3 + \\frac{1}{a_4 + \\frac{1}{a_5}}}}}$$

This looks complicated, but it's actually a very compact way to represent complex
mathematical functions. Think of it like this:

> **A standard FFN** is like describing a landscape using a fixed grid of elevation points —
> you need lots of points (parameters) to capture detail.
>
> **A continued fraction** is like describing the same landscape using a few carefully chosen
> reference points and a rule for interpolating between them — you can capture the same
> detail with far fewer numbers.

### The Key Claim

The paper from IBM Research showed that at ~1 billion parameters, CoFrGeNet-F matched a
standard model that had **1.5 billion parameters** — achieving similar quality with
**34% fewer parameters**. Our experiment tests whether this advantage holds, grows,
or disappears at much larger scales.
"""

MATH_MD = """
## The Mathematics (Simplified)

For those curious about the actual math — here's how it works under the hood.

### Continuant Polynomials

The continued fraction is computed efficiently using "continuant polynomials" — a recursive
formula that avoids doing many expensive divisions:

$$K_0 = 1, \\quad K_1 = a_d, \\quad K_k = a_{d-k+1} \\cdot K_{k-1} + K_{k-2}$$

The final result is simply:

$$\\tilde{f} = \\frac{K_{d-1}}{K_d}$$

*In plain language: instead of computing the nested fraction directly (which requires d
separate divisions), we build up two sequences of numbers using only multiplication and
addition, then do a single division at the end. This is much faster on GPU hardware.*

### The Cffn Layer

Each Cffn layer computes:

$$y = U \\cdot x + V \\cdot z$$

where:
- **U** is a direct linear connection (a safety net — if the fractions aren't helpful, the model can bypass them)
- **z** contains L independent continued fractions ("ladders"), each processing the input differently
- **V** weights how much each ladder contributes

*Think of it like having L different "experts" (the ladders), each offering their own interpretation
of the input, and the model learns how to combine their opinions.*

### Dyadic Training Schedule

One crucial trick: you don't train all the fraction depths at once. Instead, you gradually
"unfreeze" deeper levels as training progresses:

- **First half of training**: only the shallowest level learns
- **75% through**: two levels are learning
- **87.5% through**: three levels are learning
- **And so on...**

*This is like teaching someone to juggle — you start with one ball, add a second once
they're comfortable, then a third, rather than throwing all five at once.*
"""

EXPERIMENT_MD = """
## Our Experiments

We're training **matched pairs** — for each experiment, we train both a standard model
and a CoFrGeNet-F model on **exactly the same data** with **identical settings**.
The only difference is the FFN vs Cffn layer. This lets us isolate the effect of
continued fractions.

| Pair | Standard Model | CoFrGeNet-F Model | Param Reduction | Training Data | Purpose |
|:----:|:--------------:|:-----------------:|:---------------:|:------------:|:--------|
| 1 | 0.45B | 0.41B | 9.6% | 50B tokens | Sanity check — validate the pipeline |
| 3 | 7.46B | 4.83B | 35.3% | 100B tokens | **Key experiment** — 7x beyond paper's scale |
| 4 | 9.87B | 7.82B | 20.8% | 100B tokens | Does it hold with more layers? |
| 5 | 12.85B | 7.91B | 38.4% | 100B tokens | Push to the limit |

**"B" = Billions of parameters.** For context, the original ChatGPT was reportedly ~175B parameters.
Our largest experiment (12.85B) is about 1/14th that size, but large enough to show meaningful
trends in how these architectures scale.

**"Tokens"** are the basic units the model reads — roughly 3/4 of a word each.
100B tokens is approximately 75 billion words, or about 300,000 novels worth of text.
"""

METRICS_MD = """
## Understanding the Training Charts

Below are the live metrics from our training runs. Here's what each one means:
"""

LOSS_EXPLANATION = """
### Training Loss

**What it is:** A single number that measures how wrong the model's predictions are.
At each step, the model tries to predict the next word in a sentence. The loss measures
how far off those predictions were.

**What to look for:**
- **Going down = good.** The model is learning.
- **Going down fast at first, then slower** = normal. The easy patterns are learned first.
- **Sudden spike upward** = something went wrong (learning rate too high, bad data batch).
- **Flat line (plateau)** = the model has stopped improving at the current rate.

**Comparing the two models:** If CoFrGeNet-F's loss reaches the same low point as the
baseline despite having fewer parameters, that validates the continued fraction approach.
If it goes *lower*, that's a breakthrough finding.

*Technical detail: This is cross-entropy loss — it's related to perplexity by
perplexity = e^loss. A loss of 3.0 means the model is roughly as confused as
having to guess between ~20 equally likely next words.*
"""

VAL_LOSS_EXPLANATION = """
### Validation Loss

**What it is:** Same as training loss, but measured on text the model has **never seen during
training**. This is the true test of whether the model has actually learned general language
patterns vs. just memorizing the training data.

**What to look for:**
- **Close to training loss** = the model is generalizing well (learning real patterns)
- **Much higher than training loss** = overfitting (memorizing instead of learning)
- **Going up while training loss goes down** = definitely overfitting — time to stop training

**Why it matters:** Imagine studying for a test. Training loss is like your score on
practice problems you've seen before. Validation loss is your score on the actual exam
with new questions. The exam score is what really counts.
"""

LR_EXPLANATION = """
### Learning Rate

**What it is:** How big of a step the model takes when updating its knowledge after each
batch of text. Think of it like the stride length when walking toward a destination.

**The schedule we use:**
1. **Warmup** (first ~2,000 steps): Starts tiny and gradually increases. Like warming up
   before exercise — jumping in too fast can cause instability.
2. **Peak**: The maximum learning rate. The model is taking confident steps.
3. **Cosine decay**: Gradually shrinks back toward zero. As the model gets closer to
   a good solution, it takes smaller, more careful steps to avoid overshooting.

*You generally don't need to watch this closely — it follows a predetermined schedule.
But if training goes haywire, confirming the LR schedule is behaving as expected is
a good first diagnostic.*
"""

GRAD_NORM_EXPLANATION = """
### Gradient Norm

**What it is:** Measures how urgently the model wants to change its parameters at each step.
A large gradient norm means "I'm very wrong and need big corrections." A small one means
"I'm close to good and just need fine-tuning."

**What to look for:**
- **Gradually decreasing over time** = normal and healthy
- **Sudden spike** = the model hit a confusing batch of data or an instability
- **Consistently very high** = learning rate might be too high
- **Near zero** = the model has stopped learning (could be converged, or could be stuck)

*We clip the gradient norm to a maximum of 1.0 to prevent instability. When you see it
at exactly 1.0, that means clipping is active — the model wanted to make a bigger change
but we're keeping it in check.*

*Analogy: If training is like driving to a destination, the gradient norm is how hard
you're pressing the gas pedal. Clipping is the speed limiter.*
"""

PERPLEXITY_EXPLANATION = """
### Perplexity

**What it is:** A more intuitive version of loss. It answers: "how many words does the model
think are equally likely to come next?" Lower perplexity = more confident, better predictions.

**What the numbers mean:**
- **Perplexity 50** = the model is choosing between ~50 equally likely next words (early training)
- **Perplexity 20** = narrowed down to ~20 candidates (getting better)
- **Perplexity 10** = only ~10 plausible options (quite good for this scale)
- **GPT-2 (1.5B params)** achieved perplexity ~18 on WikiText-2

**The formula:** perplexity = e^loss. So a loss of 3.0 = perplexity of ~20.

**Why we track both:** Loss is what the math optimizes; perplexity is what humans can reason about.
A drop from loss 4.0 to 3.5 sounds small, but in perplexity that's 55 to 33 — the model nearly
halved its confusion.
"""

THROUGHPUT_EXPLANATION = """
### Throughput (tokens/second)

**What it is:** How fast the model is processing text — specifically, how many tokens
(word pieces) it reads and learns from per second.

**What to look for:**
- **Stable number** = everything is running smoothly
- **Sudden drop** = possible hardware issue, memory pressure, or slow data loading
- **Higher is better** = means training finishes sooner

**Context for our hardware:** On 8x NVIDIA B200 GPUs (some of the fastest AI chips
available in 2026), we expect:
- Baseline models: 300,000-500,000+ tokens/second
- CoFrGeNet-F models: somewhat lower, because continued fractions involve sequential
  operations that are harder to parallelize on GPUs

*This is why CoFrGeNet-F's parameter efficiency matters — even if it's slower per step,
if it needs fewer total steps or achieves better results with fewer parameters, the
overall cost could still be lower.*
"""

HARDWARE_MD = """
## Our Hardware

### NVIDIA DGX System with 8x B200 GPUs

We're training on a cluster of 8 **NVIDIA B200 "Blackwell" GPUs** — the latest generation
of AI accelerators released in 2025/2026.

| Spec | Per GPU | Total (8 GPUs) |
|:-----|--------:|---------------:|
| **GPU Memory** | 179 GB | 1,432 GB |
| **BF16 Performance** | 4,500 TFLOPS | 36,000 TFLOPS |
| **Architecture** | Blackwell (5th gen Tensor Cores) | |
| **Interconnect** | NVLink 5 (900 GB/s) | Full mesh |

**What does this mean in plain language?**

- **179 GB of memory per GPU** — this is where the model's "brain" lives during training.
  A 7.5 billion parameter model takes ~15 GB just for the parameters, plus much more for
  the intermediate calculations. We use a technique called FSDP (Fully Sharded Data Parallel)
  that splits the model across all 8 GPUs so they share the load.

- **36,000 TFLOPS** — that's 36 quadrillion math operations per second across all 8 GPUs.
  Training a language model requires an enormous amount of arithmetic. Our 7.5B model on
  100B tokens requires roughly 4.5 x 10^21 operations total (that's 4.5 sextillion).
  Even at this speed, training takes days, not minutes.

- **NVLink 5** — a ultra-fast direct connection between GPUs. When GPUs need to share
  information (which happens constantly during distributed training), this connection is
  ~10x faster than standard PCIe connections.

### Training Speed Estimates

| Model | Est. Training Time | Tokens/Second |
|:------|-------------------:|--------------:|
| 0.45B Baseline (sanity check) | ~3 hours | ~500K |
| 7.46B Baseline | ~3.5 days | ~300K |
| 4.83B CoFrGeNet-F | ~3 days | ~200K |
| 12.85B Baseline | ~6 days | ~200K |

*CoFrGeNet-F models are slower per step because continued fractions involve sequential
operations (each depth level depends on the previous one) that can't be fully parallelized.
This is the trade-off: fewer parameters but more serial computation per parameter.*
"""

GLOSSARY_MD = """
## Glossary

A reference for terminology used throughout this dashboard.

| Term | Plain English | Technical Detail |
|:-----|:-------------|:-----------------|
| **Parameters** | The "knowledge slots" in a model — numbers that get adjusted during training | Learnable weights and biases in the neural network |
| **Tokens** | Word pieces — roughly 3/4 of a word each | GPT-2 BPE tokenizer splits text into ~50,257 possible tokens |
| **Loss** | How wrong the model's predictions are (lower = better) | Cross-entropy between predicted and actual next-token distributions |
| **Perplexity** | How "confused" the model is — roughly, how many words it thinks are equally likely | e^loss — a perplexity of 20 means ~20 equally likely candidates |
| **Epoch** | One complete pass through all training data | We train for 1 epoch on 50-100B tokens |
| **Batch** | A chunk of text processed together in one step | ~524,288 tokens per gradient update |
| **Learning Rate** | How big of an adjustment the model makes each step | Starts at 0, warms up, then decays via cosine schedule |
| **Gradient** | The direction and magnitude of the needed adjustment | Computed via backpropagation through the network |
| **Gradient Norm** | How "urgent" the needed adjustment is | L2 norm of all gradients; clipped to 1.0 for stability |
| **FSDP** | Splits the model across multiple GPUs | Fully Sharded Data Parallel — each GPU holds 1/8th of params |
| **BF16** | Half-precision numbers for faster math | Brain Float 16 — 16-bit format with same range as float32 |
| **torch.compile** | Optimizes the model code for faster execution | PyTorch's JIT compiler fuses operations for GPU efficiency |
| **Checkpoint** | A saved snapshot of the model during training | Saved as safetensors files every 5,000 steps |
| **FFN** | The "thinking" layer in a standard Transformer | 2-layer MLP: expand 4x, apply GELU, compress back |
| **Cffn** | CoFrGeNet-F's replacement for the FFN | Uses continued fractions instead of expand-compress |
| **Ladders (L)** | Independent continued fraction "experts" in a Cffn | Each ladder computes its own fraction; outputs are combined |
| **Depth (d)** | How deep the continued fraction nesting goes | d=5 means 5 levels of 1/(a + 1/(a + ...)) |
| **Dyadic Schedule** | Gradually unfreezes deeper fraction levels | Depth i unfreezes at step t*(1 - 1/2^i) |
| **Continuant** | The efficient way to compute continued fractions | Recursive polynomial: K_k = a * K_{k-1} + K_{k-2} |
| **Weight Tying** | The input and output layers share the same parameters | Reduces param count and improves generalization |
| **Pole Avoidance** | Prevents division by zero in fractions | Clamps denominator: sgn(K_d) * max(|K_d|, 0.01) |
"""

SCALING_MD = """
## The Scaling Question

### Why Scale Matters

AI research has discovered "scaling laws" — predictable relationships between model size,
training data, and model quality. Generally:

> **Bigger models trained on more data produce better results, with diminishing returns.**

But this raises an important question for CoFrGeNet-F:

### The Paper's Finding (Small Scale)

At ~1 billion parameters, IBM showed:

| Model | Parameters | Quality (Perplexity on WikiText-2) |
|:------|----------:|-----------------------------------:|
| Standard Transformer | 1.5B | 18.30 |
| CoFrGeNet-F | 985M (34% fewer) | ~Similar or better |

This is impressive — but 1B parameters is small by modern standards.

### What We're Testing (Large Scale)

**The open question:** Does the continued fraction advantage **grow**, **shrink**, or
**stay the same** as models get bigger?

There are reasons to think it could go either way:

**Arguments it could grow:**
- Continued fractions can approximate any rational function with very few parameters
- As models get larger, the FFN's 4x expansion becomes increasingly wasteful
- The paper showed the advantage *increased* going from small to medium scale

**Arguments it could shrink:**
- Standard FFNs benefit from massive parallelism on GPUs; continued fractions are more sequential
- At very large scale, the sheer number of FFN parameters might enable representations
  that continued fractions can't match
- Training instabilities tend to get worse at scale

**Our experiment will answer this definitively** — and either answer is interesting and publishable.
If continued fractions scale well, it could change how we build efficient AI models.
If they don't, that's valuable information too.
"""

FOOTER_MD = """
---

## About This Project

**Paper:** [CoFrGeNet: From Continued Fractions to Large Language Models](https://arxiv.org/abs/2601.21766)
(IBM Research, January 2026)

**Our Code:** [github.com/cahlen/cofrgenet-f](https://github.com/cahlen/cofrgenet-f)

**Models:** [huggingface.co/cahlen/cofrgenet-f](https://huggingface.co/cahlen/cofrgenet-f)

**Hardware:** 8x NVIDIA B200 GPUs (DGX system) — 1,432 GB total GPU memory

**Training Data:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
(educational web text, curated by HuggingFace)

Built with [Trackio](https://github.com/gradio-app/trackio) for experiment tracking
and [Gradio](https://gradio.app) for this dashboard.
"""


# ── Helper functions — query Trackio Space API ──────────────

def fetch_runs():
    """Get list of runs from Trackio Space."""
    runs = _api_call("/get_runs_for_project", project=PROJECT)
    return runs or []


def fetch_metric(run_name, metric_name):
    """Fetch metric values as list of {step, value} dicts."""
    data = _api_call("/get_metric_values", project=PROJECT, run=run_name,
                     metric_name=metric_name, step=None, around_step=None,
                     at_time=None, window=None)
    return data or []


def fetch_run_summary(run_name):
    """Get summary for a run including config and last step."""
    return _api_call("/get_run_summary", project=PROJECT, run=run_name) or {}


def fetch_alerts():
    """Get alerts from Trackio Space."""
    alerts = _api_call("/get_alerts", project=PROJECT, run=None, level=None, since=None)
    return alerts or []


def fetch_system_metrics(run_name):
    """Get system metrics (GPU utilization, memory, temp, power)."""
    return _api_call("/get_system_logs", project=PROJECT, run=run_name) or []


def build_chart(metric_name, y_label):
    """Build a line chart for a metric across all runs."""
    runs = fetch_runs()
    if not runs:
        return None

    all_data = []
    for run in runs:
        values = fetch_metric(run, metric_name)
        for v in values:
            all_data.append({
                "step": v.get("step", 0),
                "value": v.get("value", 0),
                "run": run,
            })

    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    return gr.LinePlot(
        value=df, x="step", y="value", color="run",
        title=f"{y_label} vs Training Step",
        y_title=y_label, x_title="Step",
        height=350, width=700,
    )


def format_status():
    """Build status summary from Trackio Space API."""
    runs = fetch_runs()
    if not runs:
        return ("**Status:** Waiting for training to begin. Experiment configs are ready, "
                "data is downloaded. Training will start once GPUs are available.")

    lines = ["**Active Runs:**\n"]
    for run_name in runs:
        summary = fetch_run_summary(run_name)
        num_logs = summary.get("num_logs", "?")
        last_step = summary.get("last_step", "?")
        config = summary.get("config") or {}
        total_steps = config.get("total_steps", "?")
        lines.append(f"- **{run_name}** — step {last_step}/{total_steps} ({num_logs} log entries)")
    return "\n".join(lines)


def format_alerts():
    """Build alert summary from Trackio Space API."""
    alerts = fetch_alerts()
    if not alerts:
        return ("No alerts yet. Alerts will appear here when training starts — they flag "
                "important events like loss spikes, training milestones, or potential problems.")

    lines = []
    level_icons = {"info": "[INFO]", "warn": "[WARN]", "error": "[ERROR]"}
    for alert in alerts[-30:]:
        level = level_icons.get(alert.get("level", "info"), "[?]")
        title = alert.get("title", "")
        text = alert.get("text", "")
        run = alert.get("run", "")
        step = alert.get("step", "")
        step_str = f" (step {step})" if step else ""
        lines.append(f"**{level}** `{run}`{step_str} **{title}** — {text}")
    return "\n\n".join(reversed(lines))  # newest first


def build_all_charts():
    """Build all metric charts. Returns list of (df, title) for each metric."""
    runs = fetch_runs()
    empty = pd.DataFrame({"step": [], "value": [], "run": []})
    if not runs:
        return tuple([empty] * 6)

    charts = {}
    metrics_to_plot = [
        ("train/loss", "Training Loss"),
        ("val/loss", "Validation Loss"),
        ("train/perplexity", "Training Perplexity"),
        ("train/grad_norm", "Gradient Norm"),
        ("train/tokens_per_sec", "Throughput (tok/s)"),
        ("train/lr", "Learning Rate"),
    ]

    results = []
    for metric_name, label in metrics_to_plot:
        all_data = []
        for run in runs:
            values = fetch_metric(run, metric_name)
            for v in values:
                all_data.append({
                    "step": v.get("step", 0),
                    "value": v.get("value", 0),
                    "run": run,
                })
        if all_data:
            results.append(pd.DataFrame(all_data))
        else:
            results.append(pd.DataFrame({"step": [], "value": [], "run": []}))

    # Pad to 6 results
    while len(results) < 6:
        results.append(pd.DataFrame({"step": [], "value": [], "run": []}))

    return tuple(results)


def build_gpu_summary():
    """Build GPU metrics summary from system metrics."""
    runs = fetch_runs()
    if not runs:
        return "No GPU data available yet. System metrics will appear once training starts with GPU monitoring enabled."

    # Get system logs from the most recent run
    sys_logs = fetch_system_metrics(runs[-1])
    if not sys_logs:
        return "No GPU metrics logged yet. Next training run will include GPU monitoring (utilization, memory, temperature, power)."

    # Get the most recent system log entry
    latest = sys_logs[-1] if sys_logs else {}
    metrics = latest.get("metrics", {})
    if not metrics:
        return "System metrics are being collected but none received yet."

    lines = [f"**GPU Status** (latest reading from run `{runs[-1]}`):\n"]
    lines.append("| GPU | Utilization | Memory Used | Temperature | Power |")
    lines.append("|:----|:----------:|:-----------:|:----------:|:-----:|")

    for i in range(8):
        util = metrics.get(f"gpu/{i}/utilization", "?")
        mem = metrics.get(f"gpu/{i}/allocated_memory", "?")
        total_mem = metrics.get(f"gpu/{i}/total_memory", "?")
        temp = metrics.get(f"gpu/{i}/temp", "?")
        power = metrics.get(f"gpu/{i}/power", "?")

        if isinstance(mem, (int, float)) and isinstance(total_mem, (int, float)):
            mem_str = f"{mem:.1f}/{total_mem:.0f} GB"
        else:
            mem_str = "?"
        if isinstance(power, (int, float)):
            power_str = f"{power:.0f}W"
        else:
            power_str = "?"
        if isinstance(temp, (int, float)):
            temp_str = f"{temp}C"
        else:
            temp_str = "?"
        if isinstance(util, (int, float)):
            util_str = f"{util}%"
        else:
            util_str = "?"

        lines.append(f"| GPU {i} | {util_str} | {mem_str} | {temp_str} | {power_str} |")

    mean_util = metrics.get("gpu/mean_utilization", "?")
    total_power = metrics.get("gpu/total_power", "?")
    max_temp = metrics.get("gpu/max_temp", "?")
    lines.append(f"\n**Aggregate:** {mean_util}% mean utilization, "
                 f"{total_power:.0f}W total power, {max_temp}C max temp"
                 if isinstance(total_power, (int, float))
                 else f"\n**Aggregate:** {mean_util}% mean utilization")

    return "\n".join(lines)


def refresh_dashboard():
    """Refresh all dynamic content."""
    status = format_status()
    alerts = format_alerts()
    gpu_status = build_gpu_summary()
    charts = build_all_charts()
    return (status, alerts, gpu_status) + charts


# ── Build the Gradio App ─────────────────────────────────────

with gr.Blocks(title="CoFrGeNet-F Training Dashboard") as demo:

    gr.Markdown(HEADER_MD)

    with gr.Tabs():

        # ── Tab 1: Live Training ──
        with gr.Tab("Live Training"):
            gr.Markdown("## Training Status")
            gr.Markdown(
                "*This page auto-refreshes every 60 seconds. "
                "Click Refresh for an immediate update.*"
            )
            status_md = gr.Markdown(value="Loading...")

            gr.Markdown("## Live Training Charts")
            gr.Markdown(
                "Each line represents a different training run. Baseline and CoFrGeNet-F "
                "runs within the same pair are grouped together so you can compare directly. "
                "See the **Understanding the Charts** tab for what each metric means."
            )

            with gr.Row():
                loss_plot = gr.LinePlot(x="step", y="value", color="run",
                                       title="Training Loss", y_title="Loss", x_title="Step",
                                       height=300)
                val_loss_plot = gr.LinePlot(x="step", y="value", color="run",
                                            title="Validation Loss", y_title="Val Loss", x_title="Step",
                                            height=300)
            with gr.Row():
                ppl_plot = gr.LinePlot(x="step", y="value", color="run",
                                       title="Training Perplexity", y_title="Perplexity", x_title="Step",
                                       height=300)
                grad_plot = gr.LinePlot(x="step", y="value", color="run",
                                        title="Gradient Norm", y_title="Grad Norm", x_title="Step",
                                        height=300)
            with gr.Row():
                tps_plot = gr.LinePlot(x="step", y="value", color="run",
                                       title="Throughput", y_title="Tokens/sec", x_title="Step",
                                       height=300)
                lr_plot = gr.LinePlot(x="step", y="value", color="run",
                                      title="Learning Rate", y_title="LR", x_title="Step",
                                      height=300)

            gr.Markdown("## GPU Status")
            gpu_md = gr.Markdown(value="Loading...")

            gr.Markdown("## Alerts & Events")
            gr.Markdown(
                "The training code automatically fires alerts when something noteworthy happens: "
                "milestones reached, potential problems detected, or training completing. "
                "Think of these like a smart assistant watching the training and tapping you "
                "on the shoulder when something needs attention."
            )
            alerts_md = gr.Markdown(value="Loading...")

            refresh_btn = gr.Button("Refresh", variant="primary")
            all_outputs = [status_md, alerts_md, gpu_md,
                          loss_plot, val_loss_plot, ppl_plot, grad_plot, tps_plot, lr_plot]
            refresh_btn.click(fn=refresh_dashboard, outputs=all_outputs)

            timer = gr.Timer(60)
            timer.tick(fn=refresh_dashboard, outputs=all_outputs)

        # ── Tab 2: Understanding the Charts ──
        with gr.Tab("Understanding the Charts"):
            gr.Markdown(METRICS_MD)
            gr.Markdown(LOSS_EXPLANATION)
            gr.Markdown(VAL_LOSS_EXPLANATION)
            gr.Markdown(PERPLEXITY_EXPLANATION)
            gr.Markdown(LR_EXPLANATION)
            gr.Markdown(GRAD_NORM_EXPLANATION)
            gr.Markdown(THROUGHPUT_EXPLANATION)

        # ── Tab 3: How It Works ──
        with gr.Tab("How CoFrGeNet-F Works"):
            gr.Markdown(ARCHITECTURE_MD)

        # ── Tab 4: The Math ──
        with gr.Tab("The Mathematics"):
            gr.Markdown(MATH_MD)

        # ── Tab 5: Experiments ──
        with gr.Tab("Experiment Plan"):
            gr.Markdown(EXPERIMENT_MD)

        # ── Tab 6: The Scaling Question ──
        with gr.Tab("The Scaling Question"):
            gr.Markdown(SCALING_MD)

        # ── Tab 7: Hardware ──
        with gr.Tab("Our Hardware"):
            gr.Markdown(HARDWARE_MD)

        # ── Tab 8: Glossary ──
        with gr.Tab("Glossary"):
            gr.Markdown(GLOSSARY_MD)

    gr.Markdown(FOOTER_MD)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CUSTOM_CSS)
