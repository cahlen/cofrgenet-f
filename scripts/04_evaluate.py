"""Evaluate both baseline and CoFrGeNet-F models on standard benchmarks.

Benchmarks:
- WikiText-2 perplexity (stride-512)
- WikiText-103 perplexity (stride-512)
- LAMBADA perplexity + accuracy (last-word prediction)
- Throughput (tokens/sec during forward pass)
- Generation speed (ms/token)

Usage:
    python scripts/04_evaluate.py --model cofrgenet --checkpoint checkpoints/cofrgenet/final.safetensors
    python scripts/04_evaluate.py --model baseline --checkpoint checkpoints/baseline/model.safetensors
    python scripts/04_evaluate.py --model both  # evaluate both and compare
"""

import argparse
import json
import math
import os
import time

import tiktoken
import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import load_file

from src.baseline.config import BaselineConfig
from src.baseline.model import BaselineTransformer
from src.cofrgenet.config import CoFrGeNetConfig
from src.cofrgenet.model import CoFrGeNetTransformer


def load_model(model_type, checkpoint_path, device, config_overrides=None):
    """Load a model from a safetensors checkpoint."""
    config_overrides = config_overrides or {}
    if model_type == "baseline":
        config = BaselineConfig(**config_overrides)
        model = BaselineTransformer(config)
    else:
        config = CoFrGeNetConfig(**config_overrides)
        model = CoFrGeNetTransformer(config)

    state_dict = load_file(checkpoint_path)
    # Strip torch.compile (_orig_mod.) and DDP (module.) prefixes
    cleaned = {}
    for k, v in state_dict.items():
        clean_k = k.removeprefix("_orig_mod.").removeprefix("module.")
        cleaned[clean_k] = v
    # strict=False because lm_head.weight is tied to tok_emb.weight
    info = model.load_state_dict(cleaned, strict=False)
    if info.unexpected_keys:
        print(f"  WARNING: unexpected keys: {info.unexpected_keys[:5]}")
    missing_non_tied = [k for k in info.missing_keys if "lm_head" not in k]
    if missing_non_tied:
        print(f"  WARNING: missing keys (non-tied): {missing_non_tied[:5]}")
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_type} model: {total_params:,} parameters")
    return model, config, total_params


def tokenize_text(text, enc):
    """Tokenize text using tiktoken GPT-2 encoding."""
    return enc.encode(text, allowed_special=set())


def stride_perplexity(model, token_ids, block_size, stride, device):
    """Compute perplexity with sliding window and stride.

    Standard approach: slide a window of block_size over the text with given stride.
    Only score tokens that are new in each window (the last `stride` tokens),
    except for the first window where we score all tokens.
    """
    seq_len = len(token_ids)
    nlls = []
    total_tokens = 0

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + block_size, seq_len)
        input_ids = torch.tensor(token_ids[begin:end], dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(input_ids)

        # Only score tokens in the stride region (or all for first window)
        target_start = max(0, stride - (end - begin - stride))
        if begin == 0:
            target_start = 0

        # Shift: predict token[i+1] from logits[i]
        shift_logits = logits[0, target_start:-1, :]
        shift_labels = input_ids[0, target_start + 1:]

        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        num_tokens = shift_labels.numel()
        nlls.append(loss.item())
        total_tokens += num_tokens

        if end >= seq_len:
            break

    avg_nll = sum(nlls) / total_tokens
    ppl = math.exp(avg_nll)
    return ppl, total_tokens


def eval_wikitext(model, dataset_name, block_size, device):
    """Evaluate perplexity on WikiText-2 or WikiText-103."""
    enc = tiktoken.get_encoding("gpt2")

    if "103" in dataset_name:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text
    text = "\n\n".join(ds["text"])
    token_ids = tokenize_text(text, enc)
    print(f"  {dataset_name}: {len(token_ids):,} tokens")

    ppl, num_tokens = stride_perplexity(model, token_ids, block_size, stride=512, device=device)
    print(f"  {dataset_name} perplexity: {ppl:.2f} ({num_tokens:,} scored tokens)")
    return ppl


def eval_lambada(model, block_size, device):
    """Evaluate on LAMBADA — perplexity and accuracy of last-word prediction."""
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("lambada", split="test")

    total_nll = 0.0
    total_tokens = 0
    correct = 0
    total_examples = 0

    for example in ds:
        text = example["text"]
        tokens = tokenize_text(text, enc)

        if len(tokens) < 2 or len(tokens) > block_size:
            continue

        # Find last word tokens
        words = text.rstrip().rsplit(" ", 1)
        if len(words) < 2:
            continue
        last_word = " " + words[-1]  # include leading space for tokenizer
        last_word_tokens = tokenize_text(last_word, enc)
        num_last = len(last_word_tokens)

        if num_last == 0:
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(input_ids)

        # Perplexity: NLL over last-word tokens
        # logits[i] predicts token[i+1], so for last_word starting at position -num_last:
        start_pos = len(tokens) - num_last - 1
        shift_logits = logits[0, start_pos:-1, :]
        shift_labels = input_ids[0, start_pos + 1:]

        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        total_nll += loss.item()
        total_tokens += num_last

        # Accuracy: check if greedy prediction matches all last-word tokens
        predicted = shift_logits.argmax(dim=-1)
        if torch.equal(predicted, shift_labels):
            correct += 1
        total_examples += 1

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    acc = 100.0 * correct / total_examples

    print(f"  LAMBADA perplexity: {ppl:.2f} ({total_tokens:,} tokens, {total_examples:,} examples)")
    print(f"  LAMBADA accuracy: {acc:.2f}%")
    return ppl, acc


def eval_throughput(model, block_size, device, num_batches=50, batch_size=16):
    """Measure forward-pass throughput in tokens/sec."""
    model.eval()
    # Warmup
    dummy = torch.randint(0, 50257, (batch_size, block_size), device=device)
    for _ in range(5):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(dummy)
    torch.cuda.synchronize()

    total_tokens = 0
    t0 = time.time()
    for _ in range(num_batches):
        dummy = torch.randint(0, 50257, (batch_size, block_size), device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(dummy)
        total_tokens += batch_size * block_size
    torch.cuda.synchronize()
    dt = time.time() - t0

    tok_per_sec = total_tokens / dt
    print(f"  Throughput: {tok_per_sec:,.0f} tok/s ({num_batches} batches of {batch_size}x{block_size})")
    return tok_per_sec


def eval_generation_speed(model, block_size, device, num_tokens=200, num_runs=5):
    """Measure generation speed in ms/token."""
    enc = tiktoken.get_encoding("gpt2")
    prompt = enc.encode("The meaning of life is")
    prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)

    # Warmup
    model.generate(prompt_t, max_new_tokens=20, temperature=0.8, top_k=50)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        model.generate(prompt_t, max_new_tokens=num_tokens, temperature=0.8, top_k=50)
        torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)

    avg_time = sum(times) / len(times)
    ms_per_tok = (avg_time / num_tokens) * 1000
    print(f"  Generation speed: {ms_per_tok:.2f} ms/tok ({num_tokens} tokens, {num_runs} runs)")
    return ms_per_tok


def evaluate_model(model_type, checkpoint_path, device, config_overrides=None):
    """Run all benchmarks on a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    model, config, total_params = load_model(model_type, checkpoint_path, device, config_overrides)
    block_size = config.block_size

    results = {
        "model": model_type,
        "checkpoint": checkpoint_path,
        "parameters": {
            "total": total_params,
            "unique": total_params,
        },
    }

    # WikiText-2
    print("\n[WikiText-2]")
    results["wikitext2_ppl"] = eval_wikitext(model, "WikiText-2", block_size, device)

    # WikiText-103
    print("\n[WikiText-103]")
    results["wikitext103_ppl"] = eval_wikitext(model, "WikiText-103", block_size, device)

    # LAMBADA
    print("\n[LAMBADA]")
    lambada_ppl, lambada_acc = eval_lambada(model, block_size, device)
    results["lambada_ppl"] = lambada_ppl
    results["lambada_acc"] = lambada_acc

    # Throughput
    print("\n[Throughput]")
    results["throughput_tok_per_sec"] = eval_throughput(model, block_size, device)

    # Generation speed
    print("\n[Generation Speed]")
    results["generation_ms_per_tok"] = eval_generation_speed(model, block_size, device)

    # Save results
    out_dir = os.path.dirname(checkpoint_path)
    out_path = os.path.join(out_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def print_comparison(baseline_results, cofrgenet_results):
    """Print a side-by-side comparison table."""
    print(f"\n{'='*70}")
    print("HEAD-TO-HEAD COMPARISON")
    print(f"{'='*70}")

    b = baseline_results
    c = cofrgenet_results

    rows = [
        ("Parameters", f"{b['parameters']['total']:,}", f"{c['parameters']['total']:,}",
         f"{100*(1 - c['parameters']['total']/b['parameters']['total']):.1f}% fewer"),
        ("WikiText-2 PPL", f"{b['wikitext2_ppl']:.2f}", f"{c['wikitext2_ppl']:.2f}",
         f"{'better' if c['wikitext2_ppl'] < b['wikitext2_ppl'] else 'worse'}"),
        ("WikiText-103 PPL", f"{b['wikitext103_ppl']:.2f}", f"{c['wikitext103_ppl']:.2f}",
         f"{'better' if c['wikitext103_ppl'] < b['wikitext103_ppl'] else 'worse'}"),
        ("LAMBADA PPL", f"{b['lambada_ppl']:.2f}", f"{c['lambada_ppl']:.2f}",
         f"{'better' if c['lambada_ppl'] < b['lambada_ppl'] else 'worse'}"),
        ("LAMBADA Acc", f"{b['lambada_acc']:.2f}%", f"{c['lambada_acc']:.2f}%",
         f"{'better' if c['lambada_acc'] > b['lambada_acc'] else 'worse'}"),
        ("Throughput", f"{b['throughput_tok_per_sec']:,.0f} tok/s", f"{c['throughput_tok_per_sec']:,.0f} tok/s",
         f"{c['throughput_tok_per_sec']/b['throughput_tok_per_sec']:.2f}x"),
        ("Gen Speed", f"{b['generation_ms_per_tok']:.2f} ms/tok", f"{c['generation_ms_per_tok']:.2f} ms/tok",
         f"{b['generation_ms_per_tok']/c['generation_ms_per_tok']:.2f}x"),
    ]

    print(f"{'Metric':<20} {'Baseline (124M)':<22} {'CoFrGeNet-F (82M)':<22} {'Note'}")
    print("-" * 70)
    for name, bval, cval, note in rows:
        print(f"{name:<20} {bval:<22} {cval:<22} {note}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate language models on standard benchmarks")
    parser.add_argument("--model", type=str, default="both",
                        choices=["baseline", "cofrgenet", "both"],
                        help="Which model(s) to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (required for single model)")
    parser.add_argument("--baseline_checkpoint", type=str,
                        default="checkpoints/baseline/model.safetensors")
    parser.add_argument("--cofrgenet_checkpoint", type=str,
                        default="checkpoints/cofrgenet/final.safetensors")
    # Config overrides for non-default model dimensions
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    args = parser.parse_args()

    config_overrides = {}
    if args.n_embd is not None:
        config_overrides["n_embd"] = args.n_embd
    if args.n_head is not None:
        config_overrides["n_head"] = args.n_head
    if args.n_layer is not None:
        config_overrides["n_layer"] = args.n_layer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.model == "both":
        baseline_results = evaluate_model("baseline", args.baseline_checkpoint, device)
        cofrgenet_results = evaluate_model("cofrgenet", args.cofrgenet_checkpoint, device, config_overrides)
        print_comparison(baseline_results, cofrgenet_results)
    elif args.model == "baseline":
        ckpt = args.checkpoint or args.baseline_checkpoint
        evaluate_model("baseline", ckpt, device, config_overrides)
    else:
        ckpt = args.checkpoint or args.cofrgenet_checkpoint
        evaluate_model("cofrgenet", ckpt, device, config_overrides)


if __name__ == "__main__":
    main()
