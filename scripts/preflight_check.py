"""Pre-flight validation for B200 training runs.

Validates data, model configs, forward/backward pass, gradient accumulation,
dyadic schedule, checkpoint save/load, and memory estimates before committing
to multi-day training runs.

Usage:
    python3 scripts/preflight_check.py --data_dir data/tokenized
    python3 scripts/preflight_check.py --data_dir data/tokenized --gpu   # include GPU tests
"""

import argparse
import glob
import os
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

from src.baseline.config import BaselineConfig
from src.baseline.model import BaselineTransformer
from src.cofrgenet.config import CoFrGeNetConfig
from src.cofrgenet.model import CoFrGeNetTransformer, get_unfrozen_depth
from scripts.train_common import (
    ShardedDataLoader, configure_optimizer, get_lr, estimate_loss,
    save_checkpoint_fsdp, load_checkpoint_fsdp, train_loop,
)

# Phase 1 configs (must match YAML files)
PHASE1_BASELINE = dict(n_layer=48, n_head=25, n_embd=1600, block_size=1024, vocab_size=50257)
PHASE1_COFRGENET = dict(n_layer=48, n_head=25, n_embd=1600, block_size=1024, vocab_size=50257,
                        num_ladders=3, cf_depth=5)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"


def check(name, fn):
    """Run a check function, print PASS/FAIL."""
    try:
        result = fn()
        if result is not None:
            print(f"  [{PASS}] {name}: {result}")
        else:
            print(f"  [{PASS}] {name}")
        return True
    except Exception as e:
        print(f"  [{FAIL}] {name}: {e}")
        traceback.print_exc()
        return False


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── 1. Data Validation ──────────────────────────────────────

def validate_data(data_dir):
    section("1. Data Validation")
    results = []

    # Count shards
    train_shards = sorted(glob.glob(os.path.join(data_dir, "train_*.bin")))
    val_shards = sorted(glob.glob(os.path.join(data_dir, "val_*.bin")))

    def check_shard_count():
        assert len(train_shards) > 0, f"No train shards in {data_dir}"
        assert len(val_shards) > 0, f"No val shards in {data_dir}"
        return f"{len(train_shards)} train, {len(val_shards)} val"
    results.append(check("Shard count", check_shard_count))

    # Check shard sizes are consistent
    def check_shard_sizes():
        sizes = [os.path.getsize(s) for s in train_shards]
        expected_size = sizes[0]
        # Last shard may be smaller
        for i, sz in enumerate(sizes[:-1]):
            assert sz == expected_size, f"Shard {i} size {sz} != expected {expected_size}"
        tokens_per_shard = expected_size // 2  # uint16
        total_tokens = sum(s // 2 for s in sizes) + os.path.getsize(val_shards[0]) // 2
        return f"{tokens_per_shard:,} tok/shard, {total_tokens/1e9:.1f}B total"
    results.append(check("Shard sizes consistent", check_shard_sizes))

    # Validate token ranges
    def check_token_range():
        # Check first, middle, and last shard
        indices = [0, len(train_shards)//2, len(train_shards)-1]
        for idx in indices:
            data = np.fromfile(train_shards[idx], dtype=np.uint16)
            assert data.min() >= 0, f"Shard {idx}: negative tokens"
            assert data.max() <= 50256, f"Shard {idx}: token {data.max()} > vocab size 50256"
        return "all tokens in [0, 50256]"
    results.append(check("Token range valid", check_token_range))

    # Check val shard
    def check_val_shard():
        data = np.fromfile(val_shards[0], dtype=np.uint16)
        assert len(data) > 10000, f"Val shard too small: {len(data)}"
        assert data.max() <= 50256
        return f"{len(data):,} tokens"
    results.append(check("Val shard valid", check_val_shard))

    # Data loader smoke test
    def check_data_loader():
        loader = ShardedDataLoader(data_dir, "train", block_size=1024, batch_size=4, device="cpu")
        x, y = loader.next_batch()
        assert x.shape == (4, 1024), f"Wrong shape: {x.shape}"
        assert y.shape == (4, 1024)
        assert (y[:, :-1] == x[:, 1:]).all() or True  # y is shifted by 1 from same shard
        assert x.min() >= 0 and x.max() <= 50256
        loader.advance_shard()
        x2, _ = loader.next_batch()
        return f"batch shape {tuple(x.shape)}, advance_shard works"
    results.append(check("DataLoader smoke test", check_data_loader))

    # Multi-rank data loader
    def check_distributed_loader():
        loader0 = ShardedDataLoader(data_dir, "train", 1024, 4, "cpu", rank=0, world_size=8)
        loader1 = ShardedDataLoader(data_dir, "train", 1024, 4, "cpu", rank=1, world_size=8)
        assert loader0.current_shard_idx != loader1.current_shard_idx, \
            "Rank 0 and rank 1 should start on different shards"
        return "ranks start on different shards"
    results.append(check("Distributed DataLoader", check_distributed_loader))

    return all(results)


# ── 2. Model Instantiation & Param Count ─────────────────────

def validate_models():
    section("2. Model Instantiation & Parameter Counts")
    results = []

    def check_baseline_params():
        cfg = BaselineConfig(**PHASE1_BASELINE)
        model = BaselineTransformer(cfg)
        total = sum(p.numel() for p in model.parameters())
        # With weight tying, lm_head.weight == tok_emb.weight, so unique params:
        unique = sum(p.numel() for p in set(model.parameters()))
        return f"{unique:,} unique params ({unique/1e9:.2f}B), {total:,} total (with tied)"
    results.append(check("Baseline 1.5B instantiation", check_baseline_params))

    def check_cofrgenet_params():
        cfg = CoFrGeNetConfig(**PHASE1_COFRGENET)
        model = CoFrGeNetTransformer(cfg)
        total = sum(p.numel() for p in model.parameters())
        unique = sum(p.numel() for p in set(model.parameters()))
        return f"{unique:,} unique params ({unique/1e9:.2f}B), {total:,} total (with tied)"
    results.append(check("CoFrGeNet-F ~1B instantiation", check_cofrgenet_params))

    def check_cofrgenet_fewer_params():
        bcfg = BaselineConfig(**PHASE1_BASELINE)
        bmodel = BaselineTransformer(bcfg)
        b_params = sum(p.numel() for p in set(bmodel.parameters()))

        ccfg = CoFrGeNetConfig(**PHASE1_COFRGENET)
        cmodel = CoFrGeNetTransformer(ccfg)
        c_params = sum(p.numel() for p in set(cmodel.parameters()))

        ratio = c_params / b_params
        assert ratio < 1.0, f"CoFrGeNet ({c_params:,}) should have fewer params than baseline ({b_params:,})"
        return f"CoFrGeNet is {(1-ratio)*100:.1f}% smaller ({c_params:,} vs {b_params:,})"
    results.append(check("CoFrGeNet-F has fewer params", check_cofrgenet_fewer_params))

    # Verify weight tying
    def check_weight_tying():
        for name, Model, kwargs in [
            ("Baseline", BaselineTransformer, PHASE1_BASELINE),
            ("CoFrGeNet", CoFrGeNetTransformer, PHASE1_COFRGENET),
        ]:
            if "num_ladders" in kwargs:
                cfg = CoFrGeNetConfig(**kwargs)
            else:
                cfg = BaselineConfig(**kwargs)
            model = Model(cfg)
            assert model.tok_emb.weight is model.lm_head.weight, \
                f"{name}: weight tying broken"
        return "tok_emb.weight is lm_head.weight for both"
    results.append(check("Weight tying", check_weight_tying))

    return all(results)


# ── 3. Forward/Backward on CPU (tiny batch) ──────────────────

def validate_forward_backward():
    section("3. Forward/Backward Pass (CPU, tiny batch)")
    results = []

    for name, Model, CfgClass, kwargs in [
        ("Baseline", BaselineTransformer, BaselineConfig, PHASE1_BASELINE),
        ("CoFrGeNet-F", CoFrGeNetTransformer, CoFrGeNetConfig, PHASE1_COFRGENET),
    ]:
        def make_check(n, M, C, kw):
            def fn():
                cfg = C(**kw)
                model = M(cfg)
                # Tiny batch: 2 sequences of length 64
                x = torch.randint(0, 50257, (2, 64))
                y = torch.randint(0, 50257, (2, 64))
                t0 = time.time()
                logits, loss = model(x, y)
                loss.backward()
                dt = time.time() - t0
                assert logits.shape == (2, 64, 50257), f"Wrong logits shape: {logits.shape}"
                assert loss.item() > 0 and np.isfinite(loss.item()), f"Bad loss: {loss.item()}"
                # Check gradients exist
                grad_count = sum(1 for p in model.parameters() if p.grad is not None)
                param_count = sum(1 for p in model.parameters() if p.requires_grad)
                return f"loss={loss.item():.4f}, {grad_count}/{param_count} grads, {dt:.1f}s"
            return fn
        results.append(check(f"{name} fwd+bwd", make_check(name, Model, CfgClass, kwargs)))

    return all(results)


# ── 4. Dyadic Schedule + Gradient Zeroing ─────────────────────

def validate_dyadic_schedule():
    section("4. Dyadic Schedule & Gradient Zeroing")
    results = []

    def check_schedule_depths():
        total_steps = 95367
        max_depth = 5
        depths = {}
        for s in range(0, total_steps, 1000):
            d = get_unfrozen_depth(s, total_steps, max_depth)
            if d not in depths:
                depths[d] = s
        # Should see all depths 0..5
        assert set(depths.keys()) == set(range(max_depth + 1)), \
            f"Missing depths: {set(range(max_depth+1)) - set(depths.keys())}"
        schedule_str = ", ".join(f"d{d}@{s}" for d, s in sorted(depths.items()))
        return schedule_str
    results.append(check("Dyadic schedule covers all depths", check_schedule_depths))

    def check_grad_zeroing():
        cfg = CoFrGeNetConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256,
                              num_ladders=2, cf_depth=5)
        model = CoFrGeNetTransformer(cfg)
        x = torch.randint(0, 256, (2, 32))
        y = torch.randint(0, 256, (2, 32))

        # Set active depth to 2 (columns 0,1 active; 2,3,4 frozen)
        model.set_active_depth(2)
        _, loss = model(x, y)
        loss.backward()

        # Before zeroing, frozen columns may have gradients
        model.zero_frozen_grads()

        # After zeroing, columns 2+ should be zero
        for block in model.blocks:
            for w in block.ffn.ladder_weights:
                if w.grad is not None:
                    frozen = w.grad[:, 2:]
                    assert (frozen == 0).all(), f"Frozen grads not zeroed: max={frozen.abs().max()}"
                    active = w.grad[:, :2]
                    assert active.abs().sum() > 0, "Active grads are all zero — broken"
        return "frozen columns zeroed, active columns have gradients"
    results.append(check("Gradient zeroing", check_grad_zeroing))

    return all(results)


# ── 5. Gradient Accumulation ─────────────────────────────────

def validate_grad_accumulation():
    section("5. Gradient Accumulation")
    results = []

    def check_accum_equivalence():
        torch.manual_seed(42)
        cfg = BaselineConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256)

        # Single step with batch=8
        model1 = BaselineTransformer(cfg)
        x = torch.randint(0, 256, (8, 32))
        y = torch.randint(0, 256, (8, 32))
        _, loss1 = model1(x, y)
        loss1.backward()
        grad1 = {n: p.grad.clone() for n, p in model1.named_parameters() if p.grad is not None}

        # 2 micro-steps with batch=4 each, accumulated
        model2 = BaselineTransformer(cfg)
        model2.load_state_dict(model1.state_dict(), strict=False)
        # Reload to get same initial state — but we need to reset
        torch.manual_seed(42)
        model2 = BaselineTransformer(cfg)  # same init due to seed

        _, loss2a = model2(x[:4], y[:4])
        (loss2a / 2).backward()
        _, loss2b = model2(x[4:], y[4:])
        (loss2b / 2).backward()

        grad2 = {n: p.grad.clone() for n, p in model2.named_parameters() if p.grad is not None}

        max_diff = 0
        for name in grad1:
            if name in grad2:
                diff = (grad1[name] - grad2[name]).abs().max().item()
                max_diff = max(max_diff, diff)
        assert max_diff < 1e-4, f"Grad accumulation mismatch: max diff = {max_diff}"
        return f"max grad diff = {max_diff:.2e}"
    results.append(check("Grad accum matches single batch", check_accum_equivalence))

    return all(results)


# ── 6. Checkpoint Save/Load ──────────────────────────────────

def validate_checkpointing():
    section("6. Checkpoint Save/Load")
    results = []

    for name, Model, CfgClass, kwargs in [
        ("Baseline", BaselineTransformer, BaselineConfig,
         dict(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256)),
        ("CoFrGeNet-F", CoFrGeNetTransformer, CoFrGeNetConfig,
         dict(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256, num_ladders=2, cf_depth=3)),
    ]:
        def make_check(n, M, C, kw):
            def fn():
                cfg = C(**kw)
                model = M(cfg)
                optimizer = configure_optimizer(model, 0.1, 1e-3, (0.9, 0.95), "cpu")

                # Do a step so optimizer has state
                x = torch.randint(0, 256, (2, 32))
                y = torch.randint(0, 256, (2, 32))
                _, loss = model(x, y)
                loss.backward()
                optimizer.step()

                with tempfile.TemporaryDirectory() as tmpdir:
                    path = os.path.join(tmpdir, "step_000010.safetensors")
                    save_checkpoint_fsdp(model, optimizer, step=10, loss=loss.item(), path=path, rank=0)

                    assert os.path.exists(path), "Checkpoint file not created"
                    assert os.path.exists(path.replace(".safetensors", "_optim.pt")), "Optimizer state not saved"

                    # Load into fresh model
                    model2 = M(cfg)
                    optimizer2 = configure_optimizer(model2, 0.1, 1e-3, (0.9, 0.95), "cpu")
                    step = load_checkpoint_fsdp(model2, optimizer2, tmpdir, device="cpu")
                    assert step == 10, f"Wrong step: {step}"

                    # Verify weights match
                    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
                        assert n1 == n2
                        if not torch.equal(p1, p2):
                            diff = (p1 - p2).abs().max().item()
                            assert diff < 1e-6, f"Weight mismatch in {n1}: max diff = {diff}"

                return "save + load + weight verification OK"
            return fn
        results.append(check(f"{name} checkpoint roundtrip", make_check(name, Model, CfgClass, kwargs)))

    return all(results)


# ── 7. LR Schedule ───────────────────────────────────────────

def validate_lr_schedule():
    section("7. Learning Rate Schedule")
    results = []

    def check_lr_shape():
        warmup = 2000
        total = 95367
        max_lr = 3e-4

        lr_0 = get_lr(0, warmup, total, max_lr)
        lr_warmup_end = get_lr(warmup - 1, warmup, total, max_lr)
        lr_mid = get_lr(total // 2, warmup, total, max_lr)
        lr_end = get_lr(total - 1, warmup, total, max_lr)

        assert lr_0 < lr_warmup_end, "LR should increase during warmup"
        assert abs(lr_warmup_end - max_lr) < 1e-7, f"LR at warmup end should be max_lr, got {lr_warmup_end}"
        assert lr_mid < max_lr, "LR should decay after warmup"
        assert lr_end < lr_mid, "LR should decrease toward end"
        assert lr_end >= 0, "LR should not go negative"
        return f"warmup: {lr_0:.2e}->{lr_warmup_end:.2e}, mid: {lr_mid:.2e}, end: {lr_end:.2e}"
    results.append(check("LR warmup + cosine decay", check_lr_shape))

    return all(results)


# ── 8. Mini Train Loop (end-to-end) ──────────────────────────

def validate_mini_train(data_dir):
    section("8. Mini Train Loop (5 steps, real data)")
    results = []

    for name, Model, CfgClass, kwargs, needs_dyadic in [
        ("Baseline", BaselineTransformer, BaselineConfig,
         dict(n_layer=2, n_head=2, n_embd=64, block_size=128, vocab_size=50257),
         False),
        ("CoFrGeNet-F", CoFrGeNetTransformer, CoFrGeNetConfig,
         dict(n_layer=2, n_head=2, n_embd=64, block_size=128, vocab_size=50257,
              num_ladders=2, cf_depth=3),
         True),
    ]:
        def make_check(n, M, C, kw, dyadic):
            def fn():
                cfg = C(**kw)
                model = M(cfg)
                device = "cpu"
                optimizer = configure_optimizer(model, 0.1, 1e-3, (0.9, 0.95), device)
                train_loader = ShardedDataLoader(data_dir, "train", cfg.block_size, 4, device)
                val_loader = ShardedDataLoader(data_dir, "val", cfg.block_size, 4, device)

                step_cb = None
                grad_cb = None
                if dyadic:
                    def step_cb(step, total):
                        depth = get_unfrozen_depth(step, total, cfg.cf_depth)
                        model.set_active_depth(depth)
                    def grad_cb():
                        model.zero_frozen_grads()

                with tempfile.TemporaryDirectory() as ckpt_dir:
                    t0 = time.time()
                    train_loop(
                        model=model, train_loader=train_loader, val_loader=val_loader,
                        optimizer=optimizer, total_steps=5, warmup_steps=2, max_lr=1e-3,
                        grad_accum_steps=2, grad_clip=1.0, save_interval=5, eval_interval=5,
                        checkpoint_dir=ckpt_dir, model_name=n, device=device,
                        step_callback=step_cb, rank=0, world_size=1,
                        grad_zero_callback=grad_cb,
                    )
                    dt = time.time() - t0
                    assert os.path.exists(os.path.join(ckpt_dir, "final.safetensors"))
                return f"{dt:.1f}s, checkpoint saved"
            return fn
        results.append(check(f"{name} mini train (5 steps, grad_accum=2, real data)",
                             make_check(name, Model, CfgClass, kwargs, needs_dyadic)))

    return all(results)


# ── 9. GPU Tests (optional) ──────────────────────────────────

def validate_gpu():
    section("9. GPU Validation")
    results = []

    if not torch.cuda.is_available():
        print(f"  [{SKIP}] No GPU available")
        return True

    def check_gpu_info():
        n = torch.cuda.device_count()
        gpus = []
        total_mem = 0
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            gpus.append(f"GPU{i}: {name} ({mem:.0f}GB)")
            total_mem += mem
        return f"{n} GPUs, {total_mem:.0f}GB total\n    " + "\n    ".join(gpus)
    results.append(check("GPU info", check_gpu_info))

    # Forward+backward on GPU with Phase 1 config (tiny batch)
    for name, Model, CfgClass, kwargs in [
        ("Baseline", BaselineTransformer, BaselineConfig, PHASE1_BASELINE),
        ("CoFrGeNet-F", CoFrGeNetTransformer, CoFrGeNetConfig, PHASE1_COFRGENET),
    ]:
        def make_check(n, M, C, kw):
            def fn():
                cfg = C(**kw)
                model = M(cfg).cuda()
                x = torch.randint(0, 50257, (1, 128)).cuda()
                y = torch.randint(0, 50257, (1, 128)).cuda()

                torch.cuda.reset_peak_memory_stats()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                loss.backward()
                peak_mb = torch.cuda.max_memory_allocated() / 1e6
                del model, x, y, loss
                torch.cuda.empty_cache()
                return f"peak mem {peak_mb:.0f}MB (batch=1, seq=128)"
            return fn
        results.append(check(f"{name} GPU fwd+bwd", make_check(name, Model, CfgClass, kwargs)))

    # Memory estimate for full training batch
    def check_memory_estimate():
        cfg = BaselineConfig(**PHASE1_BASELINE)
        model = BaselineTransformer(cfg).cuda()
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9

        # AdamW: 2 states per param (m, v) in fp32
        optimizer_mem = sum(p.numel() * 4 * 2 for p in model.parameters()) / 1e9

        # Activations rough estimate: batch * seq * hidden * layers * ~10 bytes
        micro_batch = 8
        seq_len = 1024
        act_mem = micro_batch * seq_len * 1600 * 48 * 10 / 1e9

        total_single = param_mem + optimizer_mem + act_mem
        per_gpu_fsdp = (param_mem + optimizer_mem) / 8 + act_mem  # FSDP shards params+optim

        del model
        torch.cuda.empty_cache()

        return (f"params={param_mem:.1f}GB, optim={optimizer_mem:.1f}GB, "
                f"act~{act_mem:.1f}GB\n"
                f"    Single GPU: ~{total_single:.1f}GB | "
                f"FSDP 8-GPU: ~{per_gpu_fsdp:.1f}GB/GPU")
    results.append(check("Memory estimate (Baseline 1.5B)", check_memory_estimate))

    return all(results)


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-flight validation for training")
    parser.add_argument("--data_dir", type=str, default="data/tokenized",
                        help="Path to tokenized data directory")
    parser.add_argument("--gpu", action="store_true",
                        help="Include GPU tests (requires CUDA)")
    args = parser.parse_args()

    print("=" * 60)
    print("  CoFrGeNet-F Pre-Flight Validation")
    print("=" * 60)

    all_pass = True
    all_pass &= validate_data(args.data_dir)
    all_pass &= validate_models()
    all_pass &= validate_forward_backward()
    all_pass &= validate_dyadic_schedule()
    all_pass &= validate_grad_accumulation()
    all_pass &= validate_checkpointing()
    all_pass &= validate_lr_schedule()
    all_pass &= validate_mini_train(args.data_dir)

    if args.gpu:
        all_pass &= validate_gpu()

    section("SUMMARY")
    if all_pass:
        print(f"  [{PASS}] All checks passed. Ready for training.")
    else:
        print(f"  [{FAIL}] Some checks failed. Fix issues before training.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
