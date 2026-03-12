"""Shared training infrastructure for both baseline and CoFrGeNet-F models.

Provides:
- DataLoader: memory-mapped binary shards with random sampling
- LR schedule: linear warmup + cosine decay
- Training loop with gradient accumulation, mixed precision, logging
- Checkpointing via safetensors
"""

import os
import math
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_model, load_file, save_file
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial


def setup_torch_performance():
    """Enable performance optimizations for CUDA training."""
    # TF32: use TensorFloat32 for float32 matmuls (3x faster on Ampere+)
    torch.set_float32_matmul_precision("high")
    # cuDNN benchmark: auto-tune conv algorithms
    torch.backends.cudnn.benchmark = True
    # Allow TF32 on matmul and cuDNN
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def is_distributed():
    """Check if running in distributed mode (launched via torchrun)."""
    return "RANK" in os.environ


def setup_distributed():
    """Initialize distributed training. Returns (rank, local_rank, world_size).
    If not launched via torchrun, returns single-GPU defaults (0, 0, 1).
    """
    if not is_distributed():
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_distributed(model, device, use_gradient_checkpointing=False, force_fsdp=False):
    """Wrap a model for distributed training.

    For models <2B params: uses DDP (simpler, faster, no dtype issues with torch.compile).
    For models >=2B params: uses FSDP FULL_SHARD (required to fit in GPU memory).
    """
    from src.baseline.model import TransformerBlock

    if use_gradient_checkpointing:
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                module._original_forward = module.forward
                module.forward = partial(
                    _checkpointed_forward, module._original_forward
                )

    total_params = sum(p.numel() for p in model.parameters())
    use_fsdp = force_fsdp or total_params >= 2_000_000_000

    if use_fsdp:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock},
        )
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=device,
            use_orig_params=True,
        )
    else:
        model = model.bfloat16()
        model = DDP(model, device_ids=[device])

    return model


# Keep old name as alias for backwards compatibility
wrap_model_fsdp = wrap_model_distributed


def _checkpointed_forward(original_forward, *args, **kwargs):
    """Wrapper for gradient checkpointing."""
    from torch.utils.checkpoint import checkpoint
    return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)


class ShardedDataLoader:
    """Loads tokenized binary shards with random sampling."""

    def __init__(self, data_dir, split, block_size, batch_size, device="cpu",
                 rank=0, world_size=1):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.rank = rank
        self.world_size = world_size

        pattern = f"{split}_"
        self.shards = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.startswith(pattern) and f.endswith(".bin")
        ])
        assert len(self.shards) > 0, f"No {split} shards found in {data_dir}"

        self.current_shard_idx = 0
        self.current_data = None
        start_shard = rank % len(self.shards)
        self._load_shard(start_shard)

    def _load_shard(self, idx):
        self.current_shard_idx = idx % len(self.shards)
        self.current_data = np.memmap(
            self.shards[self.current_shard_idx], dtype=np.uint16, mode="r"
        )

    def next_batch(self):
        """Sample a random batch from the current shard."""
        data = self.current_data
        max_start = len(data) - self.block_size - 1
        if max_start <= 0:
            self._load_shard(self.current_shard_idx + 1)
            data = self.current_data
            max_start = len(data) - self.block_size - 1

        starts = np.random.randint(0, max_start, size=self.batch_size)
        x = np.stack([data[s:s + self.block_size].astype(np.int64) for s in starts])
        y = np.stack([data[s + 1:s + self.block_size + 1].astype(np.int64) for s in starts])

        x = torch.from_numpy(x).to(self.device)
        y = torch.from_numpy(y).to(self.device)
        return x, y

    def advance_shard(self):
        """Move to the next shard. In distributed mode, skip by world_size."""
        self._load_shard(self.current_shard_idx + self.world_size)


def get_lr(step, warmup_steps, total_steps, max_lr, min_lr=0.0):
    """Linear warmup then cosine decay to min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizer(model, weight_decay, lr, betas, device):
    """Configure AdamW with weight decay only on 2D parameters (weight matrices)."""
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    # Use fused AdamW if available
    use_fused = device.startswith("cuda") and torch.cuda.is_available()
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=use_fused)
    return optimizer


def save_checkpoint(model, optimizer, step, loss, path):
    """Save model checkpoint as safetensors + optimizer state as .pt."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_model(model, path)
    # Save optimizer state separately
    opt_path = path.replace(".safetensors", "_optim.pt")
    torch.save({"optimizer": optimizer.state_dict(), "step": step, "loss": loss}, opt_path)


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in a directory. Returns (step, safetensors_path) or None."""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("step_") and f.endswith(".safetensors")
    ])
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    step = int(latest.split("_")[1].split(".")[0])
    return step, os.path.join(checkpoint_dir, latest)


def resume_from_checkpoint(model, optimizer, checkpoint_dir, device):
    """Resume training from the latest checkpoint. Returns the step to resume from, or 0."""
    result = find_latest_checkpoint(checkpoint_dir)
    if result is None:
        return 0
    step, ckpt_path = result
    opt_path = ckpt_path.replace(".safetensors", "_optim.pt")

    print(f"Resuming from checkpoint: {ckpt_path} (step {step})")
    state_dict = load_file(ckpt_path)
    # Strip torch.compile _orig_mod. prefix if present
    cleaned = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)

    if os.path.exists(opt_path):
        opt_state = torch.load(opt_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(opt_state["optimizer"])
        print(f"Restored optimizer state from step {step}")

    return step


def save_checkpoint_fsdp(model, optimizer, step, loss, path, rank=0):
    """Save checkpoint, handling both FSDP and non-FSDP models.
    For FSDP: gathers full state dict on rank 0, saves there only.
    For non-FSDP: saves normally (same as existing save_checkpoint).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if isinstance(model, FSDP):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = model.state_dict()
            if rank == 0:
                save_file(state_dict, path)
        if rank == 0:
            opt_path = path.replace(".safetensors", "_optim.pt")
            optim_state = FSDP.optim_state_dict(model, optimizer)
            torch.save({"optimizer": optim_state, "step": step, "loss": loss}, opt_path)
    else:
        save_model(model, path)
        opt_path = path.replace(".safetensors", "_optim.pt")
        torch.save({"optimizer": optimizer.state_dict(), "step": step, "loss": loss}, opt_path)


def load_checkpoint_fsdp(model, optimizer, checkpoint_dir, device="cpu"):
    """Load latest checkpoint, handling both FSDP and non-FSDP models.
    Returns the step to resume from, or 0 if no checkpoint found.
    """
    result = find_latest_checkpoint(checkpoint_dir)
    if result is None:
        return 0
    step, ckpt_path = result
    opt_path = ckpt_path.replace(".safetensors", "_optim.pt")

    print(f"Resuming from checkpoint: {ckpt_path} (step {step})")

    if isinstance(model, FSDP):
        state_dict = load_file(ckpt_path)
        cleaned = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(cleaned, strict=False)
        if os.path.exists(opt_path):
            opt_state = torch.load(opt_path, map_location=device, weights_only=False)
            optim_state = FSDP.optim_state_dict_to_load(
                model, optimizer, opt_state["optimizer"]
            )
            optimizer.load_state_dict(optim_state)
    else:
        state_dict = load_file(ckpt_path)
        # Strip prefixes from torch.compile (_orig_mod.) and DDP (module.)
        cleaned = {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in state_dict.items()}
        # DDP models need loading into .module, compiled DDP into ._orig_mod.module
        target = model
        if hasattr(model, '_orig_mod'):
            target = model._orig_mod
        if hasattr(target, 'module'):
            target = target.module
        target.load_state_dict(cleaned, strict=False)
        if os.path.exists(opt_path):
            opt_state = torch.load(opt_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(opt_state["optimizer"])

    return step


def estimate_loss(model, data_loader, num_batches=20, use_autocast=True):
    """Estimate loss over num_batches."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = data_loader.next_batch()
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    total_steps,
    warmup_steps,
    max_lr,
    grad_accum_steps,
    grad_clip,
    save_interval,
    eval_interval,
    checkpoint_dir,
    model_name,
    device,
    step_callback=None,
    wandb_run=None,
    resume_step=0,
    rank=0,
    world_size=1,
    grad_zero_callback=None,
    trackio_space=None,
    loss_callback=None,
):
    """Main training loop shared by both models.

    Args:
        step_callback: Optional fn(step, total_steps) called each step,
                       e.g., for dyadic schedule updates.
        wandb_run: Optional wandb run for logging (also used by trackio).
        resume_step: Step to resume training from (0 = start from scratch).
        rank: Process rank for distributed training (0 = main process).
        world_size: Total number of processes (1 = single GPU).
        grad_zero_callback: Optional fn() called after backward to zero frozen gradients.
        trackio_space: Optional HF Space ID for periodic trackio sync.
        loss_callback: Optional fn(step, loss) called after each step for custom monitoring.
    """
    model.train()

    # FSDP MixedPrecision handles casting, so don't double-cast
    use_autocast = not isinstance(model, FSDP)

    step = resume_step
    t0 = time.time()
    tokens_processed = 0
    batch_tokens = train_loader.batch_size * train_loader.block_size

    # Alert tracking state
    prev_val_loss = None
    loss_history = []
    throughput_baseline = None  # established after torch.compile warmup
    throughput_alert_step = 0   # throttle throughput alerts

    if rank == 0:
        if resume_step > 0:
            print(f"Resuming training from step {resume_step}/{total_steps}")
        print(f"Starting training: {total_steps} steps, {grad_accum_steps} grad accum steps")
        print(f"Tokens per update: {batch_tokens * grad_accum_steps:,}")

    while step < total_steps:
        # Update learning rate
        lr = get_lr(step, warmup_steps, total_steps, max_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Call step callback (e.g., dyadic schedule)
        if step_callback is not None:
            step_callback(step, total_steps)

        # Gradient accumulation
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.item()
            loss.backward()

        # Zero frozen gradients (dyadic schedule for CoFrGeNet-F)
        if grad_zero_callback is not None:
            grad_zero_callback()

        # Gradient clipping — FSDP has its own method
        if grad_clip > 0:
            if isinstance(model, FSDP):
                grad_norm = model.clip_grad_norm_(grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            grad_norm = 0.0

        optimizer.step()

        tokens_processed += batch_tokens * grad_accum_steps * world_size
        step += 1

        # Sync loss across ranks for accurate logging
        if world_size > 1:
            loss_tensor = torch.tensor(loss_accum, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            loss_accum = loss_tensor.item()

        # Advance shard periodically
        if step % 100 == 0:
            train_loader.advance_shard()

        # ── Alerts: detect training problems (rank 0 only) ──
        if rank == 0:
            _check_training_alerts(
                step, total_steps, loss_accum, float(grad_norm),
                loss_history, warmup_steps, wandb_run, t0,
            )

        # Custom loss monitoring (e.g., dyadic depth change stability)
        if loss_callback is not None:
            loss_callback(step, loss_accum)

        # Logging
        if rank == 0 and (step % 10 == 0 or step == 1):
            dt = time.time() - t0
            tokens_per_sec = tokens_processed / dt if dt > 0 else 0
            ppl = math.exp(min(loss_accum, 20))  # cap to avoid overflow
            print(
                f"step {step:>6d}/{total_steps} | "
                f"loss {loss_accum:.4f} | ppl {ppl:.1f} | "
                f"lr {lr:.2e} | "
                f"grad_norm {float(grad_norm):.2f} | "
                f"tok/s {tokens_per_sec:,.0f}"
            )

            if wandb_run is not None:
                wandb_run.log({
                    "train/loss": loss_accum,
                    "train/perplexity": ppl,
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/tokens": tokens_processed,
                }, step=step)

            # Throughput collapse detection (after compile warmup at step 100)
            if step == 100:
                throughput_baseline = tokens_per_sec
            if (throughput_baseline is not None and tokens_per_sec < throughput_baseline * 0.5
                    and step > throughput_alert_step + 500):  # throttle to once per 500 steps
                _fire_alert(
                    "Throughput collapse — GPU may be stalled",
                    f"Throughput dropped to {tokens_per_sec:,.0f} tok/s "
                    f"({tokens_per_sec/throughput_baseline*100:.0f}% of baseline {throughput_baseline:,.0f}). "
                    f"Check for OOM swapping, NCCL hangs, or GPU thermal throttling. "
                    f"Every slow hour wastes limited cluster time.",
                    "warn", wandb_run,
                )
                throughput_alert_step = step

        # Periodic trackio sync to HF Space (every 100 steps)
        if rank == 0 and trackio_space and step % 100 == 0:
            try:
                import trackio
                trackio.sync(
                    project=wandb_run._project if hasattr(wandb_run, '_project') else "cofrgenet-f",
                    space_id=trackio_space,
                    run_in_background=True,
                )
            except Exception:
                pass  # sync failure is not fatal

        # Log val loss — ALL ranks must participate (FSDP deadlock prevention)
        if step % eval_interval == 0 or step == total_steps:
            val_loss = estimate_loss(model, val_loader, use_autocast=use_autocast)
            if world_size > 1:
                vl_tensor = torch.tensor(val_loss, device=device)
                dist.all_reduce(vl_tensor, op=dist.ReduceOp.AVG)
                val_loss = vl_tensor.item()
            if rank == 0:
                val_ppl = math.exp(min(val_loss, 20))
                print(f"  >>> val_loss: {val_loss:.4f} | val_ppl: {val_ppl:.1f}")
                if wandb_run is not None:
                    wandb_run.log({
                        "val/loss": val_loss,
                        "val/perplexity": val_ppl,
                    }, step=step)
                # Alert on val loss regression
                _check_val_loss_alert(step, val_loss, prev_val_loss, wandb_run)
                prev_val_loss = val_loss

        # Checkpoint
        if step % save_interval == 0 or step == total_steps:
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step:06d}.safetensors")
            save_checkpoint_fsdp(model, optimizer, step, loss_accum, ckpt_path, rank=rank)
            if rank == 0:
                print(f"  >>> saved checkpoint: {ckpt_path}")
                _fire_alert(
                    "Checkpoint saved",
                    f"Step {step}/{total_steps}, loss {loss_accum:.4f}, saved to {os.path.basename(ckpt_path)}",
                    "info", wandb_run,
                )

    total_time = time.time() - t0
    if rank == 0:
        final_ppl = math.exp(min(loss_accum, 20))
        avg_tps = tokens_processed / total_time
        print(f"\nTraining complete! {total_steps} steps in {total_time:.1f}s")
        print(f"Average throughput: {avg_tps:,.0f} tok/s")
        print(f"Final loss: {loss_accum:.4f} | Final perplexity: {final_ppl:.1f}")
        _fire_alert(
            "Training complete",
            f"{model_name}: {total_steps} steps in {total_time/3600:.1f}h, "
            f"final loss {loss_accum:.4f}, ppl {final_ppl:.1f}, {avg_tps:,.0f} tok/s",
            "info", wandb_run,
        )

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "final.safetensors")
    save_checkpoint_fsdp(model, optimizer, step, loss_accum, final_path, rank=rank)
    if rank == 0:
        print(f"Final checkpoint: {final_path}")


def _fire_alert(title, text, level, wandb_run):
    """Fire a trackio alert if trackio is the backend, otherwise just print."""
    try:
        import trackio
        level_map = {
            "info": trackio.AlertLevel.INFO,
            "warn": trackio.AlertLevel.WARN,
            "error": trackio.AlertLevel.ERROR,
        }
        trackio.alert(title=title, text=text, level=level_map.get(level, trackio.AlertLevel.WARN))
    except Exception:
        print(f"  [ALERT/{level.upper()}] {title}: {text}")


def _check_training_alerts(step, total_steps, loss, grad_norm, loss_history, warmup_steps, wandb_run, t0=None,
                           _alert_state={}):
    """Check for training anomalies and fire alerts.

    Comprehensive alerts tailored for CoFrGeNet-F scaling experiments:
    - ERROR: NaN/Inf, early divergence, loss explosion after dyadic depth change
    - WARN: loss spike, plateau, grad explosion, throughput collapse, loss rebound, ETA overrun
    - INFO: progress milestones with ETA
    """
    # Initialize per-run alert state (using mutable default to persist across calls)
    if "initialized" not in _alert_state or _alert_state.get("total_steps") != total_steps:
        _alert_state.clear()
        _alert_state["initialized"] = True
        _alert_state["total_steps"] = total_steps
        _alert_state["min_loss_seen"] = float("inf")
        _alert_state["loss_at_500"] = None
        _alert_state["first_loss"] = None
        _alert_state["throughput_baseline"] = None
        _alert_state["last_depth_change_step"] = None
        _alert_state["loss_before_depth_change"] = None
        _alert_state["rebound_alerted"] = False
        _alert_state["divergence_alerted"] = False
        _alert_state["throughput_alert_step"] = 0

    # ═══════════════════════════════════════════════════════════
    # ERROR: NaN/Inf loss — training is broken, stop immediately
    # ═══════════════════════════════════════════════════════════
    if math.isnan(loss) or math.isinf(loss):
        _fire_alert(
            "CRITICAL: NaN/Inf loss — stop this run",
            f"Loss is {loss} at step {step}/{total_steps}. Training is unrecoverable. "
            f"Kill this run and check: (1) learning rate too high, (2) data corruption, "
            f"(3) numerical instability in continued fractions.",
            "error", wandb_run,
        )
        return

    # Track loss history
    if _alert_state["first_loss"] is None:
        _alert_state["first_loss"] = loss
    loss_history.append(loss)
    if len(loss_history) > 200:
        loss_history.pop(0)
    _alert_state["min_loss_seen"] = min(_alert_state["min_loss_seen"], loss)

    # ═══════════════════════════════════════════════════════════
    # ERROR: Early divergence — loss hasn't dropped by step 500
    # Catches bad LR, broken data, misconfigured model before
    # wasting hours. Only fire once.
    # ═══════════════════════════════════════════════════════════
    if step == 500 and _alert_state["first_loss"] is not None:
        _alert_state["loss_at_500"] = loss
        improvement = (_alert_state["first_loss"] - loss) / _alert_state["first_loss"]
        if improvement < 0.05:  # less than 5% improvement in 500 steps
            _fire_alert(
                "CRITICAL: Early divergence — consider killing run",
                f"Loss only improved {improvement*100:.1f}% in first 500 steps "
                f"({_alert_state['first_loss']:.4f} -> {loss:.4f}). "
                f"Expected >20% drop by now. Check LR, data, or model config.",
                "error", wandb_run,
            )
            _alert_state["divergence_alerted"] = True

    # ═══════════════════════════════════════════════════════════
    # ERROR: Loss rebounding — loss consistently rising after
    # initial decrease. Run is likely doomed.
    # ═══════════════════════════════════════════════════════════
    if (step > warmup_steps * 2 and len(loss_history) >= 100
            and not _alert_state["rebound_alerted"]):
        recent_50 = sum(loss_history[-50:]) / 50
        older_50 = sum(loss_history[-100:-50]) / 50
        if recent_50 > older_50 * 1.15:  # 15% worse over 100 steps
            _fire_alert(
                "CRITICAL: Loss rebounding — training going backwards",
                f"Recent loss avg ({recent_50:.4f}) is {(recent_50/older_50 - 1)*100:.1f}% "
                f"higher than 50 steps ago ({older_50:.4f}) at step {step}. "
                f"Training is diverging. Consider stopping and reducing LR.",
                "error", wandb_run,
            )
            _alert_state["rebound_alerted"] = True

    # ═══════════════════════════════════════════════════════════
    # WARN: Loss spike — sudden 3x jump over recent average
    # ═══════════════════════════════════════════════════════════
    if step > warmup_steps and len(loss_history) > 50:
        recent_avg = sum(loss_history[-50:-1]) / 49
        if recent_avg > 0 and loss > recent_avg * 3:
            _fire_alert(
                "Loss spike detected",
                f"Loss {loss:.4f} is {loss/recent_avg:.1f}x the recent average "
                f"({recent_avg:.4f}) at step {step}. Single spikes are usually OK; "
                f"sustained increases mean trouble.",
                "warn", wandb_run,
            )

    # ═══════════════════════════════════════════════════════════
    # WARN: Loss plateau — truly stuck (no improvement over 500 steps)
    # Only fires in first 80% of training (cosine decay naturally slows late)
    # ═══════════════════════════════════════════════════════════
    if (step > warmup_steps * 3 and step < total_steps * 0.8
            and len(loss_history) >= 100):
        old_avg = sum(loss_history[:50]) / 50
        new_avg = sum(loss_history[-50:]) / 50
        # Loss must be INCREASING or truly flat (< 0.01% improvement)
        if old_avg > 0 and new_avg >= old_avg * 0.9999:
            plateau_key = f"plateau_{step // 2000}"  # at most once per 2000 steps
            if plateau_key not in _alert_state:
                _alert_state[plateau_key] = True
                _fire_alert(
                    "Loss plateau — model may be stuck",
                    f"Loss flat/increasing over last 100 steps: {old_avg:.4f} -> "
                    f"{new_avg:.4f} ({(old_avg-new_avg)/old_avg*100:.2f}%). "
                    f"If this persists, the learning rate may be too low.",
                    "warn", wandb_run,
                )

    # ═══════════════════════════════════════════════════════════
    # WARN: Gradient explosion — grad norm way above clip threshold
    # ═══════════════════════════════════════════════════════════
    if grad_norm > 10.0 and step > warmup_steps:
        _fire_alert(
            "Gradient explosion",
            f"Gradient norm {grad_norm:.2f} at step {step} (10x above clip threshold). "
            f"This can precede loss divergence. Monitor closely.",
            "warn", wandb_run,
        )

    # ═══════════════════════════════════════════════════════════
    # WARN: ETA exceeds expected training time
    # Our GPU cluster time is limited — alert if we're too slow.
    # ═══════════════════════════════════════════════════════════
    if t0 is not None and step > 0 and step % 1000 == 0:
        elapsed_h = (time.time() - t0) / 3600
        total_eta_h = elapsed_h * total_steps / step
        remaining_h = total_eta_h - elapsed_h
        # Alert if training will take >7 days (168h) — beyond reasonable cluster time
        if total_eta_h > 168:
            _fire_alert(
                "ETA exceeds cluster time budget",
                f"At current pace, training will take {total_eta_h:.1f}h total "
                f"({remaining_h:.1f}h remaining). Step {step}/{total_steps}, "
                f"{elapsed_h:.1f}h elapsed. Consider reducing total_steps or "
                f"increasing batch size to speed up.",
                "warn", wandb_run,
            )

    # ═══════════════════════════════════════════════════════════
    # INFO: Progress milestones with ETA (every 10%)
    # More granular than 25% for multi-day runs
    # ═══════════════════════════════════════════════════════════
    milestones = {int(total_steps * p): f"{int(p*100)}%"
                  for p in [0.10, 0.25, 0.50, 0.75, 0.90]}
    if step in milestones:
        elapsed_h = (time.time() - t0) / 3600 if t0 else 0
        remaining_h = elapsed_h * (total_steps - step) / step if step > 0 else 0
        ppl = math.exp(min(loss, 20))
        _fire_alert(
            f"Training {milestones[step]} complete",
            f"Step {step}/{total_steps}, loss {loss:.4f}, ppl {ppl:.1f}, "
            f"elapsed {elapsed_h:.1f}h, est. remaining {remaining_h:.1f}h",
            "info", wandb_run,
        )


def _check_val_loss_alert(step, val_loss, prev_val_loss, wandb_run):
    """Alert on validation loss regression."""
    if prev_val_loss is not None:
        if val_loss > prev_val_loss * 1.1:
            _fire_alert(
                "Validation loss regression",
                f"Val loss increased from {prev_val_loss:.4f} to {val_loss:.4f} "
                f"(+{(val_loss-prev_val_loss)/prev_val_loss*100:.1f}%) at step {step}. "
                f"If this persists across multiple evals, the model may be overfitting.",
                "warn", wandb_run,
            )
        # Also alert on first good val loss
        if prev_val_loss is not None and val_loss < prev_val_loss * 0.95:
            ppl = math.exp(min(val_loss, 20))
            _fire_alert(
                "Validation improving",
                f"Val loss improved from {prev_val_loss:.4f} to {val_loss:.4f} "
                f"(-{(1 - val_loss/prev_val_loss)*100:.1f}%), ppl {ppl:.1f} at step {step}",
                "info", wandb_run,
            )


def load_experiment_config(config_path, args, parser=None):
    """Load experiment config from YAML. CLI args take precedence over YAML values.
    Uses parser defaults to detect which args were explicitly set on the CLI.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Determine parser defaults so we can distinguish "user passed --total_steps 500"
    # from "total_steps has its argparse default of 19073"
    defaults = vars(parser.parse_args([])) if parser else {}
    for key, value in cfg.items():
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        # Override if: no parser (legacy), value is None, or value matches the default
        if parser is None:
            if current is None:
                setattr(args, key, value)
        else:
            if current is None or current == defaults.get(key):
                setattr(args, key, value)
    return args


def add_training_args(parser):
    """Add common training arguments to an argparse parser."""
    parser.add_argument("--data_dir", type=str, default="data/tokenized")
    parser.add_argument("--total_steps", type=int, default=19073)
    parser.add_argument("--warmup_steps", type=int, default=700)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--batch_tokens", type=int, default=524288,
                        help="Total tokens per gradient update")
    parser.add_argument("--micro_batch_size", type=int, default=16,
                        help="Micro batch size (sequences per forward pass)")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--wandb_project", type=str, default="cofrgenet-f")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--trackio_space", type=str, default="cahlen/cofrgenet-f-trackio",
                        help="HuggingFace Space ID for trackio dashboard")
    parser.add_argument("--trackio_group", type=str, default=None,
                        help="Group name for trackio (e.g., 'pair1', 'pair3') — groups baseline+cofrgenet together")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in checkpoint_dir")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment YAML config file")
    return parser
