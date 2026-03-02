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
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file


class ShardedDataLoader:
    """Loads tokenized binary shards with random sampling."""

    def __init__(self, data_dir, split, block_size, batch_size, device="cpu"):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        pattern = f"{split}_"
        self.shards = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.startswith(pattern) and f.endswith(".bin")
        ])
        assert len(self.shards) > 0, f"No {split} shards found in {data_dir}"

        self.current_shard_idx = 0
        self.current_data = None
        self._load_shard(0)

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
        """Move to the next shard (call after each epoch-ish)."""
        self._load_shard(self.current_shard_idx + 1)


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
    save_file(model.state_dict(), path)
    # Save optimizer state separately
    opt_path = path.replace(".safetensors", "_optim.pt")
    torch.save({"optimizer": optimizer.state_dict(), "step": step, "loss": loss}, opt_path)


def estimate_loss(model, data_loader, num_batches=20):
    """Estimate loss over num_batches."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = data_loader.next_batch()
            with torch.autocast("cuda", dtype=torch.bfloat16):
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
):
    """Main training loop shared by both models.

    Args:
        step_callback: Optional fn(step, total_steps) called each step,
                       e.g., for dyadic schedule updates.
        wandb_run: Optional wandb run for logging.
    """
    model.train()

    step = 0
    t0 = time.time()
    tokens_processed = 0
    batch_tokens = train_loader.batch_size * train_loader.block_size

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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.item()
            loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            grad_norm = 0.0

        optimizer.step()

        tokens_processed += batch_tokens * grad_accum_steps
        step += 1

        # Advance shard periodically
        if step % 100 == 0:
            train_loader.advance_shard()

        # Logging
        if step % 10 == 0 or step == 1:
            dt = time.time() - t0
            tokens_per_sec = tokens_processed / dt if dt > 0 else 0
            print(
                f"step {step:>6d}/{total_steps} | "
                f"loss {loss_accum:.4f} | "
                f"lr {lr:.2e} | "
                f"grad_norm {grad_norm:.2f} | "
                f"tok/s {tokens_per_sec:,.0f}"
            )

            if wandb_run is not None:
                wandb_run.log({
                    "train/loss": loss_accum,
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/tokens": tokens_processed,
                }, step=step)

        # Log val loss
        if step % eval_interval == 0 or step == total_steps:
            val_loss = estimate_loss(model, val_loader)
            print(f"  >>> val_loss: {val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"val/loss": val_loss}, step=step)

        # Checkpoint
        if step % save_interval == 0 or step == total_steps:
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step:06d}.safetensors")
            save_checkpoint(model, optimizer, step, loss_accum, ckpt_path)
            print(f"  >>> saved checkpoint: {ckpt_path}")

    total_time = time.time() - t0
    print(f"\nTraining complete! {total_steps} steps in {total_time:.1f}s")
    print(f"Average throughput: {tokens_processed / total_time:,.0f} tok/s")

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "final.safetensors")
    save_checkpoint(model, optimizer, step, loss_accum, final_path)
    print(f"Final checkpoint: {final_path}")


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
    return parser
