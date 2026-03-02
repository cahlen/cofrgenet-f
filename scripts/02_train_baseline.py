"""Train baseline GPT-2 Small transformer on FineWeb-Edu.

Usage:
    python scripts/02_train_baseline.py
    python scripts/02_train_baseline.py --max_steps 10 --eval_interval 5  # smoke test
"""

import argparse
import torch
import numpy as np

from src.baseline.config import BaselineConfig
from src.baseline.model import BaselineTransformer
from scripts.train_common import (
    ShardedDataLoader, configure_optimizer, train_loop, add_training_args
)


def main():
    parser = argparse.ArgumentParser(description="Train baseline transformer")
    add_training_args(parser)
    # Allow overriding total_steps via --max_steps alias
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override total_steps (alias for smoke testing)")
    args = parser.parse_args()

    if args.max_steps is not None:
        args.total_steps = args.max_steps

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Model
    config = BaselineConfig(block_size=args.block_size)
    model = BaselineTransformer(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Baseline Transformer: {total_params:,} parameters")

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Data
    grad_accum_steps = args.batch_tokens // (args.micro_batch_size * args.block_size)
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    train_loader = ShardedDataLoader(
        args.data_dir, "train", args.block_size, args.micro_batch_size, device
    )
    val_loader = ShardedDataLoader(
        args.data_dir, "val", args.block_size, args.micro_batch_size, device
    )

    # Optimizer
    optimizer = configure_optimizer(
        model, args.weight_decay, args.lr, (args.beta1, args.beta2), device
    )

    # Wandb
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or "baseline-125m",
                config=vars(args),
            )
        except Exception as e:
            print(f"wandb init failed: {e}, continuing without logging")

    # Train
    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        max_lr=args.lr,
        grad_accum_steps=grad_accum_steps,
        grad_clip=args.grad_clip,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        checkpoint_dir="checkpoints/baseline",
        model_name="baseline",
        device=device,
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
