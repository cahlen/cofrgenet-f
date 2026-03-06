"""Train CoFrGeNet-F transformer on FineWeb-Edu with dyadic schedule.

Usage:
    python scripts/03_train_cofrgenet.py
    python scripts/03_train_cofrgenet.py --max_steps 10 --eval_interval 5  # smoke test
    python scripts/03_train_cofrgenet.py --n_embd 1024 --n_head 16 --checkpoint_dir checkpoints/cofrgenet-128m
"""

import argparse
import torch
import numpy as np

from src.cofrgenet.config import CoFrGeNetConfig
from src.cofrgenet.model import CoFrGeNetTransformer, get_unfrozen_depth
from scripts.train_common import (
    ShardedDataLoader, configure_optimizer, train_loop, add_training_args
)


def main():
    parser = argparse.ArgumentParser(description="Train CoFrGeNet-F transformer")
    add_training_args(parser)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override total_steps (alias for smoke testing)")
    # Model dimension overrides
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--num_ladders", type=int, default=None)
    parser.add_argument("--cf_depth", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/cofrgenet")
    args = parser.parse_args()

    if args.max_steps is not None:
        args.total_steps = args.max_steps

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Model — apply any dimension overrides
    config_kwargs = dict(block_size=args.block_size)
    if args.n_layer is not None:
        config_kwargs["n_layer"] = args.n_layer
    if args.n_head is not None:
        config_kwargs["n_head"] = args.n_head
    if args.n_embd is not None:
        config_kwargs["n_embd"] = args.n_embd
    if args.num_ladders is not None:
        config_kwargs["num_ladders"] = args.num_ladders
    if args.cf_depth is not None:
        config_kwargs["cf_depth"] = args.cf_depth
    config = CoFrGeNetConfig(**config_kwargs)
    model = CoFrGeNetTransformer(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"CoFrGeNet-F Transformer: {total_params:,} parameters")
    print(f"Cffn config: L={config.num_ladders} ladders, d={config.cf_depth} depth")

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

    # Dyadic schedule callback
    max_depth = config.cf_depth
    last_depth = [-1]  # mutable for closure

    def dyadic_callback(step, total_steps):
        depth = get_unfrozen_depth(step, total_steps, max_depth)
        if depth != last_depth[0]:
            print(f"  >>> Dyadic schedule: unfreezing depth {depth} at step {step}")
            model.set_active_depth(depth)
            last_depth[0] = depth

    # Wandb
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"cofrgenet-f-{total_params // 1_000_000}m",
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
        checkpoint_dir=args.checkpoint_dir,
        model_name="cofrgenet-f",
        device=device,
        step_callback=dyadic_callback,
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
