"""Train CoFrGeNet-F transformer on FineWeb-Edu with dyadic schedule.

Usage:
    python scripts/03_train_cofrgenet.py
    python scripts/03_train_cofrgenet.py --max_steps 10 --eval_interval 5  # smoke test
    python scripts/03_train_cofrgenet.py --n_embd 1024 --n_head 16 --checkpoint_dir checkpoints/cofrgenet-128m
    torchrun --nproc_per_node=2 scripts/03_train_cofrgenet.py  # multi-GPU FSDP
"""

import argparse
import torch
import numpy as np

from src.cofrgenet.config import CoFrGeNetConfig
from src.cofrgenet.model import CoFrGeNetTransformer, get_unfrozen_depth
from scripts.train_common import (
    ShardedDataLoader, configure_optimizer, train_loop, add_training_args,
    setup_distributed, cleanup_distributed, wrap_model_fsdp, is_distributed,
    load_checkpoint_fsdp, load_experiment_config
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

    if args.config:
        args = load_experiment_config(args.config, args)

    if args.max_steps is not None:
        args.total_steps = args.max_steps

    # Distributed setup
    rank, local_rank, world_size = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Per-rank seeding for data diversity
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed)

    if rank == 0:
        print(f"Device: {device}, World size: {world_size}")

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
    if rank == 0:
        print(f"CoFrGeNet-F Transformer: {total_params:,} parameters")
        print(f"Cffn config: L={config.num_ladders} ladders, d={config.cf_depth} depth")

    # FSDP wrapping (before torch.compile and optimizer)
    use_grad_ckpt = total_params > 5_000_000_000
    if world_size > 1:
        model = wrap_model_fsdp(model, device, use_gradient_checkpointing=use_grad_ckpt)

    # Data
    grad_accum_steps = args.batch_tokens // (args.micro_batch_size * args.block_size)
    if rank == 0:
        print(f"Gradient accumulation steps: {grad_accum_steps}")

    train_loader = ShardedDataLoader(
        args.data_dir, "train", args.block_size, args.micro_batch_size, device,
        rank=rank, world_size=world_size
    )
    val_loader = ShardedDataLoader(
        args.data_dir, "val", args.block_size, args.micro_batch_size, device,
        rank=rank, world_size=world_size
    )

    # Optimizer
    optimizer = configure_optimizer(
        model, args.weight_decay, args.lr, (args.beta1, args.beta2), device
    )

    # Resume from checkpoint if requested (must happen before torch.compile)
    resume_step = 0
    if args.resume:
        resume_step = load_checkpoint_fsdp(model, optimizer, args.checkpoint_dir, device)

    if args.compile:
        if rank == 0:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Dyadic schedule callback
    max_depth = config.cf_depth
    last_depth = [-1]  # mutable for closure

    def dyadic_callback(step, total_steps):
        depth = get_unfrozen_depth(step, total_steps, max_depth)
        if depth != last_depth[0]:
            if rank == 0:
                print(f"  >>> Dyadic schedule: unfreezing depth {depth} at step {step}")
            unwrapped = model.module if hasattr(model, 'module') else model
            unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)
            unwrapped.set_active_depth(depth)
            last_depth[0] = depth

    def grad_zero_callback():
        unwrapped = model.module if hasattr(model, 'module') else model
        unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)
        unwrapped.zero_frozen_grads()

    # Wandb — only on rank 0
    wandb_run = None
    if rank == 0 and not args.no_wandb:
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
        resume_step=resume_step,
        rank=rank,
        world_size=world_size,
        grad_zero_callback=grad_zero_callback,
    )

    if wandb_run is not None:
        wandb_run.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
