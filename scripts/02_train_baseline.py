"""Train baseline GPT-2 Small transformer on FineWeb-Edu.

Usage:
    python scripts/02_train_baseline.py
    python scripts/02_train_baseline.py --max_steps 10 --eval_interval 5  # smoke test
    torchrun --nproc_per_node=2 scripts/02_train_baseline.py  # multi-GPU FSDP
"""

import argparse
import torch
import numpy as np

from src.baseline.config import BaselineConfig
from src.baseline.model import BaselineTransformer
from scripts.train_common import (
    ShardedDataLoader, configure_optimizer, train_loop, add_training_args,
    setup_distributed, cleanup_distributed, wrap_model_fsdp, is_distributed,
    load_checkpoint_fsdp, load_experiment_config, setup_torch_performance
)


def main():
    parser = argparse.ArgumentParser(description="Train baseline transformer")
    add_training_args(parser)
    # Allow overriding total_steps via --max_steps alias
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override total_steps (alias for smoke testing)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/baseline")
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    args = parser.parse_args()

    if args.config:
        args = load_experiment_config(args.config, args, parser=parser)

    if args.max_steps is not None:
        args.total_steps = args.max_steps

    # Performance + distributed setup
    setup_torch_performance()
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
    config = BaselineConfig(**config_kwargs)
    model = BaselineTransformer(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Baseline Transformer: {total_params:,} parameters")

    # FSDP wrapping (before torch.compile and optimizer)
    use_grad_ckpt = total_params > 5_000_000_000
    if world_size > 1:
        model = wrap_model_fsdp(model, device, use_gradient_checkpointing=use_grad_ckpt)

    if args.compile:
        if rank == 0:
            print("Compiling model with torch.compile(mode='max-autotune')...")
        model = torch.compile(model, mode="max-autotune")

    # Data — account for world_size: each GPU processes micro_batch_size independently
    grad_accum_steps = max(1, args.batch_tokens // (args.micro_batch_size * args.block_size * world_size))
    if rank == 0:
        effective_batch = args.micro_batch_size * args.block_size * world_size * grad_accum_steps
        print(f"Gradient accumulation steps: {grad_accum_steps} (effective batch: {effective_batch:,} tokens)")

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

    # Resume from checkpoint if requested
    resume_step = 0
    if args.resume:
        resume_step = load_checkpoint_fsdp(model, optimizer, args.checkpoint_dir, device)

    # Tracking — trackio (HF Spaces dashboard)
    wandb_run = None
    if rank == 0 and not args.no_wandb:
        try:
            import trackio
            init_kwargs = dict(
                project=args.wandb_project,
                name=args.wandb_run_name or f"baseline-{total_params // 1_000_000}m",
                config=vars(args),
                space_id=args.trackio_space,
                auto_log_gpu=True,
                gpu_log_interval=30.0,
            )
            if args.trackio_group:
                init_kwargs["group"] = args.trackio_group
            trackio.init(**init_kwargs)
            wandb_run = trackio  # trackio has wandb-compatible .log() API
            print(f"Trackio initialized → {args.trackio_space or 'local'}")
        except Exception as e:
            print(f"Trackio init failed: {e}, continuing without logging")

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
        model_name="baseline",
        device=device,
        wandb_run=wandb_run,
        resume_step=resume_step,
        rank=rank,
        world_size=world_size,
        trackio_space=args.trackio_space if not args.no_wandb else None,
    )

    try:
        import trackio
        trackio.finish()
    except Exception:
        pass

    cleanup_distributed()


if __name__ == "__main__":
    main()
