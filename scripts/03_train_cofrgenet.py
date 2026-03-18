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
    load_checkpoint_fsdp, load_experiment_config, setup_torch_performance
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
    if args.num_ladders is not None:
        config_kwargs["num_ladders"] = args.num_ladders
    if args.cf_depth is not None:
        config_kwargs["cf_depth"] = args.cf_depth
    config = CoFrGeNetConfig(**config_kwargs)
    model = CoFrGeNetTransformer(config)
    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"CoFrGeNet-F Transformer: {total_params:,} parameters")
        print(f"Cffn config: L={config.num_ladders} ladders, d={config.cf_depth} depth")

    # Distributed wrapping (before torch.compile and optimizer)
    # For models that fit on one GPU (<2B), uses DDP with .to(device) first
    # For larger models, uses FSDP which handles device placement
    use_grad_ckpt = total_params > 2_000_000_000
    if world_size > 1:
        if total_params < 2_000_000_000:
            model = model.to(device)
        model = wrap_model_fsdp(model, device, use_gradient_checkpointing=use_grad_ckpt)
    else:
        model = model.to(device)

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

    # Resume from checkpoint if requested (must happen before torch.compile)
    resume_step = 0
    if args.resume:
        resume_step = load_checkpoint_fsdp(model, optimizer, args.checkpoint_dir, device)

    if args.compile:
        if rank == 0:
            print("Compiling model with torch.compile(mode='max-autotune')...")
        model = torch.compile(model, mode="max-autotune")

    # Dyadic schedule callback — with loss monitoring around depth changes.
    # Unfreezing a new depth can destabilize training; we track loss before/after
    # to alert if a depth change causes divergence.
    max_depth = config.cf_depth
    last_depth = [-1]  # mutable for closure
    depth_change_state = {"step": None, "loss_before": None}  # for post-change monitoring

    def dyadic_callback(step, total_steps):
        depth = get_unfrozen_depth(step, total_steps, max_depth)
        if depth != last_depth[0]:
            if rank == 0:
                print(f"  >>> Dyadic schedule: unfreezing depth {depth} at step {step}")
                try:
                    import trackio
                    trackio.alert(
                        title=f"Dyadic depth change: depth {last_depth[0]} -> {depth}",
                        text=f"Unfreezing continued fraction depth {depth}/{max_depth} at step {step}/{total_steps}. "
                             f"Watch for loss spikes in the next ~100 steps.",
                        level=trackio.AlertLevel.INFO,
                    )
                except Exception:
                    pass
                # Record state for post-change monitoring
                depth_change_state["step"] = step
                depth_change_state["loss_before"] = None  # will be set from train loop
            unwrapped = model.module if hasattr(model, 'module') else model
            unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)
            unwrapped.set_active_depth(depth)
            last_depth[0] = depth

    def check_depth_change_stability(step, current_loss):
        """Called from train loop to monitor loss after dyadic depth changes."""
        if depth_change_state["step"] is None or rank != 0:
            return
        # Record loss at time of depth change
        if depth_change_state["loss_before"] is None:
            depth_change_state["loss_before"] = current_loss
            return
        # Check 50 steps after depth change
        steps_since = step - depth_change_state["step"]
        if steps_since == 50:
            loss_before = depth_change_state["loss_before"]
            if current_loss > loss_before * 1.5:  # 50% worse
                try:
                    import trackio
                    trackio.alert(
                        title="Loss destabilized after depth change",
                        text=f"Loss jumped from {loss_before:.4f} to {current_loss:.4f} "
                             f"({(current_loss/loss_before - 1)*100:.0f}% increase) "
                             f"within 50 steps of unfreezing depth {last_depth[0]}. "
                             f"This may recover, but if it persists, the depth change caused divergence.",
                        level=trackio.AlertLevel.WARN,
                    )
                except Exception:
                    pass
            depth_change_state["step"] = None  # reset

    def grad_zero_callback():
        unwrapped = model.module if hasattr(model, 'module') else model
        unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)
        unwrapped.zero_frozen_grads()

    # Tracking — trackio (HF Spaces dashboard)
    wandb_run = None
    if rank == 0 and not args.no_wandb:
        try:
            import trackio
            init_kwargs = dict(
                project=args.wandb_project,
                name=args.wandb_run_name or f"cofrgenet-f-{total_params // 1_000_000}m",
                config=vars(args),
                space_id=args.trackio_space,
                auto_log_gpu=True,
                gpu_log_interval=30.0,
                resume="allow" if resume_step > 0 else "never",
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
        model_name="cofrgenet-f",
        device=device,
        step_callback=dyadic_callback,
        wandb_run=wandb_run,
        resume_step=resume_step,
        rank=rank,
        world_size=world_size,
        grad_zero_callback=grad_zero_callback,
        trackio_space=args.trackio_space if not args.no_wandb else None,
        loss_callback=check_depth_change_stability,
    )

    try:
        import trackio
        trackio.finish()
    except Exception:
        pass

    cleanup_distributed()


if __name__ == "__main__":
    main()
