"""
Rectified Flow Training Script for MNIST (pixel space).
"""
import argparse
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import get_train_loader, get_test_loader
from models.unet import UNet
from models.rectified_flow import RectifiedFlow
from models.utils import (
    get_device,
    ExponentialMovingAverage,
    save_checkpoint,
    setup_experiment_folder,
    save_images,
    plot_loss_curves,
    save_denoising_progression,
    save_individual_denoising_steps,
    log_config,
    log_epoch,
    save_performance_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Rectified Flow on MNIST (pixel space)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides data module default)')
    parser.add_argument('--sampling_steps', type=int, default=50,
                        help='Number of Euler steps for sampling during training (default: 50)')
    parser.add_argument('--sample_every', type=int, default=10)
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base channel count (default: 32)')
    parser.add_argument('--channel_multipliers', type=int, nargs='+', default=[1, 2, 3, 3],
                        help='Per-level channel multipliers (default: 1 2 3 3)')
    parser.add_argument('--layers_per_block', type=int, default=1,
                        help='ResNet blocks per resolution level (default: 1)')
    parser.add_argument('--attention_levels', type=int, nargs='+', default=[0, 0, 0, 1],
                        help='Attention flags per level, 0 or 1 (default: 0 0 0 1)')
    return parser.parse_args()


def train_epoch(model, rectified_flow, optimizer, loader, device, scaler=None, ema=None):
    """Train for one epoch using Rectified Flow velocity loss.

    Rectified Flow training loss (Liu et al. 2022):
        L = E_{t, x_0, epsilon}[|| v_theta(x_t, t) - v ||^2]
    where v = epsilon - x_0 and x_t = (1-t)*x_0 + t*epsilon.
    """
    model.train()
    total_loss = 0
    use_amp = scaler is not None

    for images, _ in tqdm(loader, desc='Training', leave=False):
        images = images.to(device)

        # Sample continuous time uniformly from (epsilon, 1 - epsilon)
        # to avoid degenerate endpoints t=0 (pure data) and t=1 (pure noise)
        continuous_time = torch.rand(images.shape[0], device=device) * (1.0 - 1e-5) + 1e-5

        optimizer.zero_grad()

        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = rectified_flow.velocity_loss(model, images, continuous_time)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = rectified_flow.velocity_loss(model, images, continuous_time)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, rectified_flow, loader, device, use_amp=False):
    """Evaluate on test set using Rectified Flow velocity loss."""
    model.eval()
    total_loss = 0

    for images, _ in loader:
        images = images.to(device)
        continuous_time = torch.rand(images.shape[0], device=device) * (1.0 - 1e-5) + 1e-5

        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = rectified_flow.velocity_loss(model, images, continuous_time)
        else:
            loss = rectified_flow.velocity_loss(model, images, continuous_time)

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    args = parse_args()
    device = get_device()
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    print(f"Using device: {device}")

    # Create data loaders
    train_loader = get_train_loader(args.batch_size)
    test_loader = get_test_loader(args.batch_size)
    print(f"Batch size: {train_loader.batch_size}")

    # Setup experiment folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = setup_experiment_folder(
        f'./experiments/{timestamp}-rf',
        extra_subfolders=['denoising_steps'],
    )
    print(f"Experiment directory: {exp_dir}")

    # Parse architecture arguments
    channel_multipliers = tuple(args.channel_multipliers)
    attention_levels = tuple(bool(flag) for flag in args.attention_levels)

    # Log configuration
    config = {
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'flow_type': 'rectified_flow',
        'number_of_sampling_steps': args.sampling_steps,
        'sample_every': args.sample_every,
        'base_channels': args.base_channels,
        'channel_multipliers': list(channel_multipliers),
        'layers_per_block': args.layers_per_block,
        'attention_levels': list(attention_levels),
        'device': str(device),
        'batch_size': train_loader.batch_size,
    }
    log_config(exp_dir, config)

    # Initialize UNet (same architecture as DDPM — only loss/sampling differ)
    model = UNet(
        image_channels=1,
        base_channels=args.base_channels,
        channel_multipliers=channel_multipliers,
        layers_per_block=args.layers_per_block,
        attention_levels=attention_levels,
    ).to(device)

    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # CUDA optimizations
    scaler = None
    if device.type == 'cuda':
        model = torch.compile(model)
        scaler = torch.amp.GradScaler('cuda')
        print("CUDA optimizations enabled: torch.compile() + mixed precision (float16)")

    # Initialize Rectified Flow
    rectified_flow = RectifiedFlow(
        number_of_sampling_steps=args.sampling_steps,
    ).to(device)

    # Model config dict for checkpoint reconstruction at generation time
    model_config = {
        'image_channels': 1,
        'base_channels': args.base_channels,
        'channel_multipliers': list(channel_multipliers),
        'layers_per_block': args.layers_per_block,
        'attention_levels': list(attention_levels),
        'flow_type': 'rectified_flow',
        'number_of_sampling_steps': args.sampling_steps,
    }

    # Initialize EMA
    ema = ExponentialMovingAverage(model, decay=0.995)
    print("EMA enabled (decay=0.995)")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    train_losses, eval_losses = [], []
    start_time = time.time()
    use_amp = device.type == 'cuda'

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss = train_epoch(model, rectified_flow, optimizer, train_loader, device, scaler, ema)

        # Use EMA weights for evaluation
        ema.apply_shadow(model)
        eval_loss = evaluate(model, rectified_flow, test_loader, device, use_amp)
        ema.restore(model)

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        log_epoch(exp_dir, epoch, train_loss, eval_loss, epoch_time)

        print(f"Epoch {epoch+1}/{args.epochs} | Train: {train_loss:.6f} | "
              f"Eval: {eval_loss:.6f} | Time: {epoch_time:.1f}s")

        # Generate samples and save checkpoint periodically using EMA weights
        if (epoch + 1) % args.sample_every == 0:
            ema.apply_shadow(model)
            model.eval()
            samples = rectified_flow.euler_sample_loop(model, (10, 1, 28, 28))
            save_images(samples, f'{exp_dir}/epoch_samples/epoch_{epoch+1:03d}.pdf')

            checkpoint_path = f'{exp_dir}/checkpoints/checkpoint_epoch_{epoch+1:03d}.pt'
            save_checkpoint(model, checkpoint_path, model_config, epoch, train_loss, eval_loss)
            print(f"  Checkpoint saved: {checkpoint_path}")

            model.train()
            ema.restore(model)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")

    # Generate final outputs using EMA weights
    print("Generating final samples...")
    ema.apply_shadow(model)
    model.eval()

    # Save final checkpoint
    final_checkpoint_path = f'{exp_dir}/checkpoints/checkpoint_final.pt'
    save_checkpoint(
        model, final_checkpoint_path, model_config,
        epoch=args.epochs - 1,
        train_loss=train_losses[-1],
        eval_loss=eval_losses[-1],
    )
    print(f"Final checkpoint saved: {final_checkpoint_path}")

    # 10 final samples
    final_samples = rectified_flow.euler_sample_loop(model, (10, 1, 28, 28))
    save_images(final_samples, f'{exp_dir}/final_samples/final_grid.pdf')

    # Save individual final samples
    for sample_index in range(10):
        image = (final_samples[sample_index, 0] + 1) / 2
        image = image.clamp(0, 1).cpu().numpy()
        plt.figure(figsize=(3, 3))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.savefig(f'{exp_dir}/final_samples/sample_{sample_index}.pdf',
                    bbox_inches='tight', dpi=100)
        plt.close()

    # Denoising progression
    print("Generating denoising progression...")
    inference_start = time.time()
    samples_with_intermediates, intermediates = rectified_flow.euler_sample_loop(
        model, (10, 1, 28, 28), return_intermediates=True,
    )
    inference_time = time.time() - inference_start

    save_denoising_progression(intermediates, f'{exp_dir}/denoising_steps/progression.pdf')
    save_individual_denoising_steps(intermediates, f'{exp_dir}/denoising_steps')

    # Save loss curves
    plot_loss_curves(train_losses, eval_losses, f'{exp_dir}/loss_curves.pdf')

    # Save performance metrics
    save_performance_metrics(
        exp_dir, total_time, args.epochs, inference_time,
        train_losses[-1], eval_losses[-1],
    )

    print(f"All outputs saved to {exp_dir}")


if __name__ == '__main__':
    main()
