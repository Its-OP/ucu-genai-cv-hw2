"""
DDPM Training Script for MNIST.

Usage:
    python -m models.train --epochs 100 --lr 2e-4 --beta_schedule cosine
"""
import argparse
import time
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import get_train_loader, get_test_loader
from models.unet import UNet
from models.ddpm import DDPM
from models.utils import (
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
    parser = argparse.ArgumentParser(description='Train DDPM on MNIST')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides data module default)')
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'])
    parser.add_argument('--sample_every', type=int, default=10)
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channel count (use 64-128 for powerful GPUs)')
    return parser.parse_args()


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        # Enable TensorFloat32 for faster matmul on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_epoch(model, ddpm, optimizer, loader, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    use_amp = scaler is not None

    for x, _ in tqdm(loader, desc='Training', leave=False):
        x = x.to(device)
        t = torch.randint(0, ddpm.timesteps, (x.shape[0],), device=device)

        optimizer.zero_grad()

        if use_amp:
            # Mixed precision training for CUDA
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = ddpm.p_losses(model, x, t)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = ddpm.p_losses(model, x, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, ddpm, loader, device, use_amp=False):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0

    for x, _ in loader:
        x = x.to(device)
        t = torch.randint(0, ddpm.timesteps, (x.shape[0],), device=device)

        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = ddpm.p_losses(model, x, t)
        else:
            loss = ddpm.p_losses(model, x, t)

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    # Create data loaders (with optional batch size override)
    train_loader = get_train_loader(args.batch_size)
    test_loader = get_test_loader(args.batch_size)
    print(f"Batch size: {train_loader.batch_size}")

    # Setup experiment folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = setup_experiment_folder(f'./experiments/{timestamp}-ddpm')
    print(f"Experiment directory: {exp_dir}")

    # Log configuration
    config = {
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'timesteps': args.timesteps,
        'beta_schedule': args.beta_schedule,
        'sample_every': args.sample_every,
        'base_channels': args.base_channels,
        'device': str(device),
        'batch_size': train_loader.batch_size,
    }
    log_config(exp_dir, config)

    # Initialize model (new simplified UNet based on reference implementation)
    model = UNet(
        image_channels=1,
        base_channels=args.base_channels,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # CUDA optimizations: compile model and enable mixed precision
    scaler = None
    if device.type == 'cuda':
        model = torch.compile(model)
        scaler = torch.amp.GradScaler('cuda')
        print("CUDA optimizations enabled: torch.compile() + mixed precision (float16)")

    # Initialize DDPM
    ddpm = DDPM(timesteps=args.timesteps, beta_schedule=args.beta_schedule).to(device)

    # Optimizer and scheduler
    # AdamW with weight decay for better generalization
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine annealing: lr decays from initial to 0 over T_max epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    train_losses, eval_losses = [], []
    start_time = time.time()
    use_amp = device.type == 'cuda'

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss = train_epoch(model, ddpm, optimizer, train_loader, device, scaler)
        eval_loss = evaluate(model, ddpm, test_loader, device, use_amp)

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        log_epoch(exp_dir, epoch, train_loss, eval_loss, epoch_time)

        print(f"Epoch {epoch+1}/{args.epochs} | Train: {train_loss:.6f} | Eval: {eval_loss:.6f} | Time: {epoch_time:.1f}s")

        # Generate samples periodically
        if (epoch + 1) % args.sample_every == 0:
            model.eval()
            samples = ddpm.p_sample_loop(model, (10, 1, 28, 28))
            save_images(samples, f'{exp_dir}/epoch_samples/epoch_{epoch+1:03d}.png')
            model.train()

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")

    # Generate final outputs
    print("Generating final samples...")
    model.eval()

    # 10 final samples
    final_samples = ddpm.p_sample_loop(model, (10, 1, 28, 28))
    save_images(final_samples, f'{exp_dir}/final_samples/final_grid.png')

    # Save individual final samples
    for i in range(10):
        img = (final_samples[i, 0] + 1) / 2
        img = img.clamp(0, 1).cpu().numpy()
        import matplotlib.pyplot as plt
        plt.figure(figsize=(3, 3))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig(f'{exp_dir}/final_samples/sample_{i}.png', bbox_inches='tight', dpi=100)
        plt.close()

    # Denoising progression
    print("Generating denoising progression...")
    inference_start = time.time()
    samples_with_intermediates, intermediates = ddpm.p_sample_loop(
        model, (10, 1, 28, 28), return_intermediates=True
    )
    inference_time = time.time() - inference_start

    save_denoising_progression(intermediates, f'{exp_dir}/denoising_steps/progression.png')
    save_individual_denoising_steps(intermediates, f'{exp_dir}/denoising_steps')

    # Save loss curves
    plot_loss_curves(train_losses, eval_losses, f'{exp_dir}/loss_curves.png')

    # Save performance metrics
    save_performance_metrics(
        exp_dir,
        total_time,
        args.epochs,
        inference_time,
        train_losses[-1],
        eval_losses[-1],
    )

    print(f"All outputs saved to {exp_dir}")


if __name__ == '__main__':
    main()
