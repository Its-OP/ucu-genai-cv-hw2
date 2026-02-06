"""
DDPM Training Script for MNIST.

Usage:
    python -m models.train --epochs 100 --lr 1e-3
"""
import argparse
import copy
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


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) of model parameters.

    Maintains shadow copies of model parameters that are updated as:
        shadow_param = decay * shadow_param + (1 - decay) * param

    Used in the original DDPM paper (Ho et al. 2020) to produce smoother,
    higher-quality samples during inference.

    Args:
        model: The model whose parameters will be tracked.
        decay (float): EMA decay rate. Higher values give smoother averaging.
                       Default: 0.995 (standard for small DDPM models).
    """

    def __init__(self, model, decay=0.995):
        self.decay = decay
        # Deep copy all parameters as shadow weights
        self.shadow_parameters = [parameter.clone().detach() for parameter in model.parameters()]

    def update(self, model):
        """Update shadow parameters with current model parameters.

        Formula: shadow = decay · shadow + (1 - decay) · param
        """
        for shadow_parameter, model_parameter in zip(self.shadow_parameters, model.parameters()):
            shadow_parameter.data.mul_(self.decay).add_(
                model_parameter.data, alpha=1.0 - self.decay
            )

    def apply_shadow(self, model):
        """Replace model parameters with shadow (EMA) parameters for inference."""
        self.backup_parameters = [parameter.clone() for parameter in model.parameters()]
        for model_parameter, shadow_parameter in zip(model.parameters(), self.shadow_parameters):
            model_parameter.data.copy_(shadow_parameter.data)

    def restore(self, model):
        """Restore original model parameters after inference."""
        for model_parameter, backup_parameter in zip(model.parameters(), self.backup_parameters):
            model_parameter.data.copy_(backup_parameter.data)


def parse_args():
    parser = argparse.ArgumentParser(description='Train DDPM on MNIST')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides data module default)')
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--sample_every', type=int, default=10)
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base channel count (32 → ~3M params, 64 → ~26M params)')
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


def train_epoch(model, ddpm, optimizer, loader, device, scaler=None, ema=None):
    """Train for one epoch, updating EMA after each optimizer step."""
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

        # Update EMA shadow parameters after each optimizer step
        if ema is not None:
            ema.update(model)

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
        'beta_schedule': 'cosine',
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
    ddpm = DDPM(timesteps=args.timesteps).to(device)

    # Initialize EMA for smoother sample quality (DDPM paper, Ho et al. 2020)
    ema = ExponentialMovingAverage(model, decay=0.995)
    print("EMA enabled (decay=0.995)")

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

        train_loss = train_epoch(model, ddpm, optimizer, train_loader, device, scaler, ema)

        # Use EMA weights for evaluation (better sample quality)
        ema.apply_shadow(model)
        eval_loss = evaluate(model, ddpm, test_loader, device, use_amp)
        ema.restore(model)

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        log_epoch(exp_dir, epoch, train_loss, eval_loss, epoch_time)

        print(f"Epoch {epoch+1}/{args.epochs} | Train: {train_loss:.6f} | Eval: {eval_loss:.6f} | Time: {epoch_time:.1f}s")

        # Generate samples periodically using EMA weights
        if (epoch + 1) % args.sample_every == 0:
            ema.apply_shadow(model)
            model.eval()
            samples = ddpm.p_sample_loop(model, (10, 1, 28, 28))
            save_images(samples, f'{exp_dir}/epoch_samples/epoch_{epoch+1:03d}.png')
            model.train()
            ema.restore(model)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")

    # Generate final outputs using EMA weights
    print("Generating final samples...")
    ema.apply_shadow(model)
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
