"""
VAE Training Script for MNIST.

Trains a Variational Autoencoder to compress padded 1x32x32 MNIST images
into a 2x4x4 latent space, regularized by KL divergence against N(0, I).

Usage:
    python -m models.train_vae --epochs 100 --lr 1e-4
"""
import argparse
import copy
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import get_train_loader, get_test_loader
from models.vae import VAE
from models.utils import (
    setup_experiment_folder,
    save_images,
    plot_loss_curves,
    log_config,
    log_epoch,
    save_performance_metrics,
)


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) of model parameters.

    Maintains shadow copies of model parameters that are updated as:
        shadow_param = decay * shadow_param + (1 - decay) * param

    Used to produce smoother, higher-quality reconstructions during inference.

    Args:
        model: The model whose parameters will be tracked.
        decay (float): EMA decay rate. Higher values give smoother averaging.
                       Default: 0.995 (standard for small models).
    """

    def __init__(self, model, decay=0.995):
        self.decay = decay
        # Deep copy all parameters as shadow weights
        self.shadow_parameters = [
            parameter.clone().detach() for parameter in model.parameters()
        ]

    def update(self, model):
        """Update shadow parameters with current model parameters.

        Formula: shadow = decay * shadow + (1 - decay) * param
        """
        for shadow_parameter, model_parameter in zip(
            self.shadow_parameters, model.parameters()
        ):
            shadow_parameter.data.mul_(self.decay).add_(
                model_parameter.data, alpha=1.0 - self.decay
            )

    def apply_shadow(self, model):
        """Replace model parameters with shadow (EMA) parameters for inference."""
        self.backup_parameters = [
            parameter.clone() for parameter in model.parameters()
        ]
        for model_parameter, shadow_parameter in zip(
            model.parameters(), self.shadow_parameters
        ):
            model_parameter.data.copy_(shadow_parameter.data)

    def restore(self, model):
        """Restore original model parameters after inference."""
        for model_parameter, backup_parameter in zip(
            model.parameters(), self.backup_parameters
        ):
            model_parameter.data.copy_(backup_parameter.data)


def pad_to_32(images: torch.Tensor) -> torch.Tensor:
    """
    Pad 28x28 MNIST images to 32x32 using reflect padding.

    This matches the UNet wrapper's padding strategy so the VAE
    operates on the same 32x32 input space.

    Padding: 2 pixels on each side (left, right, top, bottom).

    Args:
        images: Tensor of shape (B, C, 28, 28).

    Returns:
        Padded tensor of shape (B, C, 32, 32).
    """
    return F.pad(images, (2, 2, 2, 2), mode="reflect")


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE on MNIST")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides data module default)",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=1e-6,
        help="KL divergence weight (beta in beta-VAE). Default: 1e-6",
    )
    parser.add_argument("--base_channels", type=int, default=64, help="Base channel count (default: 64)")
    parser.add_argument(
        "--channel_multipliers",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Per-level channel multipliers (default: 1 2 4)",
    )
    parser.add_argument(
        "--latent_channels",
        type=int,
        default=2,
        help="Latent space channels (default: 2)",
    )
    parser.add_argument(
        "--layers_per_block",
        type=int,
        default=1,
        help="ResNet blocks per encoder/decoder level (default: 1)",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=10,
        help="Generate reconstruction samples every N epochs (default: 10)",
    )
    return parser.parse_args()


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        # Enable TensorFloat32 for faster matmul on Ampere+ GPUs
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, optimizer, loader, device, kl_weight, scaler=None, ema=None):
    """Train for one epoch, returning average losses.

    The VAE training loss is:
        L = L_rec(x, x_hat) + kl_weight * D_KL(q(z|x) || N(0, I))

    where L_rec is per-pixel MSE.
    """
    model.train()
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_loss = 0.0
    use_amp = scaler is not None

    for images, _ in tqdm(loader, desc="Training", leave=False):
        # Pad 28x28 -> 32x32 for the VAE
        images = pad_to_32(images.to(device))

        optimizer.zero_grad()

        if use_amp:
            # Mixed precision training for CUDA
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                reconstruction, posterior = model(images)
                losses = model.loss(images, reconstruction, posterior, kl_weight=kl_weight)
            scaler.scale(losses["total_loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstruction, posterior = model(images)
            losses = model.loss(images, reconstruction, posterior, kl_weight=kl_weight)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Update EMA shadow parameters after each optimizer step
        if ema is not None:
            ema.update(model)

        total_reconstruction_loss += losses["reconstruction_loss"].item()
        total_kl_loss += losses["kl_loss"].item()
        total_loss += losses["total_loss"].item()

    num_batches = len(loader)
    return {
        "reconstruction_loss": total_reconstruction_loss / num_batches,
        "kl_loss": total_kl_loss / num_batches,
        "total_loss": total_loss / num_batches,
    }


@torch.no_grad()
def evaluate(model, loader, device, kl_weight, use_amp=False):
    """Evaluate on test set, returning average losses."""
    model.eval()
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_loss = 0.0

    for images, _ in loader:
        images = pad_to_32(images.to(device))

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                reconstruction, posterior = model(images)
                losses = model.loss(images, reconstruction, posterior, kl_weight=kl_weight)
        else:
            reconstruction, posterior = model(images)
            losses = model.loss(images, reconstruction, posterior, kl_weight=kl_weight)

        total_reconstruction_loss += losses["reconstruction_loss"].item()
        total_kl_loss += losses["kl_loss"].item()
        total_loss += losses["total_loss"].item()

    num_batches = len(loader)
    return {
        "reconstruction_loss": total_reconstruction_loss / num_batches,
        "kl_loss": total_kl_loss / num_batches,
        "total_loss": total_loss / num_batches,
    }


def save_reconstruction_comparison(model, loader, device, path, num_images=10):
    """
    Save a side-by-side comparison of original vs reconstructed images.

    Top row: original padded images
    Bottom row: VAE reconstructions

    Args:
        model: Trained VAE model.
        loader: Data loader to draw examples from.
        device: Torch device.
        path: File path to save the comparison image.
        num_images: Number of image pairs to display.
    """
    import matplotlib.pyplot as plt

    model.eval()

    # Get a batch of images
    images, _ = next(iter(loader))
    images = pad_to_32(images[:num_images].to(device))

    # Reconstruct using deterministic mode encoding
    with torch.no_grad():
        posterior = model.encode(images)
        latent = posterior.mode()  # Deterministic for cleaner visualization
        reconstructions = model.decode(latent)

    # Denormalize from [-1, 1] to [0, 1]
    originals = ((images + 1) / 2).clamp(0, 1).cpu()
    reconstructed = ((reconstructions + 1) / 2).clamp(0, 1).cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(2 * num_images, 4))
    for index in range(num_images):
        # Original (top row)
        axes[0, index].imshow(originals[index, 0], cmap="gray")
        axes[0, index].axis("off")
        if index == 0:
            axes[0, index].set_title("Original", fontsize=10)

        # Reconstruction (bottom row)
        axes[1, index].imshow(reconstructed[index, 0], cmap="gray")
        axes[1, index].axis("off")
        if index == 0:
            axes[1, index].set_title("Reconstructed", fontsize=10)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_checkpoint(model, checkpoint_path, config, epoch, train_losses, eval_losses):
    """Save VAE checkpoint with EMA weights and configuration.

    Note: Call this while EMA weights are applied to the model
    (after ema.apply_shadow()) so model.state_dict() returns EMA weights.

    Args:
        model: The model with EMA weights currently applied.
        checkpoint_path: Path to save the .pt file.
        config: Dict with model architecture config.
        epoch: Current epoch number (0-indexed).
        train_losses: Dict of training loss components.
        eval_losses: Dict of evaluation loss components.
    """
    # Handle torch.compile wrapper: extract original module's state_dict
    if hasattr(model, "_orig_mod"):
        model_state_dict = model._orig_mod.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        "model_state_dict": model_state_dict,
        "config": config,
        "epoch": epoch,
        "train_loss": train_losses["total_loss"],
        "eval_loss": eval_losses["total_loss"],
        "train_reconstruction_loss": train_losses["reconstruction_loss"],
        "train_kl_loss": train_losses["kl_loss"],
        "eval_reconstruction_loss": eval_losses["reconstruction_loss"],
        "eval_kl_loss": eval_losses["kl_loss"],
    }
    torch.save(checkpoint, checkpoint_path)


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    # Create data loaders (with optional batch size override)
    train_loader = get_train_loader(args.batch_size)
    test_loader = get_test_loader(args.batch_size)
    print(f"Batch size: {train_loader.batch_size}")

    # Setup experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = setup_experiment_folder(f"./experiments/{timestamp}-vae")
    print(f"Experiment directory: {exp_dir}")

    # Parse architecture arguments into tuples
    channel_multipliers = tuple(args.channel_multipliers)

    # Log configuration
    config = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "kl_weight": args.kl_weight,
        "base_channels": args.base_channels,
        "channel_multipliers": list(channel_multipliers),
        "latent_channels": args.latent_channels,
        "layers_per_block": args.layers_per_block,
        "sample_every": args.sample_every,
        "device": str(device),
        "batch_size": train_loader.batch_size,
        "image_channels": 1,
    }
    log_config(exp_dir, config)

    # Initialize VAE model
    model = VAE(
        image_channels=1,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        channel_multipliers=channel_multipliers,
        num_layers_per_block=args.layers_per_block,
    ).to(device)

    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"VAE parameters: {num_params:,}")

    # CUDA optimizations: compile model and enable mixed precision
    scaler = None
    if device.type == "cuda":
        model = torch.compile(model)
        scaler = torch.amp.GradScaler("cuda")
        print("CUDA optimizations enabled: torch.compile() + mixed precision (float16)")

    # Model config dict — saved inside checkpoints for reconstruction at generation time
    model_config = {
        "image_channels": 1,
        "latent_channels": args.latent_channels,
        "base_channels": args.base_channels,
        "channel_multipliers": list(channel_multipliers),
        "layers_per_block": args.layers_per_block,
        "kl_weight": args.kl_weight,
    }

    # Initialize EMA for smoother reconstructions
    ema = ExponentialMovingAverage(model, decay=0.995)
    print("EMA enabled (decay=0.995)")

    # Optimizer and scheduler
    # AdamW with weight decay for better generalization
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine annealing: lr decays from initial to 0 over T_max epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    train_losses_history = []
    eval_losses_history = []
    start_time = time.time()
    use_amp = device.type == "cuda"

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_losses = train_epoch(
            model, optimizer, train_loader, device, args.kl_weight, scaler, ema
        )

        # Use EMA weights for evaluation (better reconstruction quality)
        ema.apply_shadow(model)
        eval_losses = evaluate(model, test_loader, device, args.kl_weight, use_amp)
        ema.restore(model)

        train_losses_history.append(train_losses["total_loss"])
        eval_losses_history.append(eval_losses["total_loss"])
        scheduler.step()

        epoch_time = time.time() - epoch_start
        log_epoch(exp_dir, epoch, train_losses["total_loss"], eval_losses["total_loss"], epoch_time)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train: {train_losses['total_loss']:.6f} "
            f"(rec={train_losses['reconstruction_loss']:.6f}, kl={train_losses['kl_loss']:.2f}) | "
            f"Eval: {eval_losses['total_loss']:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Generate reconstruction comparisons and save checkpoint periodically
        if (epoch + 1) % args.sample_every == 0:
            ema.apply_shadow(model)
            model.eval()

            # Save side-by-side original vs reconstruction comparison
            comparison_path = f"{exp_dir}/epoch_samples/epoch_{epoch + 1:03d}.png"
            save_reconstruction_comparison(model, test_loader, device, comparison_path)

            # Save checkpoint with EMA weights (model currently has EMA applied)
            checkpoint_path = f"{exp_dir}/checkpoints/checkpoint_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(model, checkpoint_path, model_config, epoch, train_losses, eval_losses)
            print(f"  Checkpoint saved: {checkpoint_path}")

            model.train()
            ema.restore(model)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes")

    # Generate final outputs and save final checkpoint using EMA weights
    print("Generating final reconstruction comparison...")
    ema.apply_shadow(model)
    model.eval()

    # Save final checkpoint with EMA weights
    final_checkpoint_path = f"{exp_dir}/checkpoints/checkpoint_final.pt"
    save_checkpoint(
        model,
        final_checkpoint_path,
        model_config,
        epoch=args.epochs - 1,
        train_losses={
            "total_loss": train_losses_history[-1],
            "reconstruction_loss": train_losses["reconstruction_loss"],
            "kl_loss": train_losses["kl_loss"],
        },
        eval_losses={
            "total_loss": eval_losses_history[-1],
            "reconstruction_loss": eval_losses["reconstruction_loss"],
            "kl_loss": eval_losses["kl_loss"],
        },
    )
    print(f"Final checkpoint saved: {final_checkpoint_path}")

    # Save final reconstruction comparison
    save_reconstruction_comparison(
        model, test_loader, device,
        f"{exp_dir}/final_samples/final_reconstruction.png",
    )

    # Save loss curves
    plot_loss_curves(
        train_losses_history, eval_losses_history,
        f"{exp_dir}/loss_curves.png",
    )

    # Save performance metrics
    save_performance_metrics(
        exp_dir,
        total_time,
        args.epochs,
        avg_inference_time=0.0,  # VAE inference is near-instant
        final_train_loss=train_losses_history[-1],
        final_eval_loss=eval_losses_history[-1],
    )

    print(f"All outputs saved to {exp_dir}")


if __name__ == "__main__":
    main()
