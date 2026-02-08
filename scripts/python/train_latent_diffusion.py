"""
Latent Diffusion Model Training Script for MNIST.

Trains a UNet denoising model within the latent space of a pre-trained
frozen VAE (Rombach et al. 2022 — Latent Diffusion Models).

Pipeline:
    1. Load frozen VAE from checkpoint (no gradients)
    2. Encode training images to latent space: x -> z ~ q(z|x)
    3. Add noise to latent at random timestep: z_t = sqrt(alpha_bar_t) * z + sqrt(1-alpha_bar_t) * epsilon
    4. UNet predicts noise from z_t: epsilon_theta(z_t, t)
    5. Loss = MSE(epsilon, epsilon_theta)

Usage:
    python -m scripts.python.train_latent_diffusion --vae_checkpoint path/to/vae.pt --epochs 100
"""
import argparse
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import get_train_loader, get_test_loader
from models.unet import UNet
from models.vae import VAE
from models.ddpm import DDPM
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

    Args:
        model: The model whose parameters will be tracked.
        decay (float): EMA decay rate. Default: 0.995.
    """

    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow_parameters = [
            parameter.clone().detach() for parameter in model.parameters()
        ]

    def update(self, model):
        """Update shadow parameters: shadow = decay * shadow + (1 - decay) * param"""
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

    Args:
        images: Tensor of shape (B, C, 28, 28).

    Returns:
        Padded tensor of shape (B, C, 32, 32).
    """
    return F.pad(images, (2, 2, 2, 2), mode="reflect")


def load_vae(checkpoint_path: str, device: torch.device) -> VAE:
    """
    Load a pre-trained VAE from checkpoint and freeze all parameters.

    The VAE is set to eval mode with requires_grad=False so it acts
    as a fixed encoder/decoder during latent diffusion training.

    Args:
        checkpoint_path: Path to the VAE .pt checkpoint file.
        device: Device to load the VAE onto.

    Returns:
        Frozen VAE model in eval mode.
    """
    print(f"Loading VAE checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    vae_config = checkpoint["config"]
    print(
        f"  VAE config: latent_channels={vae_config['latent_channels']}, "
        f"base_channels={vae_config['base_channels']}, "
        f"channel_multipliers={vae_config['channel_multipliers']}"
    )

    # Reconstruct VAE from saved config
    vae = VAE(
        image_channels=vae_config.get("image_channels", 1),
        latent_channels=vae_config["latent_channels"],
        base_channels=vae_config["base_channels"],
        channel_multipliers=tuple(vae_config["channel_multipliers"]),
        num_layers_per_block=vae_config.get("layers_per_block", 1),
    ).to(device)

    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()
    vae.requires_grad_(False)

    num_params = sum(parameter.numel() for parameter in vae.parameters())
    print(f"  VAE parameters: {num_params:,} (frozen)")

    return vae


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Latent Diffusion Model on MNIST"
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained VAE checkpoint (.pt file)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides data module default)",
    )
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument(
        "--base_channels",
        type=int,
        default=64,
        help="UNet base channel count for latent space (default: 64)",
    )
    parser.add_argument(
        "--channel_multipliers",
        type=int,
        nargs="+",
        default=[1, 2],
        help="UNet per-level channel multipliers (default: 1 2)",
    )
    parser.add_argument(
        "--layers_per_block",
        type=int,
        default=1,
        help="ResNet blocks per UNet resolution level (default: 1)",
    )
    parser.add_argument(
        "--attention_levels",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Attention flags per UNet level, 0 or 1 (default: 0 1)",
    )
    return parser.parse_args()


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def encode_batch(vae: VAE, images: torch.Tensor) -> torch.Tensor:
    """
    Encode a batch of images to latent space using the frozen VAE.

    Uses reparameterized sampling from the posterior q(z|x):
        z = mean + std * epsilon, where epsilon ~ N(0, I)

    Args:
        vae: Frozen VAE encoder.
        images: Input images, shape (B, 1, 32, 32).

    Returns:
        Latent samples, shape (B, latent_channels, 4, 4).
    """
    posterior = vae.encode(images)
    return posterior.sample()


def train_epoch(
    unet, ddpm, vae, optimizer, loader, device, scaler=None, ema=None
):
    """Train the latent-space UNet for one epoch.

    The diffusion training loss in latent space is:
        L = E_{t, z_0, epsilon}[|| epsilon - epsilon_theta(z_t, t) ||^2]

    where z_0 = Enc(x) is the VAE-encoded input.
    """
    unet.train()
    total_loss = 0.0
    use_amp = scaler is not None

    for images, _ in tqdm(loader, desc="Training", leave=False):
        # Pad images 28x28 -> 32x32, then encode to latent space
        images = pad_to_32(images.to(device))
        latent = encode_batch(vae, images)

        # Sample random timesteps
        timestep = torch.randint(0, ddpm.timesteps, (latent.shape[0],), device=device)

        optimizer.zero_grad()

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = ddpm.p_losses(unet, latent, timestep)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = ddpm.p_losses(unet, latent, timestep)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()

        if ema is not None:
            ema.update(unet)

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(unet, ddpm, vae, loader, device, use_amp=False):
    """Evaluate latent diffusion model on test set."""
    unet.eval()
    total_loss = 0.0

    for images, _ in loader:
        images = pad_to_32(images.to(device))
        latent = encode_batch(vae, images)

        timestep = torch.randint(0, ddpm.timesteps, (latent.shape[0],), device=device)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = ddpm.p_losses(unet, latent, timestep)
        else:
            loss = ddpm.p_losses(unet, latent, timestep)

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def generate_samples(unet, ddpm, vae, num_samples, latent_channels, device):
    """
    Generate pixel-space samples via the latent diffusion pipeline.

    Pipeline:
        1. Sample noise in latent space: z_T ~ N(0, I), shape (B, latent_channels, 4, 4)
        2. Denoise via DDPM reverse process: z_T -> z_0
        3. Decode latent to pixel space: x_hat = Dec(z_0), shape (B, 1, 32, 32)
        4. Crop back to 28x28: remove the 2-pixel reflect padding

    Args:
        unet: Trained latent-space UNet.
        ddpm: DDPM scheduler.
        vae: Frozen VAE decoder.
        num_samples: Number of samples to generate.
        latent_channels: Number of latent channels.
        device: Torch device.

    Returns:
        Generated images, shape (num_samples, 1, 28, 28), in [-1, 1].
    """
    # Generate in latent space: (num_samples, latent_channels, 4, 4)
    # The UNet wrapper pads 4x4 -> 8x8 internally, so this works with channel_multipliers=(1,2)
    latent_shape = (num_samples, latent_channels, 4, 4)
    latent_samples = ddpm.p_sample_loop(unet, latent_shape)

    # Decode to pixel space: (num_samples, 1, 32, 32)
    pixel_samples = vae.decode(latent_samples)

    # Crop from 32x32 back to 28x28 (remove 2-pixel padding on each side)
    pixel_samples = pixel_samples[:, :, 2:30, 2:30]

    return pixel_samples


def save_checkpoint(unet, checkpoint_path, config, epoch, train_loss, eval_loss):
    """Save latent diffusion UNet checkpoint.

    Note: Call this while EMA weights are applied to the model.

    Args:
        unet: The UNet model with EMA weights currently applied.
        checkpoint_path: Path to save the .pt file.
        config: Dict with model architecture config.
        epoch: Current epoch number (0-indexed).
        train_loss: Training loss at this epoch.
        eval_loss: Evaluation loss at this epoch.
    """
    if hasattr(unet, "_orig_mod"):
        model_state_dict = unet._orig_mod.state_dict()
    else:
        model_state_dict = unet.state_dict()

    checkpoint = {
        "model_state_dict": model_state_dict,
        "config": config,
        "epoch": epoch,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
    }
    torch.save(checkpoint, checkpoint_path)


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    # Load frozen VAE
    vae = load_vae(args.vae_checkpoint, device)

    # Get latent channels from VAE config
    latent_channels = vae.latent_channels

    # Create data loaders
    train_loader = get_train_loader(args.batch_size)
    test_loader = get_test_loader(args.batch_size)
    print(f"Batch size: {train_loader.batch_size}")

    # Setup experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = setup_experiment_folder(f"./experiments/{timestamp}-latent-diffusion")
    print(f"Experiment directory: {exp_dir}")

    # Parse architecture arguments
    channel_multipliers = tuple(args.channel_multipliers)
    attention_levels = tuple(bool(flag) for flag in args.attention_levels)

    # Log configuration
    config = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "timesteps": args.timesteps,
        "beta_schedule": "cosine",
        "sample_every": args.sample_every,
        "base_channels": args.base_channels,
        "channel_multipliers": list(channel_multipliers),
        "layers_per_block": args.layers_per_block,
        "attention_levels": list(attention_levels),
        "device": str(device),
        "batch_size": train_loader.batch_size,
        "vae_checkpoint": args.vae_checkpoint,
        "image_channels": latent_channels,  # UNet input = latent channels
        "latent_channels": latent_channels,
    }
    log_config(exp_dir, config)

    # Initialize latent-space UNet
    # image_channels = latent_channels (UNet operates on latent representations)
    # The UNet wrapper handles padding from 4x4 to the next multiple of 8 (8x8)
    unet = UNet(
        image_channels=latent_channels,
        base_channels=args.base_channels,
        channel_multipliers=channel_multipliers,
        layers_per_block=args.layers_per_block,
        attention_levels=attention_levels,
    ).to(device)

    num_params = sum(parameter.numel() for parameter in unet.parameters())
    print(f"Latent UNet parameters: {num_params:,}")

    # CUDA optimizations
    scaler = None
    if device.type == "cuda":
        unet = torch.compile(unet)
        scaler = torch.amp.GradScaler("cuda")
        print("CUDA optimizations enabled: torch.compile() + mixed precision (float16)")

    # Initialize DDPM scheduler
    ddpm = DDPM(timesteps=args.timesteps).to(device)

    # Model config saved in checkpoints for reconstruction at generation time
    model_config = {
        "image_channels": latent_channels,
        "base_channels": args.base_channels,
        "channel_multipliers": list(channel_multipliers),
        "layers_per_block": args.layers_per_block,
        "attention_levels": list(attention_levels),
        "timesteps": args.timesteps,
        "vae_checkpoint": args.vae_checkpoint,
        "latent_channels": latent_channels,
    }

    # Initialize EMA
    ema = ExponentialMovingAverage(unet, decay=0.995)
    print("EMA enabled (decay=0.995)")

    # Optimizer and scheduler
    optimizer = AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    train_losses, eval_losses = [], []
    start_time = time.time()
    use_amp = device.type == "cuda"

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss = train_epoch(
            unet, ddpm, vae, optimizer, train_loader, device, scaler, ema
        )

        # Use EMA weights for evaluation
        ema.apply_shadow(unet)
        eval_loss = evaluate(unet, ddpm, vae, test_loader, device, use_amp)
        ema.restore(unet)

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        log_epoch(exp_dir, epoch, train_loss, eval_loss, epoch_time)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train: {train_loss:.6f} | Eval: {eval_loss:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Generate samples and save checkpoint periodically
        if (epoch + 1) % args.sample_every == 0:
            ema.apply_shadow(unet)
            unet.eval()

            # Generate samples through the full pipeline: noise -> latent -> pixel
            samples = generate_samples(unet, ddpm, vae, 10, latent_channels, device)
            save_images(samples, f"{exp_dir}/epoch_samples/epoch_{epoch + 1:03d}.png")

            # Save checkpoint with EMA weights
            checkpoint_path = (
                f"{exp_dir}/checkpoints/checkpoint_epoch_{epoch + 1:03d}.pt"
            )
            save_checkpoint(unet, checkpoint_path, model_config, epoch, train_loss, eval_loss)
            print(f"  Checkpoint saved: {checkpoint_path}")

            unet.train()
            ema.restore(unet)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes")

    # Generate final outputs using EMA weights
    print("Generating final samples...")
    ema.apply_shadow(unet)
    unet.eval()

    # Save final checkpoint
    final_checkpoint_path = f"{exp_dir}/checkpoints/checkpoint_final.pt"
    save_checkpoint(
        unet,
        final_checkpoint_path,
        model_config,
        epoch=args.epochs - 1,
        train_loss=train_losses[-1],
        eval_loss=eval_losses[-1],
    )
    print(f"Final checkpoint saved: {final_checkpoint_path}")

    # Generate final samples
    inference_start = time.time()
    final_samples = generate_samples(unet, ddpm, vae, 10, latent_channels, device)
    inference_time = time.time() - inference_start
    save_images(final_samples, f"{exp_dir}/final_samples/final_grid.png")

    # Save individual final samples
    import matplotlib.pyplot as plt

    for sample_index in range(10):
        image = (final_samples[sample_index, 0] + 1) / 2
        image = image.clamp(0, 1).cpu().numpy()
        plt.figure(figsize=(3, 3))
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.savefig(
            f"{exp_dir}/final_samples/sample_{sample_index}.png",
            bbox_inches="tight",
            dpi=100,
        )
        plt.close()

    # Save loss curves
    plot_loss_curves(train_losses, eval_losses, f"{exp_dir}/loss_curves.png")

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


if __name__ == "__main__":
    main()
