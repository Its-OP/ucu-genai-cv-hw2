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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import get_train_loader, get_test_loader
from models.unet import UNet
from models.vae import VAE
from models.ddpm import DDPM
from models.utils import (
    ExponentialMovingAverage,
    pad_to_32,
    get_device,
    load_vae_checkpoint,
    save_checkpoint,
    setup_experiment_folder,
    save_images,
    plot_loss_curves,
    log_config,
    log_epoch,
    save_performance_metrics,
)


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
        default=32,
        help="UNet base channel count for latent space (default: 32)",
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
        default=2,
        help="ResNet blocks per UNet resolution level (default: 2)",
    )
    parser.add_argument(
        "--attention_levels",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Attention flags per UNet level, 0 or 1 (default: 0 1)",
    )
    return parser.parse_args()


@torch.no_grad()
def encode_batch(
    vae: VAE, images: torch.Tensor, scaling_factor: float = 1.0,
) -> torch.Tensor:
    """
    Encode a batch of images to scaled latent space using the frozen VAE.

    Uses the deterministic posterior mode (mean) rather than stochastic sampling
    to avoid injecting VAE reparameterization noise on top of diffusion noise.
    This is standard practice for latent diffusion training (Rombach et al. 2022).

    The latents are scaled by ``scaling_factor`` (= 1/std(z)) so that the
    diffusion noise schedule — which assumes unit-variance data — is properly
    calibrated (Rombach et al. 2022, Section 3.3).

    Args:
        vae: Frozen VAE encoder.
        images: Input images, shape (B, 1, 32, 32).
        scaling_factor: Latent scaling factor (1 / std of unscaled latents).

    Returns:
        Scaled latent codes, shape (B, latent_channels, 4, 4).
    """
    posterior = vae.encode(images)
    # z = posterior mean (deterministic — no reparameterization noise)
    latent = posterior.mode()
    # Scale so that Var(z_scaled) ≈ 1, matching the noise schedule assumption
    return latent * scaling_factor


@torch.no_grad()
def compute_latent_scaling_factor(
    vae: VAE, data_loader, device: torch.device, num_batches: int = 50,
) -> float:
    """
    Compute the latent scaling factor from training data.

    The diffusion noise schedule (cosine or linear) assumes that the training
    data has approximately unit variance. VAE latents may have a very different
    variance, causing the noise schedule to be miscalibrated. This function
    computes ``scaling_factor = 1 / std(z)`` across a subset of encoded
    training images so that ``z_scaled = z * scaling_factor`` has unit variance.

    Reference: Rombach et al. 2022, Section 3.3 ("Latent Diffusion Models").

    Args:
        vae: Frozen VAE encoder.
        data_loader: Training data loader.
        device: Torch device.
        num_batches: Number of batches to use for estimation (default: 50).

    Returns:
        Scalar scaling factor (float).
    """
    all_latents = []
    for batch_index, (images, _) in enumerate(data_loader):
        if batch_index >= num_batches:
            break
        images = pad_to_32(images.to(device))
        posterior = vae.encode(images)
        all_latents.append(posterior.mode())

    # Concatenate all latent samples: shape (N, C, H, W)
    latents = torch.cat(all_latents, dim=0)
    # scaling_factor = 1 / std(z) so that z_scaled = z * scaling_factor has Var ≈ 1
    scaling_factor = 1.0 / latents.flatten().std().item()
    return scaling_factor


def train_epoch(
    unet, ddpm, vae, optimizer, loader, device,
    scaling_factor=1.0, scaler=None, ema=None,
):
    """Train the latent-space UNet for one epoch.

    The diffusion training loss in latent space is:
        L = E_{t, z_0, epsilon}[|| epsilon - epsilon_theta(z_t, t) ||^2]

    where z_0 = scaling_factor · Enc(x) is the scaled VAE-encoded input.
    """
    unet.train()
    total_loss = 0.0
    use_amp = scaler is not None

    for images, _ in tqdm(loader, desc="Training", leave=False):
        # Pad images 28x28 -> 32x32, then encode to scaled latent space
        images = pad_to_32(images.to(device))
        latent = encode_batch(vae, images, scaling_factor=scaling_factor)

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
def evaluate(unet, ddpm, vae, loader, device, scaling_factor=1.0, use_amp=False):
    """Evaluate latent diffusion model on test set."""
    unet.eval()
    total_loss = 0.0

    for images, _ in loader:
        images = pad_to_32(images.to(device))
        latent = encode_batch(vae, images, scaling_factor=scaling_factor)

        timestep = torch.randint(0, ddpm.timesteps, (latent.shape[0],), device=device)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = ddpm.p_losses(unet, latent, timestep)
        else:
            loss = ddpm.p_losses(unet, latent, timestep)

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def generate_samples(
    unet, ddpm, vae, num_samples, latent_channels, device, scaling_factor=1.0,
):
    """
    Generate pixel-space samples via the latent diffusion pipeline.

    Pipeline:
        1. Sample noise in latent space: z_T ~ N(0, I), shape (B, latent_channels, 4, 4)
        2. Denoise via DDPM reverse process (no x₀ clipping): z_T -> z_0_scaled
        3. Unscale latents: z_0 = z_0_scaled / scaling_factor
        4. Decode latent to pixel space: x_hat = Dec(z_0), shape (B, 1, 32, 32)
        5. Crop back to 28x28: remove the 2-pixel reflect padding

    Args:
        unet: Trained latent-space UNet.
        ddpm: DDPM scheduler.
        vae: Frozen VAE decoder.
        num_samples: Number of samples to generate.
        latent_channels: Number of latent channels.
        device: Torch device.
        scaling_factor: Latent scaling factor used during training (1 / std(z)).

    Returns:
        Generated images, shape (num_samples, 1, 28, 28), in [-1, 1].
    """
    # Generate in latent space: (num_samples, latent_channels, 4, 4)
    # clip_denoised=False because latent values are not bounded to [-1, 1]
    latent_shape = (num_samples, latent_channels, 4, 4)
    latent_samples = ddpm.p_sample_loop(unet, latent_shape, clip_denoised=False)

    # Unscale latents back to the original VAE latent range before decoding
    # z_0 = z_0_scaled / scaling_factor
    latent_samples = latent_samples / scaling_factor

    # Decode to pixel space: (num_samples, 1, 32, 32)
    pixel_samples = vae.decode(latent_samples)

    # Crop from 32x32 back to 28x28 (remove 2-pixel padding on each side)
    pixel_samples = pixel_samples[:, :, 2:30, 2:30]

    return pixel_samples


def main():
    args = parse_args()
    device = get_device()

    # Enable TensorFloat-32 matmul precision on CUDA for faster training
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print(f"Using device: {device}")

    # Load frozen VAE
    vae = load_vae_checkpoint(args.vae_checkpoint, device)

    # Get latent channels from VAE config
    latent_channels = vae.latent_channels

    # Create data loaders
    train_loader = get_train_loader(args.batch_size)
    test_loader = get_test_loader(args.batch_size)
    print(f"Batch size: {train_loader.batch_size}")

    # Compute latent scaling factor: 1 / std(z) across a subset of training images
    # This normalizes latent variance to ~1.0, matching the noise schedule assumption
    # (Rombach et al. 2022, Section 3.3: "Latent Diffusion Models")
    print("Computing latent scaling factor...")
    scaling_factor = compute_latent_scaling_factor(vae, train_loader, device)
    print(f"Latent scaling factor: {scaling_factor:.4f}")

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
        "latent_scaling_factor": scaling_factor,
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
        "latent_scaling_factor": scaling_factor,
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
            unet, ddpm, vae, optimizer, train_loader, device,
            scaling_factor=scaling_factor, scaler=scaler, ema=ema,
        )

        # Use EMA weights for evaluation
        ema.apply_shadow(unet)
        eval_loss = evaluate(
            unet, ddpm, vae, test_loader, device,
            scaling_factor=scaling_factor, use_amp=use_amp,
        )
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
            samples = generate_samples(
                unet, ddpm, vae, 10, latent_channels, device,
                scaling_factor=scaling_factor,
            )
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
    final_samples = generate_samples(
        unet, ddpm, vae, 10, latent_channels, device,
        scaling_factor=scaling_factor,
    )
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
