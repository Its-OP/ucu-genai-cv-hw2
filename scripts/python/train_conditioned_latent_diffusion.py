"""
Class-Conditioned Latent Diffusion Model Training Script for MNIST.

Trains a UNet denoising model within the latent space of a pre-trained
frozen VAE, with class conditioning via input channel concatenation and
classifier-free guidance (Ho & Salimans 2022).

Pipeline:
    1. Load frozen VAE from checkpoint (no gradients)
    2. Encode training images to latent space: x -> z ~ q(z|x)
    3. Build class conditioning map via learnable embedding: label -> (1, H, W)
    4. Concatenate conditioning to noisy latent: (C+1, H, W)
    5. UNet predicts noise from conditioned input: epsilon_theta(z_t || c, t)
    6. Loss = MSE(epsilon, epsilon_theta)
    7. Classifier-free dropout: randomly drop conditioning for 10% of samples

Usage:
    python -m scripts.python.train_conditioned_latent_diffusion \
        --vae_checkpoint path/to/vae.pt --epochs 100
"""
import argparse
import time
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import get_train_loader, get_test_loader
from models.vae import VAE
from models.ddpm import DDPM
from models.ddim import DDIMSampler
from models.classifier_free_guidance import (
    ClassConditionedUNet,
    ClassifierFreeGuidanceWrapper,
    create_conditioned_unet,
)
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
        description="Train Class-Conditioned Latent Diffusion Model on MNIST"
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
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale for sample generation (default: 3.0)",
    )
    parser.add_argument(
        "--unconditional_probability",
        type=float,
        default=0.1,
        help="Probability of dropping class conditioning per sample (default: 0.1)",
    )
    parser.add_argument(
        "--number_of_classes",
        type=int,
        default=10,
        help="Number of classes for conditioning (default: 10 for MNIST)",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="Number of DDIM sampling steps for epoch demonstrations (default: 100)",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.05,
        help="DDIM stochasticity parameter η for epoch demonstrations (default: 0.05)",
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
    conditioned_unet, ddpm, vae, optimizer, loader, device,
    scaling_factor=1.0, scaler=None, ema=None,
):
    """Train the class-conditioned latent-space UNet for one epoch.

    The diffusion training loss in latent space with class conditioning is:
        L = E_{t, z_0, epsilon, c}[|| epsilon - epsilon_theta(z_t || c, t) ||^2]

    where z_0 = scaling_factor · Enc(x) is the scaled VAE-encoded input,
    and c is the class conditioning map (randomly dropped with probability
    p_uncond for classifier-free guidance training).
    """
    conditioned_unet.train()
    total_loss = 0.0
    use_amp = scaler is not None

    for images, labels in tqdm(loader, desc="Training", leave=False):
        # Pad images 28x28 -> 32x32, then encode to scaled latent space
        images = pad_to_32(images.to(device))
        latent = encode_batch(vae, images, scaling_factor=scaling_factor)

        # Set class labels for conditioning (dropout handled inside wrapper)
        conditioned_unet.set_class_labels(labels.to(device))

        # Sample random timesteps
        timestep = torch.randint(
            0, ddpm.timesteps, (latent.shape[0],), device=device,
        )

        optimizer.zero_grad()

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = ddpm.p_losses(conditioned_unet, latent, timestep)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                conditioned_unet.parameters(), 1.0,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = ddpm.p_losses(conditioned_unet, latent, timestep)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                conditioned_unet.parameters(), 1.0,
            )
            optimizer.step()

        if ema is not None:
            ema.update(conditioned_unet)

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    conditioned_unet, ddpm, vae, loader, device,
    scaling_factor=1.0, use_amp=False,
):
    """Evaluate conditioned latent diffusion model on test set."""
    conditioned_unet.eval()
    total_loss = 0.0

    for images, labels in loader:
        images = pad_to_32(images.to(device))
        latent = encode_batch(vae, images, scaling_factor=scaling_factor)

        # Set class labels (no dropout in eval mode)
        conditioned_unet.set_class_labels(labels.to(device))

        timestep = torch.randint(
            0, ddpm.timesteps, (latent.shape[0],), device=device,
        )

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = ddpm.p_losses(conditioned_unet, latent, timestep)
        else:
            loss = ddpm.p_losses(conditioned_unet, latent, timestep)

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def generate_conditioned_samples(
    conditioned_unet, ddpm, vae, latent_channels, device,
    scaling_factor=1.0, guidance_scale=3.0, ddim_steps=100, ddim_eta=0.05,
):
    """
    Generate one sample per digit class (0-9) using classifier-free guidance.

    Uses DDIM sampling (Song et al. 2020) instead of full DDPM reverse diffusion
    for significantly faster generation. With 100 DDIM steps (vs 1000 DDPM steps)
    and CFG's dual forward pass, this reduces per-demo cost from 20,000 to 2,000
    UNet forward passes (~10× speedup).

    Pipeline per class:
        1. Set class label on the conditioned UNet
        2. Wrap with ClassifierFreeGuidanceWrapper for CFG sampling
        3. Sample noise in latent space: z_T ~ N(0, I)
        4. Denoise via DDIM with CFG: z_T -> z_0_scaled
        5. Unscale and decode: z_0 -> pixel space -> crop to 28x28

    Args:
        conditioned_unet: Trained ClassConditionedUNet.
        ddpm: DDPM scheduler.
        vae: Frozen VAE decoder.
        latent_channels: Number of VAE latent channels.
        device: Torch device.
        scaling_factor: Latent scaling factor (1 / std(z)).
        guidance_scale: CFG weight (higher = stronger class adherence).
        ddim_steps: Number of DDIM sampling steps (default: 100).
        ddim_eta: DDIM stochasticity parameter η (default: 0.05).
            η=0 is fully deterministic, η=1 matches DDPM variance.

    Returns:
        Generated images, shape (10, 1, 28, 28), one per digit class.
    """
    conditioned_unet.eval()
    cfg_wrapper = ClassifierFreeGuidanceWrapper(
        conditioned_unet, guidance_scale=guidance_scale,
    )

    # Use DDIM for faster sampling: ddim_steps << ddpm.timesteps
    # DDIM reuses the DDPM's trained noise schedule (alphas_cumprod) but
    # with a strided subsequence of timesteps (Song et al. 2020)
    ddim_sampler = DDIMSampler(ddpm, ddim_timesteps=ddim_steps, eta=ddim_eta)

    all_samples = []
    latent_shape = (1, latent_channels, 4, 4)

    for class_label in range(conditioned_unet.number_of_classes):
        # Set the target class label
        label_tensor = torch.tensor([class_label], device=device)
        conditioned_unet.set_class_labels(label_tensor)

        # Generate in latent space with CFG using DDIM.
        # clip_denoised=True (default) prevents the x₀ reconstruction
        # amplification cascade at late timesteps.
        latent_sample = ddim_sampler.ddim_sample_loop(
            cfg_wrapper, latent_shape,
        )

        # Unscale and decode to pixel space
        latent_sample = latent_sample / scaling_factor
        pixel_sample = vae.decode(latent_sample)
        pixel_sample = pixel_sample[:, :, 2:30, 2:30]  # Crop 32x32 -> 28x28
        all_samples.append(pixel_sample)

    return torch.cat(all_samples, dim=0)


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
    exp_dir = setup_experiment_folder(
        f"./experiments/{timestamp}-conditioned-latent-diffusion"
    )
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
        # UNet input = latent_channels + 1 (extra channel for class conditioning)
        "image_channels": latent_channels + 1,
        "output_channels": latent_channels,
        "latent_channels": latent_channels,
        "latent_scaling_factor": scaling_factor,
        "conditioned": True,
        "number_of_classes": args.number_of_classes,
        "unconditional_probability": args.unconditional_probability,
        "guidance_scale": args.guidance_scale,
        "ddim_steps": args.ddim_steps,
        "ddim_eta": args.ddim_eta,
    }
    log_config(exp_dir, config)

    # Initialize class-conditioned latent-space UNet
    # UNet input: latent_channels + 1 (class conditioning channel)
    # UNet output: latent_channels (noise prediction in latent space)
    conditioned_unet = create_conditioned_unet(
        latent_channels=latent_channels,
        number_of_classes=args.number_of_classes,
        unconditional_probability=args.unconditional_probability,
        base_channels=args.base_channels,
        channel_multipliers=channel_multipliers,
        layers_per_block=args.layers_per_block,
        attention_levels=attention_levels,
    ).to(device)

    num_params = sum(
        parameter.numel() for parameter in conditioned_unet.parameters()
    )
    print(f"Conditioned Latent UNet parameters: {num_params:,}")
    print(
        f"  Class conditioning: {args.number_of_classes} classes, "
        f"p_uncond={args.unconditional_probability}, "
        f"guidance_scale={args.guidance_scale}"
    )

    # CUDA optimizations: compile the inner UNet (not the wrapper, which has
    # control flow for conditioning dropout that torch.compile cannot handle)
    scaler = None
    if device.type == "cuda":
        conditioned_unet.unet = torch.compile(conditioned_unet.unet)
        scaler = torch.amp.GradScaler("cuda")
        print(
            "CUDA optimizations enabled: torch.compile(inner UNet) "
            "+ mixed precision (float16)"
        )

    # Initialize DDPM scheduler
    ddpm = DDPM(timesteps=args.timesteps).to(device)

    # Model config saved in checkpoints for reconstruction at generation time
    model_config = {
        "image_channels": latent_channels + 1,
        "output_channels": latent_channels,
        "base_channels": args.base_channels,
        "channel_multipliers": list(channel_multipliers),
        "layers_per_block": args.layers_per_block,
        "attention_levels": list(attention_levels),
        "timesteps": args.timesteps,
        "vae_checkpoint": args.vae_checkpoint,
        "latent_channels": latent_channels,
        "latent_scaling_factor": scaling_factor,
        "conditioned": True,
        "number_of_classes": args.number_of_classes,
        "unconditional_probability": args.unconditional_probability,
    }

    # Initialize EMA (tracks all parameters: UNet + class embedding)
    ema = ExponentialMovingAverage(conditioned_unet, decay=0.995)
    print("EMA enabled (decay=0.995)")

    # Optimizer and scheduler
    optimizer = AdamW(
        conditioned_unet.parameters(), lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # Training loop
    train_losses, eval_losses = [], []
    start_time = time.time()
    use_amp = device.type == "cuda"

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss = train_epoch(
            conditioned_unet, ddpm, vae, optimizer, train_loader, device,
            scaling_factor=scaling_factor, scaler=scaler, ema=ema,
        )

        # Use EMA weights for evaluation
        ema.apply_shadow(conditioned_unet)
        eval_loss = evaluate(
            conditioned_unet, ddpm, vae, test_loader, device,
            scaling_factor=scaling_factor, use_amp=use_amp,
        )
        ema.restore(conditioned_unet)

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
            ema.apply_shadow(conditioned_unet)
            conditioned_unet.eval()

            # Generate one sample per digit class (0-9) with CFG + DDIM
            samples = generate_conditioned_samples(
                conditioned_unet, ddpm, vae, latent_channels, device,
                scaling_factor=scaling_factor,
                guidance_scale=args.guidance_scale,
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
            )
            save_images(
                samples,
                f"{exp_dir}/epoch_samples/epoch_{epoch + 1:03d}.pdf",
                nrow=args.number_of_classes,
            )

            # Save checkpoint with EMA weights
            checkpoint_path = (
                f"{exp_dir}/checkpoints/checkpoint_epoch_{epoch + 1:03d}.pt"
            )
            save_checkpoint(
                conditioned_unet, checkpoint_path, model_config,
                epoch, train_loss, eval_loss,
            )
            print(f"  Checkpoint saved: {checkpoint_path}")

            conditioned_unet.train()
            ema.restore(conditioned_unet)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes")

    # Generate final outputs using EMA weights
    print("Generating final samples (one per class, with CFG)...")
    ema.apply_shadow(conditioned_unet)
    conditioned_unet.eval()

    # Save final checkpoint
    final_checkpoint_path = f"{exp_dir}/checkpoints/checkpoint_final.pt"
    save_checkpoint(
        conditioned_unet,
        final_checkpoint_path,
        model_config,
        epoch=args.epochs - 1,
        train_loss=train_losses[-1],
        eval_loss=eval_losses[-1],
    )
    print(f"Final checkpoint saved: {final_checkpoint_path}")

    # Generate final samples: one per digit class using DDIM
    inference_start = time.time()
    final_samples = generate_conditioned_samples(
        conditioned_unet, ddpm, vae, latent_channels, device,
        scaling_factor=scaling_factor,
        guidance_scale=args.guidance_scale,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
    )
    inference_time = time.time() - inference_start
    save_images(
        final_samples,
        f"{exp_dir}/final_samples/final_grid.pdf",
        nrow=args.number_of_classes,
    )

    # Save individual final samples labeled by class
    import matplotlib.pyplot as plt

    for class_index in range(min(10, len(final_samples))):
        image = (final_samples[class_index, 0] + 1) / 2
        image = image.clamp(0, 1).cpu().numpy()
        plt.figure(figsize=(3, 3))
        plt.imshow(image, cmap="gray")
        plt.title(f"Class {class_index}")
        plt.axis("off")
        plt.savefig(
            f"{exp_dir}/final_samples/sample_class_{class_index}.pdf",
            bbox_inches="tight",
            dpi=100,
        )
        plt.close()

    # Save loss curves
    plot_loss_curves(train_losses, eval_losses, f"{exp_dir}/loss_curves.pdf")

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
