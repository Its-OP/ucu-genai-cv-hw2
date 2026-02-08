"""
Latent Diffusion Model Sample Generation Script (DDPM / DDIM).

Loads a trained latent UNet checkpoint and a pre-trained VAE checkpoint,
generates MNIST digit samples by denoising in latent space, then decoding
to pixel space via the VAE decoder.

Pipeline:
    1. Sample noise in latent space: z_T ~ N(0, I), shape (1, latent_channels, 4, 4)
    2. Denoise via DDPM or DDIM reverse process: z_T -> z_0
    3. Decode latent to pixel space: x_hat = VAE.decode(z_0), shape (1, 1, 32, 32)
    4. Crop back to 28x28: remove the 2-pixel reflect padding

Each sample is generated individually with per-sample timing, denoising
step visualization, and dedicated subfolders. Output directories are
timestamped to avoid overwriting previous runs.

Usage:
    python -m scripts.python.generate_latent_diffusion \\
        --vae_checkpoint path/to/vae.pt \\
        --unet_checkpoint path/to/unet.pt
    python -m scripts.python.generate_latent_diffusion \\
        --vae_checkpoint path/to/vae.pt \\
        --unet_checkpoint path/to/unet.pt \\
        --mode ddim --ddim_steps 50
"""
import argparse
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from models.unet import UNet
from models.vae import VAE
from models.ddpm import DDPM
from models.ddim import DDIMSampler
from models.utils import save_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate samples from a trained Latent Diffusion Model"
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained VAE checkpoint (.pt file)",
    )
    parser.add_argument(
        "--unet_checkpoint",
        type=str,
        required=True,
        help="Path to trained latent UNet checkpoint (.pt file)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_samples",
        help="Base directory for outputs (a timestamped subfolder is created)",
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=4,
        help="Number of images per row in the grid",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (cuda, mps, cpu). Auto-detects if not specified.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim"],
        help="Sampling mode: ddpm (full T steps) or ddim (fewer steps)",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps (only used with --mode ddim)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM stochasticity: 0.0=deterministic, 1.0=DDPM-like (only with --mode ddim)",
    )
    return parser.parse_args()


def get_device(requested_device=None):
    """Get the best available device, or use the requested one."""
    if requested_device is not None:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_vae(checkpoint_path: str, device: torch.device) -> VAE:
    """
    Load a pre-trained VAE from checkpoint for decoding.

    Args:
        checkpoint_path: Path to the VAE .pt checkpoint file.
        device: Device to load the VAE onto.

    Returns:
        VAE model in eval mode (frozen).
    """
    print(f"Loading VAE checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    vae_config = checkpoint["config"]
    print(
        f"  VAE config: latent_channels={vae_config['latent_channels']}, "
        f"base_channels={vae_config['base_channels']}"
    )

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


def load_unet(checkpoint_path: str, device: torch.device):
    """
    Load a trained latent-space UNet from checkpoint.

    Args:
        checkpoint_path: Path to the latent UNet .pt checkpoint file.
        device: Device to load the model onto.

    Returns:
        Tuple of (unet, ddpm, config) ready for generation.
    """
    print(f"Loading UNet checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    config = checkpoint["config"]
    print(
        f"  UNet config: image_channels={config['image_channels']}, "
        f"base_channels={config['base_channels']}, timesteps={config['timesteps']}"
    )
    print(
        f"  Trained for {checkpoint['epoch'] + 1} epochs "
        f"(train_loss={checkpoint['train_loss']:.6f}, eval_loss={checkpoint['eval_loss']:.6f})"
    )

    # Reconstruct latent UNet from saved config
    channel_multipliers = tuple(config.get("channel_multipliers", (1, 2)))
    layers_per_block = config.get("layers_per_block", 1)
    attention_levels = tuple(config.get("attention_levels", (False, True)))

    unet = UNet(
        image_channels=config["image_channels"],
        base_channels=config["base_channels"],
        channel_multipliers=channel_multipliers,
        layers_per_block=layers_per_block,
        attention_levels=attention_levels,
    ).to(device)

    unet.load_state_dict(checkpoint["model_state_dict"])
    unet.eval()

    num_params = sum(parameter.numel() for parameter in unet.parameters())
    print(f"  UNet parameters: {num_params:,}")

    # Reconstruct DDPM scheduler
    ddpm = DDPM(timesteps=config["timesteps"]).to(device)

    return unet, ddpm, config


def compute_denoising_intermediate_steps_ddpm(total_timesteps):
    """Compute timestep indices at every 10% of the DDPM denoising process.

    Args:
        total_timesteps: Total number of DDPM timesteps (T).

    Returns:
        List of unique integer timestep indices for recording intermediates.
    """
    intermediate_steps = []
    for percentage in range(0, 101, 10):
        step_index = int((total_timesteps - 1) * (1.0 - percentage / 100.0))
        intermediate_steps.append(step_index)
    # Remove duplicates while preserving order
    seen = set()
    unique_steps = []
    for step in intermediate_steps:
        if step not in seen:
            seen.add(step)
            unique_steps.append(step)
    return unique_steps


def compute_denoising_intermediate_steps_ddim(timestep_sequence):
    """Compute timestep indices at every 10% of the DDIM denoising process.

    Args:
        timestep_sequence: The DDIM timestep subsequence tensor (ascending order).

    Returns:
        List of unique integer timestep indices for recording intermediates.
    """
    reversed_sequence = torch.flip(timestep_sequence, [0])
    total_steps = len(reversed_sequence)
    intermediate_steps = []
    for percentage in range(0, 101, 10):
        step_index = min(
            int(percentage / 100.0 * (total_steps - 1)), total_steps - 1
        )
        timestep_value = reversed_sequence[step_index].item()
        intermediate_steps.append(timestep_value)
    seen = set()
    unique_steps = []
    for step in intermediate_steps:
        if step not in seen:
            seen.add(step)
            unique_steps.append(step)
    return unique_steps


@torch.no_grad()
def decode_latent_to_pixel(vae: VAE, latent: torch.Tensor) -> torch.Tensor:
    """
    Decode latent samples to 28x28 pixel images via the VAE decoder.

    Pipeline: z -> VAE.decode(z) -> 32x32 -> crop to 28x28

    Args:
        vae: Frozen VAE decoder.
        latent: Latent tensor, shape (B, latent_channels, 4, 4).

    Returns:
        Pixel images, shape (B, 1, 28, 28), in [-1, 1].
    """
    pixel_images = vae.decode(latent)
    # Crop from 32x32 back to 28x28 (remove 2-pixel padding on each side)
    pixel_images = pixel_images[:, :, 2:30, 2:30]
    return pixel_images


def save_single_image(tensor: torch.Tensor, path: str):
    """Save a single-channel image tensor as a PDF file.

    Args:
        tensor: Image tensor of shape (1, 1, H, W) in [-1, 1].
        path: Output file path.
    """
    image = (tensor[0, 0] + 1) / 2
    image = image.clamp(0, 1).cpu().numpy()

    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()


def save_denoising_steps(intermediates, vae, sample_directory):
    """Save each denoising step (decoded to pixel space) as an individual image.

    Args:
        intermediates: List of (timestep, latent_tensor) tuples from the sampling loop.
        vae: Frozen VAE for decoding latent -> pixel.
        sample_directory: Directory path where step images are saved.
    """
    for step_number, (timestep, latent_tensor) in enumerate(intermediates):
        # Decode latent intermediate to pixel space for visualization
        pixel_tensor = decode_latent_to_pixel(vae, latent_tensor)
        step_path = os.path.join(
            sample_directory, f"step_{step_number:02d}_t{timestep:04d}.pdf"
        )
        save_single_image(pixel_tensor, step_path)


def save_denoising_progression_strip(intermediates, vae, path):
    """Save a horizontal strip showing the denoising progression (decoded to pixels).

    Args:
        intermediates: List of (timestep, latent_tensor) tuples.
        vae: Frozen VAE for decoding.
        path: Output file path for the strip image.
    """
    num_steps = len(intermediates)
    fig, axes = plt.subplots(1, num_steps, figsize=(2.5 * num_steps, 2.5))

    if num_steps == 1:
        axes = [axes]

    for index, (timestep, latent_tensor) in enumerate(intermediates):
        pixel_tensor = decode_latent_to_pixel(vae, latent_tensor)
        image = (pixel_tensor[0, 0] + 1) / 2
        image = image.clamp(0, 1).cpu().numpy()
        axes[index].imshow(image, cmap="gray")
        axes[index].set_title(f"t={timestep}", fontsize=9)
        axes[index].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def write_profile(
    profile_path,
    per_sample_times,
    mode,
    ddim_steps=None,
    eta=None,
    total_timesteps=None,
):
    """Write generation profiling information to a text file.

    Args:
        profile_path: Path to the output profile.txt file.
        per_sample_times: List of per-sample generation times in seconds.
        mode: Sampling mode ('ddpm' or 'ddim').
        ddim_steps: Number of DDIM steps (only for ddim mode).
        eta: DDIM eta parameter (only for ddim mode).
        total_timesteps: Total DDPM timesteps T.
    """
    total_time = sum(per_sample_times)
    average_time = total_time / len(per_sample_times) if per_sample_times else 0.0

    with open(profile_path, "w") as file:
        file.write("Latent Diffusion Generation Profile\n")
        file.write("=" * 50 + "\n\n")

        file.write("Configuration\n")
        file.write("-" * 50 + "\n")
        file.write(f"Sampling mode:       {mode}\n")
        if mode == "ddim":
            file.write(f"DDIM steps:          {ddim_steps}\n")
            file.write(f"DDIM eta:            {eta}\n")
        if total_timesteps is not None:
            file.write(f"Total DDPM timesteps: {total_timesteps}\n")
        file.write(f"Number of samples:   {len(per_sample_times)}\n\n")

        file.write("Per-sample timing\n")
        file.write("-" * 50 + "\n")
        for sample_index, sample_time in enumerate(per_sample_times):
            file.write(f"Sample {sample_index:3d}: {sample_time:.3f}s\n")

        file.write("\n")
        file.write("Summary\n")
        file.write("-" * 50 + "\n")
        file.write(f"Total generation time:   {total_time:.3f}s\n")
        file.write(f"Average time per sample: {average_time:.3f}s\n")
        if len(per_sample_times) > 1:
            min_time = min(per_sample_times)
            max_time = max(per_sample_times)
            file.write(f"Fastest sample:          {min_time:.3f}s\n")
            file.write(f"Slowest sample:          {max_time:.3f}s\n")


def main():
    args = parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = os.path.join(
        args.output_dir, f"{timestamp}-latent-{args.mode}"
    )
    os.makedirs(run_directory, exist_ok=True)

    # Load models
    device = get_device(args.device)
    print(f"Using device: {device}")

    vae = load_vae(args.vae_checkpoint, device)
    unet, ddpm, config = load_unet(args.unet_checkpoint, device)

    # Determine latent shape from config
    latent_channels = config.get("latent_channels", config["image_channels"])
    single_latent_shape = (1, latent_channels, 4, 4)

    # Setup sampler and compute denoising intermediate steps
    if args.mode == "ddim":
        ddim_sampler = DDIMSampler(
            ddpm, ddim_timesteps=args.ddim_steps, eta=args.eta
        ).to(device)
        intermediate_steps = compute_denoising_intermediate_steps_ddim(
            ddim_sampler.timestep_sequence
        )
        print(
            f"\nGenerating {args.num_samples} samples using DDIM "
            f"({args.ddim_steps} steps, eta={args.eta})"
        )
    else:
        intermediate_steps = compute_denoising_intermediate_steps_ddpm(ddpm.timesteps)
        print(
            f"\nGenerating {args.num_samples} samples using DDPM "
            f"({ddpm.timesteps} steps)"
        )

    print(f"Output directory: {run_directory}")
    print(f"Denoising snapshots at {len(intermediate_steps)} points (every 10%)\n")

    # Generate each sample individually
    all_samples = []
    per_sample_times = []

    for sample_index in range(args.num_samples):
        # Create per-sample subfolder
        sample_directory = os.path.join(
            run_directory, f"sample_{sample_index:03d}"
        )
        os.makedirs(sample_directory, exist_ok=True)

        # Time the generation
        sample_start_time = time.time()

        if args.mode == "ddim":
            latent_sample, intermediates = ddim_sampler.ddim_sample_loop(
                unet,
                single_latent_shape,
                return_intermediates=True,
                intermediate_steps=intermediate_steps,
            )
        else:
            latent_sample, intermediates = ddpm.p_sample_loop(
                unet,
                single_latent_shape,
                return_intermediates=True,
                intermediate_steps=intermediate_steps,
            )

        sample_elapsed_time = time.time() - sample_start_time
        per_sample_times.append(sample_elapsed_time)

        # Decode final latent to pixel space
        pixel_sample = decode_latent_to_pixel(vae, latent_sample)
        all_samples.append(pixel_sample)

        # Save final generated image (pixel space, 28x28)
        final_image_path = os.path.join(sample_directory, "final.pdf")
        save_single_image(pixel_sample, final_image_path)

        # Save each denoising step decoded to pixel space
        save_denoising_steps(intermediates, vae, sample_directory)

        # Save denoising progression strip
        progression_path = os.path.join(
            sample_directory, "denoising_progression.pdf"
        )
        save_denoising_progression_strip(intermediates, vae, progression_path)

        print(
            f"  Sample {sample_index:3d}: {sample_elapsed_time:.3f}s "
            f"({len(intermediates)} denoising steps saved)"
        )

    # Save grid of all final samples
    all_samples_tensor = torch.cat(all_samples, dim=0)
    grid_path = os.path.join(run_directory, "grid.pdf")
    save_images(all_samples_tensor, grid_path, nrow=args.nrow)
    print(f"\nSample grid saved: {grid_path}")

    # Write profiling information
    profile_path = os.path.join(run_directory, "profile.txt")
    write_profile(
        profile_path=profile_path,
        per_sample_times=per_sample_times,
        mode=args.mode,
        ddim_steps=args.ddim_steps if args.mode == "ddim" else None,
        eta=args.eta if args.mode == "ddim" else None,
        total_timesteps=ddpm.timesteps,
    )
    print(f"Profile saved: {profile_path}")

    # Print summary
    total_time = sum(per_sample_times)
    average_time = total_time / len(per_sample_times)
    print(f"\nTotal: {total_time:.3f}s | Average per sample: {average_time:.3f}s")
    print(f"All outputs saved to {run_directory}")


if __name__ == "__main__":
    main()
