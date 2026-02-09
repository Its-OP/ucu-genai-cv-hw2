"""
Latent Rectified Flow Sample Generation Script.

Loads a trained latent UNet checkpoint (Rectified Flow) and a pre-trained
VAE checkpoint, generates MNIST digit samples by Euler ODE integration
in latent space, then decoding to pixel space via the VAE decoder.

Pipeline:
    1. Sample noise in latent space: z_1 ~ N(0, I), shape (1, latent_channels, 4, 4)
    2. Euler integration from t=1 to t=0: z_1 -> z_0_scaled
    3. Unscale latents: z_0 = z_0_scaled / scaling_factor
    4. Decode to pixel space: x_hat = VAE.decode(z_0), shape (1, 1, 32, 32)
    5. Crop back to 28x28

Each sample is generated individually with per-sample timing, denoising
step visualization, and dedicated subfolders.

Usage:
    python -m scripts.python.generate_latent_rf \\
        --vae_checkpoint path/to/vae.pt \\
        --unet_checkpoint path/to/unet.pt
    python -m scripts.python.generate_latent_rf \\
        --vae_checkpoint path/to/vae.pt \\
        --unet_checkpoint path/to/unet.pt \\
        --sampling_steps 100
"""
import argparse
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from models.utils import (
    get_device,
    load_rectified_flow_checkpoint,
    load_vae_checkpoint,
    compute_denoising_intermediate_steps_rf,
    save_single_image,
    save_images,
    write_generation_profile,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate samples from a trained Latent Rectified Flow Model"
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
        help="Path to trained latent UNet (RF) checkpoint (.pt file)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,
        help="Number of Euler steps (default: from checkpoint config)",
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
    return parser.parse_args()


@torch.no_grad()
def decode_latent_to_pixel(
    vae, latent: torch.Tensor, scaling_factor: float = 1.0,
) -> torch.Tensor:
    """
    Decode latent samples to 28x28 pixel images via the VAE decoder.

    Pipeline: z_scaled -> z = z_scaled / scaling_factor -> VAE.decode(z) -> 32x32 -> crop to 28x28

    Args:
        vae: Frozen VAE decoder.
        latent: Scaled latent tensor, shape (B, latent_channels, 4, 4).
        scaling_factor: Latent scaling factor used during training.

    Returns:
        Pixel images, shape (B, 1, 28, 28), in [-1, 1].
    """
    unscaled_latent = latent / scaling_factor
    pixel_images = vae.decode(unscaled_latent)
    pixel_images = pixel_images[:, :, 2:30, 2:30]
    return pixel_images


def save_latent_denoising_steps(
    intermediates, vae, sample_directory, scaling_factor=1.0,
):
    """Save each denoising step (decoded to pixel space) as an individual image.

    Args:
        intermediates: List of (step_index, latent_tensor) tuples from the sampling loop.
        vae: Frozen VAE for decoding latent -> pixel.
        sample_directory: Directory path where step images are saved.
        scaling_factor: Latent scaling factor for unscaling before VAE decode.
    """
    for step_number, (step_index, latent_tensor) in enumerate(intermediates):
        pixel_tensor = decode_latent_to_pixel(vae, latent_tensor, scaling_factor)
        step_path = os.path.join(
            sample_directory, f"step_{step_number:02d}_i{step_index:04d}.pdf"
        )
        save_single_image(pixel_tensor, step_path)


def save_latent_denoising_progression(
    intermediates, vae, path, scaling_factor=1.0,
):
    """Save a horizontal strip showing the denoising progression (decoded to pixels).

    Args:
        intermediates: List of (step_index, latent_tensor) tuples.
        vae: Frozen VAE for decoding.
        path: Output file path for the strip image.
        scaling_factor: Latent scaling factor for unscaling before VAE decode.
    """
    num_steps = len(intermediates)
    fig, axes = plt.subplots(1, num_steps, figsize=(2.5 * num_steps, 2.5))

    if num_steps == 1:
        axes = [axes]

    for index, (step_index, latent_tensor) in enumerate(intermediates):
        pixel_tensor = decode_latent_to_pixel(vae, latent_tensor, scaling_factor)
        image = (pixel_tensor[0, 0] + 1) / 2
        image = image.clamp(0, 1).cpu().numpy()
        axes[index].imshow(image, cmap="gray")
        axes[index].set_title(f"step={step_index}", fontsize=9)
        axes[index].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = os.path.join(args.output_dir, f"{timestamp}-latent-rf")
    os.makedirs(run_directory, exist_ok=True)

    # Load models
    device = get_device(args.device)
    print(f"Using device: {device}")

    vae = load_vae_checkpoint(args.vae_checkpoint, device)
    unet, rectified_flow, config = load_rectified_flow_checkpoint(
        args.unet_checkpoint, device,
    )

    # Override sampling steps if specified
    if args.sampling_steps is not None:
        sampling_steps = args.sampling_steps
    else:
        sampling_steps = rectified_flow.number_of_sampling_steps

    # Determine latent shape and scaling factor from config
    latent_channels = config.get("latent_channels", config["image_channels"])
    single_latent_shape = (1, latent_channels, 4, 4)

    scaling_factor = config.get("latent_scaling_factor", 1.0)
    print(f"Latent scaling factor: {scaling_factor:.4f}")

    # Compute intermediate step indices for visualization
    intermediate_steps = compute_denoising_intermediate_steps_rf(sampling_steps)

    print(
        f"\nGenerating {args.num_samples} samples using Latent Rectified Flow "
        f"(Euler, {sampling_steps} steps)"
    )
    print(f"Output directory: {run_directory}")
    print(f"Denoising snapshots at {len(intermediate_steps)} points (every 10%)\n")

    # Generate each sample individually
    all_samples = []
    per_sample_times = []

    for sample_index in range(args.num_samples):
        sample_directory = os.path.join(
            run_directory, f"sample_{sample_index:03d}"
        )
        os.makedirs(sample_directory, exist_ok=True)

        sample_start_time = time.time()

        # clip_denoised=False because latent values are not bounded to [-1, 1]
        latent_sample, intermediates = rectified_flow.euler_sample_loop(
            unet,
            single_latent_shape,
            number_of_steps=sampling_steps,
            return_intermediates=True,
            intermediate_steps=intermediate_steps,
            clip_denoised=False,
        )

        sample_elapsed_time = time.time() - sample_start_time
        per_sample_times.append(sample_elapsed_time)

        # Decode final latent to pixel space
        pixel_sample = decode_latent_to_pixel(vae, latent_sample, scaling_factor)
        all_samples.append(pixel_sample)

        # Save final generated image (pixel space, 28x28)
        final_image_path = os.path.join(sample_directory, "final.pdf")
        save_single_image(pixel_sample, final_image_path)

        # Save each denoising step decoded to pixel space
        save_latent_denoising_steps(
            intermediates, vae, sample_directory, scaling_factor,
        )

        # Save denoising progression strip
        progression_path = os.path.join(
            sample_directory, "denoising_progression.pdf"
        )
        save_latent_denoising_progression(
            intermediates, vae, progression_path, scaling_factor,
        )

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
    write_generation_profile(
        profile_path=profile_path,
        per_sample_times=per_sample_times,
        mode="rf",
        rf_steps=sampling_steps,
    )
    print(f"Profile saved: {profile_path}")

    # Print summary
    total_time = sum(per_sample_times)
    average_time = total_time / len(per_sample_times)
    print(f"\nTotal: {total_time:.3f}s | Average per sample: {average_time:.3f}s")
    print(f"All outputs saved to {run_directory}")


if __name__ == "__main__":
    main()
