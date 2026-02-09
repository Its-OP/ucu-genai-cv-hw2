"""
Class-Conditioned Latent Diffusion Model Sample Generation Script (DDPM / DDIM).

Loads a trained class-conditioned latent UNet checkpoint and a pre-trained VAE
checkpoint, then generates MNIST digit samples for specific classes using
classifier-free guidance (Ho & Salimans 2022).

Pipeline per sample:
    1. Set target class label on the conditioned UNet
    2. Sample noise in latent space: z_T ~ N(0, I), shape (1, latent_channels, 4, 4)
    3. Denoise via DDPM or DDIM with CFG: z_T -> z_0
    4. Decode latent to pixel space: x_hat = VAE.decode(z_0), shape (1, 1, 32, 32)
    5. Crop back to 28x28: remove the 2-pixel reflect padding

Usage:
    python -m scripts.python.generate_conditioned_latent_diffusion \\
        --vae_checkpoint path/to/vae.pt \\
        --unet_checkpoint path/to/conditioned_unet.pt \\
        --class_label 7 --guidance_scale 3.0

    # Generate one sample per class (0-9) if --class_label is not specified:
    python -m scripts.python.generate_conditioned_latent_diffusion \\
        --vae_checkpoint path/to/vae.pt \\
        --unet_checkpoint path/to/conditioned_unet.pt
"""
import argparse
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from models.ddpm import DDPM
from models.ddim import DDIMSampler
from models.classifier_free_guidance import (
    ClassConditionedUNet,
    ClassifierFreeGuidanceWrapper,
)
from models.utils import (
    get_device,
    load_unet_checkpoint,
    load_vae_checkpoint,
    compute_denoising_intermediate_steps_ddpm,
    compute_denoising_intermediate_steps_ddim,
    save_single_image,
    save_images,
    write_generation_profile,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate class-conditioned samples from a trained "
        "Latent Diffusion Model"
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
        help="Path to trained conditioned latent UNet checkpoint (.pt file)",
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=None,
        help="Target digit class (0-9). If not specified, generates one "
        "sample per class.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate per class (default: 1)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale (default: 3.0). Higher values "
        "produce stronger class adherence.",
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
        default=10,
        help="Number of images per row in the grid (default: 10)",
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
        help="DDIM stochasticity: 0.0=deterministic, 1.0=DDPM-like "
        "(only with --mode ddim)",
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
    # Crop from 32x32 back to 28x28 (remove 2-pixel padding on each side)
    pixel_images = pixel_images[:, :, 2:30, 2:30]
    return pixel_images


def save_latent_denoising_steps(
    intermediates, vae, sample_directory, scaling_factor=1.0,
):
    """Save each denoising step (decoded to pixel space) as an individual image."""
    for step_number, (timestep, latent_tensor) in enumerate(intermediates):
        pixel_tensor = decode_latent_to_pixel(vae, latent_tensor, scaling_factor)
        step_path = os.path.join(
            sample_directory, f"step_{step_number:02d}_t{timestep:04d}.pdf"
        )
        save_single_image(pixel_tensor, step_path)


def save_latent_denoising_progression(
    intermediates, vae, path, scaling_factor=1.0,
):
    """Save a horizontal strip showing the denoising progression (decoded to pixels)."""
    num_steps = len(intermediates)
    fig, axes = plt.subplots(1, num_steps, figsize=(2.5 * num_steps, 2.5))

    if num_steps == 1:
        axes = [axes]

    for index, (timestep, latent_tensor) in enumerate(intermediates):
        pixel_tensor = decode_latent_to_pixel(vae, latent_tensor, scaling_factor)
        image = (pixel_tensor[0, 0] + 1) / 2
        image = image.clamp(0, 1).cpu().numpy()
        axes[index].imshow(image, cmap="gray")
        axes[index].set_title(f"t={timestep}", fontsize=9)
        axes[index].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    # Determine which classes to generate
    if args.class_label is not None:
        class_labels = [args.class_label]
    else:
        # Generate one sample per class (0-9) by default
        class_labels = list(range(10))

    total_samples = len(class_labels) * args.num_samples

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = os.path.join(
        args.output_dir,
        f"{timestamp}-conditioned-latent-{args.mode}",
    )
    os.makedirs(run_directory, exist_ok=True)

    # Load models
    device = get_device(args.device)
    print(f"Using device: {device}")

    vae = load_vae_checkpoint(args.vae_checkpoint, device)
    model, ddpm, config = load_unet_checkpoint(args.unet_checkpoint, device)

    # Verify checkpoint is a conditioned model
    if not isinstance(model, ClassConditionedUNet):
        raise ValueError(
            "The provided UNet checkpoint is not a conditioned model. "
            "Use generate_latent_diffusion.py for unconditional models."
        )

    conditioned_unet = model
    number_of_classes = conditioned_unet.number_of_classes

    # Determine latent shape and scaling factor from config
    latent_channels = config.get("latent_channels", config["output_channels"])
    single_latent_shape = (1, latent_channels, 4, 4)

    scaling_factor = config.get("latent_scaling_factor", 1.0)
    print(f"Latent scaling factor: {scaling_factor:.4f}")
    print(f"Number of classes: {number_of_classes}")
    print(f"Guidance scale: {args.guidance_scale}")

    # Create CFG wrapper for sampling
    cfg_wrapper = ClassifierFreeGuidanceWrapper(
        conditioned_unet, guidance_scale=args.guidance_scale,
    )

    # Setup sampler and compute denoising intermediate steps
    if args.mode == "ddim":
        ddim_sampler = DDIMSampler(
            ddpm, ddim_timesteps=args.ddim_steps, eta=args.eta,
        ).to(device)
        intermediate_steps = compute_denoising_intermediate_steps_ddim(
            ddim_sampler.timestep_sequence
        )
        print(
            f"\nGenerating {total_samples} samples using DDIM "
            f"({args.ddim_steps} steps, eta={args.eta})"
        )
    else:
        intermediate_steps = compute_denoising_intermediate_steps_ddpm(
            ddpm.timesteps
        )
        print(
            f"\nGenerating {total_samples} samples using DDPM "
            f"({ddpm.timesteps} steps)"
        )

    print(f"Classes: {class_labels}")
    print(f"Samples per class: {args.num_samples}")
    print(f"Output directory: {run_directory}")
    print(
        f"Denoising snapshots at {len(intermediate_steps)} points (every 10%)\n"
    )

    # Generate samples
    all_samples = []
    all_labels = []
    per_sample_times = []

    sample_counter = 0
    for class_label in class_labels:
        label_tensor = torch.tensor([class_label], device=device)

        for repeat_index in range(args.num_samples):
            # Create per-sample subfolder
            sample_directory = os.path.join(
                run_directory,
                f"sample_{sample_counter:03d}_class{class_label}",
            )
            os.makedirs(sample_directory, exist_ok=True)

            # Set class label for CFG
            conditioned_unet.set_class_labels(label_tensor)

            # Time the generation
            sample_start_time = time.time()

            # clip_denoised=False because latent values are not bounded to [-1, 1]
            if args.mode == "ddim":
                latent_sample, intermediates = ddim_sampler.ddim_sample_loop(
                    cfg_wrapper,
                    single_latent_shape,
                    return_intermediates=True,
                    intermediate_steps=intermediate_steps,
                    clip_denoised=False,
                )
            else:
                latent_sample, intermediates = ddpm.p_sample_loop(
                    cfg_wrapper,
                    single_latent_shape,
                    return_intermediates=True,
                    intermediate_steps=intermediate_steps,
                    clip_denoised=False,
                )

            sample_elapsed_time = time.time() - sample_start_time
            per_sample_times.append(sample_elapsed_time)

            # Decode final latent to pixel space
            pixel_sample = decode_latent_to_pixel(
                vae, latent_sample, scaling_factor,
            )
            all_samples.append(pixel_sample)
            all_labels.append(class_label)

            # Save final generated image
            final_image_path = os.path.join(sample_directory, "final.pdf")
            save_single_image(pixel_sample, final_image_path)

            # Save denoising steps and progression
            save_latent_denoising_steps(
                intermediates, vae, sample_directory, scaling_factor,
            )
            progression_path = os.path.join(
                sample_directory, "denoising_progression.pdf",
            )
            save_latent_denoising_progression(
                intermediates, vae, progression_path, scaling_factor,
            )

            print(
                f"  Sample {sample_counter:3d} (class {class_label}): "
                f"{sample_elapsed_time:.3f}s "
                f"({len(intermediates)} denoising steps saved)"
            )

            sample_counter += 1

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
