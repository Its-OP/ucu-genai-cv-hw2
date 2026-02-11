"""
Diffusion Model Sample Generation Script (DDPM / DDIM)
"""
import argparse
import os
import time
from datetime import datetime

import torch

from models.ddpm import DDPM
from models.ddim import DDIMSampler
from models.utils import (
    get_device,
    load_unet_checkpoint,
    compute_denoising_intermediate_steps_ddpm,
    compute_denoising_intermediate_steps_ddim,
    save_single_image,
    save_individual_denoising_steps,
    save_denoising_progression,
    save_images,
    write_generation_profile,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate samples from a trained DDPM/DDIM checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pt checkpoint file')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='./generated_samples',
                        help='Base directory for outputs (a timestamped subfolder is created)')
    parser.add_argument('--nrow', type=int, default=4,
                        help='Number of images per row in the grid')
    parser.add_argument('--device', type=str, default=None,
                        help='Force device (cuda, mps, cpu). Auto-detects if not specified.')
    parser.add_argument('--mode', type=str, default='ddpm', choices=['ddpm', 'ddim'],
                        help='Sampling mode: ddpm (full T steps) or ddim (fewer steps)')
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='Number of DDIM sampling steps (only used with --mode ddim)')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM stochasticity: 0.0=deterministic, 1.0=DDPM-like (only with --mode ddim)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Create timestamped output directory to avoid overwriting previous runs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_directory = os.path.join(args.output_dir, f'{timestamp}-{args.mode}')
    os.makedirs(run_directory, exist_ok=True)

    # Load model from checkpoint
    device = get_device(args.device)
    print(f"Using device: {device}")

    model, ddpm, config = load_unet_checkpoint(args.checkpoint, device)

    image_channels = config['image_channels']
    single_sample_shape = (1, image_channels, 28, 28)

    # Setup sampler and compute denoising intermediate steps (every 10% of process)
    if args.mode == 'ddim':
        ddim_sampler = DDIMSampler(
            ddpm, ddim_timesteps=args.ddim_steps, eta=args.eta,
        ).to(device)
        intermediate_steps = compute_denoising_intermediate_steps_ddim(
            ddim_sampler.timestep_sequence,
        )
        print(f"\nGenerating {args.num_samples} samples using DDIM "
              f"({args.ddim_steps} steps, eta={args.eta})")
    else:
        intermediate_steps = compute_denoising_intermediate_steps_ddpm(ddpm.timesteps)
        print(f"\nGenerating {args.num_samples} samples using DDPM "
              f"({ddpm.timesteps} steps)")

    print(f"Output directory: {run_directory}")
    print(f"Denoising snapshots at {len(intermediate_steps)} points (every 10%)\n")

    # Generate each sample individually with timing and denoising visualization
    all_samples = []
    per_sample_times = []

    for sample_index in range(args.num_samples):
        # Create per-sample subfolder
        sample_directory = os.path.join(run_directory, f'sample_{sample_index:03d}')
        os.makedirs(sample_directory, exist_ok=True)

        # Time the generation of this sample
        sample_start_time = time.time()

        if args.mode == 'ddim':
            sample, intermediates = ddim_sampler.ddim_sample_loop(
                model, single_sample_shape,
                return_intermediates=True,
                intermediate_steps=intermediate_steps,
            )
        else:
            sample, intermediates = ddpm.p_sample_loop(
                model, single_sample_shape,
                return_intermediates=True,
                intermediate_steps=intermediate_steps,
            )

        sample_elapsed_time = time.time() - sample_start_time
        per_sample_times.append(sample_elapsed_time)
        all_samples.append(sample)

        # Save final generated image
        final_image_path = os.path.join(sample_directory, 'final.pdf')
        save_single_image(sample, final_image_path)

        # Save each denoising step as an individual image
        save_individual_denoising_steps(intermediates, sample_directory)

        # Save denoising progression strip for this sample
        progression_path = os.path.join(sample_directory, 'denoising_progression.pdf')
        save_denoising_progression(intermediates, progression_path)

        print(f"  Sample {sample_index:3d}: {sample_elapsed_time:.3f}s "
              f"({len(intermediates)} denoising steps saved)")

    # Save grid of all final samples
    all_samples_tensor = torch.cat(all_samples, dim=0)
    grid_path = os.path.join(run_directory, 'grid.pdf')
    save_images(all_samples_tensor, grid_path, nrow=args.nrow)
    print(f"\nSample grid saved: {grid_path}")

    # Write profiling information
    profile_path = os.path.join(run_directory, 'profile.txt')
    write_generation_profile(
        profile_path=profile_path,
        per_sample_times=per_sample_times,
        mode=args.mode,
        ddim_steps=args.ddim_steps if args.mode == 'ddim' else None,
        eta=args.eta if args.mode == 'ddim' else None,
        total_timesteps=ddpm.timesteps,
    )
    print(f"Profile saved: {profile_path}")

    # Print summary
    total_time = sum(per_sample_times)
    average_time = total_time / len(per_sample_times)
    print(f"\nTotal: {total_time:.3f}s | Average per sample: {average_time:.3f}s")
    print(f"All outputs saved to {run_directory}")


if __name__ == '__main__':
    main()
