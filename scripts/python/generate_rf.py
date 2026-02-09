"""
Rectified Flow Sample Generation Script (pixel space).

Loads a trained Rectified Flow checkpoint and generates MNIST digit samples
using Euler ODE integration (Liu et al. 2022).

Each sample is generated individually with per-sample timing, denoising
step visualization (at every 10% of the process), and dedicated subfolders.
Output directories are timestamped to avoid overwriting previous runs.

Usage:
    python -m scripts.python.generate_rf --checkpoint path/to/checkpoint.pt
    python -m scripts.python.generate_rf --checkpoint path/to/checkpoint.pt --sampling_steps 100
    python -m scripts.python.generate_rf --checkpoint path/to/checkpoint.pt --num_samples 25 --nrow 5
"""
import argparse
import os
import time
from datetime import datetime

import torch

from models.utils import (
    get_device,
    load_rectified_flow_checkpoint,
    compute_denoising_intermediate_steps_rf,
    save_single_image,
    save_individual_denoising_steps,
    save_denoising_progression,
    save_images,
    write_generation_profile,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate samples from a trained Rectified Flow checkpoint (pixel space)'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pt checkpoint file')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to generate')
    parser.add_argument('--sampling_steps', type=int, default=None,
                        help='Number of Euler steps (default: from checkpoint config)')
    parser.add_argument('--output_dir', type=str, default='./generated_samples',
                        help='Base directory for outputs (a timestamped subfolder is created)')
    parser.add_argument('--nrow', type=int, default=4,
                        help='Number of images per row in the grid')
    parser.add_argument('--device', type=str, default=None,
                        help='Force device (cuda, mps, cpu). Auto-detects if not specified.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_directory = os.path.join(args.output_dir, f'{timestamp}-rf')
    os.makedirs(run_directory, exist_ok=True)

    # Load model from checkpoint
    device = get_device(args.device)
    print(f"Using device: {device}")

    model, rectified_flow, config = load_rectified_flow_checkpoint(args.checkpoint, device)

    # Override sampling steps if specified
    if args.sampling_steps is not None:
        sampling_steps = args.sampling_steps
    else:
        sampling_steps = rectified_flow.number_of_sampling_steps

    image_channels = config['image_channels']
    single_sample_shape = (1, image_channels, 28, 28)

    # Compute intermediate step indices for visualization
    intermediate_steps = compute_denoising_intermediate_steps_rf(sampling_steps)

    print(f"\nGenerating {args.num_samples} samples using Rectified Flow "
          f"(Euler, {sampling_steps} steps)")
    print(f"Output directory: {run_directory}")
    print(f"Denoising snapshots at {len(intermediate_steps)} points (every 10%)\n")

    # Generate each sample individually with timing and denoising visualization
    all_samples = []
    per_sample_times = []

    for sample_index in range(args.num_samples):
        sample_directory = os.path.join(run_directory, f'sample_{sample_index:03d}')
        os.makedirs(sample_directory, exist_ok=True)

        sample_start_time = time.time()

        sample, intermediates = rectified_flow.euler_sample_loop(
            model, single_sample_shape,
            number_of_steps=sampling_steps,
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

        # Save denoising progression strip
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
        mode='rf',
        rf_steps=sampling_steps,
    )
    print(f"Profile saved: {profile_path}")

    # Print summary
    total_time = sum(per_sample_times)
    average_time = total_time / len(per_sample_times)
    print(f"\nTotal: {total_time:.3f}s | Average per sample: {average_time:.3f}s")
    print(f"All outputs saved to {run_directory}")


if __name__ == '__main__':
    main()
