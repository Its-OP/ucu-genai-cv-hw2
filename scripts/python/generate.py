"""
Diffusion Model Sample Generation Script (DDPM / DDIM).

Loads a trained checkpoint and generates MNIST digit samples using either
DDPM (Ho et al. 2020) or DDIM (Song et al. 2020) sampling.

Each sample is generated individually with per-sample timing, denoising
step visualization (at every 10% of the process), and dedicated subfolders.
Output directories are timestamped to avoid overwriting previous runs.

Usage:
    python -m scripts.python.generate --checkpoint path/to/checkpoint.pt
    python -m scripts.python.generate --checkpoint path/to/checkpoint.pt --mode ddim --ddim_steps 50
    python -m scripts.python.generate --checkpoint path/to/checkpoint.pt --mode ddim --ddim_steps 100 --eta 0.5
    python -m scripts.python.generate --checkpoint path/to/checkpoint.pt --num_samples 25 --nrow 5
"""
import argparse
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from models.unet import UNet
from models.ddpm import DDPM
from models.ddim import DDIMSampler
from models.utils import save_images


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


def get_device(requested_device=None):
    """Get the best available device, or use the requested one.

    Args:
        requested_device: Optional device string to force (e.g., 'cuda', 'mps', 'cpu').

    Returns:
        torch.device for computation.
    """
    if requested_device is not None:
        return torch.device(requested_device)

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_checkpoint(checkpoint_path, device):
    """Load a DDPM checkpoint and reconstruct the model.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load the model onto.

    Returns:
        Tuple of (model, ddpm, config) ready for generation.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    config = checkpoint['config']
    print(f"  Model config: image_channels={config['image_channels']}, "
          f"base_channels={config['base_channels']}, timesteps={config['timesteps']}")
    print(f"  Trained for {checkpoint['epoch'] + 1} epochs "
          f"(train_loss={checkpoint['train_loss']:.6f}, eval_loss={checkpoint['eval_loss']:.6f})")

    # Backward compatibility: old checkpoints lack architecture fields,
    # fall back to the original DDPM defaults (Ho et al. 2020)
    channel_multipliers = tuple(config.get('channel_multipliers', (1, 2, 4, 4)))
    layers_per_block = config.get('layers_per_block', 2)
    attention_levels = tuple(config.get('attention_levels', (False, False, True, True)))

    # Reconstruct model architecture from saved config
    model = UNet(
        image_channels=config['image_channels'],
        base_channels=config['base_channels'],
        channel_multipliers=channel_multipliers,
        layers_per_block=layers_per_block,
        attention_levels=attention_levels,
    ).to(device)

    # Load EMA weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    number_of_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"  Model parameters: {number_of_parameters:,}")

    # Reconstruct DDPM diffusion scheduler
    ddpm = DDPM(timesteps=config['timesteps']).to(device)

    return model, ddpm, config


def compute_denoising_intermediate_steps_ddpm(total_timesteps):
    """Compute timestep indices at every 10% of the DDPM denoising process.

    For DDPM, the loop runs from timestep (T-1) down to 0. We capture snapshots
    at 0%, 10%, 20%, ..., 100% of progress (i.e., at timesteps T-1, ~90%*T, ..., 0).

    Args:
        total_timesteps: Total number of DDPM timesteps (T).

    Returns:
        List of integer timestep indices at which to record intermediates.
    """
    intermediate_steps = []
    for percentage in range(0, 101, 10):
        # percentage=0 means the very start (timestep T-1, pure noise)
        # percentage=100 means the very end (timestep 0, clean image)
        step_index = int((total_timesteps - 1) * (1.0 - percentage / 100.0))
        intermediate_steps.append(step_index)
    # Remove duplicates while preserving order (can happen at small T)
    seen = set()
    unique_steps = []
    for step in intermediate_steps:
        if step not in seen:
            seen.add(step)
            unique_steps.append(step)
    return unique_steps


def compute_denoising_intermediate_steps_ddim(timestep_sequence):
    """Compute timestep indices at every 10% of the DDIM denoising process.

    For DDIM, the loop runs through a subsequence in reverse. We capture snapshots
    at 0%, 10%, 20%, ..., 100% of progress through that reversed subsequence.

    Args:
        timestep_sequence: The DDIM timestep subsequence tensor (ascending order).

    Returns:
        List of integer timestep indices at which to record intermediates.
    """
    reversed_sequence = torch.flip(timestep_sequence, [0])
    total_steps = len(reversed_sequence)
    intermediate_steps = []
    for percentage in range(0, 101, 10):
        # Map percentage of progress to an index in the reversed sequence
        step_index = min(int(percentage / 100.0 * (total_steps - 1)), total_steps - 1)
        timestep_value = reversed_sequence[step_index].item()
        intermediate_steps.append(timestep_value)
    # Remove duplicates while preserving order
    seen = set()
    unique_steps = []
    for step in intermediate_steps:
        if step not in seen:
            seen.add(step)
            unique_steps.append(step)
    return unique_steps


def save_single_image(tensor, path):
    """Save a single-channel image tensor as a PDF file (for Overleaf import).

    Args:
        tensor: Image tensor of shape (1, 1, H, W) in [-1, 1].
        path: Output file path.
    """
    # Denormalize from [-1, 1] to [0, 1]
    image = (tensor[0, 0] + 1) / 2
    image = image.clamp(0, 1).cpu().numpy()

    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=100)
    plt.close()


def save_denoising_steps(intermediates, sample_directory):
    """Save each denoising step as an individual PNG in the sample directory.

    Args:
        intermediates: List of (timestep, tensor) tuples from the sampling loop.
        sample_directory: Directory path where step images are saved.
    """
    for step_number, (timestep, image_tensor) in enumerate(intermediates):
        step_path = os.path.join(sample_directory, f'step_{step_number:02d}_t{timestep:04d}.pdf')
        save_single_image(image_tensor, step_path)


def save_denoising_progression_strip(intermediates, path):
    """Save a horizontal strip showing the denoising progression for one sample.

    Args:
        intermediates: List of (timestep, tensor) tuples.
        path: Output file path for the strip image.
    """
    num_steps = len(intermediates)
    fig, axes = plt.subplots(1, num_steps, figsize=(2.5 * num_steps, 2.5))

    if num_steps == 1:
        axes = [axes]

    for index, (timestep, image_tensor) in enumerate(intermediates):
        image = (image_tensor[0, 0] + 1) / 2
        image = image.clamp(0, 1).cpu().numpy()
        axes[index].imshow(image, cmap='gray')
        axes[index].set_title(f't={timestep}', fontsize=9)
        axes[index].axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
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

    with open(profile_path, 'w') as file:
        file.write("Generation Profile\n")
        file.write("=" * 50 + "\n\n")

        # Sampling configuration
        file.write("Configuration\n")
        file.write("-" * 50 + "\n")
        file.write(f"Sampling mode:       {mode}\n")
        if mode == 'ddim':
            file.write(f"DDIM steps:          {ddim_steps}\n")
            file.write(f"DDIM eta:            {eta}\n")
        if total_timesteps is not None:
            file.write(f"Total DDPM timesteps: {total_timesteps}\n")
        file.write(f"Number of samples:   {len(per_sample_times)}\n\n")

        # Per-sample timing
        file.write("Per-sample timing\n")
        file.write("-" * 50 + "\n")
        for sample_index, sample_time in enumerate(per_sample_times):
            file.write(f"Sample {sample_index:3d}: {sample_time:.3f}s\n")

        # Summary
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

    # Create timestamped output directory to avoid overwriting previous runs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_directory = os.path.join(args.output_dir, f'{timestamp}-{args.mode}')
    os.makedirs(run_directory, exist_ok=True)

    # Load model from checkpoint
    device = get_device(args.device)
    print(f"Using device: {device}")

    model, ddpm, config = load_checkpoint(args.checkpoint, device)

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
        save_denoising_steps(intermediates, sample_directory)

        # Save denoising progression strip for this sample
        progression_path = os.path.join(sample_directory, 'denoising_progression.pdf')
        save_denoising_progression_strip(intermediates, progression_path)

        print(f"  Sample {sample_index:3d}: {sample_elapsed_time:.3f}s "
              f"({len(intermediates)} denoising steps saved)")

    # Save grid of all final samples
    all_samples_tensor = torch.cat(all_samples, dim=0)
    grid_path = os.path.join(run_directory, 'grid.pdf')
    save_images(all_samples_tensor, grid_path, nrow=args.nrow)
    print(f"\nSample grid saved: {grid_path}")

    # Write profiling information
    profile_path = os.path.join(run_directory, 'profile.txt')
    write_profile(
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
