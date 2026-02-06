"""
Diffusion Model Sample Generation Script (DDPM / DDIM).

Loads a trained checkpoint and generates MNIST digit samples using either
DDPM (Ho et al. 2020) or DDIM (Song et al. 2020) sampling.

Usage:
    python -m models.generate --checkpoint path/to/checkpoint.pt
    python -m models.generate --checkpoint path/to/checkpoint.pt --mode ddim --ddim_steps 50
    python -m models.generate --checkpoint path/to/checkpoint.pt --mode ddim --ddim_steps 100 --eta 0.5
    python -m models.generate --checkpoint path/to/checkpoint.pt --num_samples 25 --nrow 5
    python -m models.generate --checkpoint path/to/checkpoint.pt --show_denoising
"""
import argparse
import os

import matplotlib.pyplot as plt
import torch

from models.unet import UNet
from models.ddpm import DDPM
from models.ddim import DDIMSampler
from models.utils import save_images, save_denoising_progression


def parse_args():
    parser = argparse.ArgumentParser(description='Generate samples from a trained DDPM checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pt checkpoint file')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='./generated_samples',
                        help='Directory to save generated images')
    parser.add_argument('--nrow', type=int, default=4,
                        help='Number of images per row in the grid')
    parser.add_argument('--show_denoising', action='store_true',
                        help='Also save denoising progression visualization')
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


def save_individual_samples(samples, output_dir):
    """Save each generated sample as an individual PNG file.

    Args:
        samples: Tensor of shape (num_samples, channels, height, width) in [-1, 1].
        output_dir: Directory to save individual images.
    """
    for sample_index in range(samples.shape[0]):
        # Denormalize from [-1, 1] to [0, 1]
        image = (samples[sample_index, 0] + 1) / 2
        image = image.clamp(0, 1).cpu().numpy()

        plt.figure(figsize=(3, 3))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.savefig(
            f'{output_dir}/sample_{sample_index:03d}.png',
            bbox_inches='tight', dpi=100,
        )
        plt.close()


def main():
    args = parse_args()

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model from checkpoint
    device = get_device(args.device)
    print(f"Using device: {device}")

    model, ddpm, config = load_checkpoint(args.checkpoint, device)

    # Select sampling method: DDPM (full T steps) or DDIM (fewer steps)
    image_channels = config['image_channels']
    sample_shape = (args.num_samples, image_channels, 28, 28)

    if args.mode == 'ddim':
        ddim_sampler = DDIMSampler(
            ddpm, ddim_timesteps=args.ddim_steps, eta=args.eta,
        ).to(device)
        print(f"\nGenerating {args.num_samples} samples using DDIM "
              f"({args.ddim_steps} steps, eta={args.eta})...")
        samples = ddim_sampler.ddim_sample_loop(model, sample_shape)
    else:
        print(f"\nGenerating {args.num_samples} samples using DDPM "
              f"({ddpm.timesteps} steps)...")
        samples = ddpm.p_sample_loop(model, sample_shape)

    # Save grid of all samples
    grid_path = f'{args.output_dir}/grid.png'
    save_images(samples, grid_path, nrow=args.nrow)
    print(f"Sample grid saved: {grid_path}")

    # Save individual samples
    save_individual_samples(samples, args.output_dir)
    print(f"Individual samples saved: {args.output_dir}/sample_000.png ... sample_{args.num_samples - 1:03d}.png")

    # Optionally generate denoising progression
    if args.show_denoising:
        print("\nGenerating denoising progression...")
        progression_shape = (1, image_channels, 28, 28)

        if args.mode == 'ddim':
            _, intermediates = ddim_sampler.ddim_sample_loop(
                model, progression_shape, return_intermediates=True,
            )
        else:
            _, intermediates = ddpm.p_sample_loop(
                model, progression_shape, return_intermediates=True,
            )

        progression_path = f'{args.output_dir}/denoising_progression.png'
        save_denoising_progression(intermediates, progression_path)
        print(f"Denoising progression saved: {progression_path}")

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == '__main__':
    main()
