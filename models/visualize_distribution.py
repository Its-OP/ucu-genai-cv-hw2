"""
UMAP Distribution Visualization for DDPM/DDIM Generated Samples.

Compares the distribution of freshly generated MNIST samples against real MNIST
digits in UNet encoder feature space using UMAP (McInnes et al. 2018).

Features are extracted from the UNet bottleneck (mid_block output) via PyTorch
forward hooks. The bottleneck captures the model's compressed representation
after attention-equipped processing, giving the richest feature space.

For feature extraction, timestep t=0 is used (clean image encoding), which
gives the model's representation of the actual image content.

Usage:
    python -m models.visualize_distribution --checkpoint path/to/checkpoint.pt
    python -m models.visualize_distribution --checkpoint path/to/checkpoint.pt --mode ddim --ddim_steps 50
    python -m models.visualize_distribution --checkpoint path/to/checkpoint.pt --num_generated 2000 --num_real 2000
"""
import argparse
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from tqdm import tqdm

from models.generate import get_device, load_checkpoint, write_profile
from models.ddim import DDIMSampler
from models.utils import save_images
from data import test_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='UMAP visualization of generated vs real MNIST distribution',
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to .pt checkpoint file',
    )
    parser.add_argument(
        '--num_generated', type=int, default=1000,
        help='Number of samples to generate',
    )
    parser.add_argument(
        '--num_real', type=int, default=1000,
        help='Number of real MNIST samples to use',
    )
    parser.add_argument(
        '--output_dir', type=str, default='./distribution_plots',
        help='Base directory for outputs (a timestamped subfolder is created)',
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Force device (cuda, mps, cpu). Auto-detects if not specified.',
    )
    parser.add_argument(
        '--mode', type=str, default='ddim', choices=['ddpm', 'ddim'],
        help='Sampling mode: ddpm (full T steps) or ddim (fewer steps)',
    )
    parser.add_argument(
        '--ddim_steps', type=int, default=50,
        help='Number of DDIM sampling steps (only used with --mode ddim)',
    )
    parser.add_argument(
        '--eta', type=float, default=0.0,
        help='DDIM stochasticity: 0.0=deterministic, 1.0=DDPM-like (only with --mode ddim)',
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size for generation and feature extraction',
    )
    parser.add_argument(
        '--umap_neighbors', type=int, default=15,
        help='UMAP n_neighbors: larger values capture more global structure',
    )
    parser.add_argument(
        '--umap_min_dist', type=float, default=0.1,
        help='UMAP min_dist: smaller values create tighter clusters',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility',
    )
    return parser.parse_args()


@torch.no_grad()
def extract_bottleneck_features(model, images, device, batch_size=64):
    """Extract UNet bottleneck features from a batch of images using forward hooks.

    Registers a forward hook on model.model.mid_block (the diffusers UNet2DModel
    bottleneck) to capture the encoder representation. Uses timestep t=0 to encode
    clean images without noise. The hook output is global-average-pooled over spatial
    dimensions to produce a fixed-size feature vector per image.

    Feature extraction point:
        After mid_block (attention + ResNet at lowest resolution)
        Shape before pooling: (batch, bottleneck_channels, H_low, W_low)
        Shape after pooling:  (batch, bottleneck_channels)

    Args:
        model: UNet model instance (wraps diffusers.UNet2DModel as model.model).
        images: Tensor of shape (N, C, H, W) with images in [-1, 1] range.
        device: Torch device to use for computation.
        batch_size: Number of images to process in each forward pass.

    Returns:
        Feature tensor of shape (N, bottleneck_channels), where bottleneck_channels
        is the last entry of block_out_channels (e.g. 96 for default config).
    """
    model.eval()

    all_features = []

    for batch_start in range(0, len(images), batch_size):
        batch_images = images[batch_start:batch_start + batch_size].to(device)
        batch_size_actual = batch_images.shape[0]

        # Storage for hooked features
        captured_features = {}

        def hook_function(module, input_tensors, output_tensor):
            """Capture mid_block output during forward pass."""
            captured_features['bottleneck'] = output_tensor.detach()

        # Register hook on the UNet2DModel mid_block (bottleneck)
        hook_handle = model.model.mid_block.register_forward_hook(hook_function)

        try:
            # Run forward pass with timestep t=0 (clean image, no noise)
            # This gives the model's learned representation of the image content
            timestep_zero = torch.zeros(batch_size_actual, device=device, dtype=torch.long)
            model(batch_images, timestep_zero)

            # Extract bottleneck features and apply global average pooling
            # bottleneck shape: (batch, channels, height, width) e.g. (B, 96, 4, 4)
            # After pooling: (batch, channels) e.g. (B, 96)
            bottleneck_output = captured_features['bottleneck']
            pooled_features = bottleneck_output.mean(dim=[2, 3])  # Global average pool over spatial dims
            all_features.append(pooled_features.cpu())
        finally:
            hook_handle.remove()

    return torch.cat(all_features, dim=0)


@torch.no_grad()
def generate_samples_batched(model, ddpm, sampler, num_samples, device, batch_size, mode):
    """Generate fresh MNIST samples in batches using DDPM or DDIM sampling.

    Each batch is timed individually. The per-sample time is estimated by dividing
    the batch time by the batch size (samples within a batch are generated in parallel).

    Args:
        model: UNet noise prediction model.
        ddpm: DDPM diffusion scheduler instance.
        sampler: DDIMSampler instance (only used when mode='ddim'), or None.
        num_samples: Total number of samples to generate.
        device: Torch device for generation.
        batch_size: Maximum number of samples per generation batch.
        mode: Sampling mode, either 'ddpm' or 'ddim'.

    Returns:
        Tuple of (samples, per_sample_times) where:
            samples: Tensor of shape (num_samples, 1, 28, 28) in [-1, 1] range.
            per_sample_times: List of floats with estimated per-sample generation time.
    """
    model.eval()
    all_samples = []
    per_sample_times = []
    samples_remaining = num_samples

    while samples_remaining > 0:
        current_batch_size = min(batch_size, samples_remaining)
        sample_shape = (current_batch_size, 1, 28, 28)

        batch_start_time = time.time()
        if mode == 'ddim' and sampler is not None:
            batch_samples = sampler.ddim_sample_loop(model, sample_shape)
        else:
            batch_samples = ddpm.p_sample_loop(model, sample_shape)
        batch_elapsed_time = time.time() - batch_start_time

        # Estimate per-sample time by dividing batch time evenly
        time_per_sample = batch_elapsed_time / current_batch_size
        per_sample_times.extend([time_per_sample] * current_batch_size)

        all_samples.append(batch_samples.cpu())
        samples_remaining -= current_batch_size
        print(f"  Generated {num_samples - samples_remaining}/{num_samples} samples "
              f"({batch_elapsed_time:.2f}s for batch of {current_batch_size})")

    return torch.cat(all_samples, dim=0), per_sample_times


def load_real_samples(num_samples):
    """Load real MNIST test set samples with their labels.

    Args:
        num_samples: Number of real samples to load from the test set.

    Returns:
        Tuple of (images, labels) where images has shape (N, 1, H, W) in [-1, 1]
        and labels has shape (N,) with digit classes 0-9.
    """
    # Use at most the available test set size
    actual_count = min(num_samples, len(test_dataset))

    images = []
    labels = []
    for index in range(actual_count):
        image, label = test_dataset[index]
        images.append(image)
        labels.append(label)

    images_tensor = torch.stack(images, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return images_tensor, labels_tensor


def create_umap_plot(
    real_features,
    generated_features,
    real_labels,
    output_path,
    umap_neighbors=15,
    umap_min_dist=0.1,
):
    """Create a two-panel UMAP scatter plot comparing real vs generated distributions.

    UMAP (McInnes et al. 2018) reduces high-dimensional UNet bottleneck features
    to 2D for visualization. It preserves both local neighborhood structure and
    global cluster relationships.

    Left panel:  All points colored by source (real=blue, generated=red)
    Right panel: Real points colored by digit class (0-9), generated in gray

    Args:
        real_features: Numpy array of shape (N_real, feature_dim) with real sample features.
        generated_features: Numpy array of shape (N_gen, feature_dim) with generated features.
        real_labels: Numpy array of shape (N_real,) with digit class labels (0-9).
        output_path: File path for saving the plot (PDF for Overleaf import).
        umap_neighbors: UMAP n_neighbors parameter (controls local vs global balance).
        umap_min_dist: UMAP min_dist parameter (controls cluster tightness).
    """
    # Combine features for joint UMAP embedding
    combined_features = np.concatenate([real_features, generated_features], axis=0)
    num_real = len(real_features)
    num_generated = len(generated_features)

    print(f"Running UMAP on {len(combined_features)} samples "
          f"(n_neighbors={umap_neighbors}, min_dist={umap_min_dist})...")

    # Fit UMAP: reduces feature_dim → 2 dimensions
    # n_neighbors controls local neighborhood size (larger = more global structure)
    # min_dist controls minimum distance between points (smaller = tighter clusters)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric='euclidean',
        random_state=42,
    )
    embedding = reducer.fit_transform(combined_features)

    # Split embedding back into real and generated portions
    real_embedding = embedding[:num_real]
    generated_embedding = embedding[num_real:]

    # Create two-panel figure
    figure, (axis_source, axis_class) = plt.subplots(1, 2, figsize=(20, 8))

    # --- Left panel: Real vs Generated source comparison ---
    axis_source.scatter(
        generated_embedding[:, 0], generated_embedding[:, 1],
        c='red', alpha=0.3, s=8, label=f'Generated ({num_generated})',
        rasterized=True,
    )
    axis_source.scatter(
        real_embedding[:, 0], real_embedding[:, 1],
        c='blue', alpha=0.3, s=8, label=f'Real ({num_real})',
        rasterized=True,
    )
    axis_source.set_title('Real vs Generated Distribution', fontsize=14)
    axis_source.set_xlabel('UMAP Component 1')
    axis_source.set_ylabel('UMAP Component 2')
    axis_source.legend(fontsize=12, markerscale=3)

    # --- Right panel: Digit class coloring (real colored, generated gray) ---
    axis_class.scatter(
        generated_embedding[:, 0], generated_embedding[:, 1],
        c='lightgray', alpha=0.2, s=8, label='Generated',
        rasterized=True,
    )

    # Plot real samples colored by digit class using a distinct colormap
    colormap = plt.cm.tab10
    for digit in range(10):
        digit_mask = real_labels == digit
        if digit_mask.any():
            axis_class.scatter(
                real_embedding[digit_mask, 0], real_embedding[digit_mask, 1],
                c=[colormap(digit)], alpha=0.5, s=10, label=f'Digit {digit}',
                rasterized=True,
            )

    axis_class.set_title('Real Samples by Digit Class', fontsize=14)
    axis_class.set_xlabel('UMAP Component 1')
    axis_class.set_ylabel('UMAP Component 2')
    axis_class.legend(fontsize=9, markerscale=3, ncol=2, loc='best')

    plt.suptitle('UMAP Projection of UNet Bottleneck Features', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"UMAP plot saved: {output_path}")


def write_metadata(
    metadata_path,
    num_real,
    num_generated,
    mode,
    ddim_steps,
    eta,
    umap_neighbors,
    umap_min_dist,
    feature_dim,
    generation_time,
    feature_extraction_time,
    umap_time,
    total_time,
):
    """Write visualization metadata and timing to a text file.

    Args:
        metadata_path: Path to the output metadata.txt file.
        num_real: Number of real MNIST samples used.
        num_generated: Number of generated samples.
        mode: Sampling mode ('ddpm' or 'ddim').
        ddim_steps: Number of DDIM steps (None for ddpm mode).
        eta: DDIM eta parameter (None for ddpm mode).
        umap_neighbors: UMAP n_neighbors parameter.
        umap_min_dist: UMAP min_dist parameter.
        feature_dim: Dimensionality of bottleneck features.
        generation_time: Time spent generating samples (seconds).
        feature_extraction_time: Time spent extracting features (seconds).
        umap_time: Time spent on UMAP fitting and transformation (seconds).
        total_time: Total wall-clock time (seconds).
    """
    with open(metadata_path, 'w') as file:
        file.write("Distribution Visualization Metadata\n")
        file.write("=" * 50 + "\n\n")

        file.write("Configuration\n")
        file.write("-" * 50 + "\n")
        file.write(f"Sampling mode:       {mode}\n")
        if mode == 'ddim':
            file.write(f"DDIM steps:          {ddim_steps}\n")
            file.write(f"DDIM eta:            {eta}\n")
        file.write(f"Real samples:        {num_real}\n")
        file.write(f"Generated samples:   {num_generated}\n")
        file.write(f"Feature dimension:   {feature_dim}\n")
        file.write(f"UMAP n_neighbors:    {umap_neighbors}\n")
        file.write(f"UMAP min_dist:       {umap_min_dist}\n\n")

        file.write("Timing\n")
        file.write("-" * 50 + "\n")
        file.write(f"Sample generation:   {generation_time:.2f}s\n")
        file.write(f"Feature extraction:  {feature_extraction_time:.2f}s\n")
        file.write(f"UMAP computation:    {umap_time:.2f}s\n")
        file.write(f"Total time:          {total_time:.2f}s\n")


def main():
    args = parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_directory = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_directory, exist_ok=True)

    total_start_time = time.time()

    # Load model from checkpoint
    device = get_device(args.device)
    print(f"Using device: {device}")

    model, ddpm, config = load_checkpoint(args.checkpoint, device)

    # Setup sampler for generation
    sampler = None
    if args.mode == 'ddim':
        sampler = DDIMSampler(
            ddpm, ddim_timesteps=args.ddim_steps, eta=args.eta,
        ).to(device)
        print(f"\nGenerating {args.num_generated} samples using DDIM "
              f"({args.ddim_steps} steps, eta={args.eta})")
    else:
        print(f"\nGenerating {args.num_generated} samples using DDPM "
              f"({ddpm.timesteps} steps)")

    # Step 1: Generate fresh samples with per-sample timing
    print("\n--- Step 1: Generating samples ---")
    generation_start_time = time.time()
    generated_images, per_sample_times = generate_samples_batched(
        model, ddpm, sampler, args.num_generated, device, args.batch_size, args.mode,
    )
    generation_time = time.time() - generation_start_time
    print(f"Generation complete: {generation_time:.2f}s")

    # Step 2: Save sample grid and profiling info alongside the UMAP plot
    print("\n--- Step 2: Saving sample grid and profile ---")
    nrow = int(args.num_generated ** 0.5)  # Approximate square grid
    nrow = max(nrow, 1)
    grid_path = os.path.join(run_directory, 'grid.pdf')
    save_images(generated_images, grid_path, nrow=nrow)
    print(f"Sample grid saved: {grid_path}")

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

    # Step 3: Load real MNIST samples
    print("\n--- Step 3: Loading real MNIST samples ---")
    real_images, real_labels = load_real_samples(args.num_real)
    print(f"Loaded {len(real_images)} real samples with labels")

    # Step 4: Extract bottleneck features
    print("\n--- Step 4: Extracting bottleneck features ---")
    feature_extraction_start_time = time.time()

    print("  Extracting features from real images...")
    real_features = extract_bottleneck_features(
        model, real_images, device, batch_size=args.batch_size,
    )
    print(f"  Real features shape: {real_features.shape}")

    print("  Extracting features from generated images...")
    generated_features = extract_bottleneck_features(
        model, generated_images, device, batch_size=args.batch_size,
    )
    print(f"  Generated features shape: {generated_features.shape}")

    feature_extraction_time = time.time() - feature_extraction_start_time
    print(f"Feature extraction complete: {feature_extraction_time:.2f}s")

    # Step 5: Run UMAP and create visualization
    print("\n--- Step 5: Running UMAP and creating plot ---")
    umap_start_time = time.time()

    plot_path = os.path.join(run_directory, 'real_vs_generated.pdf')
    create_umap_plot(
        real_features=real_features.numpy(),
        generated_features=generated_features.numpy(),
        real_labels=real_labels.numpy(),
        output_path=plot_path,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
    )

    umap_time = time.time() - umap_start_time
    print(f"UMAP computation complete: {umap_time:.2f}s")

    total_time = time.time() - total_start_time

    # Step 6: Save metadata
    metadata_path = os.path.join(run_directory, 'metadata.txt')
    write_metadata(
        metadata_path=metadata_path,
        num_real=len(real_images),
        num_generated=len(generated_images),
        mode=args.mode,
        ddim_steps=args.ddim_steps if args.mode == 'ddim' else None,
        eta=args.eta if args.mode == 'ddim' else None,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        feature_dim=real_features.shape[1],
        generation_time=generation_time,
        feature_extraction_time=feature_extraction_time,
        umap_time=umap_time,
        total_time=total_time,
    )
    print(f"Metadata saved: {metadata_path}")

    print(f"\nTotal time: {total_time:.2f}s")
    print(f"All outputs saved to {run_directory}")


if __name__ == '__main__':
    main()
