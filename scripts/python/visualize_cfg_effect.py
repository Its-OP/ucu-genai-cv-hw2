"""
Visualize how Classifier-Free Guidance (CFG) strength affects class-conditional
image quality and diversity for both conditioning approaches.

Generates a grid of images where:
    - Rows = digit classes (0-9)
    - Columns = guidance scale values (w)
    - Each cell contains multiple samples to show intra-class diversity

This allows side-by-side comparison of how increasing the guidance scale w
trades off diversity (sample variety within a class) for quality (class
adherence / recognizability). At low w, samples are diverse but may lack
class identity; at high w, samples are sharper but more homogeneous.

Supports both conditioning approaches:
    1. Channel concatenation (ClassConditionedUNet)
    2. Cross-attention (CrossAttentionConditionedUNet)

If both model checkpoints are provided, generates comparison plots for both
in a single run.

CFG formula (Ho & Salimans 2022):
    ε_guided = ε_uncond + w × (ε_cond − ε_uncond)

where w=0 is pure unconditional, w=1 is standard conditional (no guidance),
and w>1 amplifies the class signal beyond standard conditioning.

Usage:
    # Compare both models:
    python -m scripts.python.visualize_cfg_effect \\
        --vae_checkpoint path/to/vae.pt \\
        --concat_checkpoint path/to/concat_unet.pt \\
        --cross_attention_checkpoint path/to/ca_unet.pt

    # Single model:
    python -m scripts.python.visualize_cfg_effect \\
        --vae_checkpoint path/to/vae.pt \\
        --concat_checkpoint path/to/concat_unet.pt

    # Custom guidance scales and sample count:
    python -m scripts.python.visualize_cfg_effect \\
        --vae_checkpoint path/to/vae.pt \\
        --concat_checkpoint path/to/concat_unet.pt \\
        --guidance_scales 0.0 1.0 3.0 5.0 10.0 \\
        --samples_per_cell 10
"""
import argparse
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from models.ddim import DDIMSampler
from models.classifier_free_guidance import (
    ClassConditionedUNet,
    CrossAttentionConditionedUNet,
    ClassifierFreeGuidanceWrapper,
)
from models.utils import (
    get_device,
    load_unet_checkpoint,
    load_vae_checkpoint,
)


# ── Default guidance scales to sweep ────────────────────────────────────────
DEFAULT_GUIDANCE_SCALES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 50.0, 100.0]
DEFAULT_SAMPLES_PER_CELL = 10
NUMBER_OF_CLASSES = 10  # MNIST digits 0-9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize how CFG guidance scale affects "
        "class-conditional generation quality and diversity."
    )

    # ── Model checkpoints ───────────────────────────────────────────────
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained VAE checkpoint (.pt file).",
    )
    parser.add_argument(
        "--concat_checkpoint",
        type=str,
        default=None,
        help="Path to channel-concatenation conditioned UNet checkpoint.",
    )
    parser.add_argument(
        "--cross_attention_checkpoint",
        type=str,
        default=None,
        help="Path to cross-attention conditioned UNet checkpoint.",
    )

    # ── Sweep configuration ─────────────────────────────────────────────
    parser.add_argument(
        "--guidance_scales",
        type=float,
        nargs="+",
        default=DEFAULT_GUIDANCE_SCALES,
        help=f"List of guidance scale values to sweep "
        f"(default: {DEFAULT_GUIDANCE_SCALES}).",
    )
    parser.add_argument(
        "--samples_per_cell",
        type=int,
        default=DEFAULT_SAMPLES_PER_CELL,
        help=f"Number of samples per (class, guidance_scale) combination "
        f"(default: {DEFAULT_SAMPLES_PER_CELL}).",
    )

    # ── Sampling configuration ──────────────────────────────────────────
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=100,
        help="Number of DDIM sampling steps (default: 100).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.05,
        help="DDIM stochasticity parameter η (default: 0.05).",
    )

    # ── Output ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_samples",
        help="Base directory for outputs (a timestamped subfolder is created).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (cuda, mps, cpu). Auto-detects if not specified.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42). "
        "All models share the same initial noise per cell.",
    )

    args = parser.parse_args()

    # Validate: at least one model checkpoint must be provided
    if args.concat_checkpoint is None and args.cross_attention_checkpoint is None:
        parser.error(
            "At least one of --concat_checkpoint or "
            "--cross_attention_checkpoint must be provided."
        )

    return args


@torch.no_grad()
def decode_latent_to_pixel(
    vae, latent: torch.Tensor, scaling_factor: float = 1.0,
) -> torch.Tensor:
    """
    Decode latent samples to 28x28 pixel images via the VAE decoder.

    Pipeline: z_scaled → z = z_scaled / scaling_factor → VAE.decode(z) → 32x32 → crop 28x28

    Args:
        vae: Frozen VAE decoder.
        latent: Scaled latent tensor, shape (B, latent_channels, 4, 4).
        scaling_factor: Latent scaling factor used during training.

    Returns:
        Pixel images, shape (B, 1, 28, 28), in [-1, 1].
    """
    unscaled_latent = latent / scaling_factor
    pixel_images = vae.decode(unscaled_latent)
    # Crop from 32x32 back to 28x28 (remove 2-pixel reflect padding)
    pixel_images = pixel_images[:, :, 2:30, 2:30]
    return pixel_images


@torch.no_grad()
def generate_samples_for_class(
    conditioned_unet,
    ddpm,
    vae,
    class_label: int,
    guidance_scale: float,
    number_of_samples: int,
    latent_shape: tuple,
    scaling_factor: float,
    sampling_steps: int,
    eta: float,
    device: torch.device,
    initial_noise: torch.Tensor = None,
) -> torch.Tensor:
    """
    Generate multiple samples for a single class at a given guidance scale.

    Uses DDIM sampling with the given number of steps and stochasticity η.
    Optionally accepts pre-generated initial noise for reproducibility
    across different guidance scales.

    Args:
        conditioned_unet: ClassConditionedUNet or CrossAttentionConditionedUNet.
        ddpm: DDPM diffusion scheduler (provides noise schedule buffers).
        vae: Frozen VAE for latent-to-pixel decoding.
        class_label: Target digit class (0-9).
        guidance_scale: CFG weight w.
        number_of_samples: How many samples to generate.
        latent_shape: Shape of a single latent sample (C, H, W).
        scaling_factor: Latent scaling factor from training config.
        sampling_steps: Number of DDIM denoising steps.
        eta: DDIM stochasticity parameter η.
        device: Torch device.
        initial_noise: Optional pre-generated noise tensor of shape
            (number_of_samples, *latent_shape) for reproducible comparisons.

    Returns:
        Decoded pixel images, shape (number_of_samples, 1, 28, 28), in [-1, 1].
    """
    # Create CFG wrapper with the specified guidance scale
    cfg_wrapper = ClassifierFreeGuidanceWrapper(
        conditioned_unet, guidance_scale=guidance_scale,
    )

    # Set class label for all samples in the batch
    label_tensor = torch.full(
        (number_of_samples,), class_label, dtype=torch.long, device=device,
    )
    conditioned_unet.set_class_labels(label_tensor)

    # Setup DDIM sampler
    sampler = DDIMSampler(
        ddpm, ddim_timesteps=sampling_steps, eta=eta,
    ).to(device)

    batch_latent_shape = (number_of_samples, *latent_shape)

    # Generate latent samples via DDIM.
    # clip_denoised=False because latent values are not bounded to [-1, 1].
    latent_samples = sampler.ddim_sample_loop(
        cfg_wrapper,
        batch_latent_shape,
        return_intermediates=False,
        clip_denoised=False,
        initial_noise=initial_noise,
    )

    # Decode latents to pixel space
    pixel_images = decode_latent_to_pixel(vae, latent_samples, scaling_factor)

    return pixel_images


def create_cfg_comparison_grid(
    all_images: dict,
    guidance_scales: list,
    samples_per_cell: int,
    model_label: str,
    output_path: str,
):
    """
    Create and save a large grid visualization of CFG effect.

    Layout:
        - Rows: digit classes 0-9
        - Columns: guidance scale values
        - Each cell: horizontal strip of samples_per_cell images

    The grid visually demonstrates how increasing w sharpens class identity
    at the cost of intra-class diversity.

    Args:
        all_images: Dict mapping (class_label, guidance_scale) to pixel
            tensor of shape (samples_per_cell, 1, 28, 28).
        guidance_scales: List of guidance scale values used.
        samples_per_cell: Number of samples per cell.
        model_label: Human-readable model name for the title.
        output_path: File path for the output image.
    """
    number_of_rows = NUMBER_OF_CLASSES
    number_of_columns = len(guidance_scales)

    # Each cell shows samples_per_cell images side by side
    cell_width_inches = samples_per_cell * 0.45
    cell_height_inches = 0.6
    # Extra space for labels
    left_margin_inches = 0.6
    top_margin_inches = 0.9

    figure_width = left_margin_inches + number_of_columns * cell_width_inches
    figure_height = top_margin_inches + number_of_rows * cell_height_inches

    figure = plt.figure(figsize=(figure_width, figure_height))

    # Use GridSpec for precise control over cell placement
    grid_specification = gridspec.GridSpec(
        number_of_rows,
        number_of_columns,
        figure=figure,
        left=left_margin_inches / figure_width,
        right=1.0 - 0.05,
        top=1.0 - top_margin_inches / figure_height,
        bottom=0.02,
        wspace=0.05,
        hspace=0.15,
    )

    for row_index in range(number_of_rows):
        class_label = row_index

        for column_index, guidance_scale in enumerate(guidance_scales):
            axis = figure.add_subplot(
                grid_specification[row_index, column_index]
            )

            # Retrieve images for this (class, w) cell
            pixel_tensor = all_images[(class_label, guidance_scale)]

            # Denormalize from [-1, 1] to [0, 1]
            images_normalized = (pixel_tensor + 1.0) / 2.0
            images_normalized = images_normalized.clamp(0, 1)

            # Concatenate samples horizontally into a single strip
            # Shape: (samples_per_cell, 1, 28, 28) → (28, samples_per_cell * 28)
            strip = torch.cat(
                [images_normalized[sample_index, 0]
                 for sample_index in range(images_normalized.shape[0])],
                dim=1,
            )
            strip_numpy = strip.cpu().numpy()

            axis.imshow(strip_numpy, cmap="gray", aspect="auto")
            axis.set_xticks([])
            axis.set_yticks([])

            # Column headers: guidance scale values
            if row_index == 0:
                axis.set_title(f"w={guidance_scale}", fontsize=9, pad=3)

            # Row labels: digit class
            if column_index == 0:
                axis.set_ylabel(
                    f"{class_label}", fontsize=10, rotation=0,
                    labelpad=15, va="center",
                )

    # Overall title
    figure.suptitle(
        f"CFG Guidance Scale Effect — {model_label}\n"
        f"({samples_per_cell} samples per cell)",
        fontsize=13,
        fontweight="bold",
        y=1.0 - 0.15 / figure_height,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_diversity_plot(
    all_images: dict,
    guidance_scales: list,
    model_label: str,
    output_path: str,
):
    """
    Plot intra-class pixel-space standard deviation vs. guidance scale.

    For each (class, w) combination, computes the per-pixel standard
    deviation across samples, then averages over all pixels. This gives
    a scalar measure of sample diversity.

    Higher std → more diverse samples (different pixel patterns).
    Lower std → more homogeneous samples (similar pixel patterns).

    As guidance scale increases, we expect diversity to decrease because
    CFG amplifies the conditional signal, collapsing samples toward a
    prototypical class appearance.

    Args:
        all_images: Dict mapping (class_label, guidance_scale) to pixel
            tensor of shape (N, 1, 28, 28).
        guidance_scales: List of guidance scale values.
        model_label: Human-readable model name for the plot title.
        output_path: File path for the output plot.
    """
    figure, axis = plt.subplots(1, 1, figsize=(10, 6))
    colormap = plt.cm.tab10

    for class_label in range(NUMBER_OF_CLASSES):
        diversity_values = []

        for guidance_scale in guidance_scales:
            pixel_tensor = all_images[(class_label, guidance_scale)]
            # Denormalize to [0, 1]
            images_01 = (pixel_tensor + 1.0) / 2.0
            images_01 = images_01.clamp(0, 1)

            # Per-pixel standard deviation across samples, then average
            # std(images_01, dim=0): std across the N samples for each pixel
            # .mean(): average over all pixels → single diversity scalar
            pixel_standard_deviation = images_01.std(dim=0).mean().item()
            diversity_values.append(pixel_standard_deviation)

        axis.plot(
            guidance_scales,
            diversity_values,
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=colormap(class_label),
            label=f"Digit {class_label}",
            alpha=0.8,
        )

    axis.set_xlabel("Guidance Scale (w)", fontsize=12)
    axis.set_ylabel("Mean Pixel Std Dev (diversity)", fontsize=12)
    axis.set_title(
        f"Intra-class Diversity vs. CFG Strength — {model_label}",
        fontsize=13,
        fontweight="bold",
    )
    axis.legend(fontsize=9, ncol=2, loc="best")
    axis.grid(True, alpha=0.3)
    axis.set_xlim(min(guidance_scales) - 0.3, max(guidance_scales) + 0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_samples_for_model(
    conditioned_unet,
    ddpm,
    vae,
    config: dict,
    guidance_scales: list,
    samples_per_cell: int,
    sampling_steps: int,
    eta: float,
    device: torch.device,
    seed: int,
    model_label: str,
) -> dict:
    """
    Generate samples for all (class, guidance_scale) combinations.

    For reproducibility across guidance scales, pre-generates the initial
    noise for each (class, sample_index) pair using a deterministic seed.
    Each guidance scale then starts from the same noise, so differences
    are purely due to the CFG strength.

    Args:
        conditioned_unet: The class-conditioned UNet model.
        ddpm: DDPM diffusion scheduler.
        vae: Frozen VAE decoder.
        config: Model config dict (for latent shape and scaling factor).
        guidance_scales: List of w values to sweep.
        samples_per_cell: Samples per (class, w) cell.
        sampling_steps: DDIM steps.
        eta: DDIM stochasticity η.
        device: Torch device.
        seed: Random seed for initial noise generation.
        model_label: Human-readable model name for progress messages.

    Returns:
        Dict mapping (class_label, guidance_scale) to decoded pixel
        tensor of shape (samples_per_cell, 1, 28, 28).
    """
    latent_channels = config.get("latent_channels", config["output_channels"])
    latent_shape = (latent_channels, 4, 4)
    scaling_factor = config.get("latent_scaling_factor", 1.0)

    total_cells = NUMBER_OF_CLASSES * len(guidance_scales)
    print(f"\n{'='*60}")
    print(f"Generating samples for: {model_label}")
    print(f"  Guidance scales: {guidance_scales}")
    print(f"  Samples per cell: {samples_per_cell}")
    print(f"  Total cells: {total_cells} ({NUMBER_OF_CLASSES} classes × "
          f"{len(guidance_scales)} scales)")
    print(f"  Total samples: {total_cells * samples_per_cell}")
    print(f"  DDIM steps: {sampling_steps}, η={eta}")
    print(f"  Seed: {seed}")
    print(f"{'='*60}")

    all_images = {}
    cell_counter = 0

    for class_label in range(NUMBER_OF_CLASSES):
        # Pre-generate initial noise for this class using deterministic seed.
        # Same noise is reused across all guidance scales so the only variable
        # changing between columns is the guidance strength w.
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + class_label)
        initial_noise = torch.randn(
            samples_per_cell, *latent_shape,
            device=device, generator=generator,
        )

        for guidance_scale in guidance_scales:
            cell_start_time = time.time()

            pixel_images = generate_samples_for_class(
                conditioned_unet=conditioned_unet,
                ddpm=ddpm,
                vae=vae,
                class_label=class_label,
                guidance_scale=guidance_scale,
                number_of_samples=samples_per_cell,
                latent_shape=latent_shape,
                scaling_factor=scaling_factor,
                sampling_steps=sampling_steps,
                eta=eta,
                device=device,
                initial_noise=initial_noise,
            )

            all_images[(class_label, guidance_scale)] = pixel_images
            cell_counter += 1
            cell_elapsed = time.time() - cell_start_time

            print(
                f"  [{cell_counter:3d}/{total_cells}] "
                f"class={class_label}, w={guidance_scale:5.1f}: "
                f"{cell_elapsed:.1f}s"
            )

    return all_images


def write_summary(
    output_path: str,
    args,
    model_configs: dict,
    elapsed_times: dict,
):
    """
    Write a text summary of the visualization run.

    Args:
        output_path: Path to the summary text file.
        args: Parsed command-line arguments.
        model_configs: Dict mapping model_label to its config dict.
        elapsed_times: Dict mapping model_label to total generation time.
    """
    with open(output_path, "w") as file:
        file.write("CFG Effect Visualization Summary\n")
        file.write("=" * 60 + "\n\n")

        file.write("Configuration\n")
        file.write("-" * 60 + "\n")
        file.write(f"Guidance scales:    {args.guidance_scales}\n")
        file.write(f"Samples per cell:   {args.samples_per_cell}\n")
        file.write(f"DDIM steps:         {args.sampling_steps}\n")
        file.write(f"DDIM eta:           {args.eta}\n")
        file.write(f"Seed:               {args.seed}\n")
        file.write(f"Device:             {args.device}\n\n")

        total_samples_per_model = (
            NUMBER_OF_CLASSES * len(args.guidance_scales) * args.samples_per_cell
        )
        file.write(f"Total samples per model: {total_samples_per_model}\n")
        file.write(f"  ({NUMBER_OF_CLASSES} classes × "
                   f"{len(args.guidance_scales)} scales × "
                   f"{args.samples_per_cell} samples)\n\n")

        for model_label, config in model_configs.items():
            elapsed = elapsed_times.get(model_label, 0)
            file.write(f"Model: {model_label}\n")
            file.write("-" * 60 + "\n")
            file.write(f"  Conditioning:       {config.get('conditioning_type', 'channel_concat')}\n")
            file.write(f"  Base channels:      {config.get('base_channels')}\n")
            file.write(f"  Latent channels:    {config.get('latent_channels', config.get('output_channels'))}\n")
            file.write(f"  Scaling factor:     {config.get('latent_scaling_factor', 1.0):.4f}\n")
            if config.get('conditioning_type') == 'cross_attention':
                file.write(f"  Cross-attn dim:     {config.get('cross_attention_dim')}\n")
            file.write(f"  Generation time:    {elapsed:.1f}s\n")
            file.write(f"  Avg per sample:     {elapsed / total_samples_per_model:.3f}s\n\n")


def main():
    args = parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = os.path.join(
        args.output_dir, f"{timestamp}-cfg-effect-visualization",
    )
    os.makedirs(run_directory, exist_ok=True)

    # Load device and VAE
    device = get_device(args.device)
    args.device = str(device)  # Save actual device for summary
    print(f"Using device: {device}")

    vae = load_vae_checkpoint(args.vae_checkpoint, device)

    # Track configs and timing for the summary
    model_configs = {}
    elapsed_times = {}

    # ── Process channel-concatenation model ──────────────────────────────
    if args.concat_checkpoint is not None:
        model_label = "Channel Concatenation"
        concat_model, concat_ddpm, concat_config = load_unet_checkpoint(
            args.concat_checkpoint, device,
        )

        if not isinstance(concat_model, ClassConditionedUNet):
            raise ValueError(
                f"--concat_checkpoint must be a ClassConditionedUNet, "
                f"got {type(concat_model).__name__}"
            )

        model_configs[model_label] = concat_config
        generation_start = time.time()

        concat_images = generate_all_samples_for_model(
            conditioned_unet=concat_model,
            ddpm=concat_ddpm,
            vae=vae,
            config=concat_config,
            guidance_scales=args.guidance_scales,
            samples_per_cell=args.samples_per_cell,
            sampling_steps=args.sampling_steps,
            eta=args.eta,
            device=device,
            seed=args.seed,
            model_label=model_label,
        )

        elapsed_times[model_label] = time.time() - generation_start

        # Save grid visualization
        grid_path = os.path.join(run_directory, "cfg_grid_concat.pdf")
        create_cfg_comparison_grid(
            all_images=concat_images,
            guidance_scales=args.guidance_scales,
            samples_per_cell=args.samples_per_cell,
            model_label=model_label,
            output_path=grid_path,
        )
        print(f"\nGrid saved: {grid_path}")

        # Save diversity plot
        diversity_path = os.path.join(run_directory, "diversity_concat.pdf")
        create_diversity_plot(
            all_images=concat_images,
            guidance_scales=args.guidance_scales,
            model_label=model_label,
            output_path=diversity_path,
        )
        print(f"Diversity plot saved: {diversity_path}")

    # ── Process cross-attention model ────────────────────────────────────
    if args.cross_attention_checkpoint is not None:
        model_label = "Cross-Attention"
        cross_attention_model, cross_attention_ddpm, cross_attention_config = (
            load_unet_checkpoint(args.cross_attention_checkpoint, device)
        )

        if not isinstance(cross_attention_model, CrossAttentionConditionedUNet):
            raise ValueError(
                f"--cross_attention_checkpoint must be a "
                f"CrossAttentionConditionedUNet, "
                f"got {type(cross_attention_model).__name__}"
            )

        model_configs[model_label] = cross_attention_config
        generation_start = time.time()

        cross_attention_images = generate_all_samples_for_model(
            conditioned_unet=cross_attention_model,
            ddpm=cross_attention_ddpm,
            vae=vae,
            config=cross_attention_config,
            guidance_scales=args.guidance_scales,
            samples_per_cell=args.samples_per_cell,
            sampling_steps=args.sampling_steps,
            eta=args.eta,
            device=device,
            seed=args.seed,
            model_label=model_label,
        )

        elapsed_times[model_label] = time.time() - generation_start

        # Save grid visualization
        grid_path = os.path.join(run_directory, "cfg_grid_cross_attention.pdf")
        create_cfg_comparison_grid(
            all_images=cross_attention_images,
            guidance_scales=args.guidance_scales,
            samples_per_cell=args.samples_per_cell,
            model_label=model_label,
            output_path=grid_path,
        )
        print(f"\nGrid saved: {grid_path}")

        # Save diversity plot
        diversity_path = os.path.join(
            run_directory, "diversity_cross_attention.pdf",
        )
        create_diversity_plot(
            all_images=cross_attention_images,
            guidance_scales=args.guidance_scales,
            model_label=model_label,
            output_path=diversity_path,
        )
        print(f"Diversity plot saved: {diversity_path}")

    # ── Side-by-side comparison (if both models provided) ────────────────
    if (args.concat_checkpoint is not None
            and args.cross_attention_checkpoint is not None):
        comparison_path = os.path.join(
            run_directory, "diversity_comparison.pdf",
        )
        create_side_by_side_diversity_comparison(
            concat_images=concat_images,
            cross_attention_images=cross_attention_images,
            guidance_scales=args.guidance_scales,
            output_path=comparison_path,
        )
        print(f"Comparison plot saved: {comparison_path}")

    # ── Write summary ────────────────────────────────────────────────────
    summary_path = os.path.join(run_directory, "summary.txt")
    write_summary(summary_path, args, model_configs, elapsed_times)
    print(f"\nSummary saved: {summary_path}")

    # ── Final timing ─────────────────────────────────────────────────────
    total_elapsed = sum(elapsed_times.values())
    print(f"\n{'='*60}")
    print(f"Total generation time: {total_elapsed:.1f}s")
    for label, elapsed in elapsed_times.items():
        print(f"  {label}: {elapsed:.1f}s")
    print(f"All outputs saved to: {run_directory}")
    print(f"{'='*60}")


def create_side_by_side_diversity_comparison(
    concat_images: dict,
    cross_attention_images: dict,
    guidance_scales: list,
    output_path: str,
):
    """
    Create a side-by-side diversity comparison plot for both conditioning approaches.

    Shows how intra-class diversity (mean pixel std dev) drops as guidance
    scale increases, for both channel concatenation and cross-attention
    conditioning. This reveals whether the two approaches respond similarly
    to CFG strength or have different quality-diversity trade-off curves.

    Left panel: per-class diversity curves for both models overlaid.
    Right panel: average diversity across all classes.

    Args:
        concat_images: Dict from concat model, (class, w) → pixel tensor.
        cross_attention_images: Dict from CA model, (class, w) → pixel tensor.
        guidance_scales: Guidance scale values used.
        output_path: Output file path.
    """
    figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(16, 6))
    colormap = plt.cm.tab10

    concat_averages = []
    cross_attention_averages = []

    for class_label in range(NUMBER_OF_CLASSES):
        concat_diversity = []
        cross_attention_diversity = []

        for guidance_scale in guidance_scales:
            # Concat model diversity
            concat_tensor = concat_images[(class_label, guidance_scale)]
            concat_01 = ((concat_tensor + 1.0) / 2.0).clamp(0, 1)
            concat_diversity.append(concat_01.std(dim=0).mean().item())

            # Cross-attention model diversity
            cross_attention_tensor = cross_attention_images[
                (class_label, guidance_scale)
            ]
            cross_attention_01 = (
                (cross_attention_tensor + 1.0) / 2.0
            ).clamp(0, 1)
            cross_attention_diversity.append(
                cross_attention_01.std(dim=0).mean().item()
            )

        # Left panel: per-class curves
        axis_left.plot(
            guidance_scales, concat_diversity,
            marker="o", markersize=3, linewidth=1.2,
            color=colormap(class_label), alpha=0.6,
            linestyle="-",
        )
        axis_left.plot(
            guidance_scales, cross_attention_diversity,
            marker="s", markersize=3, linewidth=1.2,
            color=colormap(class_label), alpha=0.6,
            linestyle="--",
        )

        concat_averages.append(concat_diversity)
        cross_attention_averages.append(cross_attention_diversity)

    # Add legend entries for line styles (not per-class)
    axis_left.plot([], [], color="gray", linestyle="-", label="Concat")
    axis_left.plot([], [], color="gray", linestyle="--", label="Cross-Attn")
    axis_left.set_xlabel("Guidance Scale (w)", fontsize=12)
    axis_left.set_ylabel("Mean Pixel Std Dev", fontsize=12)
    axis_left.set_title("Per-Class Diversity", fontsize=13, fontweight="bold")
    axis_left.legend(fontsize=10, loc="best")
    axis_left.grid(True, alpha=0.3)

    # Right panel: average across all classes
    # concat_averages shape: (10, len(guidance_scales)) → average over dim 0
    concat_mean_diversity = [
        sum(class_div[scale_index] for class_div in concat_averages)
        / NUMBER_OF_CLASSES
        for scale_index in range(len(guidance_scales))
    ]
    cross_attention_mean_diversity = [
        sum(class_div[scale_index] for class_div in cross_attention_averages)
        / NUMBER_OF_CLASSES
        for scale_index in range(len(guidance_scales))
    ]

    axis_right.plot(
        guidance_scales, concat_mean_diversity,
        marker="o", markersize=6, linewidth=2.0,
        color="#1f77b4", label="Channel Concat",
    )
    axis_right.plot(
        guidance_scales, cross_attention_mean_diversity,
        marker="s", markersize=6, linewidth=2.0,
        color="#ff7f0e", label="Cross-Attention",
    )

    axis_right.set_xlabel("Guidance Scale (w)", fontsize=12)
    axis_right.set_ylabel("Mean Pixel Std Dev (avg over classes)", fontsize=12)
    axis_right.set_title(
        "Average Diversity Comparison", fontsize=13, fontweight="bold",
    )
    axis_right.legend(fontsize=11, loc="best")
    axis_right.grid(True, alpha=0.3)

    figure.suptitle(
        "CFG Quality-Diversity Trade-off: Channel Concat vs Cross-Attention",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
