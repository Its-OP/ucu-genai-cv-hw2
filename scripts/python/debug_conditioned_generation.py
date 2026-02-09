"""
Diagnostic script for debugging conditioned LDM generation quality.

Generates samples in multiple modes to isolate whether the problem is in:
- The model itself (raw conditional/unconditional both bad → training issue)
- The CFG formula (raw is OK but CFG is bad → wrapper bug)
- The guidance scale (CFG scale=1 OK but scale=3 bad → amplification issue)

Also analyzes the learned class embeddings and gradient flow.

Usage:
    python -m scripts.python.debug_conditioned_generation \
        --unet_checkpoint issues/lcgf-generating-noise/checkpoints/checkpoint_epoch_040.pt \
        --vae_checkpoint experiments/vae-training/checkpoints/checkpoint_final.pt
"""
import argparse
import os

import torch
import matplotlib.pyplot as plt

from data import get_train_loader
from models.ddpm import DDPM
from models.classifier_free_guidance import (
    ClassConditionedUNet,
    ClassifierFreeGuidanceWrapper,
)
from models.utils import (
    pad_to_32,
    get_device,
    load_vae_checkpoint,
    load_unet_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug conditioned LDM generation")
    parser.add_argument(
        "--unet_checkpoint", type=str, required=True,
        help="Path to conditioned UNet checkpoint",
    )
    parser.add_argument(
        "--vae_checkpoint", type=str, required=True,
        help="Path to VAE checkpoint",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./debug_output",
        help="Directory to save diagnostic outputs",
    )
    parser.add_argument(
        "--class_label", type=int, default=7,
        help="Class label to generate (default: 7)",
    )
    return parser.parse_args()


@torch.no_grad()
def generate_single_sample_ddim(model, ddpm, latent_shape, device, clip_denoised=False):
    """Generate a single sample using DDIM sampling (100 steps for speed)."""
    from models.ddim import DDIMSampler
    ddim_sampler = DDIMSampler(ddpm, ddim_timesteps=100, eta=0.05).to(device)
    return ddim_sampler.ddim_sample_loop(model, latent_shape, clip_denoised=clip_denoised)


@torch.no_grad()
def generate_single_sample_ddpm(model, ddpm, latent_shape, device, clip_denoised=False):
    """Generate a single sample using full DDPM sampling (1000 steps)."""
    return ddpm.p_sample_loop(model, latent_shape, clip_denoised=clip_denoised)


@torch.no_grad()
def decode_and_crop(latent_sample, vae, scaling_factor):
    """Decode latent to pixel space and crop to 28x28."""
    latent_unscaled = latent_sample / scaling_factor
    pixel_sample = vae.decode(latent_unscaled)
    # Crop 32x32 -> 28x28
    pixel_sample = pixel_sample[:, :, 2:30, 2:30]
    return pixel_sample


def save_comparison_grid(samples_dict, output_path):
    """Save a comparison grid of samples from different generation modes."""
    number_of_modes = len(samples_dict)
    fig, axes = plt.subplots(1, number_of_modes, figsize=(4 * number_of_modes, 4))

    for axis_index, (mode_name, sample_tensor) in enumerate(samples_dict.items()):
        # Denormalize [-1, 1] -> [0, 1]
        image = (sample_tensor[0, 0] + 1) / 2
        image = image.clamp(0, 1).cpu().numpy()

        axes[axis_index].imshow(image, cmap="gray")
        axes[axis_index].set_title(mode_name, fontsize=10)
        axes[axis_index].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison grid: {output_path}")


def analyze_embeddings(conditioned_unet, output_dir):
    """Analyze the learned class embeddings."""
    embedding_weights = conditioned_unet.class_embedding.weight.data.cpu()
    number_of_entries, embedding_dimension = embedding_weights.shape

    print("\n=== Embedding Analysis ===")
    print(f"Embedding shape: ({number_of_entries}, {embedding_dimension})")
    print(f"  Index 0 (unconditional): mean={embedding_weights[0].mean():.4f}, "
          f"std={embedding_weights[0].std():.4f}, "
          f"norm={embedding_weights[0].norm():.4f}")

    for class_index in range(1, number_of_entries):
        weight = embedding_weights[class_index]
        print(f"  Index {class_index} (class {class_index - 1}): "
              f"mean={weight.mean():.4f}, std={weight.std():.4f}, "
              f"norm={weight.norm():.4f}")

    # Compute pairwise distances between class embeddings
    print("\n=== Pairwise Distances ===")
    # Distance between unconditional and each class
    unconditional_embedding = embedding_weights[0]
    for class_index in range(1, number_of_entries):
        distance = (embedding_weights[class_index] - unconditional_embedding).norm()
        print(f"  dist(uncond, class {class_index - 1}) = {distance:.4f}")

    # Distance between different classes
    class_embeddings = embedding_weights[1:]  # Skip unconditional
    pairwise_distances = torch.cdist(
        class_embeddings.unsqueeze(0), class_embeddings.unsqueeze(0),
    ).squeeze(0)
    mean_class_distance = pairwise_distances[
        ~torch.eye(pairwise_distances.shape[0], dtype=torch.bool)
    ].mean()
    print(f"\n  Mean inter-class distance: {mean_class_distance:.4f}")
    print(f"  Mean class-unconditional distance: "
          f"{torch.stack([((embedding_weights[i] - unconditional_embedding).norm()) for i in range(1, number_of_entries)]).mean():.4f}")

    # Visualize embeddings as spatial maps
    spatial_size = int(embedding_dimension ** 0.5)
    if spatial_size * spatial_size == embedding_dimension:
        fig, axes = plt.subplots(1, number_of_entries, figsize=(2 * number_of_entries, 2))
        for entry_index in range(number_of_entries):
            spatial_map = embedding_weights[entry_index].view(spatial_size, spatial_size)
            axes[entry_index].imshow(spatial_map.numpy(), cmap="RdBu", vmin=-2, vmax=2)
            label = "uncond" if entry_index == 0 else f"cls {entry_index - 1}"
            axes[entry_index].set_title(label, fontsize=8)
            axes[entry_index].axis("off")
        plt.tight_layout()
        embedding_path = os.path.join(output_dir, "embeddings_spatial.pdf")
        plt.savefig(embedding_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved embedding visualization: {embedding_path}")


def analyze_gradient_flow(conditioned_unet, ddpm, vae, device, scaling_factor):
    """Check gradient norms on embedding vs UNet parameters."""
    print("\n=== Gradient Flow Analysis ===")

    conditioned_unet.train()
    train_loader = get_train_loader(batch_size=64)
    images, labels = next(iter(train_loader))
    images = pad_to_32(images.to(device))
    labels = labels.to(device)

    # Encode to latent
    with torch.no_grad():
        posterior = vae.encode(images)
        latent = posterior.mode() * scaling_factor

    # Set labels and do forward/backward
    conditioned_unet.set_class_labels(labels)
    timestep = torch.randint(0, ddpm.timesteps, (latent.shape[0],), device=device)

    conditioned_unet.zero_grad()
    loss = ddpm.p_losses(conditioned_unet, latent, timestep)
    loss.backward()

    # Check gradient norms
    embedding_grad_norm = conditioned_unet.class_embedding.weight.grad.norm().item()
    unet_grad_norms = []
    for name, parameter in conditioned_unet.unet.named_parameters():
        if parameter.grad is not None:
            unet_grad_norms.append(parameter.grad.norm().item())

    mean_unet_grad = sum(unet_grad_norms) / len(unet_grad_norms)
    max_unet_grad = max(unet_grad_norms)

    print(f"  Embedding gradient norm: {embedding_grad_norm:.6f}")
    print(f"  UNet mean gradient norm: {mean_unet_grad:.6f}")
    print(f"  UNet max gradient norm:  {max_unet_grad:.6f}")
    print(f"  Ratio (embedding / unet_mean): {embedding_grad_norm / mean_unet_grad:.4f}")

    # Check the first conv layer weights for the conditioning channel
    first_conv = conditioned_unet.unet.model.input_convolution
    first_conv_weight = first_conv.weight.data  # (out_channels, in_channels, 3, 3)
    latent_channel_weights = first_conv_weight[:, :2, :, :]  # Weights for latent channels
    conditioning_channel_weights = first_conv_weight[:, 2:, :, :]  # Weights for conditioning channel

    print(f"\n  First conv layer analysis:")
    print(f"    Latent channels (0-1) weight norm:       {latent_channel_weights.norm():.4f}")
    print(f"    Conditioning channel (2) weight norm:    {conditioning_channel_weights.norm():.4f}")
    print(f"    Ratio (conditioning / latent):           "
          f"{conditioning_channel_weights.norm() / latent_channel_weights.norm():.4f}")

    conditioned_unet.eval()


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load VAE
    vae = load_vae_checkpoint(args.vae_checkpoint, device)
    latent_channels = vae.latent_channels
    print(f"VAE loaded. Latent channels: {latent_channels}")

    # Load conditioned UNet checkpoint
    model, ddpm, config = load_unet_checkpoint(args.unet_checkpoint, device)
    assert isinstance(model, ClassConditionedUNet), (
        f"Expected ClassConditionedUNet, got {type(model)}"
    )
    conditioned_unet = model
    scaling_factor = config.get("latent_scaling_factor", 1.0)
    print(f"Conditioned UNet loaded. Scaling factor: {scaling_factor:.4f}")
    print(f"Config: {config}")

    # ====================================================================
    # DIAGNOSTIC 1: Embedding Analysis
    # ====================================================================
    analyze_embeddings(conditioned_unet, args.output_dir)

    # ====================================================================
    # DIAGNOSTIC 2: Gradient Flow Analysis
    # ====================================================================
    analyze_gradient_flow(conditioned_unet, ddpm, vae, device, scaling_factor)

    # ====================================================================
    # DIAGNOSTIC 3: Generate samples in multiple modes
    # ====================================================================
    print("\n=== Generation Diagnostics ===")
    conditioned_unet.eval()

    class_label = args.class_label
    latent_shape = (1, latent_channels, 4, 4)
    label_tensor = torch.tensor([class_label], device=device)

    samples = {}

    # Debug: print device info
    print(f"\n  Model device: {next(conditioned_unet.parameters()).device}")
    print(f"  DDPM device: {ddpm.betas.device}")
    print(f"  Label device: {label_tensor.device}")
    print(f"  Embedding device: {conditioned_unet.class_embedding.weight.device}")

    # Mode 1: DDIM 100 steps - CFG w=3.0 (standard fast sampling)
    print(f"\nMode 1: DDIM 100 steps, CFG w=3.0 (class={class_label})...")
    conditioned_unet.set_class_labels(label_tensor)
    cfg_wrapper = ClassifierFreeGuidanceWrapper(conditioned_unet, guidance_scale=3.0)
    latent = generate_single_sample_ddim(cfg_wrapper, ddpm, latent_shape, device)
    samples["DDIM CFG w=3.0"] = decode_and_crop(latent, vae, scaling_factor)

    # Mode 2: DDPM 1000 steps - CFG w=3.0, clip_denoised=True (THE FIX)
    # Clamping predicted x₀ to [-1, 1] at each reverse step prevents the
    # amplification cascade at late timesteps where √(1/ᾱ_t − 1) ≈ 64,000×.
    print(f"Mode 2: DDPM 1000 steps, CFG w=3.0, clip_denoised=True (class={class_label})...")
    conditioned_unet.set_class_labels(label_tensor)
    cfg_wrapper_clip = ClassifierFreeGuidanceWrapper(
        conditioned_unet, guidance_scale=3.0,
    )
    latent = generate_single_sample_ddpm(
        cfg_wrapper_clip, ddpm, latent_shape, device, clip_denoised=True,
    )
    samples["DDPM CFG clip=True"] = decode_and_crop(latent, vae, scaling_factor)

    # Mode 3: DDPM 1000 steps - Raw conditional (no CFG), clip_denoised=True
    print(f"Mode 3: DDPM 1000 steps, raw cond, clip_denoised=True (class={class_label})...")
    conditioned_unet.set_class_labels(label_tensor)
    latent = generate_single_sample_ddpm(
        conditioned_unet, ddpm, latent_shape, device, clip_denoised=True,
    )
    samples["DDPM raw clip=True"] = decode_and_crop(latent, vae, scaling_factor)

    # Mode 4: DDPM 1000 steps - Raw conditional, clip_denoised=False (BROKEN)
    # Without x₀ clamping, prediction errors at late timesteps compound
    # catastrophically over 1000 reverse steps, producing blurry blobs.
    print(f"Mode 4: DDPM 1000 steps, raw cond, clip_denoised=False (class={class_label})...")
    conditioned_unet.set_class_labels(label_tensor)
    latent = generate_single_sample_ddpm(
        conditioned_unet, ddpm, latent_shape, device, clip_denoised=False,
    )
    samples["DDPM no clip (broken)"] = decode_and_crop(latent, vae, scaling_factor)

    # Save comparison grid
    comparison_path = os.path.join(args.output_dir, "generation_modes_comparison.pdf")
    save_comparison_grid(samples, comparison_path)

    # ====================================================================
    # DIAGNOSTIC 4: Noise prediction difference between cond and uncond
    # ====================================================================
    print("\n=== Noise Prediction Difference Analysis ===")
    conditioned_unet.eval()

    # Start from the same noise
    torch.manual_seed(42)
    test_noise = torch.randn(latent_shape, device=device)
    test_timestep = torch.tensor([500], device=device)

    # Conditional prediction
    conditioned_unet.set_class_labels(label_tensor)
    noise_prediction_conditional = conditioned_unet(test_noise, test_timestep)

    # Unconditional prediction
    conditioned_unet.set_class_labels(None)
    noise_prediction_unconditional = conditioned_unet(test_noise, test_timestep)

    # Compare
    difference = noise_prediction_conditional - noise_prediction_unconditional
    print(f"  Conditional noise pred norm:   {noise_prediction_conditional.norm():.4f}")
    print(f"  Unconditional noise pred norm: {noise_prediction_unconditional.norm():.4f}")
    print(f"  Difference norm:               {difference.norm():.4f}")
    print(f"  Difference mean abs:           {difference.abs().mean():.6f}")
    print(f"  Difference max abs:            {difference.abs().max():.6f}")
    print(f"  Relative difference:           "
          f"{difference.norm() / noise_prediction_conditional.norm():.4f}")

    # Test at multiple timesteps
    print("\n  Difference across timesteps:")
    for timestep_value in [50, 200, 500, 800, 950]:
        test_timestep = torch.tensor([timestep_value], device=device)

        conditioned_unet.set_class_labels(label_tensor)
        prediction_cond = conditioned_unet(test_noise, test_timestep)

        conditioned_unet.set_class_labels(None)
        prediction_uncond = conditioned_unet(test_noise, test_timestep)

        diff_norm = (prediction_cond - prediction_uncond).norm()
        cond_norm = prediction_cond.norm()
        print(f"    t={timestep_value:4d}: diff_norm={diff_norm:.4f}, "
              f"cond_norm={cond_norm:.4f}, "
              f"relative={diff_norm / cond_norm:.4f}")

    print(f"\nAll diagnostics saved to {args.output_dir}")


if __name__ == "__main__":
    main()
