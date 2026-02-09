#!/bin/bash
# Generate class-conditioned samples from a trained Latent Diffusion Model
# Run from the root of the repository
#
# Usage:
#   bash scripts/sh/generate-conditioned-ldm.sh <vae_checkpoint> <unet_checkpoint> [OPTIONS]
#
# Examples (generate specific digit class):
#   bash scripts/sh/generate-conditioned-ldm.sh path/to/vae.pt path/to/cond_unet.pt --class_label 7
#   bash scripts/sh/generate-conditioned-ldm.sh path/to/vae.pt path/to/cond_unet.pt --class_label 3 --guidance_scale 5.0
#
# Examples (generate one sample per class, 0-9):
#   bash scripts/sh/generate-conditioned-ldm.sh path/to/vae.pt path/to/cond_unet.pt
#   bash scripts/sh/generate-conditioned-ldm.sh path/to/vae.pt path/to/cond_unet.pt --num_samples 3
#
# Examples (DDIM sampling):
#   bash scripts/sh/generate-conditioned-ldm.sh path/to/vae.pt path/to/cond_unet.pt --mode ddim --ddim_steps 50
#
# Output structure (timestamped to avoid overwriting):
#   generated_samples/20260209_143052-conditioned-latent-ddpm/
#     grid.pdf                                  # All samples in a grid
#     profile.txt                               # Per-sample timing + averages
#     sample_000_class0/                        # Per-sample subfolder (labeled by class)
#       final.pdf                               # Final generated image
#       denoising_progression.pdf               # Denoising strip
#       step_00_t0999.pdf ... step_10_t0000.pdf # Individual denoising steps
#     sample_001_class1/
#       ...
#
# Options:
#   --class_label N         Target digit class 0-9 (default: generate all classes)
#   --num_samples N         Samples per class (default: 1)
#   --guidance_scale FLOAT  CFG guidance scale (default: 3.0)
#   --output_dir PATH       Base directory for outputs (default: ./generated_samples)
#   --nrow N                Images per row in grid (default: 10)
#   --mode MODE             Sampling mode: ddpm or ddim (default: ddpm)
#   --ddim_steps N          DDIM sampling steps (default: 50, only with --mode ddim)
#   --eta FLOAT             DDIM stochasticity 0.0-1.0 (default: 0.0, only with --mode ddim)

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: both VAE and UNet checkpoint paths are required"
    echo "Usage: bash scripts/sh/generate-conditioned-ldm.sh <vae_checkpoint> <unet_checkpoint> [OPTIONS]"
    exit 1
fi

VAE_CHECKPOINT="$1"
UNET_CHECKPOINT="$2"
shift 2  # Remove both checkpoints from args, pass remaining options through

python -m scripts.python.generate_conditioned_latent_diffusion \
    --vae_checkpoint "$VAE_CHECKPOINT" \
    --unet_checkpoint "$UNET_CHECKPOINT" \
    "$@"
