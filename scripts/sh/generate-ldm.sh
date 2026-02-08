#!/bin/bash
# Generate samples from a trained Latent Diffusion Model (LDM)
# Run from the root of the repository
#
# Usage:
#   bash scripts/sh/generate-ldm.sh <vae_checkpoint> <unet_checkpoint> [OPTIONS]
#
# Examples (DDPM — default, full 1000 steps):
#   bash scripts/sh/generate-ldm.sh path/to/vae.pt path/to/unet.pt
#   bash scripts/sh/generate-ldm.sh path/to/vae.pt path/to/unet.pt --num_samples 25 --nrow 5
#
# Examples (DDIM — faster sampling with fewer steps):
#   bash scripts/sh/generate-ldm.sh path/to/vae.pt path/to/unet.pt --mode ddim --ddim_steps 50
#   bash scripts/sh/generate-ldm.sh path/to/vae.pt path/to/unet.pt --mode ddim --ddim_steps 100 --eta 0.5
#
# Output structure (timestamped to avoid overwriting):
#   generated_samples/20260208_143052-latent-ddpm/
#     grid.pdf                          # All samples in a grid (PDF for Overleaf)
#     profile.txt                       # Per-sample timing + averages
#     sample_000/                       # Per-sample subfolder
#       final.pdf                       # Final generated image (decoded to pixel space)
#       denoising_progression.pdf       # Horizontal strip of denoising steps
#       step_00_t0999.pdf ... step_10_t0000.pdf  # Individual denoising steps (every 10%)
#     sample_001/
#       ...
#
# Options:
#   --num_samples N     Number of samples to generate (default: 16)
#   --output_dir PATH   Base directory for outputs (default: ./generated_samples)
#   --nrow N            Images per row in grid (default: 4)
#   --mode MODE         Sampling mode: ddpm or ddim (default: ddpm)
#   --ddim_steps N      DDIM sampling steps (default: 50, only with --mode ddim)
#   --eta FLOAT         DDIM stochasticity 0.0-1.0 (default: 0.0, only with --mode ddim)

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: both VAE and UNet checkpoint paths are required"
    echo "Usage: bash scripts/sh/generate-ldm.sh <vae_checkpoint> <unet_checkpoint> [OPTIONS]"
    exit 1
fi

VAE_CHECKPOINT="$1"
UNET_CHECKPOINT="$2"
shift 2  # Remove both checkpoints from args, pass remaining options through

python -m scripts.python.generate_latent_diffusion \
    --vae_checkpoint "$VAE_CHECKPOINT" \
    --unet_checkpoint "$UNET_CHECKPOINT" \
    "$@"
