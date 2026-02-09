#!/bin/bash
# Generate samples from a trained Latent Rectified Flow Model
# Run from the root of the repository
#
# Usage:
#   bash scripts/sh/generate-latent-rf.sh <vae_checkpoint> <unet_checkpoint> [OPTIONS]
#
# Examples:
#   bash scripts/sh/generate-latent-rf.sh path/to/vae.pt path/to/unet.pt
#   bash scripts/sh/generate-latent-rf.sh path/to/vae.pt path/to/unet.pt --sampling_steps 100
#   bash scripts/sh/generate-latent-rf.sh path/to/vae.pt path/to/unet.pt --num_samples 25 --nrow 5
#
# Output structure (timestamped to avoid overwriting):
#   generated_samples/20260210_143052-latent-rf/
#     grid.pdf                          # All samples in a grid (PDF for Overleaf)
#     profile.txt                       # Per-sample timing + averages
#     sample_000/                       # Per-sample subfolder
#       final.pdf                       # Final generated image (decoded to pixel space)
#       denoising_progression.pdf       # Horizontal strip of denoising steps
#       step_00_i0049.pdf ...           # Individual denoising steps (every 10%)
#     sample_001/
#       ...
#
# Options:
#   --num_samples N         Number of samples to generate (default: 16)
#   --sampling_steps N      Number of Euler steps (default: from checkpoint)
#   --output_dir PATH       Base directory for outputs (default: ./generated_samples)
#   --nrow N                Images per row in grid (default: 4)

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: both VAE and UNet checkpoint paths are required"
    echo "Usage: bash scripts/sh/generate-latent-rf.sh <vae_checkpoint> <unet_checkpoint> [OPTIONS]"
    exit 1
fi

VAE_CHECKPOINT="$1"
UNET_CHECKPOINT="$2"
shift 2  # Remove both checkpoints from args, pass remaining options through

python -m scripts.python.generate_latent_rf \
    --vae_checkpoint "$VAE_CHECKPOINT" \
    --unet_checkpoint "$UNET_CHECKPOINT" \
    "$@"
