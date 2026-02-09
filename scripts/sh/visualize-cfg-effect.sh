#!/bin/bash
# Visualize how CFG guidance scale affects class-conditional image quality
# and diversity for channel-concat and/or cross-attention models.
# Run from the root of the repository.
#
# Usage:
#   bash scripts/sh/visualize-cfg-effect.sh <vae_checkpoint> [OPTIONS]
#
# At least one of --concat_checkpoint or --cross_attention_checkpoint is required.
#
# Examples (both models):
#   bash scripts/sh/visualize-cfg-effect.sh path/to/vae.pt \
#       --concat_checkpoint path/to/concat_unet.pt \
#       --cross_attention_checkpoint path/to/ca_unet.pt
#
# Examples (single model):
#   bash scripts/sh/visualize-cfg-effect.sh path/to/vae.pt \
#       --concat_checkpoint path/to/concat_unet.pt
#
# Examples (custom sweep):
#   bash scripts/sh/visualize-cfg-effect.sh path/to/vae.pt \
#       --concat_checkpoint path/to/concat_unet.pt \
#       --guidance_scales 0.0 1.0 3.0 5.0 10.0 \
#       --samples_per_cell 10
#
# Output structure (timestamped to avoid overwriting):
#   generated_samples/20260209_143052-cfg-effect-visualization/
#     cfg_grid_concat.pdf            # Grid: rows=digits, cols=guidance scales
#     cfg_grid_cross_attention.pdf   # (if cross-attention model provided)
#     diversity_concat.pdf           # Per-class diversity vs. guidance scale
#     diversity_cross_attention.pdf  # (if cross-attention model provided)
#     diversity_comparison.pdf       # Side-by-side (if both models provided)
#     summary.txt                    # Configuration and timing
#
# Options:
#   --concat_checkpoint PATH            Channel-concat conditioned UNet checkpoint
#   --cross_attention_checkpoint PATH   Cross-attention conditioned UNet checkpoint
#   --guidance_scales FLOAT...          Guidance scales to sweep (default: 0.0 0.5 1.0 2.0 3.0 5.0 7.0 10.0 50.0 100.0)
#   --samples_per_cell N                Samples per (class, scale) cell (default: 10)
#   --sampling_steps N                  DDIM steps (default: 100)
#   --eta FLOAT                         DDIM stochasticity (default: 0.05)
#   --seed N                            Random seed (default: 42)
#   --output_dir PATH                   Base output directory (default: ./generated_samples)

set -e

if [ -z "$1" ]; then
    echo "Error: VAE checkpoint path is required"
    echo "Usage: bash scripts/sh/visualize-cfg-effect.sh <vae_checkpoint> [OPTIONS]"
    exit 1
fi

VAE_CHECKPOINT="$1"
shift 1  # Remove VAE checkpoint from args, pass remaining options through

python -m scripts.python.visualize_cfg_effect \
    --vae_checkpoint "$VAE_CHECKPOINT" \
    "$@"
