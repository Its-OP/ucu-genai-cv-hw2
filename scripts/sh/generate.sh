#!/bin/bash
# Generate samples from a trained DDPM/DDIM checkpoint
# Run from the root of the repository
#
# Usage:
#   bash scripts/sh/generate.sh <checkpoint_path> [OPTIONS]
#
# Examples (DDPM — default, full 1000 steps):
#   bash scripts/sh/generate.sh experiments/20260207-ddpm/checkpoints/checkpoint_final.pt
#   bash scripts/sh/generate.sh path/to/checkpoint.pt --num_samples 25 --nrow 5
#
# Examples (DDIM — faster sampling with fewer steps):
#   bash scripts/sh/generate.sh path/to/checkpoint.pt --mode ddim --ddim_steps 50
#   bash scripts/sh/generate.sh path/to/checkpoint.pt --mode ddim --ddim_steps 100 --eta 0.5
#
# Output structure (timestamped to avoid overwriting):
#   generated_samples/20260207_143052-ddpm/
#     grid.pdf                          # All samples in a grid (PDF for Overleaf)
#     profile.txt                       # Per-sample timing + averages
#     sample_000/                       # Per-sample subfolder
#       final.pdf                       # Final generated image
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

if [ -z "$1" ]; then
    echo "Error: checkpoint path is required"
    echo "Usage: bash scripts/sh/generate.sh <checkpoint_path> [OPTIONS]"
    exit 1
fi

CHECKPOINT="$1"
shift  # Remove checkpoint from args, pass remaining options through

python -m scripts.python.generate --checkpoint "$CHECKPOINT" "$@"
