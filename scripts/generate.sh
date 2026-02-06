#!/bin/bash
# Generate samples from a trained DDPM/DDIM checkpoint
# Run from the root of the repository
#
# Usage:
#   bash scripts/generate.sh <checkpoint_path> [OPTIONS]
#
# Examples (DDPM — default, full 1000 steps):
#   bash scripts/generate.sh experiments/20260207-ddpm/checkpoints/checkpoint_final.pt
#   bash scripts/generate.sh path/to/checkpoint.pt --num_samples 25 --nrow 5
#   bash scripts/generate.sh path/to/checkpoint.pt --show_denoising
#
# Examples (DDIM — faster sampling with fewer steps):
#   bash scripts/generate.sh path/to/checkpoint.pt --mode ddim --ddim_steps 50
#   bash scripts/generate.sh path/to/checkpoint.pt --mode ddim --ddim_steps 100 --eta 0.5
#   bash scripts/generate.sh path/to/checkpoint.pt --mode ddim --ddim_steps 25 --show_denoising
#
# Options:
#   --num_samples N     Number of samples to generate (default: 16)
#   --output_dir PATH   Where to save outputs (default: ./generated_samples)
#   --nrow N            Images per row in grid (default: 4)
#   --show_denoising    Also save denoising progression
#   --mode MODE         Sampling mode: ddpm or ddim (default: ddpm)
#   --ddim_steps N      DDIM sampling steps (default: 50, only with --mode ddim)
#   --eta FLOAT         DDIM stochasticity 0.0-1.0 (default: 0.0, only with --mode ddim)

set -e

if [ -z "$1" ]; then
    echo "Error: checkpoint path is required"
    echo "Usage: bash scripts/generate.sh <checkpoint_path> [OPTIONS]"
    exit 1
fi

CHECKPOINT="$1"
shift  # Remove checkpoint from args, pass remaining options through

python -m models.generate --checkpoint "$CHECKPOINT" "$@"
