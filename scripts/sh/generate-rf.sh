#!/bin/bash
# Generate samples from a trained Rectified Flow checkpoint (pixel space)
# Run from the root of the repository
#
# Usage:
#   bash scripts/sh/generate-rf.sh <checkpoint_path> [OPTIONS]
#
# Examples:
#   bash scripts/sh/generate-rf.sh path/to/checkpoint.pt
#   bash scripts/sh/generate-rf.sh path/to/checkpoint.pt --sampling_steps 100
#   bash scripts/sh/generate-rf.sh path/to/checkpoint.pt --num_samples 25 --nrow 5
#
# Output structure (timestamped to avoid overwriting):
#   generated_samples/20260210_143052-rf/
#     grid.pdf                          # All samples in a grid (PDF for Overleaf)
#     profile.txt                       # Per-sample timing + averages
#     sample_000/                       # Per-sample subfolder
#       final.pdf                       # Final generated image
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

if [ -z "$1" ]; then
    echo "Error: checkpoint path is required"
    echo "Usage: bash scripts/sh/generate-rf.sh <checkpoint_path> [OPTIONS]"
    exit 1
fi

CHECKPOINT="$1"
shift  # Remove checkpoint from args, pass remaining options through

python -m scripts.python.generate_rf --checkpoint "$CHECKPOINT" "$@"
