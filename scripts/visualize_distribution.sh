#!/bin/bash
# Visualize the distribution of generated vs real MNIST digits using UMAP
# Run from the root of the repository
#
# Usage:
#   bash scripts/visualize_distribution.sh <checkpoint_path> [OPTIONS]
#
# Examples:
#   bash scripts/visualize_distribution.sh path/to/checkpoint.pt
#   bash scripts/visualize_distribution.sh path/to/checkpoint.pt --mode ddim --ddim_steps 50
#   bash scripts/visualize_distribution.sh path/to/checkpoint.pt --num_generated 2000 --num_real 2000
#   bash scripts/visualize_distribution.sh path/to/checkpoint.pt --umap_neighbors 30 --umap_min_dist 0.05
#
# Output structure (timestamped to avoid overwriting):
#   distribution_plots/20260207_143052/
#     real_vs_generated.pdf     # Two-panel UMAP scatter plot (PDF for Overleaf)
#     grid.pdf                  # Grid of all generated samples
#     profile.txt               # Per-sample generation timing + averages
#     metadata.txt              # Config, timing, sample counts
#
# Options:
#   --num_generated N       Number of samples to generate (default: 1000)
#   --num_real N            Number of real MNIST samples to use (default: 1000)
#   --output_dir PATH       Base directory for outputs (default: ./distribution_plots)
#   --mode MODE             Sampling mode: ddpm or ddim (default: ddim)
#   --ddim_steps N          DDIM sampling steps (default: 50, only with --mode ddim)
#   --eta FLOAT             DDIM stochasticity 0.0-1.0 (default: 0.0, only with --mode ddim)
#   --batch_size N          Batch size for generation and feature extraction (default: 64)
#   --umap_neighbors N      UMAP n_neighbors parameter (default: 15)
#   --umap_min_dist FLOAT   UMAP min_dist parameter (default: 0.1)
#   --seed N                Random seed for reproducibility (default: 42)

set -e

if [ -z "$1" ]; then
    echo "Error: checkpoint path is required"
    echo "Usage: bash scripts/visualize_distribution.sh <checkpoint_path> [OPTIONS]"
    exit 1
fi

CHECKPOINT="$1"
shift  # Remove checkpoint from args, pass remaining options through

python -m models.visualize_distribution --checkpoint "$CHECKPOINT" "$@"
