#!/bin/bash
# Setup script for CUDA server
# Run this BEFORE the repository exists on the server
# Usage: bash setup.sh

set -e

REPO_URL="https://github.com/Its-OP/ucu-genai-cv-hw2.git"
REPO_NAME="ucu-genai-cv-hw2"
CONDA_ENV="diffusion"

echo "=========================================="
echo "Setting up DDPM training environment"
echo "=========================================="

# Install screen if not available
if ! command -v screen &> /dev/null; then
    echo "Installing screen..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y screen
    elif command -v yum &> /dev/null; then
        sudo yum install -y screen
    else
        echo "Warning: Could not install screen. Please install it manually."
    fi
else
    echo "screen is already installed."
fi

# Clone repository
if [ -d "$REPO_NAME" ]; then
    echo "Repository already exists. Pulling latest changes..."
    cd "$REPO_NAME"
    git pull
else
    echo "Cloning repository..."
    git clone "$REPO_URL"
    cd "$REPO_NAME"
fi

# Create conda environment if it doesn't exist
if conda env list | grep -q "^$CONDA_ENV "; then
    echo "Conda environment '$CONDA_ENV' already exists."
else
    echo "Creating conda environment '$CONDA_ENV'..."
    conda create -n "$CONDA_ENV" python=3.11 -y
fi

# Activate environment and install dependencies
echo "Installing dependencies..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
pip install -r requirements.txt

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "To start training:"
echo "  cd $REPO_NAME"
echo "  conda activate $CONDA_ENV"
echo "  bash scripts/train.sh --epochs 100"
echo "=========================================="
