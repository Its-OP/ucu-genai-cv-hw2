#!/bin/bash
# DDPM Training Script for MNIST
# Creates experiment folder with all outputs in ./experiments/

set -e

# Default configuration
# Note: base_channels=64 recommended for MNIST (channel multipliers: 1, 2, 4, 4)
EPOCHS=${EPOCHS:-100}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
TIMESTEPS=${TIMESTEPS:-1000}
BETA_SCHEDULE=${BETA_SCHEDULE:-cosine}
SAMPLE_EVERY=${SAMPLE_EVERY:-10}
BASE_CHANNELS=${BASE_CHANNELS:-64}

echo "=========================================="
echo "DDPM Training on MNIST"
echo "=========================================="
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Timesteps: $TIMESTEPS"
echo "Beta Schedule: $BETA_SCHEDULE"
echo "Sample Every: $SAMPLE_EVERY epochs"
echo "Base Channels: $BASE_CHANNELS"
echo "=========================================="

python -m models.train \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --timesteps $TIMESTEPS \
    --beta_schedule $BETA_SCHEDULE \
    --sample_every $SAMPLE_EVERY \
    --base_channels $BASE_CHANNELS

echo "=========================================="
echo "Training complete!"
echo "=========================================="
