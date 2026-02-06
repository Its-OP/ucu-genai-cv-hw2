#!/bin/bash
set -e

EPOCHS=${EPOCHS:-50}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
TIMESTEPS=${TIMESTEPS:-1000}
SAMPLE_EVERY=${SAMPLE_EVERY:-10}
BASE_CHANNELS=${BASE_CHANNELS:-32}

echo "=========================================="
echo "DDPM Training on MNIST"
echo "=========================================="
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Timesteps: $TIMESTEPS"
echo "Beta Schedule: cosine (hardcoded)"
echo "Sample Every: $SAMPLE_EVERY epochs"
echo "Base Channels: $BASE_CHANNELS"
echo "Model: ~3M parameters (base_channels=32)"
echo "=========================================="

python -m models.train \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --timesteps $TIMESTEPS \
    --sample_every $SAMPLE_EVERY \
    --base_channels $BASE_CHANNELS

echo "=========================================="
echo "Training complete!"
echo "=========================================="
