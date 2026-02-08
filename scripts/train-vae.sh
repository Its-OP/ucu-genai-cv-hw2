#!/bin/bash
# Train VAE model with GPU monitoring
# Run from the root of the repository
# Usage: bash scripts/train-vae.sh [OPTIONS]
#
# Options:
#   --epochs N              Number of epochs (default: 100)
#   --lr RATE               Learning rate (default: 1e-4)
#   --batch_size N          Batch size (default: 512)
#   --kl_weight FLOAT       KL divergence weight (default: 1e-6)
#   --base_channels N       Base channel count (default: 64)
#   --latent_channels N     Latent space channels (default: 2)
#   --sample_every N        Generate reconstructions every N epochs (default: 10)
#
# Note: base_channels=64, channel_multipliers=(1,2,4) → compresses 1x32x32 to 2x4x4
#
# This script launches two screen sessions:
#   - vae-train: runs the training
#   - vae-monitor: runs nvidia-smi for GPU monitoring

set -e

# Default configuration
EPOCHS=100
LEARNING_RATE="1e-4"
BATCH_SIZE=512
KL_WEIGHT="1e-6"
BASE_CHANNELS=64
LATENT_CHANNELS=2
SAMPLE_EVERY=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --kl_weight)
            KL_WEIGHT="$2"
            shift 2
            ;;
        --base_channels)
            BASE_CHANNELS="$2"
            shift 2
            ;;
        --latent_channels)
            LATENT_CHANNELS="$2"
            shift 2
            ;;
        --sample_every)
            SAMPLE_EVERY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SESSION_TRAIN="vae-train"
SESSION_MONITOR="vae-monitor"

echo "=========================================="
echo "VAE Training on MNIST"
echo "=========================================="
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE"
echo "KL Weight: $KL_WEIGHT"
echo "Base Channels: $BASE_CHANNELS"
echo "Latent Channels: $LATENT_CHANNELS"
echo "Sample Every: $SAMPLE_EVERY epochs"
echo "Target: 1x32x32 -> ${LATENT_CHANNELS}x4x4"
echo "=========================================="

# Kill existing sessions if they exist
screen -S $SESSION_TRAIN -X quit 2>/dev/null || true
screen -S $SESSION_MONITOR -X quit 2>/dev/null || true

# Start GPU monitoring session
echo "Starting GPU monitor in screen session: $SESSION_MONITOR"
screen -dmS $SESSION_MONITOR bash -c "watch -n 1 nvidia-smi"

# Start training session
echo "Starting training in screen session: $SESSION_TRAIN"
screen -dmS $SESSION_TRAIN bash -c "
    python -m models.train_vae \
        --epochs $EPOCHS \
        --lr $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --kl_weight $KL_WEIGHT \
        --base_channels $BASE_CHANNELS \
        --latent_channels $LATENT_CHANNELS \
        --sample_every $SAMPLE_EVERY
    echo ''
    echo 'Training complete! Press any key to exit.'
    read -n 1
"

echo "=========================================="
echo "Screen sessions started!"
echo ""
echo "To attach to training:   screen -r $SESSION_TRAIN"
echo "To attach to monitor:    screen -r $SESSION_MONITOR"
echo "To detach from screen:   Ctrl+A, then D"
echo "To list sessions:        screen -ls"
echo "=========================================="
