#!/bin/bash
# Train Latent Rectified Flow Model with GPU monitoring
# Run from the root of the repository
# Usage: bash scripts/sh/train-latent-rf.sh <vae_checkpoint> [OPTIONS]
#
# Required:
#   <vae_checkpoint>        Path to pre-trained VAE checkpoint (.pt file)
#
# Options:
#   --epochs N              Number of epochs (default: 100)
#   --lr RATE               Learning rate (default: 1e-3)
#   --batch_size N          Batch size (default: 512)
#   --sampling_steps N      Euler sampling steps (default: 50)
#   --base_channels N       UNet base channel count (default: 64)
#   --sample_every N        Generate samples every N epochs (default: 10)
#
# Note: The latent UNet operates on the VAE's latent space (2x4x4 by default).
#       Uses Rectified Flow (Liu et al. 2022) instead of DDPM for training.
#
# This script launches two screen sessions:
#   - latent-rf-train: runs the training
#   - latent-rf-monitor: runs nvidia-smi for GPU monitoring

set -e

if [ -z "$1" ]; then
    echo "Error: VAE checkpoint path is required"
    echo "Usage: bash scripts/sh/train-latent-rf.sh <vae_checkpoint> [OPTIONS]"
    exit 1
fi

VAE_CHECKPOINT="$1"
shift  # Remove checkpoint from args

# Default configuration
EPOCHS=100
LEARNING_RATE="1e-3"
BATCH_SIZE=512
SAMPLING_STEPS=50
BASE_CHANNELS=64
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
        --sampling_steps)
            SAMPLING_STEPS="$2"
            shift 2
            ;;
        --base_channels)
            BASE_CHANNELS="$2"
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

SESSION_TRAIN="latent-rf-train"
SESSION_MONITOR="latent-rf-monitor"

echo "=========================================="
echo "Latent Rectified Flow Training on MNIST"
echo "=========================================="
echo "VAE Checkpoint: $VAE_CHECKPOINT"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE"
echo "Sampling Steps: $SAMPLING_STEPS"
echo "Base Channels: $BASE_CHANNELS"
echo "Sample Every: $SAMPLE_EVERY epochs"
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
    python -m scripts.python.train_latent_rf \
        --vae_checkpoint $VAE_CHECKPOINT \
        --epochs $EPOCHS \
        --lr $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --sampling_steps $SAMPLING_STEPS \
        --base_channels $BASE_CHANNELS \
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
