#!/bin/bash
# Train DDPM model with GPU monitoring
# Run from the root of the repository
# Usage: bash scripts/train.sh [OPTIONS]
#
# Options:
#   --epochs N          Number of epochs (default: 100)
#   --lr RATE           Learning rate (default: 2e-4)
#   --batch_size N      Batch size (default: 512, use 1024+ for powerful GPUs)
#   --timesteps N       Diffusion timesteps (default: 1000)
#   --beta_schedule S   Beta schedule: linear or cosine (default: cosine)
#   --sample_every N    Generate samples every N epochs (default: 10)
#   --base_channels N   Base channel count (default: 64)
#
# Note: base_channels=64 recommended for MNIST (channel multipliers: 1, 2, 4, 4)
#
# This script launches two screen sessions:
#   - ddpm-train: runs the training
#   - ddpm-monitor: runs nvidia-smi for GPU monitoring

set -e

# Default configuration (optimized for powerful GPUs like 5090)
EPOCHS=100
LEARNING_RATE="2e-4"
BATCH_SIZE=512
TIMESTEPS=1000
BETA_SCHEDULE="cosine"
SAMPLE_EVERY=10
BASE_CHANNELS=64

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
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --beta_schedule)
            BETA_SCHEDULE="$2"
            shift 2
            ;;
        --sample_every)
            SAMPLE_EVERY="$2"
            shift 2
            ;;
        --base_channels)
            BASE_CHANNELS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SESSION_TRAIN="ddpm-train"
SESSION_MONITOR="ddpm-monitor"

echo "=========================================="
echo "DDPM Training on MNIST"
echo "=========================================="
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE"
echo "Timesteps: $TIMESTEPS"
echo "Beta Schedule: $BETA_SCHEDULE"
echo "Sample Every: $SAMPLE_EVERY epochs"
echo "Base Channels: $BASE_CHANNELS"
echo "Model: ~12.8M parameters"
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
    python -m models.train \
        --epochs $EPOCHS \
        --lr $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --timesteps $TIMESTEPS \
        --beta_schedule $BETA_SCHEDULE \
        --sample_every $SAMPLE_EVERY \
        --base_channels $BASE_CHANNELS
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
