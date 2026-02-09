#!/bin/bash
# Train Rectified Flow model (pixel space) with GPU monitoring
# Run from the root of the repository
# Usage: bash scripts/sh/train-rf.sh [OPTIONS]
#
# Options:
#   --epochs N              Number of epochs (default: 100)
#   --lr RATE               Learning rate (default: 1e-3)
#   --batch_size N          Batch size (default: 512)
#   --sampling_steps N      Euler sampling steps (default: 50)
#   --sample_every N        Generate samples every N epochs (default: 10)
#   --base_channels N       Base channel count (default: 32)
#
# Note: Uses the same UNet architecture as DDPM, but trains with
#       velocity prediction and linear interpolation (Liu et al. 2022).
#
# This script launches two screen sessions:
#   - rf-train: runs the training
#   - rf-monitor: runs nvidia-smi for GPU monitoring

set -e

# Default configuration
EPOCHS=100
LEARNING_RATE="1e-3"
BATCH_SIZE=512
SAMPLING_STEPS=50
SAMPLE_EVERY=10
BASE_CHANNELS=32

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

SESSION_TRAIN="rf-train"
SESSION_MONITOR="rf-monitor"

echo "=========================================="
echo "Rectified Flow Training on MNIST (pixel)"
echo "=========================================="
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE"
echo "Sampling Steps: $SAMPLING_STEPS"
echo "Sample Every: $SAMPLE_EVERY epochs"
echo "Base Channels: $BASE_CHANNELS"
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
    python -m scripts.python.train_rf \
        --epochs $EPOCHS \
        --lr $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --sampling_steps $SAMPLING_STEPS \
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
