#!/bin/bash
# Train Cross-Attention Class-Conditioned Latent Diffusion Model with GPU monitoring
# Run from the root of the repository
# Usage: bash scripts/sh/train-conditioned-ldm-ca.sh <vae_checkpoint> [OPTIONS]
#
# Required:
#   <vae_checkpoint>            Path to pre-trained VAE checkpoint (.pt file)
#
# Options:
#   --epochs N                  Number of epochs (default: 100)
#   --lr RATE                   Learning rate (default: 1e-3)
#   --batch_size N              Batch size (default: 1024)
#   --timesteps N               Diffusion timesteps (default: 1000)
#   --base_channels N           UNet base channel count (default: 64)
#   --sample_every N            Generate samples every N epochs (default: 20)
#   --cross_attention_dim N     Cross-attention embedding dimension (default: 128)
#   --guidance_scale FLOAT      CFG guidance scale for sample generation (default: 3.0)
#   --unconditional_probability FLOAT  Classifier-free dropout probability (default: 0.1)
#
# Note: The cross-attention conditioned UNet operates on VAE latent space (2x4x4).
#       Class labels are embedded as dense vectors and injected via cross-attention.
#       Classifier-free guidance allows generating specific digit classes at inference time.
#
# This script launches two screen sessions:
#   - cldm-ca-train: runs the training
#   - cldm-ca-monitor: runs nvidia-smi for GPU monitoring

set -e

if [ -z "$1" ]; then
    echo "Error: VAE checkpoint path is required"
    echo "Usage: bash scripts/sh/train-conditioned-ldm-ca.sh <vae_checkpoint> [OPTIONS]"
    exit 1
fi

VAE_CHECKPOINT="$1"
shift  # Remove checkpoint from args

# Default configuration
EPOCHS=100
LEARNING_RATE="1e-3"
BATCH_SIZE=1024
TIMESTEPS=1000
BASE_CHANNELS=64
SAMPLE_EVERY=20
CROSS_ATTENTION_DIM=128
GUIDANCE_SCALE="3.0"
UNCONDITIONAL_PROBABILITY="0.1"

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
        --base_channels)
            BASE_CHANNELS="$2"
            shift 2
            ;;
        --sample_every)
            SAMPLE_EVERY="$2"
            shift 2
            ;;
        --cross_attention_dim)
            CROSS_ATTENTION_DIM="$2"
            shift 2
            ;;
        --guidance_scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --unconditional_probability)
            UNCONDITIONAL_PROBABILITY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SESSION_TRAIN="cldm-ca-train"
SESSION_MONITOR="cldm-ca-monitor"

echo "=========================================="
echo "Cross-Attention Conditioned LDM on MNIST"
echo "=========================================="
echo "VAE Checkpoint: $VAE_CHECKPOINT"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE"
echo "Timesteps: $TIMESTEPS"
echo "Beta Schedule: cosine (hardcoded)"
echo "Base Channels: $BASE_CHANNELS"
echo "Cross-Attention Dim: $CROSS_ATTENTION_DIM"
echo "Sample Every: $SAMPLE_EVERY epochs"
echo "Guidance Scale: $GUIDANCE_SCALE"
echo "Unconditional Probability: $UNCONDITIONAL_PROBABILITY"
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
    python -m scripts.python.train_conditioned_latent_diffusion_ca \
        --vae_checkpoint $VAE_CHECKPOINT \
        --epochs $EPOCHS \
        --lr $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --timesteps $TIMESTEPS \
        --base_channels $BASE_CHANNELS \
        --sample_every $SAMPLE_EVERY \
        --cross_attention_dim $CROSS_ATTENTION_DIM \
        --guidance_scale $GUIDANCE_SCALE \
        --unconditional_probability $UNCONDITIONAL_PROBABILITY
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
