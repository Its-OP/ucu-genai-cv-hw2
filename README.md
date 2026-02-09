# Diffusion Models for MNIST

Custom PyTorch implementations of DDPM, DDIM, VAE, and Latent Diffusion Models for MNIST digit generation. Built from scratch without relying on diffusion libraries.

## Models

| Model | Description | Reference |
|-------|-------------|-----------|
| **DDPM** | Denoising Diffusion Probabilistic Model (pixel space) | Ho et al. 2020 |
| **DDIM** | Faster sampling via implicit models (50 steps vs 1000) | Song et al. 2020 |
| **VAE** | Variational Autoencoder (compresses 1x32x32 to 2x4x4) | Kingma & Welling 2013 |
| **LDM** | Latent Diffusion: trains DDPM in VAE's latent space | Rombach et al. 2022 |
| **Conditioned LDM** | Class-conditioned LDM with classifier-free guidance | Ho & Salimans 2022 |

## Setup

```bash
# Clone and create conda environment
bash scripts/sh/setup.sh

# Or manually:
conda create -n diffusion python=3.11 -y
conda activate diffusion
pip install -r requirements.txt
```

**Requirements:** Python 3.11+, PyTorch 2.10, torchvision, matplotlib, umap-learn, tqdm.

MNIST is downloaded automatically on first run.

## Training

All training scripts run from the repository root. On a GPU server, they launch `screen` sessions for training and GPU monitoring.

### DDPM (pixel-space diffusion)

```bash
bash scripts/sh/train-ddpm.sh [OPTIONS]

# Options:
#   --epochs N          (default: 100)
#   --lr RATE           (default: 1e-3)
#   --batch_size N      (default: 512)
#   --timesteps N       (default: 1000)
#   --base_channels N   (default: 32)
#   --sample_every N    (default: 10)
```

Or directly:
```bash
python -m scripts.python.train_ddpm --epochs 100 --base_channels 32
```

### VAE

```bash
bash scripts/sh/train-vae.sh [OPTIONS]

# Options:
#   --epochs N              (default: 20)
#   --lr RATE               (default: 1e-4)
#   --kl_weight FLOAT       (default: 1e-6)
#   --base_channels N       (default: 64)
#   --latent_channels N     (default: 2)
#   --sample_every N        (default: 4)
```

### Latent Diffusion Model

Requires a pre-trained VAE checkpoint:

```bash
bash scripts/sh/train-latent-diffusion.sh path/to/vae_checkpoint.pt [OPTIONS]

# Options:
#   --epochs N              (default: 100)
#   --lr RATE               (default: 1e-3)
#   --timesteps N           (default: 1000)
#   --base_channels N       (default: 64)
#   --sample_every N        (default: 10)
```

### Conditioned Latent Diffusion Model

Requires a pre-trained VAE checkpoint. Adds class conditioning via input channel concatenation with classifier-free guidance (CFG):

```bash
bash scripts/sh/train-conditioned-ldm.sh path/to/vae_checkpoint.pt [OPTIONS]

# Options:
#   --epochs N                      (default: 100)
#   --lr RATE                       (default: 1e-3)
#   --timesteps N                   (default: 1000)
#   --base_channels N               (default: 64)
#   --sample_every N                (default: 10)
#   --guidance_scale FLOAT          (default: 3.0)
#   --unconditional_probability FLOAT (default: 0.1)
```

Or directly:
```bash
python -m scripts.python.train_conditioned_latent_diffusion \
    --vae_checkpoint path/to/vae.pt --epochs 100 --guidance_scale 3.0
```

## Generation

### DDPM / DDIM samples

```bash
# DDPM (full 1000 steps)
bash scripts/sh/generate-ddpm.sh path/to/checkpoint.pt

# DDIM (faster, 50 steps)
bash scripts/sh/generate-ddpm.sh path/to/checkpoint.pt --mode ddim --ddim_steps 50

# Options: --num_samples N, --nrow N, --eta FLOAT
```

### Latent Diffusion samples

```bash
# DDPM sampling in latent space
bash scripts/sh/generate-ldm.sh path/to/vae.pt path/to/unet.pt

# DDIM sampling in latent space
bash scripts/sh/generate-ldm.sh path/to/vae.pt path/to/unet.pt --mode ddim --ddim_steps 50
```

### Conditioned Latent Diffusion samples

```bash
# Generate one sample per class (0-9)
bash scripts/sh/generate-conditioned-ldm.sh path/to/vae.pt path/to/cond_unet.pt

# Generate specific digit class
bash scripts/sh/generate-conditioned-ldm.sh path/to/vae.pt path/to/cond_unet.pt --class_label 7

# Options: --guidance_scale FLOAT, --num_samples N, --mode ddim, --ddim_steps N
```

### UMAP distribution visualization

Compares generated vs real MNIST distributions in pixel space:

```bash
bash scripts/sh/visualize_distribution.sh path/to/checkpoint.pt
bash scripts/sh/visualize_distribution.sh path/to/checkpoint.pt --mode ddim --ddim_steps 50
```

## Output Structure

Each training run creates a timestamped experiment directory under `experiments/`:

```
experiments/20260208_143052-ddpm/
  config.txt                    # Hyperparameters
  training_log.txt              # Per-epoch losses and timing
  loss_curves.png               # Train/eval loss plot
  performance_metrics.txt       # Summary statistics
  epoch_samples/                # Periodic sample grids
  final_samples/                # Final generated samples
  checkpoints/                  # Model checkpoints (.pt)
  denoising_steps/              # Denoising progression (DDPM only)
```

VAE experiments additionally include `latent_space/` with scatter plots of the 2D latent space colored by digit class.

Generation scripts output to `generated_samples/` with per-sample subfolders containing the final image, denoising progression strip, and individual denoising steps (all as PDFs).

## Testing

```bash
python -m pytest tests/ -v
```

Tests covering UNet, DDPM, DDIM, VAE, and classifier-free guidance (output shapes, gradient flow, numerical stability, determinism, CFG formula, various configurations).

## Project Structure

```
models/
  ddpm.py                       # DDPM forward/reverse diffusion (cosine schedule)
  ddim.py                       # DDIM sampler (reuses DDPM's noise network)
  unet.py                       # UNet with timestep conditioning and self-attention
  vae.py                        # VAE encoder/decoder with diagonal Gaussian posterior
  classifier_free_guidance.py   # CFG wrappers for class-conditioned generation
  utils.py                      # Shared utilities: EMA, checkpointing, visualization

scripts/
  python/           # Training and generation scripts
    train_ddpm.py
    train_vae.py
    train_latent_diffusion.py
    train_conditioned_latent_diffusion.py
    generate_ddpm.py
    generate_latent_diffusion.py
    generate_conditioned_latent_diffusion.py
    visualize_distribution.py
  sh/               # Shell wrappers (screen sessions, GPU monitoring)

data/               # MNIST data loading (auto-download, normalized to [-1, 1])
tests/              # Unit tests
```
