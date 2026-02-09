"""
Shared utilities for training and generation scripts.

Contains common functions and classes used across DDPM, VAE, and
Latent Diffusion training and generation pipelines:
  - Device selection
  - ExponentialMovingAverage for weight smoothing
  - MNIST padding (28x28 → 32x32)
  - Checkpoint saving/loading (UNet, VAE)
  - Experiment logging (config, epoch, performance metrics)
  - Image saving (grids, denoising progression, single images)
  - Generation profiling
  - Denoising intermediate step computation (DDPM/DDIM)
"""
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid


# ---------------------------------------------------------------------------
#  Device selection
# ---------------------------------------------------------------------------

def get_device(requested_device=None):
    """Get the best available device, or use the requested one.

    Priority: requested_device > CUDA > MPS > CPU.

    Args:
        requested_device: Optional device string to force (e.g., 'cuda', 'mps', 'cpu').
                          If None, auto-detects the best available device.

    Returns:
        torch.device for computation.
    """
    if requested_device is not None:
        return torch.device(requested_device)

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
#  Exponential Moving Average
# ---------------------------------------------------------------------------

class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) of model parameters.

    Maintains shadow copies of model parameters that are updated as:
        shadow_param = decay * shadow_param + (1 - decay) * param

    Used to produce smoother, higher-quality outputs during inference.

    Args:
        model: The model whose parameters will be tracked.
        decay (float): EMA decay rate. Higher values give smoother averaging.
                       Default: 0.995 (standard for small models).
    """

    def __init__(self, model, decay=0.995):
        self.decay = decay
        # Deep copy all parameters as shadow weights
        self.shadow_parameters = [
            parameter.clone().detach() for parameter in model.parameters()
        ]

    def update(self, model):
        """Update shadow parameters with current model parameters.

        Formula: shadow = decay * shadow + (1 - decay) * param
        """
        for shadow_parameter, model_parameter in zip(
            self.shadow_parameters, model.parameters()
        ):
            shadow_parameter.data.mul_(self.decay).add_(
                model_parameter.data, alpha=1.0 - self.decay
            )

    def apply_shadow(self, model):
        """Replace model parameters with shadow (EMA) parameters for inference."""
        self.backup_parameters = [
            parameter.clone() for parameter in model.parameters()
        ]
        for model_parameter, shadow_parameter in zip(
            model.parameters(), self.shadow_parameters
        ):
            model_parameter.data.copy_(shadow_parameter.data)

    def restore(self, model):
        """Restore original model parameters after inference."""
        for model_parameter, backup_parameter in zip(
            model.parameters(), self.backup_parameters
        ):
            model_parameter.data.copy_(backup_parameter.data)


# ---------------------------------------------------------------------------
#  MNIST padding
# ---------------------------------------------------------------------------

def pad_to_32(images: torch.Tensor) -> torch.Tensor:
    """
    Pad 28x28 MNIST images to 32x32 using reflect padding.

    This matches the UNet wrapper's padding strategy so the VAE
    operates on the same 32x32 input space.

    Padding: 2 pixels on each side (left, right, top, bottom).

    Args:
        images: Tensor of shape (B, C, 28, 28).

    Returns:
        Padded tensor of shape (B, C, 32, 32).
    """
    return F.pad(images, (2, 2, 2, 2), mode="reflect")


# ---------------------------------------------------------------------------
#  Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(model, checkpoint_path, config, epoch, train_loss, eval_loss,
                    extra_fields=None):
    """Save model checkpoint with EMA weights and configuration.

    The checkpoint contains everything needed to reconstruct the model:
    architecture config + EMA-smoothed weights.

    Note: Call this while EMA weights are applied to the model
    (after ema.apply_shadow()) so model.state_dict() returns EMA weights.

    Args:
        model: The model with EMA weights currently applied.
        checkpoint_path: Path to save the .pt file.
        config: Dict with model architecture config.
        epoch: Current epoch number (0-indexed).
        train_loss: Training loss at this epoch (scalar).
        eval_loss: Evaluation loss at this epoch (scalar).
        extra_fields: Optional dict of additional fields to include
                      in the checkpoint (e.g., per-component losses).
    """
    # Handle torch.compile wrapper: extract original module's state_dict
    # torch.compile may be applied to the top-level model or to submodules
    # (e.g., conditioned_unet.unet = torch.compile(conditioned_unet.unet)).
    # In both cases, state_dict keys get a '_orig_mod.' prefix that must be
    # stripped so checkpoints are loadable without recompiling.
    if hasattr(model, '_orig_mod'):
        model_state_dict = model._orig_mod.state_dict()
    else:
        model_state_dict = model.state_dict()

    # Strip any remaining '_orig_mod.' from keys (handles compiled submodules)
    cleaned_state_dict = {}
    for key, value in model_state_dict.items():
        cleaned_key = key.replace('._orig_mod', '')
        cleaned_state_dict[cleaned_key] = value
    model_state_dict = cleaned_state_dict

    checkpoint = {
        'model_state_dict': model_state_dict,
        'config': config,
        'epoch': epoch,
        'train_loss': train_loss,
        'eval_loss': eval_loss,
    }
    if extra_fields is not None:
        checkpoint.update(extra_fields)

    torch.save(checkpoint, checkpoint_path)


# ---------------------------------------------------------------------------
#  Checkpoint loading (UNet + DDPM, VAE)
# ---------------------------------------------------------------------------

def load_unet_checkpoint(checkpoint_path, device):
    """Load a UNet + DDPM checkpoint and reconstruct the models.

    Handles pixel-space DDPM checkpoints, latent-space UNet checkpoints,
    and class-conditioned latent-space checkpoints (detected via the
    ``conditioned`` flag in the saved config).

    For conditioned checkpoints, reconstructs a ClassConditionedUNet that
    wraps the UNet with a learnable class embedding.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load the model onto.

    Returns:
        Tuple of (model, ddpm, config) ready for generation.
        ``model`` is either a bare UNet or a ClassConditionedUNet.
    """
    from models.unet import UNet
    from models.ddpm import DDPM

    print(f"Loading UNet checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Strip '_orig_mod.' from state_dict keys that torch.compile adds.
    # This happens when compiled submodules (e.g., conditioned_unet.unet)
    # are saved without unwrapping first.
    state_dict = checkpoint['model_state_dict']
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key.replace('._orig_mod', '')
        cleaned_state_dict[cleaned_key] = value
    checkpoint['model_state_dict'] = cleaned_state_dict

    config = checkpoint['config']
    print(f"  UNet config: image_channels={config['image_channels']}, "
          f"base_channels={config['base_channels']}, timesteps={config['timesteps']}")
    print(f"  Trained for {checkpoint['epoch'] + 1} epochs "
          f"(train_loss={checkpoint['train_loss']:.6f}, eval_loss={checkpoint['eval_loss']:.6f})")

    # Backward compatibility: old checkpoints lack architecture fields,
    # fall back to the original DDPM defaults (Ho et al. 2020)
    channel_multipliers = tuple(config.get('channel_multipliers', (1, 2, 4, 4)))
    layers_per_block = config.get('layers_per_block', 2)
    attention_levels = tuple(config.get('attention_levels', (False, False, True, True)))

    is_conditioned = config.get('conditioned', False)

    if is_conditioned:
        # Conditioned checkpoint: reconstruct ClassConditionedUNet
        from models.classifier_free_guidance import ClassConditionedUNet

        number_of_classes = config['number_of_classes']
        output_channels = config['output_channels']
        spatial_height = config.get('spatial_height', 4)
        spatial_width = config.get('spatial_width', 4)

        print(f"  Conditioned model: {number_of_classes} classes, "
              f"output_channels={output_channels}")

        unet = UNet(
            image_channels=config['image_channels'],
            output_channels=output_channels,
            base_channels=config['base_channels'],
            channel_multipliers=channel_multipliers,
            layers_per_block=layers_per_block,
            attention_levels=attention_levels,
        ).to(device)

        model = ClassConditionedUNet(
            unet=unet,
            number_of_classes=number_of_classes,
            spatial_height=spatial_height,
            spatial_width=spatial_width,
            unconditional_probability=0.0,  # No dropout during generation
        ).to(device)

        # Load full state dict (UNet weights + class embedding weights)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        number_of_parameters = sum(
            parameter.numel() for parameter in model.parameters()
        )
        print(f"  ClassConditionedUNet parameters: {number_of_parameters:,}")

    else:
        # Unconditioned checkpoint: reconstruct bare UNet
        unet = UNet(
            image_channels=config['image_channels'],
            base_channels=config['base_channels'],
            channel_multipliers=channel_multipliers,
            layers_per_block=layers_per_block,
            attention_levels=attention_levels,
        ).to(device)

        # Load EMA weights
        unet.load_state_dict(checkpoint['model_state_dict'])
        unet.eval()

        number_of_parameters = sum(
            parameter.numel() for parameter in unet.parameters()
        )
        print(f"  UNet parameters: {number_of_parameters:,}")
        model = unet

    # Reconstruct DDPM diffusion scheduler
    ddpm = DDPM(timesteps=config['timesteps']).to(device)

    return model, ddpm, config


def load_vae_checkpoint(checkpoint_path, device):
    """Load a pre-trained VAE from checkpoint and freeze all parameters.

    The VAE is set to eval mode with requires_grad=False so it acts
    as a fixed encoder/decoder during latent diffusion training or generation.

    Args:
        checkpoint_path: Path to the VAE .pt checkpoint file.
        device: Device to load the VAE onto.

    Returns:
        Frozen VAE model in eval mode.
    """
    from models.vae import VAE

    print(f"Loading VAE checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    vae_config = checkpoint['config']
    print(f"  VAE config: latent_channels={vae_config['latent_channels']}, "
          f"base_channels={vae_config['base_channels']}, "
          f"channel_multipliers={vae_config['channel_multipliers']}")

    # Reconstruct VAE from saved config
    vae = VAE(
        image_channels=vae_config.get('image_channels', 1),
        latent_channels=vae_config['latent_channels'],
        base_channels=vae_config['base_channels'],
        channel_multipliers=tuple(vae_config['channel_multipliers']),
        num_layers_per_block=vae_config.get('layers_per_block', 1),
    ).to(device)

    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    vae.requires_grad_(False)

    number_of_parameters = sum(parameter.numel() for parameter in vae.parameters())
    print(f"  VAE parameters: {number_of_parameters:,} (frozen)")

    return vae


# ---------------------------------------------------------------------------
#  Experiment logging
# ---------------------------------------------------------------------------

def setup_experiment_folder(exp_dir: str, extra_subfolders: list = None) -> str:
    """Create experiment directory structure.

    Always creates: epoch_samples/, final_samples/, checkpoints/.
    Additional subfolders (e.g., 'denoising_steps', 'latent_space') are
    created only when explicitly requested via extra_subfolders.

    Args:
        exp_dir: Root experiment directory path.
        extra_subfolders: Optional list of additional subfolder names to create.

    Returns:
        The exp_dir path (for chaining convenience).
    """
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f'{exp_dir}/epoch_samples', exist_ok=True)
    os.makedirs(f'{exp_dir}/final_samples', exist_ok=True)
    os.makedirs(f'{exp_dir}/checkpoints', exist_ok=True)
    for subfolder in (extra_subfolders or []):
        os.makedirs(f'{exp_dir}/{subfolder}', exist_ok=True)
    return exp_dir


def log_config(exp_dir: str, config: dict):
    """Log model configuration to a text file."""
    with open(f'{exp_dir}/config.txt', 'w') as file:
        file.write("Configuration\n")
        file.write("=" * 40 + "\n")
        for key, value in config.items():
            file.write(f"{key}: {value}\n")


def log_epoch(exp_dir: str, epoch: int, train_loss: float, eval_loss: float, time_seconds: float):
    """Append epoch log entry to training_log.txt."""
    with open(f'{exp_dir}/training_log.txt', 'a') as file:
        file.write(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, "
                   f"eval_loss={eval_loss:.6f}, time={time_seconds:.2f}s\n")


def save_performance_metrics(
    exp_dir: str,
    total_time: float,
    epochs: int,
    avg_inference_time: float,
    final_train_loss: float,
    final_eval_loss: float,
):
    """Save performance metrics to a text file."""
    with open(f'{exp_dir}/performance.txt', 'w') as file:
        file.write("Performance Metrics\n")
        file.write("=" * 40 + "\n")
        file.write(f"Total training time: {total_time:.2f} seconds\n")
        file.write(f"Total training time: {total_time/60:.2f} minutes\n")
        file.write(f"Average time per epoch: {total_time/epochs:.2f} seconds\n")
        file.write(f"Average inference time (10 samples): {avg_inference_time:.2f} seconds\n")
        file.write(f"Final training loss: {final_train_loss:.6f}\n")
        file.write(f"Final evaluation loss: {final_eval_loss:.6f}\n")


# ---------------------------------------------------------------------------
#  Image saving
# ---------------------------------------------------------------------------

def save_images(images: torch.Tensor, path: str, nrow: int = 5):
    """Save batch of images as a grid.

    Args:
        images: Tensor of shape (B, C, H, W) in [-1, 1] range.
        path: File path for the output image.
        nrow: Number of images per row in the grid.
    """
    images = (images + 1) / 2  # Denormalize [-1, 1] -> [0, 1]
    images = images.clamp(0, 1)

    grid = make_grid(images, nrow=nrow, padding=2)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()


def save_single_image(tensor: torch.Tensor, path: str):
    """Save a single-channel image tensor as a PDF file (for Overleaf import).

    Args:
        tensor: Image tensor of shape (1, 1, H, W) in [-1, 1].
        path: Output file path.
    """
    # Denormalize from [-1, 1] to [0, 1]
    image = (tensor[0, 0] + 1) / 2
    image = image.clamp(0, 1).cpu().numpy()

    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=100)
    plt.close()


def plot_loss_curves(train_losses: list, eval_losses: list, path: str):
    """Plot and save training curves.

    Args:
        train_losses: List of per-epoch training losses.
        eval_losses: List of per-epoch evaluation losses.
        path: File path for the output plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(eval_losses, label='Eval Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=150)
    plt.close()


def save_denoising_progression(intermediates: list, path: str):
    """Save images at various denoising steps as a horizontal strip.

    Args:
        intermediates: List of (timestep, tensor) tuples from the sampling loop.
                       Each tensor should have shape (B, C, H, W) in [-1, 1].
        path: Output file path for the strip image.
    """
    num_steps = len(intermediates)
    fig, axes = plt.subplots(1, num_steps, figsize=(2.5 * num_steps, 2.5))

    if num_steps == 1:
        axes = [axes]

    for index, (timestep, image_tensor) in enumerate(intermediates):
        image = (image_tensor[0, 0] + 1) / 2
        image = image.clamp(0, 1).cpu().numpy()
        axes[index].imshow(image, cmap='gray')
        axes[index].set_title(f't={timestep}', fontsize=9)
        axes[index].axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_individual_denoising_steps(intermediates: list, folder: str):
    """Save each denoising step as individual PDF files.

    Args:
        intermediates: List of (timestep, tensor) tuples.
        folder: Directory path where step images are saved.
    """
    for step_number, (timestep, image_tensor) in enumerate(intermediates):
        step_path = os.path.join(folder, f'step_{step_number:02d}_t{timestep:04d}.pdf')
        save_single_image(image_tensor, step_path)


# ---------------------------------------------------------------------------
#  VAE latent space visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_latent_space_scatter(model, data_loader, device, path, num_samples=1000):
    """Visualize the VAE latent space as a 2D scatter plot colored by digit class.

    Encodes MNIST images through the VAE encoder and reduces the latent
    representation to 2D via global average pooling over the spatial grid:
        posterior.mode() -> (B, 2, 4, 4) -> mean over (H, W) -> (B, 2)

    The resulting 2D points are plotted with colors corresponding to the
    digit class (0–9), showing how well the encoder separates digit
    identities in latent space.

    Args:
        model: VAE model (with EMA weights applied for best quality).
        data_loader: DataLoader yielding (images, labels) tuples.
        device: Torch device for encoding.
        path: Output file path for the scatter plot image.
        num_samples: Maximum number of samples to encode and plot.
    """
    model.eval()

    all_latent_points = []
    all_labels = []
    collected = 0

    for images, labels in data_loader:
        if collected >= num_samples:
            break

        # Pad 28x28 → 32x32 and encode to latent space
        images = F.pad(images.to(device), (2, 2, 2, 2), mode="reflect")
        posterior = model.encode(images)

        # Deterministic encoding (mode = mean of the posterior)
        # Shape: (B, latent_channels, 4, 4)
        latent_means = posterior.mode()

        # Global average pooling over spatial dimensions:
        # (B, latent_channels, 4, 4) → (B, latent_channels)
        # This produces a single 2D point per image when latent_channels=2
        latent_2d = latent_means.mean(dim=[2, 3])

        all_latent_points.append(latent_2d.cpu())
        all_labels.append(labels)
        collected += len(images)

    latent_points = torch.cat(all_latent_points, dim=0)[:num_samples]
    labels = torch.cat(all_labels, dim=0)[:num_samples]

    # Create scatter plot colored by digit class
    fig, axis = plt.subplots(1, 1, figsize=(8, 8))
    colormap = plt.cm.tab10

    for digit in range(10):
        digit_mask = labels == digit
        if digit_mask.any():
            axis.scatter(
                latent_points[digit_mask, 0].numpy(),
                latent_points[digit_mask, 1].numpy(),
                c=[colormap(digit)], alpha=0.5, s=10,
                label=f'Digit {digit}', rasterized=True,
            )

    axis.set_xlabel('Latent dimension 1')
    axis.set_ylabel('Latent dimension 2')
    axis.set_title('VAE Latent Space (global avg pool over spatial grid)')
    axis.legend(fontsize=9, markerscale=3, ncol=2, loc='best')
    axis.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
#  Denoising intermediate step computation
# ---------------------------------------------------------------------------

def compute_denoising_intermediate_steps_ddpm(total_timesteps):
    """Compute timestep indices at every 10% of the DDPM denoising process.

    For DDPM, the loop runs from timestep (T-1) down to 0. We capture snapshots
    at 0%, 10%, 20%, ..., 100% of progress (i.e., at timesteps T-1, ~90%*T, ..., 0).

    Args:
        total_timesteps: Total number of DDPM timesteps (T).

    Returns:
        List of unique integer timestep indices at which to record intermediates.
    """
    intermediate_steps = []
    for percentage in range(0, 101, 10):
        # percentage=0 means the very start (timestep T-1, pure noise)
        # percentage=100 means the very end (timestep 0, clean image)
        step_index = int((total_timesteps - 1) * (1.0 - percentage / 100.0))
        intermediate_steps.append(step_index)
    # Remove duplicates while preserving order (can happen at small T)
    seen = set()
    unique_steps = []
    for step in intermediate_steps:
        if step not in seen:
            seen.add(step)
            unique_steps.append(step)
    return unique_steps


def compute_denoising_intermediate_steps_ddim(timestep_sequence):
    """Compute timestep indices at every 10% of the DDIM denoising process.

    For DDIM, the loop runs through a subsequence in reverse. We capture snapshots
    at 0%, 10%, 20%, ..., 100% of progress through that reversed subsequence.

    Args:
        timestep_sequence: The DDIM timestep subsequence tensor (ascending order).

    Returns:
        List of unique integer timestep indices at which to record intermediates.
    """
    reversed_sequence = torch.flip(timestep_sequence, [0])
    total_steps = len(reversed_sequence)
    intermediate_steps = []
    for percentage in range(0, 101, 10):
        # Map percentage of progress to an index in the reversed sequence
        step_index = min(int(percentage / 100.0 * (total_steps - 1)), total_steps - 1)
        timestep_value = reversed_sequence[step_index].item()
        intermediate_steps.append(timestep_value)
    # Remove duplicates while preserving order
    seen = set()
    unique_steps = []
    for step in intermediate_steps:
        if step not in seen:
            seen.add(step)
            unique_steps.append(step)
    return unique_steps


# ---------------------------------------------------------------------------
#  Generation profiling
# ---------------------------------------------------------------------------

def write_generation_profile(
    profile_path,
    per_sample_times,
    mode,
    ddim_steps=None,
    eta=None,
    total_timesteps=None,
):
    """Write generation profiling information to a text file.

    Args:
        profile_path: Path to the output profile.txt file.
        per_sample_times: List of per-sample generation times in seconds.
        mode: Sampling mode ('ddpm' or 'ddim').
        ddim_steps: Number of DDIM steps (only for ddim mode).
        eta: DDIM eta parameter (only for ddim mode).
        total_timesteps: Total DDPM timesteps T.
    """
    total_time = sum(per_sample_times)
    average_time = total_time / len(per_sample_times) if per_sample_times else 0.0

    with open(profile_path, 'w') as file:
        file.write("Generation Profile\n")
        file.write("=" * 50 + "\n\n")

        # Sampling configuration
        file.write("Configuration\n")
        file.write("-" * 50 + "\n")
        file.write(f"Sampling mode:       {mode}\n")
        if mode == 'ddim':
            file.write(f"DDIM steps:          {ddim_steps}\n")
            file.write(f"DDIM eta:            {eta}\n")
        if total_timesteps is not None:
            file.write(f"Total DDPM timesteps: {total_timesteps}\n")
        file.write(f"Number of samples:   {len(per_sample_times)}\n\n")

        # Per-sample timing
        file.write("Per-sample timing\n")
        file.write("-" * 50 + "\n")
        for sample_index, sample_time in enumerate(per_sample_times):
            file.write(f"Sample {sample_index:3d}: {sample_time:.3f}s\n")

        # Summary
        file.write("\n")
        file.write("Summary\n")
        file.write("-" * 50 + "\n")
        file.write(f"Total generation time:   {total_time:.3f}s\n")
        file.write(f"Average time per sample: {average_time:.3f}s\n")
        if len(per_sample_times) > 1:
            min_time = min(per_sample_times)
            max_time = max(per_sample_times)
            file.write(f"Fastest sample:          {min_time:.3f}s\n")
            file.write(f"Slowest sample:          {max_time:.3f}s\n")
