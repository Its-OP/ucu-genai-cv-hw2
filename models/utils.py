import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def setup_experiment_folder(exp_dir: str) -> str:
    """Create experiment directory structure."""
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f'{exp_dir}/epoch_samples', exist_ok=True)
    os.makedirs(f'{exp_dir}/final_samples', exist_ok=True)
    os.makedirs(f'{exp_dir}/denoising_steps', exist_ok=True)
    return exp_dir


def save_images(images: torch.Tensor, path: str, nrow: int = 5):
    """Save batch of images as a grid."""
    images = (images + 1) / 2  # Denormalize [-1, 1] -> [0, 1]
    images = images.clamp(0, 1)

    grid = make_grid(images, nrow=nrow, padding=2)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_loss_curves(train_losses: list, eval_losses: list, path: str):
    """Plot and save training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(eval_losses, label='Eval Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('DDPM Training Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=150)
    plt.close()


def save_denoising_progression(intermediates: list, path: str):
    """Save images at various denoising steps."""
    n_steps = len(intermediates)
    fig, axes = plt.subplots(1, n_steps, figsize=(3 * n_steps, 3))

    if n_steps == 1:
        axes = [axes]

    for idx, (t, img) in enumerate(intermediates):
        img_denorm = (img[0, 0] + 1) / 2  # First image, single channel
        axes[idx].imshow(img_denorm.cpu().numpy(), cmap='gray')
        axes[idx].set_title(f't={t}')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_individual_denoising_steps(intermediates: list, folder: str):
    """Save each denoising step as individual images."""
    for t, images in intermediates:
        for i in range(min(10, images.shape[0])):
            img = (images[i, 0] + 1) / 2
            img = img.clamp(0, 1).cpu().numpy()

            plt.figure(figsize=(3, 3))
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.savefig(f'{folder}/step_{t:04d}_sample_{i}.png', bbox_inches='tight', dpi=100)
            plt.close()


def log_config(exp_dir: str, config: dict):
    """Log model configuration."""
    with open(f'{exp_dir}/config.txt', 'w') as f:
        f.write("DDPM Configuration\n")
        f.write("=" * 40 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


def log_epoch(exp_dir: str, epoch: int, train_loss: float, eval_loss: float, time_s: float):
    """Append epoch log."""
    with open(f'{exp_dir}/training_log.txt', 'a') as f:
        f.write(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, eval_loss={eval_loss:.6f}, time={time_s:.2f}s\n")


def save_performance_metrics(
    exp_dir: str,
    total_time: float,
    epochs: int,
    avg_inference_time: float,
    final_train_loss: float,
    final_eval_loss: float,
):
    """Save performance metrics."""
    with open(f'{exp_dir}/performance.txt', 'w') as f:
        f.write("Performance Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total training time: {total_time:.2f} seconds\n")
        f.write(f"Total training time: {total_time/60:.2f} minutes\n")
        f.write(f"Average time per epoch: {total_time/epochs:.2f} seconds\n")
        f.write(f"Average inference time (10 samples): {avg_inference_time:.2f} seconds\n")
        f.write(f"Final training loss: {final_train_loss:.6f}\n")
        f.write(f"Final evaluation loss: {final_eval_loss:.6f}\n")
