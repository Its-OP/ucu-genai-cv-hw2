import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = "./data/datasets"
DEFAULT_BATCH_SIZE = 512  # Larger batch for better GPU utilization

# Use multiple workers on Linux/CUDA, but 0 on macOS (where it's slower)
NUM_WORKERS = 4 if torch.cuda.is_available() else 0

# Pin memory for faster CPU->GPU transfers (only useful with CUDA)
PIN_MEMORY = torch.cuda.is_available()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = datasets.MNIST(root=DATA_ROOT, train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=False, transform=transform)


def get_train_loader(batch_size: int = None) -> DataLoader:
    """Create training data loader with specified batch size."""
    batch_size = batch_size or DEFAULT_BATCH_SIZE
    num_workers = NUM_WORKERS
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=num_workers > 0,
    )


def get_test_loader(batch_size: int = None) -> DataLoader:
    """Create test data loader with specified batch size."""
    batch_size = batch_size or DEFAULT_BATCH_SIZE
    num_workers = NUM_WORKERS
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=num_workers > 0,
    )


# Default loaders for backward compatibility
train_loader = get_train_loader()
test_loader = get_test_loader()
