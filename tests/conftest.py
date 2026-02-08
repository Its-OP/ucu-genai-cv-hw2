"""
Shared fixtures for DDPM unit tests.

All fixtures use CPU and fixed seeds for deterministic testing.
"""
import pytest
import torch


@pytest.fixture
def device():
    """Use CPU for deterministic testing."""
    return torch.device('cpu')


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    return 42


@pytest.fixture
def sample_image(device, seed):
    """MNIST-like image batch: (4, 1, 28, 28)."""
    return torch.randn(4, 1, 28, 28, device=device)


@pytest.fixture
def sample_timesteps(device):
    """Random timesteps for batch of 4."""
    torch.manual_seed(42)
    return torch.randint(0, 1000, (4,), device=device)


@pytest.fixture
def feature_map_small(device, seed):
    """Small feature map: (4, 64, 8, 8)."""
    return torch.randn(4, 64, 8, 8, device=device)


@pytest.fixture
def feature_map_medium(device, seed):
    """Medium feature map: (4, 64, 16, 16)."""
    return torch.randn(4, 64, 16, 16, device=device)


@pytest.fixture
def time_embedding(device, seed):
    """Time embedding vector: (4, 128)."""
    return torch.randn(4, 128, device=device)


@pytest.fixture
def sample_image_padded(device, seed):
    """Padded MNIST-like image batch: (4, 1, 32, 32)."""
    return torch.randn(4, 1, 32, 32, device=device)


@pytest.fixture
def sample_latent(device, seed):
    """Latent space sample: (4, 2, 4, 4)."""
    return torch.randn(4, 2, 4, 4, device=device)
