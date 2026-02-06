"""
Unit tests for UMAP distribution visualization.

Tests: pixel-space feature flattening (shape, determinism, values),
batched sample generation (shape, various batch sizes, timing).
All tests follow the AAA pattern (Arrange, Act, Assert).
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from models.ddpm import DDPM
from models.ddim import DDIMSampler
from models.visualize_distribution import (
    flatten_images_to_pixels,
    generate_samples_batched,
)


class TestFlattenImagesToPixels:
    """Tests for pixel-space feature flattening."""

    def test_output_shape_mnist(self, seed):
        """Flattened MNIST images should have shape (N, 784) = (N, 1*28*28)."""
        # Arrange
        images = torch.randn(8, 1, 28, 28)

        # Act
        features = flatten_images_to_pixels(images)

        # Assert — 1 channel * 28 height * 28 width = 784
        assert features.shape == (8, 784)

    def test_output_is_numpy(self, seed):
        """Returned features should be a numpy array (ready for UMAP)."""
        # Arrange
        images = torch.randn(4, 1, 28, 28)

        # Act
        features = flatten_images_to_pixels(images)

        # Assert
        assert isinstance(features, np.ndarray)

    def test_deterministic(self, seed):
        """Same input should produce identical features across calls."""
        # Arrange
        images = torch.randn(4, 1, 28, 28)

        # Act
        features_first = flatten_images_to_pixels(images)
        features_second = flatten_images_to_pixels(images)

        # Assert
        np.testing.assert_array_equal(features_first, features_second)

    def test_values_preserved(self, seed):
        """Flattening should preserve pixel values exactly."""
        # Arrange
        images = torch.randn(2, 1, 28, 28)

        # Act
        features = flatten_images_to_pixels(images)

        # Assert — first pixel of first image should match
        assert features[0, 0] == pytest.approx(images[0, 0, 0, 0].item())
        # Last pixel of first image
        assert features[0, -1] == pytest.approx(images[0, 0, -1, -1].item())

    def test_different_inputs_produce_different_features(self, seed):
        """Different images should produce different feature vectors."""
        # Arrange
        images_a = torch.randn(4, 1, 28, 28)
        images_b = torch.randn(4, 1, 28, 28) * 2 + 1

        # Act
        features_a = flatten_images_to_pixels(images_a)
        features_b = flatten_images_to_pixels(images_b)

        # Assert
        assert not np.allclose(features_a, features_b)

    def test_features_are_finite(self, seed):
        """Flattened features should contain no NaN or Inf values."""
        # Arrange
        images = torch.randn(4, 1, 28, 28)

        # Act
        features = flatten_images_to_pixels(images)

        # Assert
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()

    def test_single_image(self, seed):
        """Should work with a single image."""
        # Arrange
        images = torch.randn(1, 1, 28, 28)

        # Act
        features = flatten_images_to_pixels(images)

        # Assert
        assert features.shape == (1, 784)


class TestGenerateSamplesBatched:
    """Tests for batched sample generation using DDPM/DDIM."""

    def _create_simple_model(self, device):
        """Create a simple model for testing (avoids slow UNet)."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.convolution = nn.Conv2d(1, 1, 3, padding=1)

            def forward(self, x, timestep):
                return self.convolution(x)

        return SimpleModel().to(device)

    def test_output_shape_ddim(self, device, seed):
        """Generated samples should have shape (num_samples, 1, 28, 28) with DDIM."""
        # Arrange
        ddpm = DDPM(timesteps=100).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=5, eta=0.0).to(device)
        model = self._create_simple_model(device)
        model.eval()
        num_samples = 6

        # Act
        samples, per_sample_times = generate_samples_batched(
            model, ddpm, sampler, num_samples, device, batch_size=4, mode='ddim',
        )

        # Assert
        assert samples.shape == (6, 1, 28, 28)

    def test_output_shape_ddpm(self, device, seed):
        """Generated samples should have shape (num_samples, 1, 28, 28) with DDPM."""
        # Arrange
        ddpm = DDPM(timesteps=10).to(device)
        model = self._create_simple_model(device)
        model.eval()
        num_samples = 4

        # Act
        samples, per_sample_times = generate_samples_batched(
            model, ddpm, None, num_samples, device, batch_size=4, mode='ddpm',
        )

        # Assert
        assert samples.shape == (4, 1, 28, 28)

    def test_batch_size_larger_than_num_samples(self, device, seed):
        """Should work when batch_size exceeds num_samples."""
        # Arrange
        ddpm = DDPM(timesteps=100).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=5, eta=0.0).to(device)
        model = self._create_simple_model(device)
        model.eval()

        # Act — batch_size=64 but only 3 samples requested
        samples, per_sample_times = generate_samples_batched(
            model, ddpm, sampler, 3, device, batch_size=64, mode='ddim',
        )

        # Assert
        assert samples.shape == (3, 1, 28, 28)

    def test_batch_size_smaller_than_num_samples(self, device, seed):
        """Should correctly combine multiple batches when batch_size < num_samples."""
        # Arrange
        ddpm = DDPM(timesteps=100).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=5, eta=0.0).to(device)
        model = self._create_simple_model(device)
        model.eval()

        # Act — batch_size=2 but 7 samples requested → 4 batches (2+2+2+1)
        samples, per_sample_times = generate_samples_batched(
            model, ddpm, sampler, 7, device, batch_size=2, mode='ddim',
        )

        # Assert
        assert samples.shape == (7, 1, 28, 28)

    def test_samples_are_finite(self, device, seed):
        """Generated samples should contain no NaN or Inf values."""
        # Arrange
        ddpm = DDPM(timesteps=100).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=5, eta=0.0).to(device)
        model = self._create_simple_model(device)
        model.eval()

        # Act
        samples, per_sample_times = generate_samples_batched(
            model, ddpm, sampler, 4, device, batch_size=4, mode='ddim',
        )

        # Assert
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()

    def test_samples_on_cpu(self, device, seed):
        """Returned samples should be on CPU (for downstream numpy conversion)."""
        # Arrange
        ddpm = DDPM(timesteps=100).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=5, eta=0.0).to(device)
        model = self._create_simple_model(device)
        model.eval()

        # Act
        samples, per_sample_times = generate_samples_batched(
            model, ddpm, sampler, 4, device, batch_size=4, mode='ddim',
        )

        # Assert
        assert samples.device == torch.device('cpu')

    def test_per_sample_times_length(self, device, seed):
        """Per-sample times list should have exactly num_samples entries."""
        # Arrange
        ddpm = DDPM(timesteps=100).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=5, eta=0.0).to(device)
        model = self._create_simple_model(device)
        model.eval()

        # Act
        samples, per_sample_times = generate_samples_batched(
            model, ddpm, sampler, 7, device, batch_size=3, mode='ddim',
        )

        # Assert
        assert len(per_sample_times) == 7
        assert all(t > 0 for t in per_sample_times)
