"""
Unit tests for UMAP distribution visualization.

Tests: bottleneck feature extraction (shape, determinism, no gradients,
batch processing), batched sample generation (shape, various batch sizes).
All tests follow the AAA pattern (Arrange, Act, Assert).
"""
import pytest
import torch
import torch.nn as nn

from models.unet import UNet
from models.ddpm import DDPM
from models.ddim import DDIMSampler
from models.visualize_distribution import (
    extract_bottleneck_features,
    generate_samples_batched,
)


class TestExtractBottleneckFeatures:
    """Tests for UNet bottleneck feature extraction via forward hooks."""

    def test_output_shape(self, device, seed):
        """Extracted features should have shape (N, bottleneck_channels)."""
        # Arrange
        model = UNet(image_channels=1, base_channels=32).to(device)
        model.eval()
        images = torch.randn(8, 1, 28, 28, device=device)

        # Act
        features = extract_bottleneck_features(model, images, device, batch_size=4)

        # Assert — bottleneck channels = base_channels * last multiplier = 32 * 3 = 96
        assert features.shape == (8, 96)

    def test_no_gradient_computation(self, device, seed):
        """Extracted features should not have gradient tracking."""
        # Arrange
        model = UNet(image_channels=1, base_channels=32).to(device)
        model.eval()
        images = torch.randn(4, 1, 28, 28, device=device)

        # Act
        features = extract_bottleneck_features(model, images, device, batch_size=4)

        # Assert
        assert features.grad_fn is None
        assert not features.requires_grad

    def test_deterministic(self, device):
        """Same input should produce identical features across calls."""
        # Arrange
        torch.manual_seed(42)
        model = UNet(image_channels=1, base_channels=32).to(device)
        model.eval()
        images = torch.randn(4, 1, 28, 28, device=device)

        # Act
        features_first = extract_bottleneck_features(model, images, device, batch_size=4)
        features_second = extract_bottleneck_features(model, images, device, batch_size=4)

        # Assert
        torch.testing.assert_close(features_first, features_second)

    def test_batch_processing_consistency(self, device):
        """Features should be identical regardless of batch_size used for extraction."""
        # Arrange
        torch.manual_seed(42)
        model = UNet(image_channels=1, base_channels=32).to(device)
        model.eval()
        images = torch.randn(8, 1, 28, 28, device=device)

        # Act — process all at once vs in batches of 2
        features_single_batch = extract_bottleneck_features(
            model, images, device, batch_size=8,
        )
        features_small_batches = extract_bottleneck_features(
            model, images, device, batch_size=2,
        )

        # Assert
        torch.testing.assert_close(features_single_batch, features_small_batches)

    def test_features_vary_across_different_inputs(self, device, seed):
        """Different input images should produce different feature vectors."""
        # Arrange
        model = UNet(image_channels=1, base_channels=32).to(device)
        model.eval()
        images_a = torch.randn(4, 1, 28, 28, device=device)
        images_b = torch.randn(4, 1, 28, 28, device=device) * 2 + 1

        # Act
        features_a = extract_bottleneck_features(model, images_a, device, batch_size=4)
        features_b = extract_bottleneck_features(model, images_b, device, batch_size=4)

        # Assert
        assert not torch.allclose(features_a, features_b)

    def test_features_are_finite(self, device, seed):
        """Extracted features should contain no NaN or Inf values."""
        # Arrange
        model = UNet(image_channels=1, base_channels=32).to(device)
        model.eval()
        images = torch.randn(4, 1, 28, 28, device=device)

        # Act
        features = extract_bottleneck_features(model, images, device, batch_size=4)

        # Assert
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()

    def test_hook_cleanup(self, device, seed):
        """Forward hooks should be removed after feature extraction."""
        # Arrange
        model = UNet(image_channels=1, base_channels=32).to(device)
        model.eval()
        images = torch.randn(4, 1, 28, 28, device=device)

        # Count hooks before
        hooks_before = len(model.model.mid_block._forward_hooks)

        # Act
        extract_bottleneck_features(model, images, device, batch_size=4)

        # Assert — hooks should be cleaned up
        hooks_after = len(model.model.mid_block._forward_hooks)
        assert hooks_after == hooks_before


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
        samples = generate_samples_batched(
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
        samples = generate_samples_batched(
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
        samples = generate_samples_batched(
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
        samples = generate_samples_batched(
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
        samples = generate_samples_batched(
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
        samples = generate_samples_batched(
            model, ddpm, sampler, 4, device, batch_size=4, mode='ddim',
        )

        # Assert
        assert samples.device == torch.device('cpu')
