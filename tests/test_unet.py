"""
Unit tests for UNet wrapper around diffusers.UNet2DModel.

Tests: Output shape (including pad/crop transparency), gradient flow,
determinism, numerical stability, timestep sensitivity, architecture.
All tests follow the AAA pattern (Arrange, Act, Assert).
"""
import pytest
import torch

from models.unet import UNet, UNet2DModel


class TestUNetOutputShape:
    """Tests for UNet output shape properties."""

    def test_output_shape_mnist(self, device, seed):
        """Output shape should match MNIST input shape (28x28)."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)
        model.eval()
        x = torch.randn(4, 1, 28, 28, device=device)
        timestep = torch.randint(0, 1000, (4,), device=device)

        # Act
        output = model(x, timestep)

        # Assert
        assert output.shape == x.shape

    def test_different_batch_sizes(self, device, seed):
        """Should work with different batch sizes."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)
        model.eval()
        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 1, 28, 28, device=device)
            timestep = torch.randint(0, 1000, (batch_size,), device=device)

            # Act
            output = model(x, timestep)

            # Assert
            assert output.shape == (batch_size, 1, 28, 28)

    def test_rgb_images(self, device, seed):
        """Should work with 3-channel (RGB) images."""
        # Arrange
        model = UNet(
            image_channels=3,
            base_channels=64,
        ).to(device)
        model.eval()
        x = torch.randn(2, 3, 28, 28, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)

        # Act
        output = model(x, timestep)

        # Assert
        assert output.shape == x.shape

    def test_padding_cropping_transparent_for_32x32(self, device, seed):
        """32×32 input should pass through without padding/cropping."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)
        model.eval()
        x = torch.randn(2, 1, 32, 32, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)

        # Act
        output = model(x, timestep)

        # Assert
        assert output.shape == (2, 1, 32, 32)


class TestUNetGradientFlow:
    """Tests for gradient flow through UNet."""

    def test_backward_pass(self, device, seed):
        """Backward pass should complete without errors."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)
        x = torch.randn(2, 1, 28, 28, device=device, requires_grad=True)
        timestep = torch.randint(0, 1000, (2,), device=device)

        # Act
        output = model(x, timestep)
        loss = output.sum()
        loss.backward()

        # Assert
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_all_parameters_receive_gradients(self, device, seed):
        """All model parameters should receive gradients."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)

        # Act
        output = model(x, timestep)
        loss = output.sum()
        loss.backward()

        # Assert
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, f"No gradient for {name}"


class TestUNetDeterminism:
    """Tests for deterministic behavior."""

    def test_deterministic_eval_mode(self, device):
        """Same input should produce same output in eval mode."""
        # Arrange
        torch.manual_seed(42)
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)
        model.eval()

        x = torch.randn(2, 1, 28, 28, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)

        # Act
        output_1 = model(x, timestep)
        output_2 = model(x, timestep)

        # Assert
        torch.testing.assert_close(output_1, output_2)


class TestUNetConfigurations:
    """Tests for different UNet configurations."""

    def test_default_base_channels(self, device, seed):
        """Should work with default base channel count (32)."""
        # Arrange — use default base_channels (32)
        model = UNet(
            image_channels=1,
        ).to(device)
        model.eval()
        x = torch.randn(2, 1, 28, 28, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)

        # Act
        output = model(x, timestep)

        # Assert
        assert output.shape == x.shape

    def test_default_base_channels_parameter_count(self, device, seed):
        """Default config should give <3M parameters (lightweight for MNIST)."""
        # Arrange — use all defaults: base_channels=32, multipliers=(1,2,3,3), layers=1
        model = UNet(
            image_channels=1,
        ).to(device)

        # Act
        number_of_parameters = sum(p.numel() for p in model.parameters())

        # Assert — ~2.7M params with (32, 64, 96, 96) channels, 1 layer/block
        assert 1_000_000 < number_of_parameters < 3_000_000

    def test_large_base_channels(self, device, seed):
        """Should work with base_channels=64 for higher capacity."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)
        model.eval()
        x = torch.randn(2, 1, 28, 28, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)

        # Act
        output = model(x, timestep)

        # Assert
        assert output.shape == x.shape

    def test_grayscale_channel(self, device, seed):
        """Should work with 1-channel (grayscale) images."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)
        model.eval()
        x = torch.randn(2, 1, 28, 28, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)

        # Act
        output = model(x, timestep)

        # Assert
        assert output.shape == x.shape

    def test_rgb_channel(self, device, seed):
        """Should work with 3-channel (RGB) images."""
        # Arrange
        model = UNet(
            image_channels=3,
            base_channels=64,
        ).to(device)
        model.eval()
        x = torch.randn(2, 3, 28, 28, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)

        # Act
        output = model(x, timestep)

        # Assert
        assert output.shape == x.shape

    def test_custom_architecture_parameters(self, device, seed):
        """Should work with custom channel multipliers, layers, and attention levels."""
        # Arrange — use the original DDPM (Ho et al. 2020) config for comparison
        model = UNet(
            image_channels=1,
            base_channels=32,
            channel_multipliers=(1, 2, 4, 4),
            layers_per_block=2,
            attention_levels=(False, False, True, True),
        ).to(device)
        model.eval()
        x = torch.randn(2, 1, 28, 28, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)

        # Act
        output = model(x, timestep)

        # Assert
        assert output.shape == x.shape
        number_of_parameters = sum(p.numel() for p in model.parameters())
        # Original DDPM config with base_channels=32 should give ~6.7M params
        assert 5_000_000 < number_of_parameters < 8_000_000

    def test_three_level_architecture(self, device, seed):
        """Should work with 3 resolution levels instead of 4."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=32,
            channel_multipliers=(1, 2, 4),
            layers_per_block=1,
            attention_levels=(False, False, True),
        ).to(device)
        model.eval()
        x = torch.randn(2, 1, 28, 28, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)

        # Act
        output = model(x, timestep)

        # Assert
        assert output.shape == x.shape


class TestUNetNumericalStability:
    """Tests for numerical stability."""

    def test_no_nan_output(self, device, seed):
        """Output should not contain NaN values."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)
        model.eval()
        x = torch.randn(4, 1, 28, 28, device=device)
        timestep = torch.randint(0, 1000, (4,), device=device)

        # Act
        output = model(x, timestep)

        # Assert
        assert not torch.isnan(output).any()

    def test_no_inf_output(self, device, seed):
        """Output should not contain Inf values."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)
        model.eval()
        x = torch.randn(4, 1, 28, 28, device=device)
        timestep = torch.randint(0, 1000, (4,), device=device)

        # Act
        output = model(x, timestep)

        # Assert
        assert not torch.isinf(output).any()

    def test_bounded_output_range(self, device, seed):
        """Output should have reasonable magnitude."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)
        model.eval()
        x = torch.randn(4, 1, 28, 28, device=device)
        timestep = torch.randint(0, 1000, (4,), device=device)

        # Act
        output = model(x, timestep)

        # Assert
        # Output should be within reasonable bounds (not exploding)
        assert output.abs().max() < 100


class TestUNetSkipConnections:
    """Tests for skip connection behavior."""

    def test_skip_connections_preserve_information(self, device, seed):
        """Skip connections should help preserve spatial information."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)
        model.eval()

        # Create input with known spatial pattern
        x = torch.zeros(1, 1, 28, 28, device=device)
        x[:, :, 10:18, 10:18] = 1.0  # Square in the center
        timestep = torch.tensor([0], device=device)  # t=0 means minimal noise

        # Act
        output = model(x, timestep)

        # Assert
        # The model should produce meaningful output (not all zeros or constant)
        assert output.std() > 0.01  # Output has variation


class TestUNetArchitecture:
    """Tests for UNet architectural properties."""

    def test_model_wraps_unet2d_model(self, device, seed):
        """Model should wrap a diffusers UNet2DModel instance."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)

        # Assert
        assert hasattr(model, 'model'), "UNet should have a .model attribute"
        assert isinstance(model.model, UNet2DModel), "model.model should be a UNet2DModel"

    def test_different_timesteps_produce_different_outputs(self, device, seed):
        """Different timesteps should produce different outputs."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)
        model.eval()
        x = torch.randn(1, 1, 28, 28, device=device)

        timestep_0 = torch.tensor([0], device=device)
        timestep_500 = torch.tensor([500], device=device)
        timestep_999 = torch.tensor([999], device=device)

        # Act
        output_0 = model(x, timestep_0)
        output_500 = model(x, timestep_500)
        output_999 = model(x, timestep_999)

        # Assert
        assert not torch.allclose(output_0, output_500)
        assert not torch.allclose(output_500, output_999)
        assert not torch.allclose(output_0, output_999)

    def test_parameter_count_reasonable(self, device, seed):
        """Model should have a reasonable number of parameters."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)

        # Act
        number_of_parameters = sum(p.numel() for p in model.parameters())

        # Assert
        # Model should have between 1M and 50M parameters for MNIST
        assert 1_000_000 < number_of_parameters < 50_000_000
