"""
Unit tests for UNet architecture.

Tests: Shape transformations, gradient flow, different configurations.
All tests follow the AAA pattern (Arrange, Act, Assert).
"""
import pytest
import torch

from models.unet import UNet


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
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


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
        """Should work with default base channel count (64)."""
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

    def test_model_has_encoder_decoder_structure(self, device, seed):
        """Model should have encoder and decoder blocks."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)

        # Assert
        assert hasattr(model, 'encoder_blocks'), "Model should have encoder_blocks"
        assert hasattr(model, 'decoder_blocks'), "Model should have decoder_blocks"
        assert hasattr(model, 'bottleneck'), "Model should have bottleneck"
        assert len(model.encoder_blocks) == len(model.decoder_blocks)

    def test_model_has_time_embedding(self, device, seed):
        """Model should have time embedding mechanism."""
        # Arrange
        model = UNet(
            image_channels=1,
            base_channels=64,
        ).to(device)

        # Assert
        assert hasattr(model, 'positional_encoding'), "Model should have positional_encoding"

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
        num_params = sum(p.numel() for p in model.parameters())

        # Assert
        # Model should have between 1M and 50M parameters for MNIST
        assert 1_000_000 < num_params < 50_000_000
