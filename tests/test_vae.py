"""
Unit tests for VAE (Variational Autoencoder) model.

Tests: Encoder/decoder output shapes, full forward pass, diagonal Gaussian
distribution, gradient flow, loss computation, configurations, numerical stability.
All tests follow the AAA pattern (Arrange, Act, Assert).
"""
import pytest
import torch

from models.vae import (
    VAE,
    VAEEncoder,
    VAEDecoder,
    DiagonalGaussianDistribution,
    VAEResidualBlock,
    VAEAttentionBlock,
    VAEMidBlock,
)


class TestVAEEncoderOutputShape:
    """Tests for VAE encoder output shape properties."""

    def test_encoder_output_shape_default_config(self, device, seed):
        """Encoder should produce correct latent parameter shape (B, 2, 4, 4)."""
        # Arrange
        encoder = VAEEncoder(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        encoder.eval()
        x = torch.randn(4, 1, 32, 32, device=device)

        # Act
        posterior = encoder(x)

        # Assert — spatial compression 32→4 (factor 8), channel = latent_channels
        assert posterior.mean.shape == (4, 2, 4, 4)
        assert posterior.logvar.shape == (4, 2, 4, 4)

    def test_encoder_different_batch_sizes(self, device, seed):
        """Should work with different batch sizes."""
        # Arrange
        encoder = VAEEncoder(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        encoder.eval()
        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 1, 32, 32, device=device)

            # Act
            posterior = encoder(x)

            # Assert
            assert posterior.mean.shape == (batch_size, 2, 4, 4)

    def test_encoder_spatial_compression_ratio(self, device, seed):
        """Spatial dimensions should be compressed by factor of 8 (32->4)."""
        # Arrange
        encoder = VAEEncoder(
            image_channels=1, latent_channels=2, base_channels=32,
            channel_multipliers=(1, 2, 4),
        ).to(device)
        encoder.eval()
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        posterior = encoder(x)

        # Assert — 3 levels of 2x downsampling: 32 / (2^3) = 4
        assert posterior.mean.shape[2] == 4
        assert posterior.mean.shape[3] == 4

    def test_encoder_different_latent_channels(self, device, seed):
        """Should work with different latent channel counts."""
        # Arrange
        encoder = VAEEncoder(
            image_channels=1, latent_channels=4, base_channels=32
        ).to(device)
        encoder.eval()
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        posterior = encoder(x)

        # Assert
        assert posterior.mean.shape == (2, 4, 4, 4)


class TestVAEDecoderOutputShape:
    """Tests for VAE decoder output shape properties."""

    def test_decoder_output_shape_default_config(self, device, seed):
        """Decoder should reconstruct to correct image shape (B, 1, 32, 32)."""
        # Arrange
        decoder = VAEDecoder(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        decoder.eval()
        z = torch.randn(4, 2, 4, 4, device=device)

        # Act
        output = decoder(z)

        # Assert
        assert output.shape == (4, 1, 32, 32)

    def test_decoder_different_batch_sizes(self, device, seed):
        """Should work with different batch sizes."""
        # Arrange
        decoder = VAEDecoder(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        decoder.eval()
        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            z = torch.randn(batch_size, 2, 4, 4, device=device)

            # Act
            output = decoder(z)

            # Assert
            assert output.shape == (batch_size, 1, 32, 32)

    def test_decoder_spatial_expansion_ratio(self, device, seed):
        """Spatial dimensions should be expanded by factor of 8 (4->32)."""
        # Arrange
        decoder = VAEDecoder(
            image_channels=1, latent_channels=2, base_channels=32,
            channel_multipliers=(1, 2, 4),
        ).to(device)
        decoder.eval()
        z = torch.randn(2, 2, 4, 4, device=device)

        # Act
        output = decoder(z)

        # Assert — 3 levels of 2x upsampling: 4 * (2^3) = 32
        assert output.shape[2] == 32
        assert output.shape[3] == 32


class TestVAEForwardPass:
    """Tests for full VAE encode -> sample -> decode pipeline."""

    def test_forward_output_shapes(self, device, seed):
        """Forward pass should return reconstruction and posterior with correct shapes."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        vae.eval()
        x = torch.randn(4, 1, 32, 32, device=device)

        # Act
        reconstruction, posterior = vae(x)

        # Assert
        assert reconstruction.shape == (4, 1, 32, 32)
        assert posterior.mean.shape == (4, 2, 4, 4)
        assert posterior.logvar.shape == (4, 2, 4, 4)

    def test_encode_decode_roundtrip_shape(self, device, seed):
        """Encoding then decoding should produce same-shaped output."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        vae.eval()
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        posterior = vae.encode(x)
        z = posterior.sample()
        reconstruction = vae.decode(z)

        # Assert
        assert reconstruction.shape == x.shape

    def test_mode_encoding_deterministic(self, device, seed):
        """Using mode() for encoding should be deterministic."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        vae.eval()
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        posterior = vae.encode(x)
        z_1 = posterior.mode()
        z_2 = posterior.mode()

        # Assert — mode returns mean, which is deterministic
        torch.testing.assert_close(z_1, z_2)

    def test_encode_returns_distribution(self, device, seed):
        """Encode should return a DiagonalGaussianDistribution instance."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        vae.eval()
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        posterior = vae.encode(x)

        # Assert
        assert isinstance(posterior, DiagonalGaussianDistribution)


class TestDiagonalGaussianDistribution:
    """Tests for DiagonalGaussianDistribution."""

    def test_sample_shape(self, device, seed):
        """Sample should have shape (B, C, H, W) where C = latent_channels."""
        # Arrange — parameters have 2*latent_channels along dim=1
        parameters = torch.randn(4, 4, 4, 4, device=device)
        distribution = DiagonalGaussianDistribution(parameters)

        # Act
        sample = distribution.sample()

        # Assert — split in half: 4/2 = 2 channels
        assert sample.shape == (4, 2, 4, 4)

    def test_kl_non_negative(self, device, seed):
        """KL divergence should be non-negative."""
        # Arrange
        parameters = torch.randn(4, 4, 4, 4, device=device)
        distribution = DiagonalGaussianDistribution(parameters)

        # Act
        kl_value = distribution.kl()

        # Assert
        assert kl_value >= 0

    def test_kl_approximately_zero_for_standard_normal(self, device, seed):
        """KL should be approximately 0 when mean=0 and logvar=0 (standard normal)."""
        # Arrange — mean=0, logvar=0 → N(0,1), so KL(N(0,1) || N(0,1)) = 0
        parameters = torch.zeros(4, 4, 4, 4, device=device)
        distribution = DiagonalGaussianDistribution(parameters)

        # Act
        kl_value = distribution.kl()

        # Assert
        assert kl_value.abs() < 1e-5

    def test_mode_returns_mean(self, device, seed):
        """Mode should return the mean of the distribution."""
        # Arrange
        parameters = torch.randn(4, 4, 4, 4, device=device)
        distribution = DiagonalGaussianDistribution(parameters)

        # Act
        mode = distribution.mode()

        # Assert
        torch.testing.assert_close(mode, distribution.mean)

    def test_logvar_clamping(self, device, seed):
        """Log-variance should be clamped to [-30, 20]."""
        # Arrange — extreme values that would overflow without clamping
        parameters = torch.randn(2, 4, 4, 4, device=device) * 100
        distribution = DiagonalGaussianDistribution(parameters)

        # Assert
        assert distribution.logvar.min() >= -30.0
        assert distribution.logvar.max() <= 20.0

    def test_reparameterization_produces_different_samples(self, device):
        """Different calls to sample() should produce different results (stochastic)."""
        # Arrange — non-zero variance so samples differ
        parameters = torch.randn(4, 4, 4, 4, device=device)
        distribution = DiagonalGaussianDistribution(parameters)

        # Act
        sample_1 = distribution.sample()
        sample_2 = distribution.sample()

        # Assert — samples should differ (extremely unlikely to be identical)
        assert not torch.allclose(sample_1, sample_2)

    def test_sample_mean_close_to_distribution_mean(self, device, seed):
        """Many samples should average close to the distribution mean."""
        # Arrange
        parameters = torch.cat([
            torch.ones(1, 2, 4, 4, device=device) * 3.0,   # mean = 3.0
            torch.zeros(1, 2, 4, 4, device=device),          # logvar = 0 -> std = 1
        ], dim=1)
        distribution = DiagonalGaussianDistribution(parameters)

        # Act — average many samples
        samples = torch.stack([distribution.sample() for _ in range(1000)], dim=0)
        sample_mean = samples.mean(dim=0)

        # Assert — should be close to 3.0 (within 0.2 for 1000 samples)
        assert (sample_mean - 3.0).abs().mean() < 0.2


class TestVAEGradientFlow:
    """Tests for gradient flow through the full VAE."""

    def test_backward_pass_completes(self, device, seed):
        """Backward pass through reconstruction should complete without errors."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        x = torch.randn(2, 1, 32, 32, device=device, requires_grad=True)

        # Act
        reconstruction, posterior = vae(x)
        loss = reconstruction.sum()
        loss.backward()

        # Assert
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_all_parameters_receive_gradients(self, device, seed):
        """All model parameters should receive gradients during training."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        reconstruction, posterior = vae(x)
        losses = vae.loss(x, reconstruction, posterior, kl_weight=1e-4)
        losses["total_loss"].backward()

        # Assert
        for name, parameter in vae.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, f"No gradient for {name}"

    def test_kl_loss_gradient_flows_through_encoder(self, device, seed):
        """KL loss alone should produce gradients for encoder parameters."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act — only compute KL loss (no reconstruction)
        posterior = vae.encode(x)
        kl_loss = posterior.kl()
        kl_loss.backward()

        # Assert — encoder parameters should have gradients
        for name, parameter in vae.encoder.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, f"No gradient for encoder.{name}"


class TestVAELoss:
    """Tests for VAE loss computation."""

    def test_reconstruction_loss_non_negative(self, device, seed):
        """Reconstruction loss (MSE) should be non-negative."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        reconstruction, posterior = vae(x)
        losses = vae.loss(x, reconstruction, posterior)

        # Assert
        assert losses["reconstruction_loss"] >= 0

    def test_kl_loss_non_negative(self, device, seed):
        """KL loss should be non-negative."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        reconstruction, posterior = vae(x)
        losses = vae.loss(x, reconstruction, posterior)

        # Assert
        assert losses["kl_loss"] >= 0

    def test_total_loss_is_weighted_sum(self, device, seed):
        """Total loss should equal reconstruction_loss + kl_weight * kl_loss."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        x = torch.randn(2, 1, 32, 32, device=device)
        kl_weight = 1e-4

        # Act
        reconstruction, posterior = vae(x)
        losses = vae.loss(x, reconstruction, posterior, kl_weight=kl_weight)

        # Assert
        expected = losses["reconstruction_loss"] + kl_weight * losses["kl_loss"]
        torch.testing.assert_close(losses["total_loss"], expected)

    def test_zero_kl_weight_ignores_kl(self, device, seed):
        """With kl_weight=0, total loss should equal reconstruction loss."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        reconstruction, posterior = vae(x)
        losses = vae.loss(x, reconstruction, posterior, kl_weight=0.0)

        # Assert
        torch.testing.assert_close(
            losses["total_loss"], losses["reconstruction_loss"]
        )

    def test_loss_returns_all_components(self, device, seed):
        """Loss should return a dict with all three components."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        reconstruction, posterior = vae(x)
        losses = vae.loss(x, reconstruction, posterior)

        # Assert
        assert "reconstruction_loss" in losses
        assert "kl_loss" in losses
        assert "total_loss" in losses


class TestVAEConfigurations:
    """Tests for different VAE configurations."""

    def test_base_channels_32(self, device, seed):
        """Should work with base_channels=32."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        vae.eval()
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        reconstruction, posterior = vae(x)

        # Assert
        assert reconstruction.shape == x.shape

    def test_base_channels_64(self, device, seed):
        """Should work with base_channels=64."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=64
        ).to(device)
        vae.eval()
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        reconstruction, posterior = vae(x)

        # Assert
        assert reconstruction.shape == x.shape

    def test_different_latent_channels(self, device, seed):
        """Should work with latent_channels=4."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=4, base_channels=32
        ).to(device)
        vae.eval()
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        reconstruction, posterior = vae(x)

        # Assert
        assert reconstruction.shape == x.shape
        assert posterior.mean.shape == (2, 4, 4, 4)

    def test_different_channel_multipliers(self, device, seed):
        """Should work with channel_multipliers=(1, 2, 2)."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32,
            channel_multipliers=(1, 2, 2),
        ).to(device)
        vae.eval()
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        reconstruction, posterior = vae(x)

        # Assert
        assert reconstruction.shape == x.shape

    def test_multiple_layers_per_block(self, device, seed):
        """Should work with num_layers_per_block=2."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32,
            num_layers_per_block=2,
        ).to(device)
        vae.eval()
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        reconstruction, posterior = vae(x)

        # Assert
        assert reconstruction.shape == x.shape

    def test_rgb_images(self, device, seed):
        """Should work with 3-channel (RGB) images."""
        # Arrange
        vae = VAE(
            image_channels=3, latent_channels=2, base_channels=32
        ).to(device)
        vae.eval()
        x = torch.randn(2, 3, 32, 32, device=device)

        # Act
        reconstruction, posterior = vae(x)

        # Assert
        assert reconstruction.shape == x.shape


class TestVAENumericalStability:
    """Tests for numerical stability."""

    def test_no_nan_in_reconstruction(self, device, seed):
        """Reconstruction should not contain NaN values."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        vae.eval()
        x = torch.randn(4, 1, 32, 32, device=device)

        # Act
        reconstruction, _ = vae(x)

        # Assert
        assert not torch.isnan(reconstruction).any()

    def test_no_inf_in_reconstruction(self, device, seed):
        """Reconstruction should not contain Inf values."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        vae.eval()
        x = torch.randn(4, 1, 32, 32, device=device)

        # Act
        reconstruction, _ = vae(x)

        # Assert
        assert not torch.isinf(reconstruction).any()

    def test_no_nan_in_latent_samples(self, device, seed):
        """Latent samples should not contain NaN values."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        vae.eval()
        x = torch.randn(4, 1, 32, 32, device=device)

        # Act
        posterior = vae.encode(x)
        z = posterior.sample()

        # Assert
        assert not torch.isnan(z).any()

    def test_bounded_reconstruction_range(self, device, seed):
        """Reconstruction should have reasonable magnitude."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        vae.eval()
        x = torch.randn(4, 1, 32, 32, device=device)

        # Act
        reconstruction, _ = vae(x)

        # Assert — output should not be exploding
        assert reconstruction.abs().max() < 100

    def test_loss_is_finite(self, device, seed):
        """All loss components should be finite."""
        # Arrange
        vae = VAE(
            image_channels=1, latent_channels=2, base_channels=32
        ).to(device)
        x = torch.randn(2, 1, 32, 32, device=device)

        # Act
        reconstruction, posterior = vae(x)
        losses = vae.loss(x, reconstruction, posterior, kl_weight=1e-6)

        # Assert
        assert torch.isfinite(losses["total_loss"])
        assert torch.isfinite(losses["reconstruction_loss"])
        assert torch.isfinite(losses["kl_loss"])
