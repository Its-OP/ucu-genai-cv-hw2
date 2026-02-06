"""
Unit tests for model building blocks.

Tests: TransformerPositionalEmbedding, ResNetBlock, SelfAttentionBlock,
       ConvDownBlock, ConvUpBlock, AttentionDownBlock, AttentionUpBlock,
       DownsampleBlock, UpsampleBlock.
All tests follow the AAA pattern (Arrange, Act, Assert).
"""
import pytest
import torch

from models.blocks import (
    TransformerPositionalEmbedding,
    ConvBlock,
    ResNetBlock,
    SelfAttentionBlock,
    DownsampleBlock,
    UpsampleBlock,
    ConvDownBlock,
    ConvUpBlock,
    AttentionDownBlock,
    AttentionUpBlock,
)


class TestTransformerPositionalEmbedding:
    """Tests for TransformerPositionalEmbedding class."""

    def test_output_shape(self, device):
        """Output shape should be (batch_size, dim)."""
        # Arrange
        embedding_dim = 128
        batch_size = 4
        embeddings = TransformerPositionalEmbedding(dimension=embedding_dim)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)

        # Act
        output = embeddings(timesteps)

        # Assert
        assert output.shape == (batch_size, embedding_dim)

    def test_output_range(self, device):
        """All output values should be in [-1, 1] (sin/cos range)."""
        # Arrange
        embeddings = TransformerPositionalEmbedding(dimension=128)
        timesteps = torch.randint(0, 1000, (100,), device=device)

        # Act
        output = embeddings(timesteps)

        # Assert
        assert output.min() >= -1.0
        assert output.max() <= 1.0

    def test_different_timesteps_produce_different_embeddings(self, device):
        """Different timesteps should produce distinct embeddings."""
        # Arrange
        embeddings = TransformerPositionalEmbedding(dimension=128)
        timestep_0 = torch.tensor([0], device=device)
        timestep_500 = torch.tensor([500], device=device)
        timestep_999 = torch.tensor([999], device=device)

        # Act
        embedding_0 = embeddings(timestep_0)
        embedding_500 = embeddings(timestep_500)
        embedding_999 = embeddings(timestep_999)

        # Assert
        assert not torch.allclose(embedding_0, embedding_500)
        assert not torch.allclose(embedding_500, embedding_999)
        assert not torch.allclose(embedding_0, embedding_999)

    def test_deterministic(self, device):
        """Same input should produce same output."""
        # Arrange
        embeddings = TransformerPositionalEmbedding(dimension=128)
        timesteps = torch.tensor([0, 100, 500, 999], device=device)

        # Act
        output_1 = embeddings(timesteps)
        output_2 = embeddings(timesteps)

        # Assert
        torch.testing.assert_close(output_1, output_2)


class TestConvBlock:
    """Tests for ConvBlock class."""

    def test_output_shape(self, device, seed):
        """Output should have correct shape."""
        # Arrange
        block = ConvBlock(in_channels=64, out_channels=128, num_groups=32)
        x = torch.randn(4, 64, 16, 16, device=device)

        # Act
        output = block(x)

        # Assert
        assert output.shape == (4, 128, 16, 16)

    def test_spatial_dims_preserved(self, device, seed):
        """Spatial dimensions should remain unchanged."""
        # Arrange
        block = ConvBlock(in_channels=64, out_channels=64, num_groups=32)
        x = torch.randn(2, 64, 8, 8, device=device)

        # Act
        output = block(x)

        # Assert
        assert output.shape[2:] == x.shape[2:]


class TestResNetBlock:
    """Tests for ResNetBlock class."""

    def test_output_shape_same_channels(self, device, seed):
        """Output shape should match input when channels are the same."""
        # Arrange
        in_channels = 64
        out_channels = 64
        time_embedding_channels = 256
        block = ResNetBlock(in_channels, out_channels, time_embedding_channels)
        x = torch.randn(4, in_channels, 16, 16, device=device)
        time_embedding = torch.randn(4, time_embedding_channels, device=device)

        # Act
        output = block(x, time_embedding)

        # Assert
        assert output.shape == (4, out_channels, 16, 16)

    def test_output_shape_different_channels(self, device, seed):
        """Output should have out_channels when different from in_channels."""
        # Arrange
        in_channels = 64
        out_channels = 128
        time_embedding_channels = 256
        block = ResNetBlock(in_channels, out_channels, time_embedding_channels)
        x = torch.randn(4, in_channels, 16, 16, device=device)
        time_embedding = torch.randn(4, time_embedding_channels, device=device)

        # Act
        output = block(x, time_embedding)

        # Assert
        assert output.shape == (4, out_channels, 16, 16)

    def test_spatial_dims_preserved(self, device, seed):
        """Spatial dimensions (H, W) should remain unchanged."""
        # Arrange
        block = ResNetBlock(64, 64, 256)
        heights_widths = [(8, 8), (16, 16), (32, 32)]

        for height, width in heights_widths:
            x = torch.randn(2, 64, height, width, device=device)
            time_embedding = torch.randn(2, 256, device=device)

            # Act
            output = block(x, time_embedding)

            # Assert
            assert output.shape[2] == height
            assert output.shape[3] == width

    def test_time_embedding_integration(self, device, seed):
        """Different time embeddings should produce different outputs."""
        # Arrange
        block = ResNetBlock(64, 64, 256)
        block.eval()
        x = torch.randn(2, 64, 8, 8, device=device)
        time_embedding_1 = torch.randn(2, 256, device=device)
        time_embedding_2 = torch.randn(2, 256, device=device)

        # Act
        output_1 = block(x, time_embedding_1)
        output_2 = block(x, time_embedding_2)

        # Assert
        assert not torch.allclose(output_1, output_2)

    def test_deterministic_eval_mode(self, device):
        """Same inputs should produce same outputs in eval mode."""
        # Arrange
        torch.manual_seed(42)
        block = ResNetBlock(64, 64, 256)
        block.eval()
        x = torch.randn(2, 64, 8, 8, device=device)
        time_embedding = torch.randn(2, 256, device=device)

        # Act
        output_1 = block(x, time_embedding)
        output_2 = block(x, time_embedding)

        # Assert
        torch.testing.assert_close(output_1, output_2)


class TestSelfAttentionBlock:
    """Tests for SelfAttentionBlock class."""

    def test_output_shape(self, device, seed):
        """Output shape should match input shape."""
        # Arrange
        channels = 64
        block = SelfAttentionBlock(in_channels=channels, num_heads=4, num_groups=32)
        x = torch.randn(4, channels, 8, 8, device=device)

        # Act
        output = block(x)

        # Assert
        assert output.shape == x.shape

    def test_residual_connection(self, device, seed):
        """Gradient should flow through the residual connection."""
        # Arrange
        channels = 64
        block = SelfAttentionBlock(in_channels=channels, num_heads=4, num_groups=32)
        x = torch.randn(2, channels, 4, 4, device=device, requires_grad=True)

        # Act
        output = block(x)
        # Note: Using squared sum because GroupNorm output has zero mean,
        # which causes sum() to have near-zero gradients
        loss = (output ** 2).sum()
        loss.backward()

        # Assert
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_no_nan_or_inf(self, device, seed):
        """Output should not contain NaN or Inf values."""
        # Arrange
        channels = 64
        block = SelfAttentionBlock(in_channels=channels, num_heads=4, num_groups=32)
        x = torch.randn(4, channels, 8, 8, device=device)

        # Act
        output = block(x)

        # Assert
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_different_spatial_sizes(self, device, seed):
        """Should work with different spatial dimensions."""
        # Arrange
        channels = 64
        block = SelfAttentionBlock(in_channels=channels, num_heads=4, num_groups=32)
        spatial_sizes = [(4, 4), (8, 8), (16, 16)]

        for height, width in spatial_sizes:
            x = torch.randn(2, channels, height, width, device=device)

            # Act
            output = block(x)

            # Assert
            assert output.shape == x.shape


class TestDownsampleBlock:
    """Tests for DownsampleBlock class."""

    def test_spatial_halving(self, device, seed):
        """Spatial dimensions should be halved."""
        # Arrange
        channels = 64
        downsample = DownsampleBlock(in_channels=channels, out_channels=channels)
        x = torch.randn(4, channels, 16, 16, device=device)

        # Act
        output = downsample(x)

        # Assert
        assert output.shape == (4, channels, 8, 8)

    def test_channel_change(self, device, seed):
        """Should support different input/output channels."""
        # Arrange
        downsample = DownsampleBlock(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 32, 32, device=device)

        # Act
        output = downsample(x)

        # Assert
        assert output.shape == (2, 128, 16, 16)

    def test_gradient_flow(self, device, seed):
        """Gradients should flow through the downsample operation."""
        # Arrange
        downsample = DownsampleBlock(in_channels=64, out_channels=64)
        x = torch.randn(2, 64, 16, 16, device=device, requires_grad=True)

        # Act
        output = downsample(x)
        loss = output.sum()
        loss.backward()

        # Assert
        assert x.grad is not None


class TestUpsampleBlock:
    """Tests for UpsampleBlock class."""

    def test_spatial_doubling(self, device, seed):
        """Spatial dimensions should be doubled."""
        # Arrange
        channels = 64
        upsample = UpsampleBlock(in_channels=channels, out_channels=channels)
        x = torch.randn(4, channels, 8, 8, device=device)

        # Act
        output = upsample(x)

        # Assert
        assert output.shape == (4, channels, 16, 16)

    def test_channel_change(self, device, seed):
        """Should support different input/output channels."""
        # Arrange
        upsample = UpsampleBlock(in_channels=128, out_channels=64)
        x = torch.randn(2, 128, 8, 8, device=device)

        # Act
        output = upsample(x)

        # Assert
        assert output.shape == (2, 64, 16, 16)

    def test_gradient_flow(self, device, seed):
        """Gradients should flow through the upsample operation."""
        # Arrange
        upsample = UpsampleBlock(in_channels=64, out_channels=64)
        x = torch.randn(2, 64, 8, 8, device=device, requires_grad=True)

        # Act
        output = upsample(x)
        loss = output.sum()
        loss.backward()

        # Assert
        assert x.grad is not None


class TestConvDownBlock:
    """Tests for ConvDownBlock class."""

    def test_output_shape_with_downsample(self, device, seed):
        """Output should have halved spatial dims when downsample=True."""
        # Arrange
        block = ConvDownBlock(
            in_channels=64, out_channels=128, num_layers=2,
            time_embedding_channels=256, downsample=True
        )
        x = torch.randn(4, 64, 16, 16, device=device)
        time_emb = torch.randn(4, 256, device=device)

        # Act
        output, intermediates = block(x, time_emb)

        # Assert
        assert output.shape == (4, 128, 8, 8)

    def test_output_shape_without_downsample(self, device, seed):
        """Output should preserve spatial dims when downsample=False."""
        # Arrange
        block = ConvDownBlock(
            in_channels=64, out_channels=128, num_layers=2,
            time_embedding_channels=256, downsample=False
        )
        x = torch.randn(4, 64, 16, 16, device=device)
        time_emb = torch.randn(4, 256, device=device)

        # Act
        output, intermediates = block(x, time_emb)

        # Assert
        assert output.shape == (4, 128, 16, 16)

    def test_intermediates_count_matches_num_layers(self, device, seed):
        """Should return one intermediate per ResNet block."""
        # Arrange
        num_layers = 2
        block = ConvDownBlock(
            in_channels=64, out_channels=128, num_layers=num_layers,
            time_embedding_channels=256, downsample=True
        )
        x = torch.randn(2, 64, 16, 16, device=device)
        time_emb = torch.randn(2, 256, device=device)

        # Act
        output, intermediates = block(x, time_emb)

        # Assert
        assert len(intermediates) == num_layers
        # Intermediates should be at pre-downsample resolution
        for intermediate in intermediates:
            assert intermediate.shape == (2, 128, 16, 16)


class TestConvUpBlock:
    """Tests for ConvUpBlock class."""

    def test_output_shape_with_upsample(self, device, seed):
        """Output should have doubled spatial dims when upsample=True."""
        # Arrange
        skip_channels = 64
        block = ConvUpBlock(
            in_channels=128, out_channels=64, skip_channels=skip_channels,
            num_layers=2, time_embedding_channels=256, upsample=True
        )
        x = torch.randn(4, 128, 8, 8, device=device)
        time_emb = torch.randn(4, 256, device=device)
        skip_connections = [torch.randn(4, skip_channels, 8, 8, device=device) for _ in range(2)]

        # Act
        output = block(x, time_emb, skip_connections)

        # Assert
        assert output.shape == (4, 64, 16, 16)

    def test_output_shape_without_upsample(self, device, seed):
        """Output should preserve spatial dims when upsample=False."""
        # Arrange
        skip_channels = 64
        block = ConvUpBlock(
            in_channels=128, out_channels=64, skip_channels=skip_channels,
            num_layers=2, time_embedding_channels=256, upsample=False
        )
        x = torch.randn(4, 128, 8, 8, device=device)
        time_emb = torch.randn(4, 256, device=device)
        skip_connections = [torch.randn(4, skip_channels, 8, 8, device=device) for _ in range(2)]

        # Act
        output = block(x, time_emb, skip_connections)

        # Assert
        assert output.shape == (4, 64, 8, 8)


class TestAttentionDownBlock:
    """Tests for AttentionDownBlock class."""

    def test_output_shape_with_downsample(self, device, seed):
        """Output should have halved spatial dims when downsample=True."""
        # Arrange
        block = AttentionDownBlock(
            in_channels=64, out_channels=128, num_layers=2,
            time_embedding_channels=256, num_attention_heads=4, downsample=True
        )
        x = torch.randn(2, 64, 16, 16, device=device)
        time_emb = torch.randn(2, 256, device=device)

        # Act
        output, intermediates = block(x, time_emb)

        # Assert
        assert output.shape == (2, 128, 8, 8)

    def test_output_shape_without_downsample(self, device, seed):
        """Output should preserve spatial dims when downsample=False."""
        # Arrange
        block = AttentionDownBlock(
            in_channels=64, out_channels=128, num_layers=2,
            time_embedding_channels=256, num_attention_heads=4, downsample=False
        )
        x = torch.randn(2, 64, 16, 16, device=device)
        time_emb = torch.randn(2, 256, device=device)

        # Act
        output, intermediates = block(x, time_emb)

        # Assert
        assert output.shape == (2, 128, 16, 16)

    def test_intermediates_count_matches_num_layers(self, device, seed):
        """Should return one intermediate per ResNet+Attention pair."""
        # Arrange
        num_layers = 2
        block = AttentionDownBlock(
            in_channels=64, out_channels=128, num_layers=num_layers,
            time_embedding_channels=256, num_attention_heads=4, downsample=True
        )
        x = torch.randn(2, 64, 16, 16, device=device)
        time_emb = torch.randn(2, 256, device=device)

        # Act
        output, intermediates = block(x, time_emb)

        # Assert
        assert len(intermediates) == num_layers
        for intermediate in intermediates:
            assert intermediate.shape == (2, 128, 16, 16)


class TestAttentionUpBlock:
    """Tests for AttentionUpBlock class."""

    def test_output_shape_with_upsample(self, device, seed):
        """Output should have doubled spatial dims when upsample=True."""
        # Arrange
        skip_channels = 64
        block = AttentionUpBlock(
            in_channels=128, out_channels=64, skip_channels=skip_channels,
            num_layers=2, time_embedding_channels=256,
            num_attention_heads=4, upsample=True
        )
        x = torch.randn(2, 128, 8, 8, device=device)
        time_emb = torch.randn(2, 256, device=device)
        skip_connections = [torch.randn(2, skip_channels, 8, 8, device=device) for _ in range(2)]

        # Act
        output = block(x, time_emb, skip_connections)

        # Assert
        assert output.shape == (2, 64, 16, 16)

    def test_output_shape_without_upsample(self, device, seed):
        """Output should preserve spatial dims when upsample=False."""
        # Arrange
        skip_channels = 64
        block = AttentionUpBlock(
            in_channels=128, out_channels=64, skip_channels=skip_channels,
            num_layers=2, time_embedding_channels=256,
            num_attention_heads=4, upsample=False
        )
        x = torch.randn(2, 128, 8, 8, device=device)
        time_emb = torch.randn(2, 256, device=device)
        skip_connections = [torch.randn(2, skip_channels, 8, 8, device=device) for _ in range(2)]

        # Act
        output = block(x, time_emb, skip_connections)

        # Assert
        assert output.shape == (2, 64, 8, 8)
