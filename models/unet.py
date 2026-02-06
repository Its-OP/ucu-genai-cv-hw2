"""
U-Net architecture for DDPM noise prediction.

Based on the DDPM paper (Ho et al., 2020), Appendix B.
Adapted for MNIST (28x28, 1 channel).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    TransformerPositionalEmbedding,
    ConvDownBlock,
    ConvUpBlock,
    AttentionDownBlock,
    AttentionUpBlock,
)


class UNet(nn.Module):
    """
    U-Net architecture for DDPM noise prediction.

    Architecture for 28x28 MNIST (padded to 32x32):
        Spatial:  32x32 -> 16x16 -> 8x8 -> 4x4 (bottleneck) -> 8x8 -> 16x16 -> 32x32
        Channels: C -> C -> 2C -> 4C -> 4C (bottleneck) -> 4C -> 2C -> C
        (where C = base_channels)

    Channel multipliers: (1, 2, 4, 4)
    Attention at resolutions: 8x8 and 4x4

    Skip connections are collected after EACH ResNet block (before downsampling)
    and popped by each decoder ResNet block, providing high-resolution spatial
    detail for noise prediction.
    """
    def __init__(self, image_channels=1, base_channels=64):
        super().__init__()

        # Channel multipliers: (1, 2, 4, 4) following the DDPM paper
        channel_multipliers = (1, 2, 4, 4)
        num_residual_blocks = 2
        num_groups = min(32, base_channels)
        time_embedding_channels = base_channels * 4

        # Compute channel counts per encoder level
        channels_per_level = [base_channels * mult for mult in channel_multipliers]
        # channels_per_level = [64, 128, 256, 256] for base_channels=64

        # Time embedding: sinusoidal positional encoding + MLP
        # t → sinusoidal(t) → Linear → SiLU → Linear
        self.positional_encoding = nn.Sequential(
            TransformerPositionalEmbedding(dimension=base_channels),
            nn.Linear(base_channels, time_embedding_channels),
            nn.SiLU(),
            nn.Linear(time_embedding_channels, time_embedding_channels),
        )

        # Initial convolution: image_channels -> base_channels
        self.initial_convolution = nn.Conv2d(
            in_channels=image_channels,
            out_channels=base_channels,
            kernel_size=3,
            padding=1,
        )

        # Encoder (downsampling path)
        # 4 levels: 32x32 -> 16x16 -> 8x8 -> 4x4 (level 3 has no downsample)
        # Attention at resolutions 8x8 (level 2) and 4x4 (level 3)
        self.encoder_blocks = nn.ModuleList()
        input_channels = base_channels
        for level, multiplier in enumerate(channel_multipliers):
            output_channels = base_channels * multiplier
            use_attention = level >= 2  # attention at 8x8 and 4x4
            downsample = level < len(channel_multipliers) - 1  # no downsample at last level

            if use_attention:
                self.encoder_blocks.append(AttentionDownBlock(
                    in_channels=input_channels, out_channels=output_channels,
                    num_layers=num_residual_blocks,
                    time_embedding_channels=time_embedding_channels,
                    num_attention_heads=4, num_groups=num_groups,
                    downsample=downsample,
                ))
            else:
                self.encoder_blocks.append(ConvDownBlock(
                    in_channels=input_channels, out_channels=output_channels,
                    num_layers=num_residual_blocks,
                    time_embedding_channels=time_embedding_channels,
                    num_groups=num_groups, downsample=downsample,
                ))
            input_channels = output_channels

        # Bottleneck at 4x4: ResNet + Attention + ResNet (no downsampling, no skips)
        bottleneck_channels = channels_per_level[-1]
        self.bottleneck = AttentionDownBlock(
            in_channels=bottleneck_channels, out_channels=bottleneck_channels,
            num_layers=num_residual_blocks,
            time_embedding_channels=time_embedding_channels,
            num_attention_heads=4, num_groups=num_groups, downsample=False,
        )

        # Decoder (upsampling path)
        # Mirrors encoder: 4 levels in reverse order
        # Each ResBlock receives a skip connection (concatenated) from the encoder
        reversed_channels = list(reversed(channels_per_level))
        self.decoder_blocks = nn.ModuleList()
        input_channels = bottleneck_channels
        for level in range(len(channel_multipliers)):
            output_channels = reversed_channels[level]
            skip_channels = reversed_channels[level]  # skips come from matching encoder level
            use_attention = level <= 1  # attention at 4x4 (level 0) and 8x8 (level 1)
            upsample = level < len(channel_multipliers) - 1  # no upsample at last level

            if use_attention:
                self.decoder_blocks.append(AttentionUpBlock(
                    in_channels=input_channels, out_channels=output_channels,
                    skip_channels=skip_channels, num_layers=num_residual_blocks,
                    time_embedding_channels=time_embedding_channels,
                    num_attention_heads=4, num_groups=num_groups,
                    upsample=upsample,
                ))
            else:
                self.decoder_blocks.append(ConvUpBlock(
                    in_channels=input_channels, out_channels=output_channels,
                    skip_channels=skip_channels, num_layers=num_residual_blocks,
                    time_embedding_channels=time_embedding_channels,
                    num_groups=num_groups, upsample=upsample,
                ))
            input_channels = output_channels

        # Output: GroupNorm -> SiLU -> Conv
        self.output_norm = nn.GroupNorm(num_groups, base_channels)
        self.output_convolution = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x, timestep):
        """
        Forward pass predicting noise ε_θ(x_t, t).

        Args:
            x: Noisy image x_t, shape (batch_size, channels, 28, 28)
            timestep: Timestep, shape (batch_size,)

        Returns:
            Predicted noise, shape (batch_size, channels, 28, 28)
        """
        # Pad 28x28 to 32x32 for clean downsampling (32 -> 16 -> 8 -> 4)
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')

        # Time embedding
        time_encoded = self.positional_encoding(timestep)

        # Initial convolution
        x = self.initial_convolution(x)

        # Encoder: collect per-ResBlock skip connections (before downsampling)
        skip_connections = []
        for block in self.encoder_blocks:
            x, intermediates = block(x, time_encoded)
            skip_connections.extend(intermediates)

        # Bottleneck (no skip connections collected)
        x, _ = self.bottleneck(x, time_encoded)

        # Decoder: each ResBlock pops one skip connection
        for block in self.decoder_blocks:
            x = block(x, time_encoded, skip_connections)

        # Output
        x = self.output_norm(x)
        x = F.silu(x)
        x = self.output_convolution(x)

        # Crop 32x32 back to 28x28
        x = x[:, :, 2:-2, 2:-2]

        return x
