"""
U-Net architecture for DDPM noise prediction.

Based on: https://github.com/mattroz/diffusion-ddpm/blob/main/src/model/unet.py
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
    U-Net architecture for DDPM as described in the DDPM paper, Appendix B.

    Architecture for 28x28 MNIST (padded to 32x32):
        Spatial:  32x32 -> 16x16 -> 8x8 -> 4x4 (bottleneck) -> 8x8 -> 16x16 -> 32x32
        Channels: C -> C -> 2C -> 4C -> 4C -> 2C -> C  (where C = base_channels)

    Skip connections are collected after each encoder block and concatenated
    with decoder features before each decoder block.
    """
    def __init__(self, image_channels=1, base_channels=64):
        super().__init__()

        # Channel progression: C -> C -> 2C -> 4C (encoder), mirrored in decoder
        channel_1 = base_channels          # e.g. 64
        channel_2 = base_channels * 2      # e.g. 128
        channel_3 = base_channels * 4      # e.g. 256
        time_embedding_channels = base_channels * 4
        num_groups = min(32, base_channels)

        # Time embedding: sinusoidal positional encoding + MLP
        self.positional_encoding = nn.Sequential(
            TransformerPositionalEmbedding(dimension=base_channels),
            nn.Linear(base_channels, time_embedding_channels),
            nn.GELU(),
            nn.Linear(time_embedding_channels, time_embedding_channels),
        )

        # Initial convolution: image_channels -> base_channels
        self.initial_conv = nn.Conv2d(
            in_channels=image_channels,
            out_channels=base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Encoder (downsampling path)
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.downsample_blocks = nn.ModuleList([
            ConvDownBlock(
                in_channels=channel_1, out_channels=channel_1, num_layers=2,
                time_embedding_channels=time_embedding_channels,
                num_groups=num_groups, downsample=True
            ),  # 32 -> 16
            ConvDownBlock(
                in_channels=channel_1, out_channels=channel_2, num_layers=2,
                time_embedding_channels=time_embedding_channels,
                num_groups=num_groups, downsample=True
            ),  # 16 -> 8
            AttentionDownBlock(
                in_channels=channel_2, out_channels=channel_3, num_layers=2,
                time_embedding_channels=time_embedding_channels,
                num_attention_heads=4, num_groups=num_groups, downsample=True
            ),  # 8 -> 4, with attention
        ])

        # Bottleneck at 4x4 with attention
        self.bottleneck = AttentionDownBlock(
            in_channels=channel_3, out_channels=channel_3, num_layers=2,
            time_embedding_channels=time_embedding_channels,
            num_attention_heads=4, num_groups=num_groups, downsample=False
        )

        # Decoder (upsampling path)
        # 4x4 -> 8x8 -> 16x16 -> 32x32
        # Note: input channels include skip connection concatenation
        self.upsample_blocks = nn.ModuleList([
            ConvUpBlock(
                in_channels=channel_3 + channel_3, out_channels=channel_3, num_layers=2,
                time_embedding_channels=time_embedding_channels,
                num_groups=num_groups, upsample=True
            ),  # 4 -> 8
            AttentionUpBlock(
                in_channels=channel_3 + channel_2, out_channels=channel_2, num_layers=2,
                time_embedding_channels=time_embedding_channels,
                num_attention_heads=4, num_groups=num_groups, upsample=True
            ),  # 8 -> 16, with attention
            ConvUpBlock(
                in_channels=channel_2 + channel_1, out_channels=channel_1, num_layers=2,
                time_embedding_channels=time_embedding_channels,
                num_groups=num_groups, upsample=True
            ),  # 16 -> 32
        ])

        # Output convolution: base_channels -> image_channels
        # Final skip connection adds base_channels, so input is channel_1 + channel_1
        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_channels=channel_1 + channel_1, num_groups=num_groups),
            nn.SiLU(),
            nn.Conv2d(channel_1 + channel_1, image_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timestep):
        """
        Forward pass predicting noise ε_θ(x_t, t).

        Args:
            x: Noisy image x_t, shape (batch_size, channels, 28, 28)
            timestep: Timestep, shape (batch_size,)

        Returns:
            Predicted noise, shape (batch_size, channels, 28, 28)
        """
        # Pad 28x28 to 32x32 for clean downsampling
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')

        # Time embedding
        time_encoded = self.positional_encoding(timestep)

        # Initial convolution
        x = self.initial_conv(x)

        # Store skip connections: initial features + after each encoder block
        skip_connections = [x]

        # Encoder: collect skip connections after each block
        for block in self.downsample_blocks:
            x = block(x, time_encoded)
            skip_connections.append(x)

        # Reverse skip connections for decoder (last encoder output first)
        skip_connections = list(reversed(skip_connections))

        # Bottleneck
        x = self.bottleneck(x, time_encoded)

        # Decoder: concatenate skip connections before each block
        for i, block in enumerate(self.upsample_blocks):
            x = torch.cat([x, skip_connections[i]], dim=1)
            x = block(x, time_encoded)

        # Final skip connection (from initial conv) and output
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.output_conv(x)

        # Crop 32x32 back to 28x28
        x = x[:, :, 2:-2, 2:-2]

        return x
