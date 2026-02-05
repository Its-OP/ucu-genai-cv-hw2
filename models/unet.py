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
        Channels: 64 -> 64 -> 128 -> 256 -> 256 -> 128 -> 64

    Skip connections are collected after each encoder block (before downsampling)
    and used in the decoder (after upsampling, before processing).
    """
    def __init__(self, image_channels=1, base_channels=64):
        super().__init__()

        time_embedding_channels = base_channels * 4  # 256

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
                in_channels=64, out_channels=64, num_layers=2,
                time_embedding_channels=time_embedding_channels, downsample=True
            ),  # 32 -> 16
            ConvDownBlock(
                in_channels=64, out_channels=128, num_layers=2,
                time_embedding_channels=time_embedding_channels, downsample=True
            ),  # 16 -> 8
            AttentionDownBlock(
                in_channels=128, out_channels=256, num_layers=2,
                time_embedding_channels=time_embedding_channels,
                num_attention_heads=4, downsample=True
            ),  # 8 -> 4, with attention
        ])

        # Bottleneck at 4x4 with attention
        self.bottleneck = AttentionDownBlock(
            in_channels=256, out_channels=256, num_layers=2,
            time_embedding_channels=time_embedding_channels,
            num_attention_heads=4, downsample=False
        )

        # Decoder (upsampling path)
        # 4x4 -> 8x8 -> 16x16 -> 32x32
        # Note: input channels include skip connection concatenation
        self.upsample_blocks = nn.ModuleList([
            ConvUpBlock(
                in_channels=256 + 256, out_channels=256, num_layers=2,
                time_embedding_channels=time_embedding_channels, upsample=True
            ),  # 4 -> 8
            AttentionUpBlock(
                in_channels=256 + 128, out_channels=128, num_layers=2,
                time_embedding_channels=time_embedding_channels,
                num_attention_heads=4, upsample=True
            ),  # 8 -> 16, with attention
            ConvUpBlock(
                in_channels=128 + 64, out_channels=64, num_layers=2,
                time_embedding_channels=time_embedding_channels, upsample=True
            ),  # 16 -> 32
        ])

        # Output convolution: base_channels -> image_channels
        # Note: final skip connection adds base_channels, so input is 64 + 64 = 128
        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_channels=64 + 64, num_groups=32),
            nn.SiLU(),
            nn.Conv2d(64 + 64, image_channels, kernel_size=3, padding=1),
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
