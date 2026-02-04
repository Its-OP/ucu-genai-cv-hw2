import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    SinusoidalPositionEmbeddings,
    ResidualBlock,
    AttentionBlock,
    Downsample,
    Upsample,
)


class UNet(nn.Module):
    """
    U-Net architecture for DDPM noise prediction.

    Architecture for 28x28 MNIST (padded to 32x32):
        - Encoder: 32->16->8->4 with increasing channels
        - Bottleneck at 4x4 with attention
        - Decoder: 4->8->16->32 with skip connections
        - Attention at 8x8 and 4x4 resolutions
    """
    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4, 4),
        num_residual_blocks: int = 2,
        attention_resolutions: tuple = (8, 4),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.image_channels = image_channels
        self.base_channels = base_channels
        self.num_residual_blocks = num_residual_blocks
        time_embedding_dim = base_channels * 4

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        # Initial convolution
        self.initial_convolution = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        # Track channels at each resolution for skip connections
        channels_per_level = [base_channels * multiplier for multiplier in channel_multipliers]
        spatial_resolutions = [32, 16, 8, 4]

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        input_channels = base_channels
        for level, multiplier in enumerate(channel_multipliers):
            output_channels = base_channels * multiplier
            resolution = spatial_resolutions[level]
            use_attention = resolution in attention_resolutions

            level_blocks = nn.ModuleList()
            for _ in range(num_residual_blocks):
                level_blocks.append(ResidualBlock(input_channels, output_channels, time_embedding_dim, dropout))
                if use_attention:
                    level_blocks.append(AttentionBlock(output_channels))
                input_channels = output_channels

            self.encoder_blocks.append(level_blocks)

            if level < len(channel_multipliers) - 1:
                self.downsample_layers.append(Downsample(output_channels))
            else:
                self.downsample_layers.append(None)

        # Bottleneck
        middle_channels = channels_per_level[-1]
        self.middle_block1 = ResidualBlock(middle_channels, middle_channels, time_embedding_dim, dropout)
        self.middle_attention = AttentionBlock(middle_channels)
        self.middle_block2 = ResidualBlock(middle_channels, middle_channels, time_embedding_dim, dropout)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        reversed_channels = list(reversed(channels_per_level))
        reversed_resolutions = list(reversed(spatial_resolutions))

        input_channels = middle_channels
        for level in range(len(channel_multipliers)):
            output_channels = reversed_channels[level]
            resolution = reversed_resolutions[level]
            use_attention = resolution in attention_resolutions

            level_blocks = nn.ModuleList()
            for _ in range(num_residual_blocks):
                # Each ResBlock receives skip connection (concatenated)
                skip_channels = reversed_channels[level]
                level_blocks.append(ResidualBlock(input_channels + skip_channels, output_channels, time_embedding_dim, dropout))
                if use_attention:
                    level_blocks.append(AttentionBlock(output_channels))
                input_channels = output_channels

            self.decoder_blocks.append(level_blocks)

            if level < len(channel_multipliers) - 1:
                self.upsample_layers.append(Upsample(output_channels))
                input_channels = output_channels
            else:
                self.upsample_layers.append(None)

        # Output layers
        self.output_norm = nn.GroupNorm(32, base_channels)
        self.output_convolution = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting noise ε_θ(x_t, t).

        Args:
            x: Noisy image x_t, shape (batch_size, channels, 28, 28)
            timestep: Timestep, shape (batch_size,)

        Returns:
            Predicted noise, shape (batch_size, channels, 28, 28)
        """
        # Pad 28x28 to 32x32
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')

        # Time embedding
        time_embedding = self.time_mlp(timestep)

        # Initial convolution
        hidden = self.initial_convolution(x)

        # Encoder - collect skip connections
        skip_connections = []
        for level_blocks, downsample in zip(self.encoder_blocks, self.downsample_layers):
            for block in level_blocks:
                if isinstance(block, ResidualBlock):
                    hidden = block(hidden, time_embedding)
                    skip_connections.append(hidden)
                else:
                    hidden = block(hidden)
            if downsample is not None:
                hidden = downsample(hidden)

        # Bottleneck
        hidden = self.middle_block1(hidden, time_embedding)
        hidden = self.middle_attention(hidden)
        hidden = self.middle_block2(hidden, time_embedding)

        # Decoder - use skip connections in reverse
        for level_blocks, upsample in zip(self.decoder_blocks, self.upsample_layers):
            for block in level_blocks:
                if isinstance(block, ResidualBlock):
                    skip = skip_connections.pop()
                    hidden = torch.cat([hidden, skip], dim=1)
                    hidden = block(hidden, time_embedding)
                else:
                    hidden = block(hidden)
            if upsample is not None:
                hidden = upsample(hidden)

        # Output
        hidden = self.output_norm(hidden)
        hidden = F.silu(hidden)
        hidden = self.output_convolution(hidden)

        # Crop back to 28x28
        hidden = hidden[:, :, 2:-2, 2:-2]

        return hidden
