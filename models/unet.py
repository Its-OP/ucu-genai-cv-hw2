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
        channel_mults: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (8, 4),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.image_channels = image_channels
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks
        time_emb_dim = base_channels * 4

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        # Track channels at each resolution for skip connections
        # channels[i] = output channels after processing at resolution i
        channels = [base_channels * m for m in channel_mults]
        resolutions = [32, 16, 8, 4]

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        in_ch = base_channels
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            res = resolutions[level]
            use_attn = res in attention_resolutions

            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                level_blocks.append(ResidualBlock(in_ch, out_ch, time_emb_dim, dropout))
                if use_attn:
                    level_blocks.append(AttentionBlock(out_ch))
                in_ch = out_ch

            self.down_blocks.append(level_blocks)

            if level < len(channel_mults) - 1:
                self.down_samples.append(Downsample(out_ch))
            else:
                self.down_samples.append(None)

        # Bottleneck
        mid_ch = channels[-1]
        self.mid_block1 = ResidualBlock(mid_ch, mid_ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid_block2 = ResidualBlock(mid_ch, mid_ch, time_emb_dim, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        reversed_channels = list(reversed(channels))
        reversed_resolutions = list(reversed(resolutions))

        in_ch = mid_ch
        for level in range(len(channel_mults)):
            out_ch = reversed_channels[level]
            res = reversed_resolutions[level]
            use_attn = res in attention_resolutions

            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                # Each ResBlock receives skip connection (concatenated)
                skip_ch = reversed_channels[level]
                level_blocks.append(ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim, dropout))
                if use_attn:
                    level_blocks.append(AttentionBlock(out_ch))
                in_ch = out_ch

            self.up_blocks.append(level_blocks)

            if level < len(channel_mults) - 1:
                next_ch = reversed_channels[level + 1]
                self.up_samples.append(Upsample(out_ch))
                in_ch = out_ch
            else:
                self.up_samples.append(None)

        # Output layers
        self.out_norm = nn.GroupNorm(32, base_channels)
        self.out_conv = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting noise ε_θ(x_t, t).

        Args:
            x: Noisy image x_t, shape (B, C, 28, 28)
            t: Timestep, shape (B,)

        Returns:
            Predicted noise, shape (B, C, 28, 28)
        """
        # Pad 28x28 to 32x32
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')

        # Time embedding
        t_emb = self.time_mlp(t)

        # Initial conv
        h = self.init_conv(x)

        # Encoder - collect skip connections
        skips = []
        for level_blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in level_blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, t_emb)
                    skips.append(h)
                else:
                    h = block(h)
            if downsample is not None:
                h = downsample(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder - use skip connections in reverse
        for level_blocks, upsample in zip(self.up_blocks, self.up_samples):
            for block in level_blocks:
                if isinstance(block, ResidualBlock):
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = block(h, t_emb)
                else:
                    h = block(h)
            if upsample is not None:
                h = upsample(h)

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)

        # Crop back to 28x28
        h = h[:, :, 2:-2, 2:-2]

        return h
