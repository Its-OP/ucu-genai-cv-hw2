import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timestep conditioning.

    Formula:
        emb(t, 2i)   = sin(t / 10000^(2i/d))
        emb(t, 2i+1) = cos(t / 10000^(2i/d))
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2

        # emb = log(10000) / (half_dim - 1)
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        # emb = exp(-emb * [0, 1, 2, ..., half_dim-1])
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # emb = t[:, None] * emb[None, :]
        emb = t[:, None].float() * emb[None, :]
        # Concatenate sin and cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding conditioning.

    Structure:
        1. GroupNorm -> SiLU -> Conv
        2. Add time embedding (projected to channels)
        3. GroupNorm -> SiLU -> Dropout -> Conv
        4. Residual connection (with projection if channels differ)
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)

        # First conv block
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding: project and broadcast to spatial dims
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb

        # Second conv block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class AttentionBlock(nn.Module):
    """
    Self-attention block.

    Formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape
        residual = x

        x = self.norm(x)

        # Compute Q, K, V
        query_key_value = self.qkv(x)
        query_key_value = query_key_value.reshape(batch_size, 3, num_channels, height * width)
        query, key, value = query_key_value[:, 0], query_key_value[:, 1], query_key_value[:, 2]

        # Attention: softmax(QK^T / sqrt(d)) V
        # query, key, value: (batch_size, num_channels, height*width)
        attention_weights = torch.bmm(query.transpose(1, 2), key) * self.scale
        attention_weights = F.softmax(attention_weights, dim=-1)

        output = torch.bmm(value, attention_weights.transpose(1, 2))
        output = output.reshape(batch_size, num_channels, height, width)

        output = self.proj(output)
        return output + residual


class Downsample(nn.Module):
    """Spatial downsampling by factor of 2 using strided convolution."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling by factor of 2 using nearest interpolation + conv."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
