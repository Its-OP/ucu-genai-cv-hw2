"""
Building blocks for DDPM U-Net.

Based on: https://github.com/mattroz/diffusion-ddpm/blob/main/src/model/layers.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings for timestep encoding.
    From paper "Attention Is All You Need", section 3.5

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    def __init__(self, dimension, max_timesteps=1000):
        super(TransformerPositionalEmbedding, self).__init__()
        assert dimension % 2 == 0, "Embedding dimension must be even"
        self.dimension = dimension

        # Pre-compute positional embeddings
        pe_matrix = torch.zeros(max_timesteps, dimension)

        even_indices = torch.arange(0, self.dimension, 2)
        log_term = torch.log(torch.tensor(10000.0)) / self.dimension
        div_term = torch.exp(even_indices * -log_term)

        timesteps = torch.arange(max_timesteps).unsqueeze(1)
        pe_matrix[:, 0::2] = torch.sin(timesteps * div_term)
        pe_matrix[:, 1::2] = torch.cos(timesteps * div_term)

        # Register as buffer so it moves with the model to GPU/MPS
        self.register_buffer('pe_matrix', pe_matrix)

    def forward(self, timestep):
        return self.pe_matrix[timestep]


class ConvBlock(nn.Module):
    """Single convolution block: Conv -> GroupNorm -> SiLU."""
    def __init__(self, in_channels, out_channels, num_groups=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResNetBlock(nn.Module):
    """
    Wide ResNet block with time embedding injection.

    Structure:
        1. ConvBlock (in_channels -> out_channels)
        2. Add time embedding (projected to out_channels)
        3. ConvBlock (out_channels -> out_channels)
        4. Residual connection (with 1x1 conv if channels differ)
    """
    def __init__(self, in_channels, out_channels, time_embedding_channels, num_groups=32):
        super(ResNetBlock, self).__init__()
        self.time_embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_channels, out_channels)
        )

        self.block1 = ConvBlock(in_channels, out_channels, num_groups=num_groups)
        self.block2 = ConvBlock(out_channels, out_channels, num_groups=num_groups)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_embedding):
        residual = x

        x = self.block1(x)

        # Add time embedding: project and broadcast to spatial dimensions
        time_emb = self.time_embedding_projection(time_embedding)
        time_emb = time_emb[:, :, None, None]
        x = x + time_emb

        x = self.block2(x)

        return x + self.residual_conv(residual)


class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention block.
    Based on "Attention Is All You Need" (Vaswani et al., 2017)

    Formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    def __init__(self, in_channels, num_heads=4, num_groups=32, embedding_dim=None):
        super(SelfAttentionBlock, self).__init__()
        embedding_dim = embedding_dim or in_channels
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads

        self.query_projection = nn.Linear(in_channels, embedding_dim)
        self.key_projection = nn.Linear(in_channels, embedding_dim)
        self.value_projection = nn.Linear(in_channels, embedding_dim)

        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.GroupNorm(num_channels=embedding_dim, num_groups=num_groups)

    def forward(self, input_tensor):
        batch_size, channels, height, width = input_tensor.shape

        # Reshape to (batch, height*width, channels) for attention
        x = input_tensor.view(batch_size, channels, height * width).transpose(1, 2)

        # Project to Q, K, V
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)

        # Split into multiple heads: (batch, num_heads, seq_len, head_dim)
        queries = queries.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2)) * scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)

        # Concatenate heads: (batch, seq_len, embedding_dim)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, height * width, self.embedding_dim)

        # Output projection and reshape back to image format
        output = self.output_projection(attention_output)
        output = output.transpose(1, 2).view(batch_size, self.embedding_dim, height, width)

        # Residual connection with normalization
        return self.norm(output + input_tensor)


class DownsampleBlock(nn.Module):
    """Spatial downsampling by factor of 2 using strided convolution."""
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """Spatial upsampling by factor of 2 using interpolation + convolution."""
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        return self.conv(x)


class ConvDownBlock(nn.Module):
    """
    Encoder block: multiple ResNet blocks followed by optional downsampling.
    """
    def __init__(self, in_channels, out_channels, num_layers, time_embedding_channels, num_groups=32, downsample=True):
        super(ConvDownBlock, self).__init__()

        resnet_blocks = []
        for i in range(num_layers):
            input_ch = in_channels if i == 0 else out_channels
            resnet_blocks.append(
                ResNetBlock(input_ch, out_channels, time_embedding_channels, num_groups)
            )

        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.downsample = DownsampleBlock(out_channels, out_channels) if downsample else None

    def forward(self, x, time_embedding):
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, time_embedding)
        if self.downsample:
            x = self.downsample(x)
        return x


class ConvUpBlock(nn.Module):
    """
    Decoder block: multiple ResNet blocks followed by optional upsampling.
    """
    def __init__(self, in_channels, out_channels, num_layers, time_embedding_channels, num_groups=32, upsample=True):
        super(ConvUpBlock, self).__init__()

        resnet_blocks = []
        for i in range(num_layers):
            input_ch = in_channels if i == 0 else out_channels
            resnet_blocks.append(
                ResNetBlock(input_ch, out_channels, time_embedding_channels, num_groups)
            )

        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.upsample = UpsampleBlock(out_channels, out_channels) if upsample else None

    def forward(self, x, time_embedding):
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, time_embedding)
        if self.upsample:
            x = self.upsample(x)
        return x


class AttentionDownBlock(nn.Module):
    """
    Encoder block with self-attention: ResNet + Attention blocks, then optional downsampling.
    """
    def __init__(self, in_channels, out_channels, num_layers, time_embedding_channels,
                 num_attention_heads=4, num_groups=32, downsample=True):
        super(AttentionDownBlock, self).__init__()

        resnet_blocks = []
        attention_blocks = []
        for i in range(num_layers):
            input_ch = in_channels if i == 0 else out_channels
            resnet_blocks.append(
                ResNetBlock(input_ch, out_channels, time_embedding_channels, num_groups)
            )
            attention_blocks.append(
                SelfAttentionBlock(out_channels, num_attention_heads, num_groups)
            )

        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.downsample = DownsampleBlock(out_channels, out_channels) if downsample else None

    def forward(self, x, time_embedding):
        for resnet_block, attention_block in zip(self.resnet_blocks, self.attention_blocks):
            x = resnet_block(x, time_embedding)
            x = attention_block(x)
        if self.downsample:
            x = self.downsample(x)
        return x


class AttentionUpBlock(nn.Module):
    """
    Decoder block with self-attention: ResNet + Attention blocks, then optional upsampling.
    """
    def __init__(self, in_channels, out_channels, num_layers, time_embedding_channels,
                 num_attention_heads=4, num_groups=32, upsample=True):
        super(AttentionUpBlock, self).__init__()

        resnet_blocks = []
        attention_blocks = []
        for i in range(num_layers):
            input_ch = in_channels if i == 0 else out_channels
            resnet_blocks.append(
                ResNetBlock(input_ch, out_channels, time_embedding_channels, num_groups)
            )
            attention_blocks.append(
                SelfAttentionBlock(out_channels, num_attention_heads, num_groups)
            )

        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.upsample = UpsampleBlock(out_channels, out_channels) if upsample else None

    def forward(self, x, time_embedding):
        for resnet_block, attention_block in zip(self.resnet_blocks, self.attention_blocks):
            x = resnet_block(x, time_embedding)
            x = attention_block(x)
        if self.upsample:
            x = self.upsample(x)
        return x
