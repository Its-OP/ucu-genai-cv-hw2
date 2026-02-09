"""
UNet noise prediction model for DDPM.

Custom PyTorch implementation of a UNet2D architecture following
Ho et al. 2020 "Denoising Diffusion Probabilistic Models", with
automatic padding/cropping for MNIST's 28x28 images. Images are
padded to 32x32 to support 4-level downsampling (32 -> 16 -> 8 -> 4).

Components (bottom-up):
    1. SinusoidalPositionEmbedding  — timestep -> sinusoidal vector
    2. TimestepEmbeddingMLP         — sinusoidal vector -> dense embedding
    3. ResidualBlock                — ResNet block with timestep conditioning
    4. SelfAttentionBlock           — multi-head self-attention with residual
    5. Downsample2D                 — strided convolution (spatial /2)
    6. Upsample2D                   — nearest interpolation + convolution (spatial *2)
    7. DownBlock                    — encoder block: ResNets + optional attention + optional downsample
    8. UpBlock                      — decoder block: ResNets + skip concat + optional attention + optional upsample
    9. UNetMidBlock                 — bottleneck: ResNet -> Attention -> ResNet
   10. UNet2DModel                  — full UNet combining all components
   11. UNet                         — padding/cropping wrapper (public API)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timesteps.

    Converts a scalar timestep into a fixed-dimensional vector using
    sinusoidal functions at different frequencies (Vaswani et al. 2017).

    Formula:
        embedding[2i]   = sin(t / 10000^(2i / d))
        embedding[2i+1] = cos(t / 10000^(2i / d))

    where d is the embedding_dimension and i ranges over [0, d/2).

    Args:
        embedding_dimension: Size of the output embedding vector.
    """

    def __init__(self, embedding_dimension: int):
        super().__init__()
        self.embedding_dimension = embedding_dimension

        # Precompute frequency terms: exp(-ln(10000) * 2i / d) = 1 / 10000^(2i/d)
        half_dimension = embedding_dimension // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half_dimension, dtype=torch.float32)
            / half_dimension
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestep: Timestep values, shape (batch_size,).

        Returns:
            Sinusoidal embedding, shape (batch_size, embedding_dimension).
        """
        # arguments shape: (batch_size, half_dim)
        arguments = timestep[:, None].float() * self.frequencies[None, :]
        # Concatenate sin and cos: (batch_size, embedding_dimension)
        embedding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=-1)

        # If embedding_dimension is odd, pad with a zero column
        if self.embedding_dimension % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class TimestepEmbeddingMLP(nn.Module):
    """
    Two-layer MLP that projects sinusoidal timestep embeddings into a
    dense conditioning vector used throughout the UNet.

    Architecture:
        Linear(input_dimension -> time_embedding_dimension)
        -> SiLU
        -> Linear(time_embedding_dimension -> time_embedding_dimension)

    Args:
        input_dimension: Dimension of the sinusoidal embedding input.
        time_embedding_dimension: Output dimension (typically base_channels * 4).
    """

    def __init__(self, input_dimension: int, time_embedding_dimension: int):
        super().__init__()
        self.linear_1 = nn.Linear(input_dimension, time_embedding_dimension)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(time_embedding_dimension, time_embedding_dimension)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: Sinusoidal embedding, shape (batch_size, input_dimension).

        Returns:
            Dense timestep conditioning vector, shape (batch_size, time_embedding_dimension).
        """
        embedding = self.linear_1(embedding)
        embedding = self.activation(embedding)
        embedding = self.linear_2(embedding)
        return embedding


class ResidualBlock(nn.Module):
    """
    Residual block with timestep conditioning for diffusion UNets.

    Architecture (Ho et al. 2020):
        GroupNorm(input) -> SiLU -> Conv3x3 (channels: in -> out)
        -> add Linear(SiLU(time_embedding)) (broadcast over spatial dims)
        -> GroupNorm -> SiLU -> Dropout -> Conv3x3 (channels: out -> out)
        -> + residual_shortcut(input)

    The residual shortcut uses a 1x1 convolution when input and output
    channel counts differ, otherwise it is the identity.

    Args:
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        time_embedding_dimension: Dimension of the timestep conditioning vector.
        norm_num_groups: Number of groups for GroupNorm.
        norm_epsilon: Epsilon for GroupNorm.
        dropout_rate: Dropout probability (0.0 = no dropout).
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_embedding_dimension: int = 128,
        norm_num_groups: int = 32,
        norm_epsilon: float = 1e-5,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        # First convolution path: GroupNorm -> SiLU -> Conv3x3
        self.norm_1 = nn.GroupNorm(norm_num_groups, input_channels, eps=norm_epsilon)
        self.activation_1 = nn.SiLU()
        self.convolution_1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding=1
        )

        # Timestep conditioning: SiLU is applied to time_embedding before this linear
        # projection (following the diffusers ResnetBlock2D convention)
        self.time_embedding_projection = nn.Linear(
            time_embedding_dimension, output_channels
        )

        # Second convolution path: GroupNorm -> SiLU -> Dropout -> Conv3x3
        self.norm_2 = nn.GroupNorm(norm_num_groups, output_channels, eps=norm_epsilon)
        self.activation_2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.convolution_2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1
        )

        # Residual shortcut: 1x1 conv when channel counts differ, identity otherwise
        if input_channels != output_channels:
            self.residual_convolution = nn.Conv2d(
                input_channels, output_channels, kernel_size=1
            )
        else:
            self.residual_convolution = nn.Identity()

    def forward(
        self, hidden_states: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Feature map, shape (batch, input_channels, height, width).
            time_embedding: Timestep conditioning vector, shape (batch, time_embedding_dim).

        Returns:
            Output feature map, shape (batch, output_channels, height, width).
        """
        residual = hidden_states

        # First convolution path
        hidden_states = self.norm_1(hidden_states)
        hidden_states = self.activation_1(hidden_states)
        hidden_states = self.convolution_1(hidden_states)

        # Timestep conditioning: project SiLU(temb) and add (broadcast over H, W)
        # time_projection shape: (batch, output_channels) -> (batch, output_channels, 1, 1)
        time_projection = self.time_embedding_projection(F.silu(time_embedding))
        hidden_states = hidden_states + time_projection[:, :, None, None]

        # Second convolution path
        hidden_states = self.norm_2(hidden_states)
        hidden_states = self.activation_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.convolution_2(hidden_states)

        return hidden_states + self.residual_convolution(residual)


class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention block with GroupNorm and residual connection.

    Architecture:
        GroupNorm -> reshape (B,C,H,W) to (B,H*W,C)
        -> Q, K, V linear projections
        -> scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V
        -> output linear projection
        -> reshape back to (B,C,H,W) -> + residual

    Uses PyTorch's F.scaled_dot_product_attention for optimal performance
    (automatically selects FlashAttention / memory-efficient attention).

    Note: Following diffusers UNet2DModel convention, the default
    attention_head_dimension is 1, meaning number_of_heads = channels.

    Args:
        channels: Number of input/output channels.
        norm_num_groups: Number of groups for GroupNorm.
        norm_epsilon: Epsilon for GroupNorm.
        attention_head_dimension: Dimension per attention head (default: 1).
    """

    def __init__(
        self,
        channels: int,
        norm_num_groups: int = 32,
        norm_epsilon: float = 1e-5,
        attention_head_dimension: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.number_of_heads = channels // attention_head_dimension
        self.head_dimension = attention_head_dimension

        self.group_norm = nn.GroupNorm(norm_num_groups, channels, eps=norm_epsilon)

        # Q, K, V projections (Linear layers, matching diffusers convention)
        self.query_projection = nn.Linear(channels, channels)
        self.key_projection = nn.Linear(channels, channels)
        self.value_projection = nn.Linear(channels, channels)

        # Output projection
        self.output_projection = nn.Linear(channels, channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Feature map, shape (batch, channels, height, width).

        Returns:
            Attended feature map, shape (batch, channels, height, width).
        """
        residual = hidden_states
        batch_size, channels, height, width = hidden_states.shape
        sequence_length = height * width

        # GroupNorm then reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, channels, sequence_length)
        hidden_states = hidden_states.permute(0, 2, 1)

        # Compute Q, K, V projections: (B, H*W, C)
        query = self.query_projection(hidden_states)
        key = self.key_projection(hidden_states)
        value = self.value_projection(hidden_states)

        # Reshape for multi-head attention:
        # (B, seq_len, C) -> (B, num_heads, seq_len, head_dim)
        query = query.reshape(
            batch_size, sequence_length, self.number_of_heads, self.head_dimension
        ).permute(0, 2, 1, 3)
        key = key.reshape(
            batch_size, sequence_length, self.number_of_heads, self.head_dimension
        ).permute(0, 2, 1, 3)
        value = value.reshape(
            batch_size, sequence_length, self.number_of_heads, self.head_dimension
        ).permute(0, 2, 1, 3)

        # Scaled dot-product attention:
        #   Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        attention_output = F.scaled_dot_product_attention(query, key, value)

        # Reshape back: (B, num_heads, seq_len, head_dim) -> (B, seq_len, C)
        attention_output = attention_output.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, channels
        )

        # Output projection
        attention_output = self.output_projection(attention_output)

        # Reshape to spatial: (B, H*W, C) -> (B, C, H, W)
        attention_output = attention_output.permute(0, 2, 1).reshape(
            batch_size, channels, height, width
        )

        return attention_output + residual


class CrossAttentionBlock(nn.Module):
    """
    Multi-head cross-attention block with GroupNorm and residual connection.

    Queries are derived from UNet feature maps while keys and values come
    from external conditioning context (e.g., class embeddings, text tokens).

    Architecture:
        GroupNorm -> reshape (B,C,H,W) to (B,H*W,C)
        -> Q projection from features
        -> K, V projections from context (B, S, context_dim)
        -> scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V
        -> output linear projection
        -> reshape back to (B,C,H,W) -> + residual

    Cross-attention formula (Vaswani et al. 2017):
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        where Q = W_q · features, K = W_k · context, V = W_v · context

    Uses PyTorch's F.scaled_dot_product_attention for optimal performance.

    Args:
        channels: Number of input/output feature map channels (query dimension).
        context_dim: Dimension of the external context vectors.
        norm_num_groups: Number of groups for GroupNorm.
        norm_epsilon: Epsilon for GroupNorm.
        attention_head_dimension: Dimension per attention head (default: 1).
    """

    def __init__(
        self,
        channels: int,
        context_dim: int,
        norm_num_groups: int = 32,
        norm_epsilon: float = 1e-5,
        attention_head_dimension: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.number_of_heads = channels // attention_head_dimension
        self.head_dimension = attention_head_dimension

        self.group_norm = nn.GroupNorm(norm_num_groups, channels, eps=norm_epsilon)

        # Q projection from UNet features (channels -> channels)
        self.query_projection = nn.Linear(channels, channels)
        # K, V projections from external context (context_dim -> channels)
        self.key_projection = nn.Linear(context_dim, channels)
        self.value_projection = nn.Linear(context_dim, channels)

        # Output projection
        self.output_projection = nn.Linear(channels, channels)

    def forward(
        self, hidden_states: torch.Tensor, context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Feature map, shape (batch, channels, height, width).
            context: External conditioning context, shape (batch, sequence_length, context_dim).

        Returns:
            Attended feature map, shape (batch, channels, height, width).
        """
        residual = hidden_states
        batch_size, channels, height, width = hidden_states.shape
        sequence_length = height * width

        # GroupNorm then reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, channels, sequence_length)
        hidden_states = hidden_states.permute(0, 2, 1)

        # Q from UNet features: (B, H*W, C)
        query = self.query_projection(hidden_states)

        # K, V from external context: (B, S, context_dim) -> (B, S, C)
        context_sequence_length = context.shape[1]
        key = self.key_projection(context)
        value = self.value_projection(context)

        # Reshape for multi-head attention:
        # Q: (B, H*W, C) -> (B, num_heads, H*W, head_dim)
        query = query.reshape(
            batch_size, sequence_length, self.number_of_heads, self.head_dimension
        ).permute(0, 2, 1, 3)
        # K, V: (B, S, C) -> (B, num_heads, S, head_dim)
        key = key.reshape(
            batch_size, context_sequence_length, self.number_of_heads, self.head_dimension
        ).permute(0, 2, 1, 3)
        value = value.reshape(
            batch_size, context_sequence_length, self.number_of_heads, self.head_dimension
        ).permute(0, 2, 1, 3)

        # Scaled dot-product cross-attention:
        #   Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        attention_output = F.scaled_dot_product_attention(query, key, value)

        # Reshape back: (B, num_heads, H*W, head_dim) -> (B, H*W, C)
        attention_output = attention_output.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, channels
        )

        # Output projection
        attention_output = self.output_projection(attention_output)

        # Reshape to spatial: (B, H*W, C) -> (B, C, H, W)
        attention_output = attention_output.permute(0, 2, 1).reshape(
            batch_size, channels, height, width
        )

        return attention_output + residual


class Downsample2D(nn.Module):
    """
    Spatial downsampling by factor 2 using strided convolution.

    Args:
        channels: Number of input/output channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.convolution = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, C, H/2, W/2)"""
        return self.convolution(hidden_states)


class Upsample2D(nn.Module):
    """
    Spatial upsampling by factor 2 using nearest-neighbor interpolation
    followed by a convolution.

    Args:
        channels: Number of input/output channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.convolution = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, C, 2H, 2W)"""
        hidden_states = F.interpolate(
            hidden_states, scale_factor=2.0, mode="nearest"
        )
        return self.convolution(hidden_states)


class DownBlock(nn.Module):
    """
    Encoder block: ResidualBlocks with optional self-attention, optional
    cross-attention, and downsampling.

    Each block contains `num_layers` ResidualBlocks, each optionally followed by
    a SelfAttentionBlock and then optionally a CrossAttentionBlock (when
    cross_attention_dim is set). Then an optional Downsample2D at the end.

    Skip connections are collected from each ResidualBlock output (after attention)
    and from the Downsample output (if present).

    Args:
        input_channels: Number of channels from the previous block.
        output_channels: Number of channels for this resolution level.
        time_embedding_dimension: Timestep conditioning vector dimension.
        num_layers: Number of ResidualBlocks in this block.
        use_attention: Whether to add SelfAttention after each ResidualBlock.
        add_downsample: Whether to add Downsample2D at the end.
        norm_num_groups: Number of groups for GroupNorm.
        norm_epsilon: Epsilon for GroupNorm.
        attention_head_dimension: Dimension per attention head.
        cross_attention_dim: Dimension of external context for cross-attention.
            If None, no cross-attention blocks are created (default behavior).
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_embedding_dimension: int,
        num_layers: int,
        use_attention: bool,
        add_downsample: bool,
        norm_num_groups: int = 32,
        norm_epsilon: float = 1e-5,
        attention_head_dimension: int = 1,
        cross_attention_dim: int = None,
    ):
        super().__init__()
        self.residual_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.cross_attention_blocks = nn.ModuleList()

        for layer_index in range(num_layers):
            resnet_input_channels = (
                input_channels if layer_index == 0 else output_channels
            )
            self.residual_blocks.append(
                ResidualBlock(
                    input_channels=resnet_input_channels,
                    output_channels=output_channels,
                    time_embedding_dimension=time_embedding_dimension,
                    norm_num_groups=norm_num_groups,
                    norm_epsilon=norm_epsilon,
                )
            )
            if use_attention:
                self.attention_blocks.append(
                    SelfAttentionBlock(
                        channels=output_channels,
                        norm_num_groups=norm_num_groups,
                        norm_epsilon=norm_epsilon,
                        attention_head_dimension=attention_head_dimension,
                    )
                )
                # Cross-attention follows self-attention when context is provided
                if cross_attention_dim is not None:
                    self.cross_attention_blocks.append(
                        CrossAttentionBlock(
                            channels=output_channels,
                            context_dim=cross_attention_dim,
                            norm_num_groups=norm_num_groups,
                            norm_epsilon=norm_epsilon,
                            attention_head_dimension=attention_head_dimension,
                        )
                    )

        self.downsampler = Downsample2D(output_channels) if add_downsample else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_embedding: torch.Tensor,
        context: torch.Tensor = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """
        Args:
            hidden_states: Feature map from previous block.
            time_embedding: Timestep conditioning vector.
            context: Optional external conditioning context for cross-attention,
                shape (batch, sequence_length, context_dim). Ignored when no
                cross-attention blocks exist.

        Returns:
            Tuple of (output_hidden_states, skip_connections) where
            skip_connections is a tuple of tensors for the decoder path.
        """
        skip_connections = ()

        for layer_index, residual_block in enumerate(self.residual_blocks):
            hidden_states = residual_block(hidden_states, time_embedding)
            if self.attention_blocks:
                hidden_states = self.attention_blocks[layer_index](hidden_states)
                if self.cross_attention_blocks and context is not None:
                    hidden_states = self.cross_attention_blocks[layer_index](
                        hidden_states, context,
                    )
            skip_connections = skip_connections + (hidden_states,)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)
            skip_connections = skip_connections + (hidden_states,)

        return hidden_states, skip_connections


class UpBlock(nn.Module):
    """
    Decoder block: ResidualBlocks with skip concatenation, optional self-attention,
    optional cross-attention, and optional upsampling.

    Each block contains `num_layers` ResidualBlocks. Before each ResidualBlock, the
    current feature map is concatenated with a popped skip connection (from the
    encoder). Optionally, each ResidualBlock is followed by a SelfAttentionBlock
    and then a CrossAttentionBlock (when cross_attention_dim is set).
    At the end, an optional Upsample2D doubles the spatial resolution.

    The number of layers is typically `layers_per_block + 1` to account for the
    extra skip connection from the encoder's Downsample operation.

    Args:
        input_channels: Skip connection channels from the symmetric encoder level.
        previous_output_channels: Channels from the previous (deeper) up block.
        output_channels: Target output channels for this level.
        time_embedding_dimension: Timestep conditioning vector dimension.
        num_layers: Number of ResidualBlocks (typically layers_per_block + 1).
        use_attention: Whether to add SelfAttention after each ResidualBlock.
        add_upsample: Whether to add Upsample2D at the end.
        norm_num_groups: Number of groups for GroupNorm.
        norm_epsilon: Epsilon for GroupNorm.
        attention_head_dimension: Dimension per attention head.
        cross_attention_dim: Dimension of external context for cross-attention.
            If None, no cross-attention blocks are created (default behavior).
    """

    def __init__(
        self,
        input_channels: int,
        previous_output_channels: int,
        output_channels: int,
        time_embedding_dimension: int,
        num_layers: int,
        use_attention: bool,
        add_upsample: bool,
        norm_num_groups: int = 32,
        norm_epsilon: float = 1e-5,
        attention_head_dimension: int = 1,
        cross_attention_dim: int = None,
    ):
        super().__init__()
        self.residual_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.cross_attention_blocks = nn.ModuleList()

        for layer_index in range(num_layers):
            # Determine skip connection channel count for this layer:
            # The last layer in the block uses input_channels (from encoder's first output)
            # Other layers use output_channels (from encoder at this same level)
            skip_channels = (
                input_channels
                if layer_index == num_layers - 1
                else output_channels
            )

            # Determine the feature map channels entering this resnet:
            # First layer receives from the previous (deeper) up block
            # Subsequent layers receive output_channels from the previous layer
            resnet_input_channels = (
                previous_output_channels if layer_index == 0 else output_channels
            )

            # Total input = feature map + skip (concatenated along channel dim)
            total_input_channels = resnet_input_channels + skip_channels

            self.residual_blocks.append(
                ResidualBlock(
                    input_channels=total_input_channels,
                    output_channels=output_channels,
                    time_embedding_dimension=time_embedding_dimension,
                    norm_num_groups=norm_num_groups,
                    norm_epsilon=norm_epsilon,
                )
            )
            if use_attention:
                self.attention_blocks.append(
                    SelfAttentionBlock(
                        channels=output_channels,
                        norm_num_groups=norm_num_groups,
                        norm_epsilon=norm_epsilon,
                        attention_head_dimension=attention_head_dimension,
                    )
                )
                # Cross-attention follows self-attention when context is provided
                if cross_attention_dim is not None:
                    self.cross_attention_blocks.append(
                        CrossAttentionBlock(
                            channels=output_channels,
                            context_dim=cross_attention_dim,
                            norm_num_groups=norm_num_groups,
                            norm_epsilon=norm_epsilon,
                            attention_head_dimension=attention_head_dimension,
                        )
                    )

        self.upsampler = Upsample2D(output_channels) if add_upsample else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        skip_connections: tuple[torch.Tensor, ...],
        time_embedding: torch.Tensor,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Feature map from the previous (deeper) up block or mid block.
            skip_connections: Tuple of skip connection tensors from the encoder.
                             Consumed from the end (last element first).
            time_embedding: Timestep conditioning vector.
            context: Optional external conditioning context for cross-attention,
                shape (batch, sequence_length, context_dim). Ignored when no
                cross-attention blocks exist.

        Returns:
            Output feature map after processing and optional upsampling.
        """
        for layer_index, residual_block in enumerate(self.residual_blocks):
            # Pop the last skip connection and concatenate along channel dim
            skip_connection = skip_connections[-1]
            skip_connections = skip_connections[:-1]

            hidden_states = torch.cat([hidden_states, skip_connection], dim=1)
            hidden_states = residual_block(hidden_states, time_embedding)

            if self.attention_blocks:
                hidden_states = self.attention_blocks[layer_index](hidden_states)
                if self.cross_attention_blocks and context is not None:
                    hidden_states = self.cross_attention_blocks[layer_index](
                        hidden_states, context,
                    )

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)

        return hidden_states


class UNetMidBlock(nn.Module):
    """
    UNet bottleneck block: ResidualBlock -> SelfAttention -> [CrossAttention] -> ResidualBlock.

    Operates at the lowest spatial resolution of the UNet without changing
    the spatial dimensions or channel count. Optionally includes a
    CrossAttentionBlock between self-attention and the second residual block.

    Args:
        channels: Number of channels at the bottleneck.
        time_embedding_dimension: Timestep conditioning vector dimension.
        norm_num_groups: Number of groups for GroupNorm.
        norm_epsilon: Epsilon for GroupNorm.
        attention_head_dimension: Dimension per attention head.
        cross_attention_dim: Dimension of external context for cross-attention.
            If None, no cross-attention block is created (default behavior).
    """

    def __init__(
        self,
        channels: int,
        time_embedding_dimension: int,
        norm_num_groups: int = 32,
        norm_epsilon: float = 1e-5,
        attention_head_dimension: int = 1,
        cross_attention_dim: int = None,
    ):
        super().__init__()
        self.residual_block_1 = ResidualBlock(
            input_channels=channels,
            output_channels=channels,
            time_embedding_dimension=time_embedding_dimension,
            norm_num_groups=norm_num_groups,
            norm_epsilon=norm_epsilon,
        )
        self.attention_block = SelfAttentionBlock(
            channels=channels,
            norm_num_groups=norm_num_groups,
            norm_epsilon=norm_epsilon,
            attention_head_dimension=attention_head_dimension,
        )
        self.cross_attention_block = (
            CrossAttentionBlock(
                channels=channels,
                context_dim=cross_attention_dim,
                norm_num_groups=norm_num_groups,
                norm_epsilon=norm_epsilon,
                attention_head_dimension=attention_head_dimension,
            )
            if cross_attention_dim is not None
            else None
        )
        self.residual_block_2 = ResidualBlock(
            input_channels=channels,
            output_channels=channels,
            time_embedding_dimension=time_embedding_dimension,
            norm_num_groups=norm_num_groups,
            norm_epsilon=norm_epsilon,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_embedding: torch.Tensor,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Bottleneck feature map.
            time_embedding: Timestep conditioning vector.
            context: Optional external conditioning context for cross-attention,
                shape (batch, sequence_length, context_dim). Ignored when no
                cross-attention block exists.

        Returns:
            Processed bottleneck feature map (same shape).
        """
        hidden_states = self.residual_block_1(hidden_states, time_embedding)
        hidden_states = self.attention_block(hidden_states)
        if self.cross_attention_block is not None and context is not None:
            hidden_states = self.cross_attention_block(hidden_states, context)
        hidden_states = self.residual_block_2(hidden_states, time_embedding)
        return hidden_states


class UNet2DModel(nn.Module):
    """
    Full UNet2D architecture for diffusion noise prediction.

    Architecture:
        Timestep embedding (sinusoidal + MLP)
        -> Input convolution
        -> Encoder (DownBlocks with skip connections)
        -> Bottleneck (UNetMidBlock)
        -> Decoder (UpBlocks consuming skip connections)
        -> Output (GroupNorm -> SiLU -> Conv)

    Args:
        sample_size: Expected spatial size of input (informational, not enforced).
        input_channels: Number of input image channels.
        output_channels: Number of output image channels.
        block_output_channels: Tuple of channel counts per resolution level.
        layers_per_block: Number of ResidualBlocks per encoder level.
        attention_levels: Tuple of booleans for whether to use attention at each level.
        norm_num_groups: Number of groups for GroupNorm.
        norm_epsilon: Epsilon for GroupNorm.
        attention_head_dimension: Dimension per attention head (default: 1).
        cross_attention_dim: Dimension of external context for cross-attention.
            If None, no cross-attention blocks are created (default behavior).
            Cross-attention blocks are added only at levels where
            use_attention=True, following the self-attention placement.
    """

    def __init__(
        self,
        sample_size: int,
        input_channels: int,
        output_channels: int,
        block_output_channels: tuple[int, ...],
        layers_per_block: int,
        attention_levels: tuple[bool, ...],
        norm_num_groups: int = 32,
        norm_epsilon: float = 1e-5,
        attention_head_dimension: int = 1,
        cross_attention_dim: int = None,
    ):
        super().__init__()

        # Timestep embedding dimension = base_channels * 4
        time_embedding_dimension = block_output_channels[0] * 4

        # --- Timestep embedding ---
        self.time_position_embedding = SinusoidalPositionEmbedding(
            block_output_channels[0]
        )
        self.time_mlp = TimestepEmbeddingMLP(
            block_output_channels[0], time_embedding_dimension
        )

        # --- Input convolution ---
        self.input_convolution = nn.Conv2d(
            input_channels, block_output_channels[0], kernel_size=3, padding=1
        )

        # --- Down blocks (encoder) ---
        self.down_blocks = nn.ModuleList()
        current_channels = block_output_channels[0]

        for level_index, level_output_channels in enumerate(block_output_channels):
            is_final_block = level_index == len(block_output_channels) - 1
            # Only pass cross_attention_dim to levels that have attention enabled
            level_cross_attention_dim = (
                cross_attention_dim if attention_levels[level_index] else None
            )
            self.down_blocks.append(
                DownBlock(
                    input_channels=current_channels,
                    output_channels=level_output_channels,
                    time_embedding_dimension=time_embedding_dimension,
                    num_layers=layers_per_block,
                    use_attention=attention_levels[level_index],
                    add_downsample=not is_final_block,
                    norm_num_groups=norm_num_groups,
                    norm_epsilon=norm_epsilon,
                    attention_head_dimension=attention_head_dimension,
                    cross_attention_dim=level_cross_attention_dim,
                )
            )
            current_channels = level_output_channels

        # --- Mid block (bottleneck) ---
        # Mid block always has self-attention, so include cross-attention if specified
        self.mid_block = UNetMidBlock(
            channels=block_output_channels[-1],
            time_embedding_dimension=time_embedding_dimension,
            norm_num_groups=norm_num_groups,
            norm_epsilon=norm_epsilon,
            attention_head_dimension=attention_head_dimension,
            cross_attention_dim=cross_attention_dim,
        )

        # --- Up blocks (decoder) ---
        # The decoder mirrors the encoder in reverse. Each up block has
        # layers_per_block + 1 ResidualBlocks to consume all skip connections.
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(block_output_channels))

        previous_output_channel = reversed_channels[0]
        for level_index in range(len(block_output_channels)):
            is_final_block = level_index == len(block_output_channels) - 1
            output_channel = reversed_channels[level_index]
            # input_channel: the channels from the symmetric encoder level's
            # first output (used for the last skip connection in this up block)
            input_channel = reversed_channels[
                min(level_index + 1, len(block_output_channels) - 1)
            ]

            # Mirror the encoder's attention level for the decoder
            decoder_attention_level = attention_levels[
                len(block_output_channels) - 1 - level_index
            ]
            level_cross_attention_dim = (
                cross_attention_dim if decoder_attention_level else None
            )

            self.up_blocks.append(
                UpBlock(
                    input_channels=input_channel,
                    previous_output_channels=previous_output_channel,
                    output_channels=output_channel,
                    time_embedding_dimension=time_embedding_dimension,
                    num_layers=layers_per_block + 1,
                    use_attention=decoder_attention_level,
                    add_upsample=not is_final_block,
                    norm_num_groups=norm_num_groups,
                    norm_epsilon=norm_epsilon,
                    attention_head_dimension=attention_head_dimension,
                    cross_attention_dim=level_cross_attention_dim,
                )
            )
            previous_output_channel = output_channel

        # --- Output projection ---
        self.output_norm = nn.GroupNorm(
            norm_num_groups, block_output_channels[0], eps=norm_epsilon
        )
        self.output_activation = nn.SiLU()
        self.output_convolution = nn.Conv2d(
            block_output_channels[0], output_channels, kernel_size=3, padding=1
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predict noise for the given noisy sample and timestep.

        Args:
            sample: Noisy images, shape (batch, input_channels, height, width).
            timestep: Diffusion timesteps, shape (batch,).
            context: Optional external conditioning context for cross-attention,
                shape (batch, sequence_length, context_dim). Only used when
                cross_attention_dim was set during construction.

        Returns:
            Predicted noise tensor, same spatial shape as sample.
        """
        # --- Timestep embedding ---
        time_embedding = self.time_position_embedding(timestep.float())
        time_embedding = self.time_mlp(time_embedding)

        # --- Input convolution ---
        sample = self.input_convolution(sample)

        # --- Encoder (down path) ---
        # Collect skip connections, starting with the input conv output
        down_block_residual_samples = (sample,)

        for down_block in self.down_blocks:
            sample, residual_samples = down_block(
                sample, time_embedding, context=context,
            )
            down_block_residual_samples += residual_samples

        # --- Bottleneck ---
        sample = self.mid_block(sample, time_embedding, context=context)

        # --- Decoder (up path) ---
        for up_block in self.up_blocks:
            # Slice off the skip connections for this up block
            num_resnets = len(up_block.residual_blocks)
            skip_connections = down_block_residual_samples[-num_resnets:]
            down_block_residual_samples = down_block_residual_samples[:-num_resnets]

            sample = up_block(
                sample, skip_connections, time_embedding, context=context,
            )

        # --- Output ---
        sample = self.output_norm(sample)
        sample = self.output_activation(sample)
        sample = self.output_convolution(sample)

        return sample


class UNet(nn.Module):
    """
    Wrapper around UNet2DModel for DDPM noise prediction.

    Pads 28x28 MNIST images to 32x32 internally (following the DDPM paper),
    runs through a multi-level UNet, then crops back to 28x28. This makes the
    padding/cropping transparent to the DDPM training and sampling loops.

    Architecture (based on Ho et al. 2020):
        - Configurable resolution levels with channel multipliers
        - Configurable ResNet blocks per level
        - Self-attention at selected resolution levels
        - Optional cross-attention for external conditioning
        - Sinusoidal timestep embeddings

    Args:
        image_channels (int): Number of input image channels. Default: 1 (grayscale).
        output_channels (int or None): Number of output image channels. If None, defaults
                             to image_channels (symmetric input/output). Set explicitly when
                             input has extra conditioning channels (e.g., class-conditioned
                             diffusion where input_channels = latent_channels + 1).
        base_channels (int): Base channel count, multiplied by each entry in channel_multipliers.
                             Default: 32.
        channel_multipliers (tuple[int]): Per-level channel multipliers applied to base_channels.
                             Default: (1, 2, 3, 3) -> channels (32, 64, 96, 96) -> ~2.7M params.
        layers_per_block (int): Number of ResNet blocks at each resolution level.
                             Default: 1.
        attention_levels (tuple[bool]): Whether to use self-attention at each resolution level.
                             Default: (False, False, False, True) -> attention only at 4x4 bottleneck.
        norm_num_groups (int): Number of groups for GroupNorm. Must divide all channel counts.
                             Default: 32.
        cross_attention_dim (int or None): Dimension of external context for cross-attention.
                             If None (default), no cross-attention blocks are created.
                             When set, cross-attention blocks are added after self-attention
                             at levels where attention is enabled.
    """

    def __init__(
        self,
        image_channels=1,
        output_channels=None,
        base_channels=32,
        channel_multipliers=(1, 2, 3, 3),
        layers_per_block=1,
        attention_levels=(False, False, False, True),
        norm_num_groups=32,
        cross_attention_dim=None,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.output_channels = output_channels if output_channels is not None else image_channels

        # Compute per-level channel counts: base_channels * multiplier for each level
        block_output_channels = tuple(
            base_channels * multiplier for multiplier in channel_multipliers
        )

        self.model = UNet2DModel(
            sample_size=32,
            input_channels=image_channels,
            output_channels=self.output_channels,
            block_output_channels=block_output_channels,
            layers_per_block=layers_per_block,
            attention_levels=attention_levels,
            norm_num_groups=norm_num_groups,
            norm_epsilon=1e-5,
            cross_attention_dim=cross_attention_dim,
        )

    def forward(self, x, timestep, context=None):
        """
        Predict noise epsilon_theta(x_t, t) for the given noisy input and timestep.

        Internally pads input to the next multiple of 8 (e.g. 28x28 -> 32x32),
        runs through UNet2DModel, then crops back to the original dimensions.

        Args:
            x: Noisy images, shape (batch_size, channels, height, width).
               Typically (B, 1, 28, 28) for MNIST.
            timestep: Diffusion timesteps, shape (batch_size,).
            context: Optional external conditioning context for cross-attention,
                shape (batch, sequence_length, context_dim). Only used when
                cross_attention_dim was set during construction.

        Returns:
            Predicted noise tensor with the same shape as x.
        """
        original_height = x.shape[2]
        original_width = x.shape[3]

        # Pad to the next multiple of 8 for multi-level UNet compatibility
        # For 28x28 -> 32x32: pad 2 pixels on each side
        # F.pad format: (left, right, top, bottom)
        pad_height = (8 - original_height % 8) % 8
        pad_width = (8 - original_width % 8) % 8
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        if pad_height > 0 or pad_width > 0:
            x_padded = F.pad(
                x, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect"
            )
        else:
            x_padded = x

        # Run through UNet and get predicted noise tensor
        noise_prediction = self.model(x_padded, timestep, context=context)

        # Crop back to original spatial dimensions
        if pad_height > 0 or pad_width > 0:
            noise_prediction = noise_prediction[
                :,
                :,
                pad_top : pad_top + original_height,
                pad_left : pad_left + original_width,
            ]

        return noise_prediction
