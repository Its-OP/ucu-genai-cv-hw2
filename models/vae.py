"""
Variational Autoencoder (VAE) for latent diffusion.

Custom PyTorch implementation following the CompVis Latent Diffusion Model
(Rombach et al. 2022). Compresses images (e.g., 1x32x32 padded MNIST) into
a regularized Gaussian latent space (e.g., 2x4x4) suitable for training
a diffusion model in the compressed representation.

Components (bottom-up):
    1.  VAEResidualBlock           — ResNet block (no timestep conditioning)
    2.  VAEAttentionBlock          — multi-head self-attention with residual
    3.  VAEDownsample              — strided convolution (spatial /2)
    4.  VAEUpsample                — nearest interpolation + convolution (spatial *2)
    5.  VAEEncoderBlock            — encoder level: ResNets + optional attention + optional downsample
    6.  VAEDecoderBlock            — decoder level: ResNets + optional attention + optional upsample
    7.  VAEMidBlock                — bottleneck: ResNet -> Attention -> ResNet
    8.  DiagonalGaussianDistribution — latent posterior: mean, logvar, sample, kl
    9.  VAEEncoder                 — full encoder network
    10. VAEDecoder                 — full decoder network
    11. VAE                        — public API: encode, decode, forward, loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEResidualBlock(nn.Module):
    """
    Residual block for the VAE encoder/decoder (no timestep conditioning).

    Architecture (following CompVis LDM):
        GroupNorm(input) -> SiLU -> Conv3x3 (channels: in -> out)
        -> GroupNorm -> SiLU -> Dropout -> Conv3x3 (channels: out -> out)
        -> + residual_shortcut(input)

    Unlike the UNet's ResidualBlock, this block has no timestep embedding
    injection, making it suitable for the unconditional VAE architecture.

    Args:
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        norm_num_groups: Number of groups for GroupNorm.
        norm_epsilon: Epsilon for GroupNorm.
        dropout_rate: Dropout probability (0.0 = no dropout).
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        norm_num_groups: int = 32,
        norm_epsilon: float = 1e-6,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        # First convolution path: GroupNorm -> SiLU -> Conv3x3
        self.norm_1 = nn.GroupNorm(norm_num_groups, input_channels, eps=norm_epsilon)
        self.activation_1 = nn.SiLU()
        self.convolution_1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding=1
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Feature map, shape (batch, input_channels, height, width).

        Returns:
            Output feature map, shape (batch, output_channels, height, width).
        """
        residual = hidden_states

        # First convolution path
        hidden_states = self.norm_1(hidden_states)
        hidden_states = self.activation_1(hidden_states)
        hidden_states = self.convolution_1(hidden_states)

        # Second convolution path
        hidden_states = self.norm_2(hidden_states)
        hidden_states = self.activation_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.convolution_2(hidden_states)

        return hidden_states + self.residual_convolution(residual)


class VAEAttentionBlock(nn.Module):
    """
    Self-attention block for the VAE with GroupNorm and residual connection.

    Architecture:
        GroupNorm -> reshape (B,C,H,W) to (B,H*W,C)
        -> Q, K, V linear projections
        -> scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V
        -> output linear projection
        -> reshape back to (B,C,H,W) -> + residual

    Uses PyTorch's F.scaled_dot_product_attention for optimal performance
    (automatically selects FlashAttention / memory-efficient attention).

    Following the CompVis LDM convention, uses a single attention head
    (head_dim = channels) at the bottleneck resolution.

    Args:
        channels: Number of input/output channels.
        norm_num_groups: Number of groups for GroupNorm.
        norm_epsilon: Epsilon for GroupNorm.
    """

    def __init__(
        self,
        channels: int,
        norm_num_groups: int = 32,
        norm_epsilon: float = 1e-6,
    ):
        super().__init__()
        self.channels = channels

        self.group_norm = nn.GroupNorm(norm_num_groups, channels, eps=norm_epsilon)

        # Q, K, V projections (Linear layers, matching CompVis LDM convention)
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

        # Single-head attention (head_dim = channels):
        # Add head dimension: (B, H*W, C) -> (B, 1, H*W, C)
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        # Scaled dot-product attention:
        #   Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        attention_output = F.scaled_dot_product_attention(query, key, value)

        # Remove head dimension and reshape: (B, 1, H*W, C) -> (B, H*W, C)
        attention_output = attention_output.squeeze(1)

        # Output projection
        attention_output = self.output_projection(attention_output)

        # Reshape to spatial: (B, H*W, C) -> (B, C, H, W)
        attention_output = attention_output.permute(0, 2, 1).reshape(
            batch_size, channels, height, width
        )

        return attention_output + residual


class VAEDownsample(nn.Module):
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


class VAEUpsample(nn.Module):
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


class VAEEncoderBlock(nn.Module):
    """
    VAE encoder block: ResidualBlocks with optional downsampling.

    Each block contains `num_layers` VAEResidualBlocks, then an optional
    VAEDownsample at the end. Unlike the UNet's DownBlock, no skip
    connections are collected (the VAE uses a bottleneck architecture).

    Args:
        input_channels: Number of channels from the previous block.
        output_channels: Number of channels for this resolution level.
        num_layers: Number of VAEResidualBlocks in this block.
        add_downsample: Whether to add VAEDownsample at the end.
        norm_num_groups: Number of groups for GroupNorm.
        norm_epsilon: Epsilon for GroupNorm.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_layers: int,
        add_downsample: bool,
        norm_num_groups: int = 32,
        norm_epsilon: float = 1e-6,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.residual_blocks = nn.ModuleList()

        for layer_index in range(num_layers):
            resnet_input_channels = (
                input_channels if layer_index == 0 else output_channels
            )
            self.residual_blocks.append(
                VAEResidualBlock(
                    input_channels=resnet_input_channels,
                    output_channels=output_channels,
                    norm_num_groups=norm_num_groups,
                    norm_epsilon=norm_epsilon,
                    dropout_rate=dropout_rate,
                )
            )

        self.downsampler = VAEDownsample(output_channels) if add_downsample else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Feature map from previous block.

        Returns:
            Processed feature map (optionally downsampled).
        """
        for residual_block in self.residual_blocks:
            hidden_states = residual_block(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)

        return hidden_states


class VAEDecoderBlock(nn.Module):
    """
    VAE decoder block: ResidualBlocks with optional upsampling.

    Each block contains `num_layers` VAEResidualBlocks, then an optional
    VAEUpsample at the end. Unlike the UNet's UpBlock, no skip connections
    are consumed (the VAE uses a bottleneck architecture).

    Args:
        input_channels: Number of channels from the previous block.
        output_channels: Number of channels for this resolution level.
        num_layers: Number of VAEResidualBlocks in this block.
        add_upsample: Whether to add VAEUpsample at the end.
        norm_num_groups: Number of groups for GroupNorm.
        norm_epsilon: Epsilon for GroupNorm.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_layers: int,
        add_upsample: bool,
        norm_num_groups: int = 32,
        norm_epsilon: float = 1e-6,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.residual_blocks = nn.ModuleList()

        for layer_index in range(num_layers):
            resnet_input_channels = (
                input_channels if layer_index == 0 else output_channels
            )
            self.residual_blocks.append(
                VAEResidualBlock(
                    input_channels=resnet_input_channels,
                    output_channels=output_channels,
                    norm_num_groups=norm_num_groups,
                    norm_epsilon=norm_epsilon,
                    dropout_rate=dropout_rate,
                )
            )

        self.upsampler = VAEUpsample(output_channels) if add_upsample else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Feature map from the previous block.

        Returns:
            Processed feature map (optionally upsampled).
        """
        for residual_block in self.residual_blocks:
            hidden_states = residual_block(hidden_states)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)

        return hidden_states


class VAEMidBlock(nn.Module):
    """
    VAE bottleneck block: VAEResidualBlock -> VAEAttentionBlock -> VAEResidualBlock.

    Operates at the lowest spatial resolution of the VAE without changing
    the spatial dimensions or channel count.

    Args:
        channels: Number of channels at the bottleneck.
        norm_num_groups: Number of groups for GroupNorm.
        norm_epsilon: Epsilon for GroupNorm.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        channels: int,
        norm_num_groups: int = 32,
        norm_epsilon: float = 1e-6,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.residual_block_1 = VAEResidualBlock(
            input_channels=channels,
            output_channels=channels,
            norm_num_groups=norm_num_groups,
            norm_epsilon=norm_epsilon,
            dropout_rate=dropout_rate,
        )
        self.attention_block = VAEAttentionBlock(
            channels=channels,
            norm_num_groups=norm_num_groups,
            norm_epsilon=norm_epsilon,
        )
        self.residual_block_2 = VAEResidualBlock(
            input_channels=channels,
            output_channels=channels,
            norm_num_groups=norm_num_groups,
            norm_epsilon=norm_epsilon,
            dropout_rate=dropout_rate,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Bottleneck feature map.

        Returns:
            Processed bottleneck feature map (same shape).
        """
        hidden_states = self.residual_block_1(hidden_states)
        hidden_states = self.attention_block(hidden_states)
        hidden_states = self.residual_block_2(hidden_states)
        return hidden_states


class DiagonalGaussianDistribution:
    """
    Diagonal Gaussian distribution parameterized by mean and log-variance.

    Used as the VAE posterior q(z|x). Supports reparameterized sampling
    and KL divergence computation against a standard normal prior N(0,I).

    Following the CompVis LDM implementation, the log-variance is clamped
    to [-30, 20] for numerical stability.

    Args:
        parameters: Tensor of shape (B, 2*C, H, W) containing concatenated
                   mean and log-variance along the channel dimension.
    """

    def __init__(self, parameters: torch.Tensor):
        # Split parameters into mean and log-variance along channel dimension
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)

        # Clamp log-variance for numerical stability (CompVis LDM convention)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)

        # Precompute standard deviation and variance
        # var = exp(logvar), std = exp(0.5 * logvar)
        self.var = torch.exp(self.logvar)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self) -> torch.Tensor:
        """
        Sample from the distribution using the reparameterization trick.

        Formula:
            z = mean + std * epsilon, where epsilon ~ N(0, I)

        This allows gradients to flow through the sampling operation
        back to the encoder parameters (mean and logvar).

        Returns:
            Sampled latent tensor, shape (B, C, H, W).
        """
        # Reparameterization trick: z = mu + sigma * epsilon
        epsilon = torch.randn_like(self.std)
        return self.mean + self.std * epsilon

    def kl(self) -> torch.Tensor:
        """
        Compute KL divergence against the standard normal prior N(0, I).

        Formula:
            D_KL(q(z|x) || p(z)) = 0.5 * sum(mean^2 + var - 1 - logvar)

        where the sum is over all latent dimensions (C, H, W) and the
        result is averaged over the batch.

        Returns:
            Scalar KL divergence loss (averaged over batch).
        """
        # KL divergence between N(mu, sigma^2) and N(0, 1):
        #   D_KL = 0.5 * sum(mu^2 + sigma^2 - 1 - log(sigma^2))
        #        = 0.5 * sum(mean^2 + var - 1 - logvar)
        kl_per_sample = 0.5 * torch.sum(
            self.mean.pow(2) + self.var - 1.0 - self.logvar,
            dim=[1, 2, 3],
        )
        return kl_per_sample.mean()

    def mode(self) -> torch.Tensor:
        """
        Return the mode of the distribution (deterministic encoding).

        For a Gaussian distribution, the mode equals the mean.

        Returns:
            Mean tensor, shape (B, C, H, W).
        """
        return self.mean


class VAEEncoder(nn.Module):
    """
    VAE Encoder: compresses input images to latent distribution parameters.

    Architecture:
        Input Conv3x3 (image_channels -> base_channels)
        -> VAEEncoderBlocks with progressive channel expansion and downsampling
        -> VAEMidBlock at lowest resolution
        -> GroupNorm -> SiLU -> Conv3x3 (-> 2 * latent_channels for mean + logvar)
        -> quantization_convolution: Conv1x1 (learnable projection)

    For MNIST (1x32x32 input, channel_multipliers=(1,2,4), base_channels=64):
        1x32x32 -> 64x32x32 -> [Block0+down] 64x16x16
        -> [Block1+down] 128x8x8 -> [Block2+down] 256x4x4
        -> MidBlock 256x4x4 -> norm -> SiLU -> Conv -> 4x4x4
        -> quant_conv -> split -> mean(2x4x4), logvar(2x4x4)

    Args:
        image_channels: Number of input image channels (1 for MNIST).
        latent_channels: Number of latent space channels.
        base_channels: Base channel count multiplied by channel_multipliers.
        channel_multipliers: Per-level channel multipliers.
        num_layers_per_block: Number of VAEResidualBlocks per encoder level.
        norm_num_groups: Number of groups for GroupNorm.
        norm_epsilon: Epsilon for GroupNorm.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        image_channels: int = 1,
        latent_channels: int = 2,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        num_layers_per_block: int = 1,
        norm_num_groups: int = 32,
        norm_epsilon: float = 1e-6,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        # Compute per-level channel counts
        block_channels = tuple(
            base_channels * multiplier for multiplier in channel_multipliers
        )

        # --- Input convolution ---
        self.input_convolution = nn.Conv2d(
            image_channels, block_channels[0], kernel_size=3, padding=1
        )

        # --- Encoder blocks (progressive downsampling) ---
        self.encoder_blocks = nn.ModuleList()
        current_channels = block_channels[0]

        for level_index, level_channels in enumerate(block_channels):
            # All levels downsample (each reduces spatial dims by 2x)
            self.encoder_blocks.append(
                VAEEncoderBlock(
                    input_channels=current_channels,
                    output_channels=level_channels,
                    num_layers=num_layers_per_block,
                    add_downsample=True,
                    norm_num_groups=norm_num_groups,
                    norm_epsilon=norm_epsilon,
                    dropout_rate=dropout_rate,
                )
            )
            current_channels = level_channels

        # --- Mid block (bottleneck with attention) ---
        self.mid_block = VAEMidBlock(
            channels=block_channels[-1],
            norm_num_groups=norm_num_groups,
            norm_epsilon=norm_epsilon,
            dropout_rate=dropout_rate,
        )

        # --- Output projection ---
        # GroupNorm -> SiLU -> Conv3x3 -> 2 * latent_channels (mean + logvar)
        self.output_norm = nn.GroupNorm(
            norm_num_groups, block_channels[-1], eps=norm_epsilon
        )
        self.output_activation = nn.SiLU()
        self.output_convolution = nn.Conv2d(
            block_channels[-1], 2 * latent_channels, kernel_size=3, padding=1
        )

        # --- Quantization convolution (CompVis LDM convention) ---
        # Learnable 1x1 projection from encoder output to latent parameters
        self.quantization_convolution = nn.Conv2d(
            2 * latent_channels, 2 * latent_channels, kernel_size=1
        )

    def forward(self, pixel_values: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        Encode images to a diagonal Gaussian distribution in latent space.

        Args:
            pixel_values: Input images, shape (B, image_channels, H, W).

        Returns:
            DiagonalGaussianDistribution parameterized by the encoder output.
        """
        hidden_states = self.input_convolution(pixel_values)

        for encoder_block in self.encoder_blocks:
            hidden_states = encoder_block(hidden_states)

        hidden_states = self.mid_block(hidden_states)

        hidden_states = self.output_norm(hidden_states)
        hidden_states = self.output_activation(hidden_states)
        hidden_states = self.output_convolution(hidden_states)

        hidden_states = self.quantization_convolution(hidden_states)

        return DiagonalGaussianDistribution(hidden_states)


class VAEDecoder(nn.Module):
    """
    VAE Decoder: reconstructs images from latent space samples.

    Architecture:
        post_quantization_convolution: Conv1x1 (latent_channels -> latent_channels)
        -> Input Conv3x3 (latent_channels -> highest_channels)
        -> VAEMidBlock at lowest resolution
        -> VAEDecoderBlocks with progressive channel reduction and upsampling
        -> GroupNorm -> SiLU -> Output Conv3x3 (-> image_channels)

    For MNIST (2x4x4 latent, channel_multipliers=(1,2,4), base_channels=64):
        2x4x4 -> Conv1x1 -> 2x4x4 -> Conv3x3 -> 256x4x4
        -> MidBlock 256x4x4 -> [Block0+up] 256x8x8
        -> [Block1+up] 128x16x16 -> [Block2+up] 64x32x32
        -> norm -> SiLU -> Conv -> 1x32x32

    Args:
        image_channels: Number of output image channels (1 for MNIST).
        latent_channels: Number of latent space channels.
        base_channels: Base channel count multiplied by channel_multipliers.
        channel_multipliers: Per-level channel multipliers.
        num_layers_per_block: Number of VAEResidualBlocks per decoder level.
        norm_num_groups: Number of groups for GroupNorm.
        norm_epsilon: Epsilon for GroupNorm.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        image_channels: int = 1,
        latent_channels: int = 2,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        num_layers_per_block: int = 1,
        norm_num_groups: int = 32,
        norm_epsilon: float = 1e-6,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        # Compute per-level channel counts
        block_channels = tuple(
            base_channels * multiplier for multiplier in channel_multipliers
        )
        highest_channels = block_channels[-1]

        # --- Post-quantization convolution (CompVis LDM convention) ---
        # Learnable 1x1 projection from latent to decoder input
        self.post_quantization_convolution = nn.Conv2d(
            latent_channels, latent_channels, kernel_size=1
        )

        # --- Input convolution ---
        self.input_convolution = nn.Conv2d(
            latent_channels, highest_channels, kernel_size=3, padding=1
        )

        # --- Mid block (bottleneck with attention) ---
        self.mid_block = VAEMidBlock(
            channels=highest_channels,
            norm_num_groups=norm_num_groups,
            norm_epsilon=norm_epsilon,
            dropout_rate=dropout_rate,
        )

        # --- Decoder blocks (progressive upsampling, reverse channel order) ---
        # Reversed channels: e.g., (256, 128, 64) for multipliers (1, 2, 4)
        reversed_channels = list(reversed(block_channels))

        self.decoder_blocks = nn.ModuleList()
        current_channels = highest_channels

        for level_index, level_channels in enumerate(reversed_channels):
            # All levels upsample (each doubles spatial dims)
            self.decoder_blocks.append(
                VAEDecoderBlock(
                    input_channels=current_channels,
                    output_channels=level_channels,
                    num_layers=num_layers_per_block,
                    add_upsample=True,
                    norm_num_groups=norm_num_groups,
                    norm_epsilon=norm_epsilon,
                    dropout_rate=dropout_rate,
                )
            )
            current_channels = level_channels

        # --- Output projection ---
        output_channels = block_channels[0]
        self.output_norm = nn.GroupNorm(
            norm_num_groups, output_channels, eps=norm_epsilon
        )
        self.output_activation = nn.SiLU()
        self.output_convolution = nn.Conv2d(
            output_channels, image_channels, kernel_size=3, padding=1
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent samples to reconstructed images.

        Args:
            latent: Latent samples, shape (B, latent_channels, H_latent, W_latent).

        Returns:
            Reconstructed images, shape (B, image_channels, H_image, W_image).
        """
        hidden_states = self.post_quantization_convolution(latent)
        hidden_states = self.input_convolution(hidden_states)

        hidden_states = self.mid_block(hidden_states)

        for decoder_block in self.decoder_blocks:
            hidden_states = decoder_block(hidden_states)

        hidden_states = self.output_norm(hidden_states)
        hidden_states = self.output_activation(hidden_states)
        hidden_states = self.output_convolution(hidden_states)

        return hidden_states


class VAE(nn.Module):
    """
    Variational Autoencoder for image compression to latent space.

    Compresses input images (e.g., 1x32x32 padded MNIST) into a
    lower-dimensional latent representation (e.g., 2x4x4) using an
    encoder-decoder architecture with a diagonal Gaussian latent
    distribution regularized by KL divergence against N(0, I).

    Following the CompVis Latent Diffusion Model (Rombach et al. 2022):
        - encode(x) returns a DiagonalGaussianDistribution
        - decode(z) returns the reconstructed image
        - forward(x) returns (reconstruction, posterior) for training
        - loss() computes reconstruction_loss + kl_weight * kl_loss

    The VAE training loss is:
        L = L_rec(x, x_hat) + beta * D_KL(q(z|x) || N(0, I))

    where L_rec is MSE reconstruction loss and beta (kl_weight) is
    typically very small (e.g., 1e-6) to prevent posterior collapse
    while maintaining latent space regularity.

    Args:
        image_channels: Input/output image channels (1 for MNIST).
        latent_channels: Latent space channels (2 for 2x4x4 latent).
        base_channels: Base channel count for encoder/decoder.
        channel_multipliers: Per-level channel multipliers.
        num_layers_per_block: VAEResidualBlocks per encoder/decoder level.
        norm_num_groups: Groups for GroupNorm.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        image_channels: int = 1,
        latent_channels: int = 2,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        num_layers_per_block: int = 1,
        norm_num_groups: int = 32,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.latent_channels = latent_channels

        self.encoder = VAEEncoder(
            image_channels=image_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_layers_per_block=num_layers_per_block,
            norm_num_groups=norm_num_groups,
            dropout_rate=dropout_rate,
        )
        self.decoder = VAEDecoder(
            image_channels=image_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_layers_per_block=num_layers_per_block,
            norm_num_groups=norm_num_groups,
            dropout_rate=dropout_rate,
        )

    def encode(self, pixel_values: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        Encode images to a diagonal Gaussian distribution in latent space.

        Args:
            pixel_values: Input images, shape (B, image_channels, H, W).

        Returns:
            DiagonalGaussianDistribution over the latent space.
        """
        return self.encoder(pixel_values)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent samples to reconstructed images.

        Args:
            latent: Latent samples, shape (B, latent_channels, H_latent, W_latent).

        Returns:
            Reconstructed images, shape (B, image_channels, H, W).
        """
        return self.decoder(latent)

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """
        Full VAE forward pass: encode -> sample -> decode.

        Args:
            pixel_values: Input images, shape (B, image_channels, H, W).

        Returns:
            Tuple of (reconstruction, posterior) where:
                - reconstruction: Decoded images, same shape as input
                - posterior: DiagonalGaussianDistribution from encoder
        """
        posterior = self.encode(pixel_values)
        latent = posterior.sample()
        reconstruction = self.decode(latent)
        return reconstruction, posterior

    @staticmethod
    def loss(
        pixel_values: torch.Tensor,
        reconstruction: torch.Tensor,
        posterior: DiagonalGaussianDistribution,
        kl_weight: float = 1e-6,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the VAE training loss.

        Formula:
            L = L_rec(x, x_hat) + kl_weight * D_KL(q(z|x) || N(0, I))

        where L_rec is per-pixel MSE and D_KL is the KL divergence
        of the encoder posterior against the standard normal prior.

        Args:
            pixel_values: Original input images.
            reconstruction: Reconstructed images from the decoder.
            posterior: Encoder's posterior distribution.
            kl_weight: Weight for the KL divergence term (beta in beta-VAE).

        Returns:
            Dictionary with keys:
                - 'reconstruction_loss': MSE between input and reconstruction
                - 'kl_loss': KL divergence against N(0, I)
                - 'total_loss': reconstruction_loss + kl_weight * kl_loss
        """
        reconstruction_loss = F.mse_loss(reconstruction, pixel_values)
        kl_loss = posterior.kl()
        total_loss = reconstruction_loss + kl_weight * kl_loss

        return {
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
        }
