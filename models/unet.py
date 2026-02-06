"""
UNet noise prediction model for DDPM.

Wraps Hugging Face's diffusers.UNet2DModel with automatic padding/cropping
for MNIST's 28×28 images. Following the DDPM paper (Ho et al. 2020), images
are padded to 32×32 to support 4-level downsampling (32 → 16 → 8 → 4).
"""
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel


class UNet(nn.Module):
    """
    Wrapper around diffusers.UNet2DModel for DDPM noise prediction.

    Pads 28×28 MNIST images to 32×32 internally (following the DDPM paper),
    runs through a multi-level UNet, then crops back to 28×28. This makes the
    padding/cropping transparent to the DDPM training and sampling loops.

    Architecture (based on Ho et al. 2020):
        - Configurable resolution levels with channel multipliers
        - Configurable ResNet blocks per level
        - Self-attention at selected resolution levels
        - Sinusoidal timestep embeddings

    Args:
        image_channels (int): Number of input/output image channels. Default: 1 (grayscale).
        base_channels (int): Base channel count, multiplied by each entry in channel_multipliers.
                             Default: 32.
        channel_multipliers (tuple[int]): Per-level channel multipliers applied to base_channels.
                             Default: (1, 2, 3, 3) → channels (32, 64, 96, 96) → ~2.7M params.
        layers_per_block (int): Number of ResNet blocks at each resolution level.
                             Default: 1.
        attention_levels (tuple[bool]): Whether to use self-attention at each resolution level.
                             Default: (False, False, False, True) → attention only at 4×4 bottleneck.
        norm_num_groups (int): Number of groups for GroupNorm. Must divide all channel counts.
                             Default: 32.
    """

    def __init__(
        self,
        image_channels=1,
        base_channels=32,
        channel_multipliers=(1, 2, 3, 3),
        layers_per_block=1,
        attention_levels=(False, False, False, True),
        norm_num_groups=32,
    ):
        super().__init__()
        self.image_channels = image_channels

        # Compute per-level channel counts: base_channels × multiplier for each level
        block_output_channels = tuple(
            base_channels * multiplier for multiplier in channel_multipliers
        )

        # Build block type tuples from attention_levels flags
        # True → attention block (AttnDownBlock2D / AttnUpBlock2D)
        # False → plain convolution block (DownBlock2D / UpBlock2D)
        down_block_types = tuple(
            "AttnDownBlock2D" if use_attention else "DownBlock2D"
            for use_attention in attention_levels
        )
        up_block_types = tuple(
            "AttnUpBlock2D" if use_attention else "UpBlock2D"
            for use_attention in reversed(attention_levels)
        )

        self.model = UNet2DModel(
            sample_size=32,
            in_channels=image_channels,
            out_channels=image_channels,
            block_out_channels=block_output_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            norm_eps=1e-5,
        )

    def forward(self, x, timestep):
        """
        Predict noise ε_θ(x_t, t) for the given noisy input and timestep.

        Internally pads input to the next multiple of 8 (e.g. 28×28 → 32×32),
        runs through UNet2DModel, then crops back to the original dimensions.

        Args:
            x: Noisy images, shape (batch_size, channels, height, width).
               Typically (B, 1, 28, 28) for MNIST.
            timestep: Diffusion timesteps, shape (batch_size,).

        Returns:
            Predicted noise tensor with the same shape as x.
        """
        original_height = x.shape[2]
        original_width = x.shape[3]

        # Pad to the next multiple of 8 for 4-level UNet compatibility
        # For 28×28 → 32×32: pad 2 pixels on each side
        # F.pad format: (left, right, top, bottom)
        pad_height = (8 - original_height % 8) % 8
        pad_width = (8 - original_width % 8) % 8
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        if pad_height > 0 or pad_width > 0:
            x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        else:
            x_padded = x

        # UNet2DModel.forward returns UNet2DOutput namedtuple;
        # extract the .sample field to get the predicted noise tensor
        noise_prediction = self.model(x_padded, timestep, return_dict=True).sample

        # Crop back to original spatial dimensions
        if pad_height > 0 or pad_width > 0:
            noise_prediction = noise_prediction[
                :, :,
                pad_top: pad_top + original_height,
                pad_left: pad_left + original_width,
            ]

        return noise_prediction
