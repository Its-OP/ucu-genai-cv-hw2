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
    runs through a 4-level UNet, then crops back to 28×28. This makes the
    padding/cropping transparent to the DDPM training and sampling loops.

    Architecture (Ho et al. 2020):
        - 4 resolution levels with channel multipliers (1×, 2×, 4×, 4×)
        - 2 ResNet blocks per level
        - Self-attention at 8×8 and 4×4 resolutions
        - Sinusoidal timestep embeddings

    Args:
        image_channels (int): Number of input/output image channels. Default: 1 (grayscale).
        base_channels (int): Base channel count. Multiplied by (1, 2, 4, 4) for each level.
                             Default: 32 → channels (32, 64, 128, 128) → ~3M params.
                             Use 64 for higher capacity: (64, 128, 256, 256) → ~26M params.
    """

    def __init__(self, image_channels=1, base_channels=32):
        super().__init__()
        self.image_channels = image_channels

        # Channel multipliers (1×, 2×, 4×, 4×) following Ho et al. 2020
        channel_level_0 = base_channels          # 32 (default)
        channel_level_1 = base_channels * 2      # 64
        channel_level_2 = base_channels * 4      # 128
        channel_level_3 = base_channels * 4      # 128

        self.model = UNet2DModel(
            sample_size=32,
            in_channels=image_channels,
            out_channels=image_channels,
            # 4 levels: 32×32 → 16×16 → 8×8 → 4×4
            block_out_channels=(
                channel_level_0,
                channel_level_1,
                channel_level_2,
                channel_level_3,
            ),
            # Encoder: plain convolutions at high res, attention at low res (8×8, 4×4)
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            # Decoder: attention at low res (4×4, 8×8), plain convolutions at high res
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            layers_per_block=2,
            norm_num_groups=32,
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
