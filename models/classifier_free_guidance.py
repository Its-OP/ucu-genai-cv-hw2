"""
Classifier-Free Guidance (CFG) for class-conditioned latent diffusion.

Implements two conditioning approaches for Ho & Salimans 2022:
    "Classifier-Free Diffusion Guidance" (arXiv:2207.12598)

1. Channel concatenation (ClassConditionedUNet):
   The class label is encoded via a learnable embedding into a single-channel
   spatial map and concatenated to the noisy latent before the UNet processes it.

2. Cross-attention (CrossAttentionConditionedUNet):
   The class label is embedded as a dense vector and injected into the UNet
   via cross-attention layers (Rombach et al. 2022, "Latent Diffusion Models").
   Queries come from UNet features, keys/values from the class embedding.

During sampling, the CFG formula steers generation toward the target class:

    ε_guided = ε_uncond + w × (ε_cond − ε_uncond)

where w is the guidance scale.

Components:
    1. ClassConditionedUNet  — training-time wrapper using channel concatenation
    2. CrossAttentionConditionedUNet — training-time wrapper using cross-attention
    3. ClassifierFreeGuidanceWrapper — sampling-time wrapper that performs
       dual forward passes (conditional + unconditional) and applies CFG
    4. create_conditioned_unet — factory for channel-concat conditioned UNet
    5. create_cross_attention_conditioned_unet — factory for cross-attention conditioned UNet
"""
import torch
import torch.nn as nn

from models.unet import UNet


class ClassConditionedUNet(nn.Module):
    """
    Wrapper that adds class conditioning to a UNet via input channel concatenation.

    Encodes each class label as a learnable spatial pattern (1 × H × W) and
    concatenates it to the noisy latent along the channel dimension before
    forwarding through the UNet. Conforms to the ``model(x, t)`` interface
    expected by DDPM and DDIM samplers, so no changes are needed in those modules.

    The embedding table has ``number_of_classes + 1`` entries:
        - Index 0: unconditional token (used during classifier-free dropout
          and for unconditional generation)
        - Indices 1 .. number_of_classes: class tokens (digit 0 → index 1, etc.)

    During training, a fraction of samples in each batch have their class
    conditioning replaced with the unconditional token (index 0). This trains
    the model to predict noise both conditionally and unconditionally, which
    is required for classifier-free guidance at sampling time.

    Args:
        unet: UNet model with ``image_channels = latent_channels + 1`` input
              channels and ``output_channels = latent_channels``.
        number_of_classes: Number of distinct classes (default: 10 for MNIST digits).
        spatial_height: Height of the latent spatial grid (default: 4).
        spatial_width: Width of the latent spatial grid (default: 4).
        unconditional_probability: Probability of dropping class conditioning
            per sample during training (default: 0.1).
    """

    def __init__(
        self,
        unet: UNet,
        number_of_classes: int = 10,
        spatial_height: int = 4,
        spatial_width: int = 4,
        unconditional_probability: float = 0.1,
    ):
        super().__init__()
        self.unet = unet
        self.number_of_classes = number_of_classes
        self.spatial_height = spatial_height
        self.spatial_width = spatial_width
        self.unconditional_probability = unconditional_probability

        # Learnable class embedding: each class (and the unconditional token)
        # maps to a full H × W spatial pattern.
        # Index 0 = unconditional, indices 1..number_of_classes = class tokens
        self.class_embedding = nn.Embedding(
            num_embeddings=number_of_classes + 1,
            embedding_dim=spatial_height * spatial_width,
        )

        # Internal state: class labels for the current batch, set externally
        # before each forward call via set_class_labels()
        self._class_labels = None

    def set_class_labels(self, labels: torch.Tensor):
        """
        Store class labels for the next forward pass.

        Args:
            labels: Integer class labels, shape (batch_size,), values in
                    [0, number_of_classes). Set to None for fully unconditional.
        """
        self._class_labels = labels

    def _build_conditioning_map(
        self, batch_size: int, device: torch.device,
    ) -> torch.Tensor:
        """
        Build the class conditioning channel to concatenate with the noisy latent.

        If class labels are set, looks up the corresponding embedding and
        optionally applies unconditional dropout during training. If no labels
        are set, returns the unconditional embedding for all samples.

        Classifier-free dropout (Ho & Salimans 2022):
            During training, each sample in the batch independently has its
            class conditioning replaced with the unconditional token (index 0)
            with probability ``unconditional_probability``. This teaches the
            model to predict noise both with and without class information.

        Args:
            batch_size: Number of samples in the batch.
            device: Torch device for tensor creation.

        Returns:
            Conditioning map, shape (batch_size, 1, spatial_height, spatial_width).
        """
        if self._class_labels is not None:
            # Shift labels by +1: digit 0 → index 1, ..., digit 9 → index 10
            embedding_indices = self._class_labels + 1

            # Classifier-free dropout: randomly replace some samples with
            # the unconditional token (index 0) during training
            if self.training and self.unconditional_probability > 0:
                drop_mask = (
                    torch.rand(batch_size, device=device)
                    < self.unconditional_probability
                )
                embedding_indices = embedding_indices.masked_fill(drop_mask, 0)
        else:
            # No labels → fully unconditional (all index 0)
            embedding_indices = torch.zeros(
                batch_size, dtype=torch.long, device=device,
            )

        # Look up embedding: (batch_size,) → (batch_size, H * W)
        conditioning = self.class_embedding(embedding_indices)

        # Reshape to spatial map: (batch_size, H * W) → (batch_size, 1, H, W)
        conditioning_map = conditioning.view(
            batch_size, 1, self.spatial_height, self.spatial_width,
        )

        return conditioning_map

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Concatenate class conditioning to input, then forward through UNet.

        Conforms to the ``model(x, t)`` interface used by DDPM.p_losses,
        DDPM.p_sample, and DDIMSampler.ddim_sample.

        Args:
            x: Noisy latent tensor, shape (batch_size, latent_channels, H, W).
            timestep: Diffusion timesteps, shape (batch_size,).

        Returns:
            Predicted noise, shape (batch_size, latent_channels, H, W).
        """
        batch_size = x.shape[0]
        conditioning_map = self._build_conditioning_map(batch_size, x.device)

        # Concatenate conditioning channel: (B, C, H, W) + (B, 1, H, W) → (B, C+1, H, W)
        x_conditioned = torch.cat([x, conditioning_map], dim=1)

        return self.unet(x_conditioned, timestep)


class ClassifierFreeGuidanceWrapper(nn.Module):
    """
    Sampling-time wrapper that applies classifier-free guidance.

    Performs two forward passes per denoising step — one conditional and one
    unconditional — and combines them using the CFG formula:

        ε_guided = ε_uncond + guidance_scale × (ε_cond − ε_uncond)

    Conforms to the ``model(x, t)`` interface so it can be passed directly
    to ``DDPM.p_sample_loop()`` or ``DDIMSampler.ddim_sample_loop()`` without
    any changes to those modules.

    Args:
        conditioned_unet: A ClassConditionedUNet with class labels already set.
        guidance_scale: CFG weight (w). Higher values produce stronger class
            adherence at the cost of diversity. Typical values: 1.0–7.0.
            w=1.0 is equivalent to standard conditional sampling (no guidance).
            w=0.0 is equivalent to unconditional sampling.
    """

    def __init__(
        self,
        conditioned_unet: ClassConditionedUNet,
        guidance_scale: float = 3.0,
    ):
        super().__init__()
        self.conditioned_unet = conditioned_unet
        self.guidance_scale = guidance_scale

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Apply classifier-free guidance.

        Formula (Ho & Salimans 2022):
            ε_guided = ε_uncond + w × (ε_cond − ε_uncond)

        Two forward passes are performed:
            1. Conditional: uses the class labels stored in conditioned_unet
            2. Unconditional: temporarily sets labels to None (index 0)

        Args:
            x: Noisy latent tensor, shape (batch_size, latent_channels, H, W).
            timestep: Diffusion timesteps, shape (batch_size,).

        Returns:
            Guided noise prediction, shape (batch_size, latent_channels, H, W).
        """
        # Save the real labels for restoration after both passes
        real_labels = self.conditioned_unet._class_labels

        # --- Conditional forward pass ---
        # Disable training-time dropout for sampling: we want the pure
        # conditional prediction, not a randomly-dropped version
        original_probability = self.conditioned_unet.unconditional_probability
        self.conditioned_unet.unconditional_probability = 0.0
        self.conditioned_unet.set_class_labels(real_labels)
        noise_prediction_conditional = self.conditioned_unet(x, timestep)

        # --- Unconditional forward pass ---
        self.conditioned_unet.set_class_labels(None)
        noise_prediction_unconditional = self.conditioned_unet(x, timestep)

        # --- Restore state ---
        self.conditioned_unet.set_class_labels(real_labels)
        self.conditioned_unet.unconditional_probability = original_probability

        # --- CFG formula ---
        # ε_guided = ε_uncond + w × (ε_cond − ε_uncond)
        guided_prediction = (
            noise_prediction_unconditional
            + self.guidance_scale * (
                noise_prediction_conditional - noise_prediction_unconditional
            )
        )

        return guided_prediction


def create_conditioned_unet(
    latent_channels: int,
    number_of_classes: int = 10,
    spatial_height: int = 4,
    spatial_width: int = 4,
    unconditional_probability: float = 0.1,
    base_channels: int = 64,
    channel_multipliers: tuple = (1, 2),
    layers_per_block: int = 1,
    attention_levels: tuple = (False, True),
    norm_num_groups: int = 32,
) -> ClassConditionedUNet:
    """
    Factory function to create a class-conditioned UNet for latent diffusion.

    Constructs a UNet with ``latent_channels + 1`` input channels (the extra
    channel carries the class conditioning map) and ``latent_channels`` output
    channels (noise prediction in latent space only), then wraps it in a
    ClassConditionedUNet.

    Args:
        latent_channels: Number of VAE latent channels (typically 2 for MNIST).
        number_of_classes: Number of distinct classes (default: 10).
        spatial_height: Latent spatial height (default: 4).
        spatial_width: Latent spatial width (default: 4).
        unconditional_probability: Classifier-free dropout rate (default: 0.1).
        base_channels: UNet base channel count (default: 64).
        channel_multipliers: UNet per-level channel multipliers (default: (1, 2)).
        layers_per_block: ResNet blocks per UNet level (default: 1).
        attention_levels: Attention flags per UNet level (default: (False, True)).
        norm_num_groups: GroupNorm groups (default: 32).

    Returns:
        ClassConditionedUNet ready for training.
    """
    unet = UNet(
        image_channels=latent_channels + 1,
        output_channels=latent_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        layers_per_block=layers_per_block,
        attention_levels=attention_levels,
        norm_num_groups=norm_num_groups,
    )

    return ClassConditionedUNet(
        unet=unet,
        number_of_classes=number_of_classes,
        spatial_height=spatial_height,
        spatial_width=spatial_width,
        unconditional_probability=unconditional_probability,
    )


class CrossAttentionConditionedUNet(nn.Module):
    """
    Wrapper that adds class conditioning to a UNet via cross-attention.

    Encodes each class label as a dense embedding vector using nn.Embedding,
    then passes it as a single-token context sequence to the UNet's
    cross-attention layers. This follows the conditioning approach from
    Rombach et al. 2022 ("Latent Diffusion Models"), where:

        Q = W_q · features,  K = W_k · context,  V = W_v · context
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    The embedding table has ``number_of_classes + 1`` entries:
        - Index 0: unconditional token (used during classifier-free dropout
          and for unconditional generation)
        - Indices 1 .. number_of_classes: class tokens (digit 0 -> index 1, etc.)

    Unlike ClassConditionedUNet (channel concatenation), this approach does NOT
    add extra input channels. The UNet's input and output channel counts both
    equal latent_channels. Conditioning enters via cross-attention inside the
    network, not through the input tensor.

    Conforms to the ``model(x, t)`` interface expected by DDPM and DDIM samplers.

    Args:
        unet: UNet model with ``cross_attention_dim`` set to match ``context_dim``.
        number_of_classes: Number of distinct classes (default: 10 for MNIST digits).
        context_dim: Dimension of the dense class embedding vectors.
        unconditional_probability: Probability of dropping class conditioning
            per sample during training (default: 0.1).
    """

    def __init__(
        self,
        unet: UNet,
        number_of_classes: int = 10,
        context_dim: int = 128,
        unconditional_probability: float = 0.1,
    ):
        super().__init__()
        self.unet = unet
        self.number_of_classes = number_of_classes
        self.context_dim = context_dim
        self.unconditional_probability = unconditional_probability

        # Learnable class embedding: each class (and the unconditional token)
        # maps to a dense vector of dimension context_dim.
        # Index 0 = unconditional, indices 1..number_of_classes = class tokens
        self.class_embedding = nn.Embedding(
            num_embeddings=number_of_classes + 1,
            embedding_dim=context_dim,
        )

        # Internal state: class labels for the current batch, set externally
        # before each forward call via set_class_labels()
        self._class_labels = None

    def set_class_labels(self, labels: torch.Tensor):
        """
        Store class labels for the next forward pass.

        Args:
            labels: Integer class labels, shape (batch_size,), values in
                    [0, number_of_classes). Set to None for fully unconditional.
        """
        self._class_labels = labels

    def _build_context_tokens(
        self, batch_size: int, device: torch.device,
    ) -> torch.Tensor:
        """
        Build the context token sequence for cross-attention conditioning.

        If class labels are set, looks up the corresponding embedding and
        optionally applies unconditional dropout during training. If no labels
        are set, returns the unconditional embedding for all samples.

        Classifier-free dropout (Ho & Salimans 2022):
            During training, each sample in the batch independently has its
            class conditioning replaced with the unconditional token (index 0)
            with probability ``unconditional_probability``. This teaches the
            model to predict noise both with and without class information.

        Args:
            batch_size: Number of samples in the batch.
            device: Torch device for tensor creation.

        Returns:
            Context tokens, shape (batch_size, 1, context_dim).
            The sequence length is 1 because each class label produces
            a single embedding vector.
        """
        if self._class_labels is not None:
            # Shift labels by +1: digit 0 -> index 1, ..., digit 9 -> index 10
            embedding_indices = self._class_labels + 1

            # Classifier-free dropout: randomly replace some samples with
            # the unconditional token (index 0) during training
            if self.training and self.unconditional_probability > 0:
                drop_mask = (
                    torch.rand(batch_size, device=device)
                    < self.unconditional_probability
                )
                embedding_indices = embedding_indices.masked_fill(drop_mask, 0)
        else:
            # No labels -> fully unconditional (all index 0)
            embedding_indices = torch.zeros(
                batch_size, dtype=torch.long, device=device,
            )

        # Look up embedding: (batch_size,) -> (batch_size, context_dim)
        context_vectors = self.class_embedding(embedding_indices)

        # Reshape to token sequence: (batch_size, context_dim) -> (batch_size, 1, context_dim)
        context_tokens = context_vectors.unsqueeze(1)

        return context_tokens

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Build cross-attention context from class labels, then forward through UNet.

        Conforms to the ``model(x, t)`` interface used by DDPM.p_losses,
        DDPM.p_sample, and DDIMSampler.ddim_sample.

        Args:
            x: Noisy latent tensor, shape (batch_size, latent_channels, H, W).
            timestep: Diffusion timesteps, shape (batch_size,).

        Returns:
            Predicted noise, shape (batch_size, latent_channels, H, W).
        """
        batch_size = x.shape[0]
        context_tokens = self._build_context_tokens(batch_size, x.device)

        return self.unet(x, timestep, context=context_tokens)


def create_cross_attention_conditioned_unet(
    latent_channels: int,
    number_of_classes: int = 10,
    context_dim: int = 128,
    unconditional_probability: float = 0.1,
    base_channels: int = 64,
    channel_multipliers: tuple = (1, 2),
    layers_per_block: int = 1,
    attention_levels: tuple = (False, True),
    norm_num_groups: int = 32,
) -> CrossAttentionConditionedUNet:
    """
    Factory function to create a cross-attention class-conditioned UNet for latent diffusion.

    Constructs a UNet with ``cross_attention_dim=context_dim`` so that
    cross-attention blocks are created at levels with attention enabled.
    The UNet has ``latent_channels`` for both input and output (no extra
    conditioning channel, unlike the channel-concat approach).

    Wraps the UNet in a CrossAttentionConditionedUNet that manages class
    label embedding and classifier-free dropout.

    Args:
        latent_channels: Number of VAE latent channels (typically 2 for MNIST).
        number_of_classes: Number of distinct classes (default: 10).
        context_dim: Dimension of the dense class embedding (default: 128).
        unconditional_probability: Classifier-free dropout rate (default: 0.1).
        base_channels: UNet base channel count (default: 64).
        channel_multipliers: UNet per-level channel multipliers (default: (1, 2)).
        layers_per_block: ResNet blocks per UNet level (default: 1).
        attention_levels: Attention flags per UNet level (default: (False, True)).
        norm_num_groups: GroupNorm groups (default: 32).

    Returns:
        CrossAttentionConditionedUNet ready for training.
    """
    unet = UNet(
        image_channels=latent_channels,
        output_channels=latent_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        layers_per_block=layers_per_block,
        attention_levels=attention_levels,
        norm_num_groups=norm_num_groups,
        cross_attention_dim=context_dim,
    )

    return CrossAttentionConditionedUNet(
        unet=unet,
        number_of_classes=number_of_classes,
        context_dim=context_dim,
        unconditional_probability=unconditional_probability,
    )
