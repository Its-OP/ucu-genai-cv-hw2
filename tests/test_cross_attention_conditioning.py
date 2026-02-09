"""
Tests for cross-attention conditioning module.

Tests cover:
    - CrossAttentionBlock: output shape, context dependency, residual connection
    - CrossAttentionConditionedUNet: output shape, conditioning, dropout, gradient flow
    - ClassifierFreeGuidanceWrapper with cross-attention: CFG formula, guidance scale
    - create_cross_attention_conditioned_unet: factory function correctness
    - UNet with cross_attention_dim: forward with context, backward compat without context
"""
import pytest
import torch

from models.unet import UNet, CrossAttentionBlock
from models.ddpm import DDPM
from models.classifier_free_guidance import (
    CrossAttentionConditionedUNet,
    ClassifierFreeGuidanceWrapper,
    create_cross_attention_conditioned_unet,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cross_attention_unet(device):
    """Create a small CrossAttentionConditionedUNet for testing."""
    return create_cross_attention_conditioned_unet(
        latent_channels=2,
        number_of_classes=10,
        context_dim=64,
        unconditional_probability=0.1,
        base_channels=32,
        channel_multipliers=(1, 2),
        layers_per_block=1,
        attention_levels=(False, True),
        norm_num_groups=32,
    ).to(device)


@pytest.fixture
def sample_latent_batch(device, seed):
    """Latent batch for conditioned UNet testing: (4, 2, 4, 4)."""
    return torch.randn(4, 2, 4, 4, device=device)


@pytest.fixture
def sample_labels(device):
    """Class labels for batch of 4: digits 0, 3, 7, 9."""
    return torch.tensor([0, 3, 7, 9], device=device)


@pytest.fixture
def sample_timesteps_ca(device):
    """Timesteps for batch of 4."""
    torch.manual_seed(42)
    return torch.randint(0, 1000, (4,), device=device)


@pytest.fixture
def sample_context(device, seed):
    """Context tokens: (4, 1, 64)."""
    return torch.randn(4, 1, 64, device=device)


# ---------------------------------------------------------------------------
#  TestCrossAttentionBlock
# ---------------------------------------------------------------------------

class TestCrossAttentionBlock:
    """Tests for the CrossAttentionBlock in unet.py."""

    def test_output_shape(self, device, seed):
        """Output shape should match input feature map shape."""
        block = CrossAttentionBlock(
            channels=64,
            context_dim=128,
            norm_num_groups=32,
            attention_head_dimension=1,
        ).to(device)

        features = torch.randn(2, 64, 4, 4, device=device)
        context = torch.randn(2, 1, 128, device=device)
        output = block(features, context)
        assert output.shape == (2, 64, 4, 4)

    def test_output_shape_longer_context_sequence(self, device, seed):
        """Output shape should be correct with longer context sequences."""
        block = CrossAttentionBlock(
            channels=64,
            context_dim=128,
            norm_num_groups=32,
            attention_head_dimension=1,
        ).to(device)

        features = torch.randn(2, 64, 4, 4, device=device)
        context = torch.randn(2, 5, 128, device=device)
        output = block(features, context)
        assert output.shape == (2, 64, 4, 4)

    def test_different_context_produces_different_output(self, device, seed):
        """Different context should produce different outputs."""
        block = CrossAttentionBlock(
            channels=64,
            context_dim=128,
            norm_num_groups=32,
            attention_head_dimension=1,
        ).to(device)
        block.eval()

        features = torch.randn(2, 64, 4, 4, device=device)
        context_a = torch.randn(2, 1, 128, device=device)
        context_b = torch.randn(2, 1, 128, device=device)

        output_a = block(features, context_a)
        output_b = block(features, context_b)
        assert not torch.allclose(output_a, output_b, atol=1e-6)

    def test_residual_connection(self, device, seed):
        """Output should include a residual connection from input."""
        block = CrossAttentionBlock(
            channels=64,
            context_dim=128,
            norm_num_groups=32,
            attention_head_dimension=1,
        ).to(device)

        # Zero out all projections so attention output is zero
        with torch.no_grad():
            block.output_projection.weight.zero_()
            block.output_projection.bias.zero_()

        features = torch.randn(2, 64, 4, 4, device=device)
        context = torch.randn(2, 1, 128, device=device)
        output = block(features, context)

        # With zero output projection, output = 0 + residual = features
        assert torch.allclose(output, features, atol=1e-5)

    def test_gradient_flow(self, device, seed):
        """Gradients should flow through the cross-attention block."""
        block = CrossAttentionBlock(
            channels=64,
            context_dim=128,
            norm_num_groups=32,
            attention_head_dimension=1,
        ).to(device)

        features = torch.randn(2, 64, 4, 4, device=device, requires_grad=True)
        context = torch.randn(2, 1, 128, device=device, requires_grad=True)

        output = block(features, context)
        loss = output.sum()
        loss.backward()

        assert features.grad is not None
        assert context.grad is not None
        assert features.grad.abs().sum() > 0
        assert context.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
#  TestCrossAttentionConditionedUNet
# ---------------------------------------------------------------------------

class TestCrossAttentionConditionedUNet:
    """Tests for the CrossAttentionConditionedUNet wrapper."""

    def test_output_shape(
        self, small_cross_attention_unet, sample_latent_batch,
        sample_labels, sample_timesteps_ca, device,
    ):
        """Output shape should be (B, latent_channels, H, W) = (4, 2, 4, 4)."""
        small_cross_attention_unet.set_class_labels(sample_labels)
        output = small_cross_attention_unet(
            sample_latent_batch, sample_timesteps_ca,
        )
        assert output.shape == (4, 2, 4, 4)

    def test_unconditional_output_shape(
        self, small_cross_attention_unet, sample_latent_batch,
        sample_timesteps_ca, device,
    ):
        """Output shape should be correct when no labels are set (unconditional)."""
        small_cross_attention_unet.set_class_labels(None)
        output = small_cross_attention_unet(
            sample_latent_batch, sample_timesteps_ca,
        )
        assert output.shape == (4, 2, 4, 4)

    def test_different_labels_produce_different_outputs(
        self, small_cross_attention_unet, sample_latent_batch,
        sample_timesteps_ca, device,
    ):
        """Different class labels should produce different noise predictions."""
        small_cross_attention_unet.eval()

        small_cross_attention_unet.set_class_labels(
            torch.tensor([0, 0, 0, 0], device=device)
        )
        output_zeros = small_cross_attention_unet(
            sample_latent_batch, sample_timesteps_ca,
        ).clone()

        small_cross_attention_unet.set_class_labels(
            torch.tensor([9, 9, 9, 9], device=device)
        )
        output_nines = small_cross_attention_unet(
            sample_latent_batch, sample_timesteps_ca,
        ).clone()

        assert not torch.allclose(output_zeros, output_nines, atol=1e-6)

    def test_unconditional_differs_from_conditional(
        self, small_cross_attention_unet, sample_latent_batch,
        sample_timesteps_ca, device,
    ):
        """Unconditional output should differ from conditional output."""
        small_cross_attention_unet.eval()

        small_cross_attention_unet.set_class_labels(
            torch.tensor([5, 5, 5, 5], device=device)
        )
        output_conditional = small_cross_attention_unet(
            sample_latent_batch, sample_timesteps_ca,
        ).clone()

        small_cross_attention_unet.set_class_labels(None)
        output_unconditional = small_cross_attention_unet(
            sample_latent_batch, sample_timesteps_ca,
        ).clone()

        assert not torch.allclose(
            output_conditional, output_unconditional, atol=1e-6,
        )

    def test_embedding_has_correct_size(self, small_cross_attention_unet):
        """Embedding should have num_classes + 1 entries of size context_dim."""
        embedding = small_cross_attention_unet.class_embedding
        # 10 classes + 1 unconditional = 11 entries, each of size 64
        assert embedding.num_embeddings == 11
        assert embedding.embedding_dim == 64

    def test_no_extra_input_channel(self, small_cross_attention_unet):
        """Cross-attention UNet should NOT have extra input channels."""
        # Unlike channel-concat which adds +1 input channel,
        # cross-attention keeps input_channels == output_channels == latent_channels
        unet = small_cross_attention_unet.unet
        assert unet.image_channels == 2  # latent_channels, no +1
        assert unet.output_channels == 2

    def test_eval_mode_no_dropout(
        self, small_cross_attention_unet, sample_latent_batch,
        sample_labels, sample_timesteps_ca, device, seed,
    ):
        """In eval mode, outputs should be deterministic (no conditioning dropout)."""
        small_cross_attention_unet.eval()
        small_cross_attention_unet.set_class_labels(sample_labels)

        torch.manual_seed(42)
        output_1 = small_cross_attention_unet(
            sample_latent_batch, sample_timesteps_ca,
        ).clone()

        torch.manual_seed(42)
        output_2 = small_cross_attention_unet(
            sample_latent_batch, sample_timesteps_ca,
        ).clone()

        assert torch.allclose(output_1, output_2)

    def test_training_mode_has_dropout_effect(self, device, seed):
        """In training mode, unconditional dropout should cause variation."""
        conditioned_unet = create_cross_attention_conditioned_unet(
            latent_channels=2,
            number_of_classes=10,
            context_dim=64,
            unconditional_probability=0.5,  # 50% dropout for strong effect
            base_channels=32,
            channel_multipliers=(1, 2),
            layers_per_block=1,
            attention_levels=(False, True),
            norm_num_groups=32,
        ).to(device)

        conditioned_unet.train()
        latent = torch.randn(8, 2, 4, 4, device=device)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
        timestep = torch.randint(0, 1000, (8,), device=device)

        conditioned_unet.set_class_labels(labels)
        output_1 = conditioned_unet(latent, timestep).clone()

        conditioned_unet.set_class_labels(labels)
        output_2 = conditioned_unet(latent, timestep).clone()

        # With 50% dropout on 8 samples, the two passes should almost certainly
        # produce different outputs due to different dropout masks
        assert not torch.allclose(output_1, output_2)

    def test_gradient_flow(
        self, small_cross_attention_unet, sample_latent_batch,
        sample_labels, sample_timesteps_ca, device,
    ):
        """Gradients should flow through the wrapper to all parameters."""
        small_cross_attention_unet.train()
        small_cross_attention_unet.set_class_labels(sample_labels)

        output = small_cross_attention_unet(
            sample_latent_batch, sample_timesteps_ca,
        )
        loss = output.sum()
        loss.backward()

        # Check that UNet parameters receive gradients
        unet_has_grads = any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in small_cross_attention_unet.unet.parameters()
        )
        assert unet_has_grads

        # Check that embedding receives gradients
        embedding_grad = small_cross_attention_unet.class_embedding.weight.grad
        assert embedding_grad is not None
        assert embedding_grad.abs().sum() > 0

    def test_compatible_with_ddpm_p_losses(
        self, small_cross_attention_unet, sample_latent_batch,
        sample_labels, device,
    ):
        """The wrapper should work with DDPM.p_losses without modification."""
        ddpm = DDPM(timesteps=100).to(device)
        small_cross_attention_unet.set_class_labels(sample_labels)

        timestep = torch.randint(
            0, ddpm.timesteps, (sample_latent_batch.shape[0],), device=device,
        )
        loss = ddpm.p_losses(
            small_cross_attention_unet, sample_latent_batch, timestep,
        )

        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # MSE is non-negative
        assert loss.requires_grad


# ---------------------------------------------------------------------------
#  TestCFGWithCrossAttention
# ---------------------------------------------------------------------------

class TestCFGWithCrossAttention:
    """Tests for ClassifierFreeGuidanceWrapper with cross-attention model."""

    def test_output_shape(
        self, small_cross_attention_unet, sample_latent_batch,
        sample_labels, sample_timesteps_ca, device,
    ):
        """Output shape should match input latent shape."""
        small_cross_attention_unet.eval()
        small_cross_attention_unet.set_class_labels(sample_labels)
        cfg_wrapper = ClassifierFreeGuidanceWrapper(
            small_cross_attention_unet, guidance_scale=3.0,
        )
        output = cfg_wrapper(sample_latent_batch, sample_timesteps_ca)
        assert output.shape == (4, 2, 4, 4)

    def test_guidance_scale_zero_equals_unconditional(
        self, small_cross_attention_unet, sample_latent_batch,
        sample_labels, sample_timesteps_ca, device,
    ):
        """With guidance_scale=0, output should equal unconditional prediction."""
        small_cross_attention_unet.eval()

        # Get unconditional prediction directly
        small_cross_attention_unet.set_class_labels(None)
        unconditional_output = small_cross_attention_unet(
            sample_latent_batch, sample_timesteps_ca,
        ).clone()

        # Get CFG output with scale=0
        small_cross_attention_unet.set_class_labels(sample_labels)
        cfg_wrapper = ClassifierFreeGuidanceWrapper(
            small_cross_attention_unet, guidance_scale=0.0,
        )
        cfg_output = cfg_wrapper(sample_latent_batch, sample_timesteps_ca)

        # ε_guided = ε_uncond + 0 * (ε_cond - ε_uncond) = ε_uncond
        assert torch.allclose(cfg_output, unconditional_output, atol=1e-5)

    def test_guidance_scale_one_equals_conditional(
        self, small_cross_attention_unet, sample_latent_batch,
        sample_labels, sample_timesteps_ca, device,
    ):
        """With guidance_scale=1, output should equal conditional prediction."""
        small_cross_attention_unet.eval()

        # Get conditional prediction directly
        small_cross_attention_unet.set_class_labels(sample_labels)
        conditional_output = small_cross_attention_unet(
            sample_latent_batch, sample_timesteps_ca,
        ).clone()

        # Get CFG output with scale=1
        small_cross_attention_unet.set_class_labels(sample_labels)
        cfg_wrapper = ClassifierFreeGuidanceWrapper(
            small_cross_attention_unet, guidance_scale=1.0,
        )
        cfg_output = cfg_wrapper(sample_latent_batch, sample_timesteps_ca)

        # ε_guided = ε_uncond + 1 * (ε_cond - ε_uncond) = ε_cond
        assert torch.allclose(cfg_output, conditional_output, atol=1e-5)

    def test_different_guidance_scales_produce_different_outputs(
        self, small_cross_attention_unet, sample_latent_batch,
        sample_labels, sample_timesteps_ca, device,
    ):
        """Different guidance scales should produce different outputs."""
        small_cross_attention_unet.eval()
        small_cross_attention_unet.set_class_labels(sample_labels)

        cfg_low = ClassifierFreeGuidanceWrapper(
            small_cross_attention_unet, guidance_scale=1.0,
        )
        output_low = cfg_low(
            sample_latent_batch, sample_timesteps_ca,
        ).clone()

        cfg_high = ClassifierFreeGuidanceWrapper(
            small_cross_attention_unet, guidance_scale=5.0,
        )
        output_high = cfg_high(
            sample_latent_batch, sample_timesteps_ca,
        ).clone()

        assert not torch.allclose(output_low, output_high, atol=1e-6)


# ---------------------------------------------------------------------------
#  TestCreateCrossAttentionConditionedUNet
# ---------------------------------------------------------------------------

class TestCreateCrossAttentionConditionedUNet:
    """Tests for the create_cross_attention_conditioned_unet factory function."""

    def test_factory_creates_correct_model(self, device):
        """Factory should create a CrossAttentionConditionedUNet with correct config."""
        conditioned_unet = create_cross_attention_conditioned_unet(
            latent_channels=2,
            number_of_classes=10,
            context_dim=64,
            base_channels=32,
            channel_multipliers=(1, 2),
            layers_per_block=1,
            attention_levels=(False, True),
        ).to(device)

        assert isinstance(conditioned_unet, CrossAttentionConditionedUNet)
        assert conditioned_unet.number_of_classes == 10
        assert conditioned_unet.context_dim == 64
        # Input and output channels should both be latent_channels (no +1)
        assert conditioned_unet.unet.image_channels == 2
        assert conditioned_unet.unet.output_channels == 2

    def test_factory_different_context_dim(self, device):
        """Factory should handle different context dimensions."""
        conditioned_unet = create_cross_attention_conditioned_unet(
            latent_channels=2,
            number_of_classes=10,
            context_dim=256,
            base_channels=32,
            channel_multipliers=(1, 2),
            layers_per_block=1,
            attention_levels=(False, True),
        ).to(device)

        assert conditioned_unet.context_dim == 256
        assert conditioned_unet.class_embedding.embedding_dim == 256

        latent = torch.randn(2, 2, 4, 4, device=device)
        labels = torch.tensor([0, 5], device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)

        conditioned_unet.set_class_labels(labels)
        output = conditioned_unet(latent, timestep)
        assert output.shape == (2, 2, 4, 4)

    def test_factory_different_number_of_classes(self, device):
        """Factory should handle different class counts."""
        conditioned_unet = create_cross_attention_conditioned_unet(
            latent_channels=2,
            number_of_classes=5,
            context_dim=64,
            base_channels=32,
            channel_multipliers=(1, 2),
            layers_per_block=1,
            attention_levels=(False, True),
        ).to(device)

        assert conditioned_unet.number_of_classes == 5
        # 5 classes + 1 unconditional = 6 entries
        assert conditioned_unet.class_embedding.num_embeddings == 6


# ---------------------------------------------------------------------------
#  TestUNetWithCrossAttention
# ---------------------------------------------------------------------------

class TestUNetWithCrossAttention:
    """Tests for UNet forward pass with cross-attention context."""

    def test_forward_with_context(self, device, seed):
        """UNet should accept and use cross-attention context."""
        unet = UNet(
            image_channels=2,
            output_channels=2,
            base_channels=32,
            channel_multipliers=(1, 2),
            layers_per_block=1,
            attention_levels=(False, True),
            norm_num_groups=32,
            cross_attention_dim=64,
        ).to(device)

        x = torch.randn(2, 2, 4, 4, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)
        context = torch.randn(2, 1, 64, device=device)

        output = unet(x, timestep, context=context)
        assert output.shape == (2, 2, 4, 4)

    def test_forward_without_context_backward_compatible(self, device, seed):
        """UNet without cross_attention_dim should work without context."""
        unet = UNet(
            image_channels=2,
            output_channels=2,
            base_channels=32,
            channel_multipliers=(1, 2),
            layers_per_block=1,
            attention_levels=(False, True),
            norm_num_groups=32,
        ).to(device)

        x = torch.randn(2, 2, 4, 4, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)

        output = unet(x, timestep)
        assert output.shape == (2, 2, 4, 4)

    def test_context_affects_output(self, device, seed):
        """Different contexts should produce different outputs."""
        unet = UNet(
            image_channels=2,
            output_channels=2,
            base_channels=32,
            channel_multipliers=(1, 2),
            layers_per_block=1,
            attention_levels=(False, True),
            norm_num_groups=32,
            cross_attention_dim=64,
        ).to(device)
        unet.eval()

        x = torch.randn(2, 2, 4, 4, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)
        context_a = torch.randn(2, 1, 64, device=device)
        context_b = torch.randn(2, 1, 64, device=device)

        output_a = unet(x, timestep, context=context_a)
        output_b = unet(x, timestep, context=context_b)
        assert not torch.allclose(output_a, output_b, atol=1e-6)

    def test_mnist_backward_compatible(self, device, seed):
        """Existing MNIST usage (no cross-attention) should still work."""
        unet = UNet(
            image_channels=1,
            base_channels=32,
            channel_multipliers=(1, 2, 3, 3),
            layers_per_block=1,
            attention_levels=(False, False, False, True),
            norm_num_groups=32,
        ).to(device)

        x = torch.randn(2, 1, 28, 28, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)
        output = unet(x, timestep)
        assert output.shape == (2, 1, 28, 28)
