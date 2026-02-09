"""
Tests for classifier-free guidance (CFG) conditioning module.

Tests cover:
    - ClassConditionedUNet: output shape, conditioning, dropout, gradient flow
    - ClassifierFreeGuidanceWrapper: CFG formula, guidance scale behavior
    - create_conditioned_unet: factory function correctness
    - UNet output_channels: separate input/output channel support
"""
import pytest
import torch

from models.unet import UNet
from models.ddpm import DDPM
from models.classifier_free_guidance import (
    ClassConditionedUNet,
    ClassifierFreeGuidanceWrapper,
    create_conditioned_unet,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_conditioned_unet(device):
    """Create a small ClassConditionedUNet for testing (latent_channels=2, 10 classes)."""
    return create_conditioned_unet(
        latent_channels=2,
        number_of_classes=10,
        spatial_height=4,
        spatial_width=4,
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
def sample_timesteps_cfg(device):
    """Timesteps for batch of 4."""
    torch.manual_seed(42)
    return torch.randint(0, 1000, (4,), device=device)


# ---------------------------------------------------------------------------
#  TestClassConditionedUNet
# ---------------------------------------------------------------------------

class TestClassConditionedUNet:
    """Tests for the ClassConditionedUNet wrapper."""

    def test_output_shape(
        self, small_conditioned_unet, sample_latent_batch,
        sample_labels, sample_timesteps_cfg, device,
    ):
        """Output shape should be (B, latent_channels, H, W) = (4, 2, 4, 4)."""
        small_conditioned_unet.set_class_labels(sample_labels)
        output = small_conditioned_unet(sample_latent_batch, sample_timesteps_cfg)
        assert output.shape == (4, 2, 4, 4)

    def test_unconditional_output_shape(
        self, small_conditioned_unet, sample_latent_batch,
        sample_timesteps_cfg, device,
    ):
        """Output shape should be correct when no labels are set (unconditional)."""
        small_conditioned_unet.set_class_labels(None)
        output = small_conditioned_unet(sample_latent_batch, sample_timesteps_cfg)
        assert output.shape == (4, 2, 4, 4)

    def test_different_labels_produce_different_outputs(
        self, small_conditioned_unet, sample_latent_batch,
        sample_timesteps_cfg, device,
    ):
        """Different class labels should produce different noise predictions."""
        small_conditioned_unet.eval()

        small_conditioned_unet.set_class_labels(
            torch.tensor([0, 0, 0, 0], device=device)
        )
        output_zeros = small_conditioned_unet(
            sample_latent_batch, sample_timesteps_cfg,
        ).clone()

        small_conditioned_unet.set_class_labels(
            torch.tensor([9, 9, 9, 9], device=device)
        )
        output_nines = small_conditioned_unet(
            sample_latent_batch, sample_timesteps_cfg,
        ).clone()

        assert not torch.allclose(output_zeros, output_nines, atol=1e-6)

    def test_unconditional_differs_from_conditional(
        self, small_conditioned_unet, sample_latent_batch,
        sample_timesteps_cfg, device,
    ):
        """Unconditional output should differ from conditional output."""
        small_conditioned_unet.eval()

        small_conditioned_unet.set_class_labels(
            torch.tensor([5, 5, 5, 5], device=device)
        )
        output_conditional = small_conditioned_unet(
            sample_latent_batch, sample_timesteps_cfg,
        ).clone()

        small_conditioned_unet.set_class_labels(None)
        output_unconditional = small_conditioned_unet(
            sample_latent_batch, sample_timesteps_cfg,
        ).clone()

        assert not torch.allclose(
            output_conditional, output_unconditional, atol=1e-6,
        )

    def test_embedding_has_correct_size(self, small_conditioned_unet):
        """Embedding should have num_classes + 1 entries of size H * W."""
        embedding = small_conditioned_unet.class_embedding
        # 10 classes + 1 unconditional = 11 entries, each of size 4*4 = 16
        assert embedding.num_embeddings == 11
        assert embedding.embedding_dim == 16

    def test_eval_mode_no_dropout(
        self, small_conditioned_unet, sample_latent_batch,
        sample_labels, sample_timesteps_cfg, device, seed,
    ):
        """In eval mode, outputs should be deterministic (no conditioning dropout)."""
        small_conditioned_unet.eval()
        small_conditioned_unet.set_class_labels(sample_labels)

        torch.manual_seed(42)
        output_1 = small_conditioned_unet(
            sample_latent_batch, sample_timesteps_cfg,
        ).clone()

        torch.manual_seed(42)
        output_2 = small_conditioned_unet(
            sample_latent_batch, sample_timesteps_cfg,
        ).clone()

        assert torch.allclose(output_1, output_2)

    def test_training_mode_has_dropout_effect(self, device, seed):
        """
        In training mode, unconditional dropout should cause variation.

        Over many forward passes, the outputs should not be identical because
        different samples get their conditioning randomly dropped.
        """
        conditioned_unet = create_conditioned_unet(
            latent_channels=2,
            number_of_classes=10,
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
        self, small_conditioned_unet, sample_latent_batch,
        sample_labels, sample_timesteps_cfg, device,
    ):
        """Gradients should flow through the wrapper to all parameters."""
        small_conditioned_unet.train()
        small_conditioned_unet.set_class_labels(sample_labels)

        output = small_conditioned_unet(sample_latent_batch, sample_timesteps_cfg)
        loss = output.sum()
        loss.backward()

        # Check that UNet parameters receive gradients
        unet_has_grads = any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in small_conditioned_unet.unet.parameters()
        )
        assert unet_has_grads

        # Check that embedding receives gradients
        embedding_grad = small_conditioned_unet.class_embedding.weight.grad
        assert embedding_grad is not None
        assert embedding_grad.abs().sum() > 0

    def test_compatible_with_ddpm_p_losses(
        self, small_conditioned_unet, sample_latent_batch,
        sample_labels, device,
    ):
        """The wrapper should work with DDPM.p_losses without modification."""
        ddpm = DDPM(timesteps=100).to(device)
        small_conditioned_unet.set_class_labels(sample_labels)

        timestep = torch.randint(
            0, ddpm.timesteps, (sample_latent_batch.shape[0],), device=device,
        )
        loss = ddpm.p_losses(small_conditioned_unet, sample_latent_batch, timestep)

        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # MSE is non-negative
        assert loss.requires_grad


# ---------------------------------------------------------------------------
#  TestClassifierFreeGuidanceWrapper
# ---------------------------------------------------------------------------

class TestClassifierFreeGuidanceWrapper:
    """Tests for the CFG sampling wrapper."""

    def test_output_shape(
        self, small_conditioned_unet, sample_latent_batch,
        sample_labels, sample_timesteps_cfg, device,
    ):
        """Output shape should match input latent shape."""
        small_conditioned_unet.eval()
        small_conditioned_unet.set_class_labels(sample_labels)
        cfg_wrapper = ClassifierFreeGuidanceWrapper(
            small_conditioned_unet, guidance_scale=3.0,
        )
        output = cfg_wrapper(sample_latent_batch, sample_timesteps_cfg)
        assert output.shape == (4, 2, 4, 4)

    def test_guidance_scale_zero_equals_unconditional(
        self, small_conditioned_unet, sample_latent_batch,
        sample_labels, sample_timesteps_cfg, device,
    ):
        """With guidance_scale=0 and no rescaling, output equals unconditional."""
        small_conditioned_unet.eval()
        small_conditioned_unet.set_class_labels(sample_labels)

        # Get unconditional prediction directly
        small_conditioned_unet.set_class_labels(None)
        unconditional_output = small_conditioned_unet(
            sample_latent_batch, sample_timesteps_cfg,
        ).clone()

        # Get CFG output with scale=0, rescale_cfg=False to test raw formula
        small_conditioned_unet.set_class_labels(sample_labels)
        cfg_wrapper = ClassifierFreeGuidanceWrapper(
            small_conditioned_unet, guidance_scale=0.0, rescale_cfg=False,
        )
        cfg_output = cfg_wrapper(sample_latent_batch, sample_timesteps_cfg)

        # ε_guided = ε_uncond + 0 * (ε_cond - ε_uncond) = ε_uncond
        assert torch.allclose(cfg_output, unconditional_output, atol=1e-5)

    def test_guidance_scale_one_equals_conditional(
        self, small_conditioned_unet, sample_latent_batch,
        sample_labels, sample_timesteps_cfg, device,
    ):
        """With guidance_scale=1, output equals conditional prediction.

        Holds for both rescaled and unrescaled CFG, because at w=1:
            ε_guided = ε_uncond + 1 × (ε_cond − ε_uncond) = ε_cond
        and rescaling ε_cond to match std(ε_cond) is a no-op.
        """
        small_conditioned_unet.eval()

        # Get conditional prediction directly (no dropout in eval mode)
        small_conditioned_unet.set_class_labels(sample_labels)
        conditional_output = small_conditioned_unet(
            sample_latent_batch, sample_timesteps_cfg,
        ).clone()

        # Get CFG output with scale=1 (rescale_cfg=True by default — still works)
        small_conditioned_unet.set_class_labels(sample_labels)
        cfg_wrapper = ClassifierFreeGuidanceWrapper(
            small_conditioned_unet, guidance_scale=1.0,
        )
        cfg_output = cfg_wrapper(sample_latent_batch, sample_timesteps_cfg)

        # ε_guided = ε_uncond + 1 * (ε_cond - ε_uncond) = ε_cond
        assert torch.allclose(cfg_output, conditional_output, atol=1e-5)

    def test_different_guidance_scales_produce_different_outputs(
        self, small_conditioned_unet, sample_latent_batch,
        sample_labels, sample_timesteps_cfg, device,
    ):
        """Different guidance scales should produce different outputs."""
        small_conditioned_unet.eval()
        small_conditioned_unet.set_class_labels(sample_labels)

        cfg_low = ClassifierFreeGuidanceWrapper(
            small_conditioned_unet, guidance_scale=1.0,
        )
        output_low = cfg_low(sample_latent_batch, sample_timesteps_cfg).clone()

        cfg_high = ClassifierFreeGuidanceWrapper(
            small_conditioned_unet, guidance_scale=5.0,
        )
        output_high = cfg_high(sample_latent_batch, sample_timesteps_cfg).clone()

        assert not torch.allclose(output_low, output_high, atol=1e-6)

    def test_no_gradients_in_eval(
        self, small_conditioned_unet, sample_latent_batch,
        sample_labels, sample_timesteps_cfg, device,
    ):
        """CFG wrapper should not compute gradients during sampling."""
        small_conditioned_unet.eval()
        small_conditioned_unet.set_class_labels(sample_labels)
        cfg_wrapper = ClassifierFreeGuidanceWrapper(
            small_conditioned_unet, guidance_scale=3.0,
        )

        with torch.no_grad():
            output = cfg_wrapper(sample_latent_batch, sample_timesteps_cfg)
        assert not output.requires_grad

    def test_rescale_cfg_matches_conditional_std(
        self, small_conditioned_unet, sample_latent_batch,
        sample_labels, sample_timesteps_cfg, device,
    ):
        """With rescale_cfg=True, guided prediction std matches conditional std.

        The rescaled CFG formula normalizes the guided noise prediction so
        its per-sample standard deviation matches the conditional prediction's
        std, preventing norm inflation that destabilizes DDPM sampling.
        """
        small_conditioned_unet.eval()
        small_conditioned_unet.set_class_labels(sample_labels)

        # Get conditional prediction directly
        conditional_output = small_conditioned_unet(
            sample_latent_batch, sample_timesteps_cfg,
        ).clone()

        # Get rescaled CFG output with high guidance scale
        small_conditioned_unet.set_class_labels(sample_labels)
        cfg_wrapper = ClassifierFreeGuidanceWrapper(
            small_conditioned_unet, guidance_scale=5.0, rescale_cfg=True,
        )
        cfg_output = cfg_wrapper(sample_latent_batch, sample_timesteps_cfg)

        # Per-sample std should match within tolerance
        reduce_dims = tuple(range(1, cfg_output.ndim))
        std_conditional = conditional_output.std(dim=reduce_dims)
        std_guided = cfg_output.std(dim=reduce_dims)

        assert torch.allclose(std_guided, std_conditional, rtol=1e-4)

    def test_rescale_cfg_false_preserves_inflated_norm(
        self, small_conditioned_unet, sample_latent_batch,
        sample_labels, sample_timesteps_cfg, device,
    ):
        """With rescale_cfg=False, the raw CFG formula is used unchanged.

        At high guidance scales, the unrescaled output should have a larger
        standard deviation than the conditional prediction.
        """
        small_conditioned_unet.eval()
        small_conditioned_unet.set_class_labels(sample_labels)

        # Get conditional prediction
        conditional_output = small_conditioned_unet(
            sample_latent_batch, sample_timesteps_cfg,
        ).clone()

        # Get unrescaled CFG output with high guidance scale
        small_conditioned_unet.set_class_labels(sample_labels)
        cfg_wrapper = ClassifierFreeGuidanceWrapper(
            small_conditioned_unet, guidance_scale=5.0, rescale_cfg=False,
        )
        cfg_output = cfg_wrapper(sample_latent_batch, sample_timesteps_cfg)

        # Unrescaled CFG with w=5 should inflate the overall std
        std_conditional = conditional_output.std().item()
        std_guided = cfg_output.std().item()

        # With w=5, the guided prediction norm should be noticeably larger
        assert std_guided > std_conditional * 0.9

    def test_rescale_cfg_preserves_direction(
        self, small_conditioned_unet, sample_latent_batch,
        sample_labels, sample_timesteps_cfg, device,
    ):
        """Rescaled and unrescaled CFG should produce outputs with same direction.

        Rescaling only changes the magnitude (std), not the direction. The
        cosine similarity between rescaled and unrescaled outputs should be
        close to 1.0.
        """
        small_conditioned_unet.eval()
        small_conditioned_unet.set_class_labels(sample_labels)

        cfg_rescaled = ClassifierFreeGuidanceWrapper(
            small_conditioned_unet, guidance_scale=3.0, rescale_cfg=True,
        )
        output_rescaled = cfg_rescaled(
            sample_latent_batch, sample_timesteps_cfg,
        ).clone()

        small_conditioned_unet.set_class_labels(sample_labels)
        cfg_unrescaled = ClassifierFreeGuidanceWrapper(
            small_conditioned_unet, guidance_scale=3.0, rescale_cfg=False,
        )
        output_unrescaled = cfg_unrescaled(
            sample_latent_batch, sample_timesteps_cfg,
        ).clone()

        # Flatten and compute cosine similarity per sample
        output_rescaled_flat = output_rescaled.flatten(start_dim=1)
        output_unrescaled_flat = output_unrescaled.flatten(start_dim=1)
        cosine_similarity = torch.nn.functional.cosine_similarity(
            output_rescaled_flat, output_unrescaled_flat, dim=1,
        )

        # Direction should be preserved (cosine similarity ≈ 1.0)
        assert (cosine_similarity > 0.99).all()


# ---------------------------------------------------------------------------
#  TestCreateConditionedUNet
# ---------------------------------------------------------------------------

class TestCreateConditionedUNet:
    """Tests for the create_conditioned_unet factory function."""

    def test_factory_creates_correct_model(self, device):
        """Factory should create a ClassConditionedUNet with correct config."""
        conditioned_unet = create_conditioned_unet(
            latent_channels=2,
            number_of_classes=10,
            base_channels=32,
            channel_multipliers=(1, 2),
            layers_per_block=1,
            attention_levels=(False, True),
        ).to(device)

        assert isinstance(conditioned_unet, ClassConditionedUNet)
        assert conditioned_unet.number_of_classes == 10
        # Inner UNet should have input_channels=3 (2 latent + 1 conditioning)
        assert conditioned_unet.unet.image_channels == 3
        # Inner UNet should have output_channels=2 (latent only)
        assert conditioned_unet.unet.output_channels == 2

    def test_factory_different_latent_channels(self, device):
        """Factory should handle different latent channel counts."""
        conditioned_unet = create_conditioned_unet(
            latent_channels=4,
            number_of_classes=10,
            base_channels=32,
            channel_multipliers=(1, 2),
            layers_per_block=1,
            attention_levels=(False, True),
        ).to(device)

        assert conditioned_unet.unet.image_channels == 5  # 4 + 1
        assert conditioned_unet.unet.output_channels == 4

        latent = torch.randn(2, 4, 4, 4, device=device)
        labels = torch.tensor([0, 5], device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)

        conditioned_unet.set_class_labels(labels)
        output = conditioned_unet(latent, timestep)
        assert output.shape == (2, 4, 4, 4)

    def test_factory_different_number_of_classes(self, device):
        """Factory should handle different class counts."""
        conditioned_unet = create_conditioned_unet(
            latent_channels=2,
            number_of_classes=5,
            base_channels=32,
            channel_multipliers=(1, 2),
            layers_per_block=1,
            attention_levels=(False, True),
        ).to(device)

        assert conditioned_unet.number_of_classes == 5
        # 5 classes + 1 unconditional = 6 entries
        assert conditioned_unet.class_embedding.num_embeddings == 6


# ---------------------------------------------------------------------------
#  TestUNetSeparateOutputChannels
# ---------------------------------------------------------------------------

class TestUNetSeparateOutputChannels:
    """Tests for the UNet output_channels parameter."""

    def test_separate_input_output_channels(self, device, seed):
        """UNet should support different input and output channel counts."""
        unet = UNet(
            image_channels=3,
            output_channels=2,
            base_channels=32,
            channel_multipliers=(1, 2),
            layers_per_block=1,
            attention_levels=(False, True),
            norm_num_groups=32,
        ).to(device)

        # Input has 3 channels, output should have 2
        x = torch.randn(2, 3, 4, 4, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)
        output = unet(x, timestep)

        assert output.shape == (2, 2, 4, 4)

    def test_default_output_channels_equals_input(self, device, seed):
        """Without output_channels, output should match input channels."""
        unet = UNet(
            image_channels=2,
            base_channels=32,
            channel_multipliers=(1, 2),
            layers_per_block=1,
            attention_levels=(False, True),
            norm_num_groups=32,
        ).to(device)

        assert unet.output_channels == 2

        x = torch.randn(2, 2, 4, 4, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)
        output = unet(x, timestep)

        assert output.shape == (2, 2, 4, 4)

    def test_backward_compatible_with_mnist(self, device, seed):
        """Existing MNIST usage (image_channels=1) should still work."""
        unet = UNet(
            image_channels=1,
            base_channels=32,
            channel_multipliers=(1, 2, 3, 3),
            layers_per_block=1,
            attention_levels=(False, False, False, True),
            norm_num_groups=32,
        ).to(device)

        assert unet.image_channels == 1
        assert unet.output_channels == 1

        x = torch.randn(2, 1, 28, 28, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)
        output = unet(x, timestep)

        assert output.shape == (2, 1, 28, 28)
