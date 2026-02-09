"""
Unit tests for DDIM (Denoising Diffusion Implicit Models) sampler.

Tests: timestep subsequence computation, sigma computation, ddim_sample,
ddim_sample_loop, determinism/stochasticity, numerical stability.
All tests follow the AAA pattern (Arrange, Act, Assert).
"""
import pytest
import torch
import torch.nn as nn

from models.ddpm import DDPM
from models.ddim import DDIMSampler


class TestDDIMTimestepSequence:
    """Tests for DDIM timestep subsequence computation."""

    def test_sequence_length(self, device):
        """Timestep sequence should have exactly ddim_timesteps elements."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50)

        # Act
        sequence_length = len(sampler.timestep_sequence)

        # Assert
        assert sequence_length == 50

    def test_sequence_values_in_range(self, device):
        """All timestep values should be in [0, ddpm_timesteps)."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50)

        # Assert
        assert (sampler.timestep_sequence >= 0).all()
        assert (sampler.timestep_sequence < 1000).all()

    def test_sequence_monotonically_increasing(self, device):
        """Timestep sequence should be monotonically increasing."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50)

        # Act
        differences = sampler.timestep_sequence[1:] - sampler.timestep_sequence[:-1]

        # Assert
        assert (differences > 0).all()

    def test_sequence_uniform_spacing(self, device):
        """Timestep sequence should be uniformly spaced."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50)

        # Act — spacing should be 1000 // 50 = 20
        differences = sampler.timestep_sequence[1:] - sampler.timestep_sequence[:-1]

        # Assert
        expected_step_size = 1000 // 50
        assert (differences == expected_step_size).all()

    def test_sequence_starts_at_zero(self, device):
        """Timestep sequence should start at 0."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50)

        # Assert
        assert sampler.timestep_sequence[0] == 0


class TestDDIMSigmaComputation:
    """Tests for DDIM sigma (noise coefficient) computation."""

    def test_sigma_zero_when_eta_zero(self, device):
        """When eta=0, sigma should be exactly 0 for all timesteps."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50, eta=0.0)

        alpha_cumprod_current = torch.tensor(0.5, device=device)
        alpha_cumprod_previous = torch.tensor(0.6, device=device)

        # Act
        sigma = sampler._compute_sigma(alpha_cumprod_current, alpha_cumprod_previous)

        # Assert
        assert sigma == 0.0

    def test_sigma_positive_when_eta_nonzero(self, device):
        """When eta>0, sigma should be positive for valid alpha_cumprod values."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50, eta=1.0)

        # Use realistic alpha_cumprod values (current < previous since later timesteps have less signal)
        alpha_cumprod_current = torch.tensor(0.3, device=device)
        alpha_cumprod_previous = torch.tensor(0.5, device=device)

        # Act
        sigma = sampler._compute_sigma(alpha_cumprod_current, alpha_cumprod_previous)

        # Assert
        assert sigma > 0

    def test_sigma_scales_linearly_with_eta(self, device):
        """Sigma should scale linearly with eta."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler_half = DDIMSampler(ddpm, ddim_timesteps=50, eta=0.5)
        sampler_full = DDIMSampler(ddpm, ddim_timesteps=50, eta=1.0)

        alpha_cumprod_current = torch.tensor(0.3, device=device)
        alpha_cumprod_previous = torch.tensor(0.5, device=device)

        # Act
        sigma_half = sampler_half._compute_sigma(alpha_cumprod_current, alpha_cumprod_previous)
        sigma_full = sampler_full._compute_sigma(alpha_cumprod_current, alpha_cumprod_previous)

        # Assert — sigma(eta=0.5) should be exactly half of sigma(eta=1.0)
        torch.testing.assert_close(sigma_half, sigma_full * 0.5)


class TestDDIMSample:
    """Tests for DDIMSampler.ddim_sample (single reverse step)."""

    def _create_simple_model(self, device):
        """Create a simple model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.convolution = nn.Conv2d(1, 1, 3, padding=1)

            def forward(self, x, timestep):
                return self.convolution(x)

        return SimpleModel().to(device)

    def test_output_shape(self, device, seed):
        """Output shape should match input shape."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50, eta=0.0)
        model = self._create_simple_model(device)
        model.eval()
        x_current = torch.randn(4, 1, 28, 28, device=device)

        # Act
        x_previous = sampler.ddim_sample(model, x_current, timestep_current=500, timestep_previous=480)

        # Assert
        assert x_previous.shape == x_current.shape

    def test_deterministic_when_eta_zero(self, device, seed):
        """With eta=0, ddim_sample should be fully deterministic."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50, eta=0.0)
        model = self._create_simple_model(device)
        model.eval()
        x_current = torch.randn(4, 1, 28, 28, device=device)

        # Act
        x_previous_1 = sampler.ddim_sample(model, x_current, timestep_current=500, timestep_previous=480)
        x_previous_2 = sampler.ddim_sample(model, x_current, timestep_current=500, timestep_previous=480)

        # Assert
        torch.testing.assert_close(x_previous_1, x_previous_2)

    def test_stochastic_when_eta_nonzero(self, device, seed):
        """With eta>0, ddim_sample should be stochastic (different seeds produce different outputs)."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50, eta=1.0)
        model = self._create_simple_model(device)
        model.eval()
        x_current = torch.randn(4, 1, 28, 28, device=device)

        # Act
        torch.manual_seed(1)
        x_previous_1 = sampler.ddim_sample(model, x_current, timestep_current=500, timestep_previous=480)
        torch.manual_seed(2)
        x_previous_2 = sampler.ddim_sample(model, x_current, timestep_current=500, timestep_previous=480)

        # Assert
        assert not torch.allclose(x_previous_1, x_previous_2)

    def test_numerical_stability(self, device, seed):
        """ddim_sample should produce bounded output even with extreme noise predictions."""
        # Arrange
        class ExtremeNoiseModel(nn.Module):
            """Model that outputs extreme noise predictions to test clipping."""
            def forward(self, x, timestep):
                return torch.ones_like(x) * 100.0

        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50, eta=0.0)
        model = ExtremeNoiseModel().to(device)
        model.eval()
        x_current = torch.randn(4, 1, 28, 28, device=device)

        # Act
        x_previous = sampler.ddim_sample(model, x_current, timestep_current=500, timestep_previous=480)

        # Assert
        assert not torch.isnan(x_previous).any()
        assert not torch.isinf(x_previous).any()
        assert x_previous.abs().max() < 100

    def test_no_gradient_computation(self, device, seed):
        """ddim_sample should not compute gradients (uses @torch.no_grad())."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50, eta=0.0)
        model = self._create_simple_model(device)
        model.eval()
        x_current = torch.randn(4, 1, 28, 28, device=device, requires_grad=True)

        # Act
        x_previous = sampler.ddim_sample(model, x_current, timestep_current=500, timestep_previous=480)

        # Assert
        assert x_previous.grad_fn is None

    def test_final_step_with_sentinel(self, device, seed):
        """Final step (timestep_previous=-1) should work correctly."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50, eta=0.0)
        model = self._create_simple_model(device)
        model.eval()
        x_current = torch.randn(4, 1, 28, 28, device=device)

        # Act — sentinel -1 means stepping to clean image (ᾱ_{-1} = 1.0)
        x_previous = sampler.ddim_sample(model, x_current, timestep_current=0, timestep_previous=-1)

        # Assert
        assert x_previous.shape == x_current.shape
        assert not torch.isnan(x_previous).any()

    def test_clip_sample_range_clamps_predicted_x0(self, device, seed):
        """clip_sample_range should control the clamp bounds for predicted x̂₀.

        In the DDIM formula: x_{prev} = √ᾱ_{prev} · x̂₀ + √(1 − ᾱ_{prev}) · ε_θ,
        we use a moderate noise prediction so the √ᾱ_{prev} · x̂₀ term (which is
        affected by clipping) is not swamped by the direction term.

        At timestep_previous=-1 (sentinel, ᾱ_{prev}=1.0), the formula becomes
        x_{prev} = x̂₀, so clipping directly controls the output.
        """
        # Arrange — moderate noise prediction so x̂₀ reconstruction is large
        # but the formula's x̂₀ term isn't overwhelmed by the ε_θ direction term.
        class ModerateNoiseModel(nn.Module):
            """Model that outputs moderate noise predictions."""
            def forward(self, x, timestep):
                return torch.ones_like(x) * 5.0

        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50, eta=0.0)
        model = ModerateNoiseModel().to(device)
        model.eval()
        x_current = torch.randn(4, 1, 28, 28, device=device)

        # Act — at the final step (timestep_previous=-1, ᾱ_{prev}=1.0), the
        # output equals x̂₀ directly, so clip_sample_range directly bounds it.
        x_prev_narrow = sampler.ddim_sample(
            model, x_current, timestep_current=0, timestep_previous=-1,
            clip_denoised=True, clip_sample_range=0.5,
        )
        x_prev_wide = sampler.ddim_sample(
            model, x_current, timestep_current=0, timestep_previous=-1,
            clip_denoised=True, clip_sample_range=2.0,
        )

        # Assert — narrow range should clamp more tightly
        assert x_prev_narrow.abs().max() <= 0.5 + 1e-6
        assert x_prev_wide.abs().max() <= 2.0 + 1e-6
        assert x_prev_narrow.abs().max() <= x_prev_wide.abs().max()

    def test_clip_denoised_false_ignores_clip_sample_range(self, device, seed):
        """When clip_denoised=False, clip_sample_range should be ignored."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50, eta=0.0)
        model = self._create_simple_model(device)
        model.eval()
        x_current = torch.randn(4, 1, 28, 28, device=device)

        # Act — clip_denoised=False with different ranges
        x_prev_a = sampler.ddim_sample(
            model, x_current, timestep_current=500, timestep_previous=480,
            clip_denoised=False, clip_sample_range=0.5,
        )
        x_prev_b = sampler.ddim_sample(
            model, x_current, timestep_current=500, timestep_previous=480,
            clip_denoised=False, clip_sample_range=5.0,
        )

        # Assert — should be identical since clipping is disabled
        torch.testing.assert_close(x_prev_a, x_prev_b)


class TestDDIMSampleLoop:
    """Tests for DDIMSampler.ddim_sample_loop (full reverse diffusion loop)."""

    def _create_simple_model(self, device):
        """Create a simple model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.convolution = nn.Conv2d(1, 1, 3, padding=1)

            def forward(self, x, timestep):
                return self.convolution(x)

        return SimpleModel().to(device)

    def test_output_shape(self, device, seed):
        """Output shape should match the requested sample shape."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=50, eta=0.0)
        model = self._create_simple_model(device)
        model.eval()
        shape = (4, 1, 28, 28)

        # Act
        samples = sampler.ddim_sample_loop(model, shape)

        # Assert
        assert samples.shape == shape

    def test_deterministic_full_loop_eta_zero(self, device):
        """With eta=0, the full DDIM loop should be fully deterministic."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=10, eta=0.0)
        model = self._create_simple_model(device)
        model.eval()
        shape = (2, 1, 28, 28)

        # Act
        torch.manual_seed(42)
        samples_1 = sampler.ddim_sample_loop(model, shape)
        torch.manual_seed(42)
        samples_2 = sampler.ddim_sample_loop(model, shape)

        # Assert
        torch.testing.assert_close(samples_1, samples_2)

    def test_stochastic_full_loop_eta_nonzero(self, device):
        """With eta>0, the full DDIM loop should be stochastic."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=10, eta=1.0)
        model = self._create_simple_model(device)
        model.eval()
        shape = (2, 1, 28, 28)

        # Act
        torch.manual_seed(1)
        samples_1 = sampler.ddim_sample_loop(model, shape)
        torch.manual_seed(2)
        samples_2 = sampler.ddim_sample_loop(model, shape)

        # Assert
        assert not torch.allclose(samples_1, samples_2)

    def test_return_intermediates(self, device, seed):
        """return_intermediates=True should return a tuple of (samples, intermediates)."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=10, eta=0.0)
        model = self._create_simple_model(device)
        model.eval()
        shape = (2, 1, 28, 28)

        # Act
        result = sampler.ddim_sample_loop(model, shape, return_intermediates=True)

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        samples, intermediates = result
        assert samples.shape == shape
        assert isinstance(intermediates, list)
        assert len(intermediates) > 0

    def test_intermediates_contain_tuples(self, device, seed):
        """Each intermediate should be a (timestep, tensor) tuple."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=10, eta=0.0)
        model = self._create_simple_model(device)
        model.eval()
        shape = (2, 1, 28, 28)

        # Act
        _, intermediates = sampler.ddim_sample_loop(model, shape, return_intermediates=True)

        # Assert
        for timestep, tensor in intermediates:
            assert isinstance(timestep, int)
            assert tensor.shape == shape

    def test_different_ddim_steps(self, device, seed):
        """Should work with various numbers of DDIM steps."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        model = self._create_simple_model(device)
        model.eval()
        shape = (2, 1, 28, 28)

        for ddim_steps in [10, 50, 100, 250]:
            # Act
            sampler = DDIMSampler(ddpm, ddim_timesteps=ddim_steps, eta=0.0)
            samples = sampler.ddim_sample_loop(model, shape)

            # Assert
            assert samples.shape == shape
            assert not torch.isnan(samples).any()

    def test_single_step(self, device, seed):
        """Should work with ddim_timesteps=1 (single denoising step)."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        sampler = DDIMSampler(ddpm, ddim_timesteps=1, eta=0.0)
        model = self._create_simple_model(device)
        model.eval()
        shape = (2, 1, 28, 28)

        # Act
        samples = sampler.ddim_sample_loop(model, shape)

        # Assert
        assert samples.shape == shape
        assert not torch.isnan(samples).any()
