"""
Unit tests for DDPM diffusion process.

Tests: Beta schedules, q_sample, p_losses, p_sample.
All tests follow the AAA pattern (Arrange, Act, Assert).
"""
import pytest
import torch
import torch.nn as nn

from models.ddpm import DDPM, linear_beta_schedule, cosine_beta_schedule


class TestLinearBetaSchedule:
    """Tests for linear_beta_schedule function."""

    def test_length(self):
        """Output should have exactly `timesteps` elements."""
        # Arrange
        timesteps = 1000

        # Act
        betas = linear_beta_schedule(timesteps)

        # Assert
        assert len(betas) == timesteps

    def test_bounds(self):
        """All values should be in [1e-4, 0.02]."""
        # Arrange
        timesteps = 1000

        # Act
        betas = linear_beta_schedule(timesteps)

        # Assert
        assert betas.min() >= 1e-4
        assert betas.max() <= 0.02

    def test_monotonic_increasing(self):
        """Values should be strictly increasing."""
        # Arrange
        timesteps = 1000

        # Act
        betas = linear_beta_schedule(timesteps)

        # Assert
        differences = betas[1:] - betas[:-1]
        assert (differences > 0).all()

    def test_different_timesteps(self):
        """Should work with different timestep counts."""
        # Arrange
        timestep_counts = [10, 100, 1000, 2000]

        for count in timestep_counts:
            # Act
            betas = linear_beta_schedule(count)

            # Assert
            assert len(betas) == count


class TestCosineBetaSchedule:
    """Tests for cosine_beta_schedule function."""

    def test_length(self):
        """Output should have exactly `timesteps` elements."""
        # Arrange
        timesteps = 1000

        # Act
        betas = cosine_beta_schedule(timesteps)

        # Assert
        assert len(betas) == timesteps

    def test_bounds(self):
        """All values should be clamped to [0.0001, 0.9999]."""
        # Arrange
        timesteps = 1000

        # Act
        betas = cosine_beta_schedule(timesteps)

        # Assert
        assert betas.min() >= 0.0001
        assert betas.max() <= 0.9999

    def test_no_nan_values(self):
        """Output should not contain NaN values."""
        # Arrange
        timesteps = 1000

        # Act
        betas = cosine_beta_schedule(timesteps)

        # Assert
        assert not torch.isnan(betas).any()

    def test_different_s_values(self):
        """Different s values should produce different schedules."""
        # Arrange
        timesteps = 1000

        # Act
        betas_s1 = cosine_beta_schedule(timesteps, s=0.001)
        betas_s2 = cosine_beta_schedule(timesteps, s=0.01)
        betas_s3 = cosine_beta_schedule(timesteps, s=0.1)

        # Assert
        assert not torch.allclose(betas_s1, betas_s2)
        assert not torch.allclose(betas_s2, betas_s3)


class TestDDPMInitialization:
    """Tests for DDPM class initialization."""

    def test_buffer_shapes(self, device):
        """All buffers should have shape (timesteps,)."""
        # Arrange
        timesteps = 1000

        # Act
        ddpm = DDPM(timesteps=timesteps).to(device)

        # Assert
        assert ddpm.betas.shape == (timesteps,)
        assert ddpm.alphas.shape == (timesteps,)
        assert ddpm.alphas_cumprod.shape == (timesteps,)
        assert ddpm.alphas_cumprod_prev.shape == (timesteps,)
        assert ddpm.sqrt_alphas_cumprod.shape == (timesteps,)
        assert ddpm.sqrt_one_minus_alphas_cumprod.shape == (timesteps,)
        assert ddpm.sqrt_recip_alphas.shape == (timesteps,)
        assert ddpm.posterior_variance.shape == (timesteps,)

    def test_alphas_cumprod_decreasing(self, device):
        """alphas_cumprod should be monotonically decreasing."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)

        # Act
        differences = ddpm.alphas_cumprod[1:] - ddpm.alphas_cumprod[:-1]

        # Assert
        assert (differences < 0).all()

    def test_alphas_cumprod_prev_first_element(self, device):
        """alphas_cumprod_prev[0] should be 1.0."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)

        # Act
        first_element = ddpm.alphas_cumprod_prev[0]

        # Assert
        assert abs(first_element - 1.0) < 1e-6

    def test_linear_vs_cosine_schedule(self, device):
        """Different schedules should produce different betas."""
        # Arrange & Act
        ddpm_linear = DDPM(timesteps=1000, beta_schedule='linear').to(device)
        ddpm_cosine = DDPM(timesteps=1000, beta_schedule='cosine').to(device)

        # Assert
        assert not torch.allclose(ddpm_linear.betas, ddpm_cosine.betas)


class TestDDPMQSample:
    """Tests for DDPM.q_sample (forward diffusion)."""

    def test_output_shape(self, device, seed):
        """Output shape should match x_0 shape."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        x_0 = torch.randn(4, 1, 28, 28, device=device)
        t = torch.randint(0, 1000, (4,), device=device)

        # Act
        x_t = ddpm.q_sample(x_0, t)

        # Assert
        assert x_t.shape == x_0.shape

    def test_t_zero_minimal_noise(self, device, seed):
        """At t=0, x_t should be very close to x_0."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        x_0 = torch.randn(4, 1, 28, 28, device=device)
        t = torch.zeros(4, dtype=torch.long, device=device)

        # Act
        x_t = ddpm.q_sample(x_0, t)

        # Assert
        # sqrt_alpha_cumprod[0] ≈ 1, sqrt_one_minus_alpha_cumprod[0] ≈ 0
        torch.testing.assert_close(x_t, x_0, rtol=0.1, atol=0.1)

    def test_t_max_mostly_noise(self, device, seed):
        """At t=T-1, x_t should be mostly noise (far from x_0)."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        x_0 = torch.randn(4, 1, 28, 28, device=device)
        t = torch.full((4,), 999, dtype=torch.long, device=device)
        noise = torch.randn_like(x_0)

        # Act
        x_t = ddpm.q_sample(x_0, t, noise=noise)

        # Assert
        # At high timesteps, x_t should be much closer to noise than to x_0
        distance_to_noise = (x_t - noise).abs().mean()
        distance_to_x0 = (x_t - x_0).abs().mean()
        assert distance_to_noise < distance_to_x0

    def test_deterministic_with_provided_noise(self, device, seed):
        """Same inputs should produce same outputs when noise is provided."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        x_0 = torch.randn(4, 1, 28, 28, device=device)
        t = torch.randint(0, 1000, (4,), device=device)
        noise = torch.randn_like(x_0)

        # Act
        x_t_1 = ddpm.q_sample(x_0, t, noise=noise)
        x_t_2 = ddpm.q_sample(x_0, t, noise=noise)

        # Assert
        torch.testing.assert_close(x_t_1, x_t_2)

    def test_coefficient_property(self, device):
        """sqrt_alpha_cumprod^2 + sqrt_one_minus_alpha_cumprod^2 should equal 1."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)

        # Act
        sum_of_squares = (
            ddpm.sqrt_alphas_cumprod ** 2 +
            ddpm.sqrt_one_minus_alphas_cumprod ** 2
        )

        # Assert
        torch.testing.assert_close(
            sum_of_squares,
            torch.ones_like(sum_of_squares),
            rtol=1e-5,
            atol=1e-5
        )


class TestDDPMPLosses:
    """Tests for DDPM.p_losses (training loss)."""

    def _create_simple_model(self, device):
        """Create a simple model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3, padding=1)

            def forward(self, x, t):
                return self.conv(x)

        return SimpleModel().to(device)

    def test_output_is_scalar(self, device, seed):
        """Loss should be a 0-dimensional tensor (scalar)."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        model = self._create_simple_model(device)
        x_0 = torch.randn(4, 1, 28, 28, device=device)
        t = torch.randint(0, 1000, (4,), device=device)

        # Act
        loss = ddpm.p_losses(model, x_0, t)

        # Assert
        assert loss.dim() == 0

    def test_loss_non_negative(self, device, seed):
        """Loss should be non-negative (MSE is always >= 0)."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        model = self._create_simple_model(device)
        x_0 = torch.randn(4, 1, 28, 28, device=device)
        t = torch.randint(0, 1000, (4,), device=device)

        # Act
        loss = ddpm.p_losses(model, x_0, t)

        # Assert
        assert loss >= 0

    def test_loss_differentiable(self, device, seed):
        """Loss should have a gradient function."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        model = self._create_simple_model(device)
        x_0 = torch.randn(4, 1, 28, 28, device=device)
        t = torch.randint(0, 1000, (4,), device=device)

        # Act
        loss = ddpm.p_losses(model, x_0, t)

        # Assert
        assert loss.grad_fn is not None

    def test_deterministic_with_seed(self, device):
        """Same inputs with same seed should produce same loss."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)

        torch.manual_seed(42)
        model_1 = self._create_simple_model(device)
        torch.manual_seed(42)
        model_2 = self._create_simple_model(device)

        torch.manual_seed(123)
        x_0 = torch.randn(4, 1, 28, 28, device=device)
        t = torch.randint(0, 1000, (4,), device=device)

        # Act
        torch.manual_seed(456)
        loss_1 = ddpm.p_losses(model_1, x_0, t)
        torch.manual_seed(456)
        loss_2 = ddpm.p_losses(model_2, x_0, t)

        # Assert
        torch.testing.assert_close(loss_1, loss_2)


class TestDDPMPSample:
    """Tests for DDPM.p_sample (reverse diffusion step)."""

    def _create_simple_model(self, device):
        """Create a simple model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3, padding=1)

            def forward(self, x, t):
                return self.conv(x)

        return SimpleModel().to(device)

    def test_output_shape(self, device, seed):
        """Output shape should match input shape."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        model = self._create_simple_model(device)
        model.eval()
        x_t = torch.randn(4, 1, 28, 28, device=device)
        t = torch.full((4,), 500, dtype=torch.long, device=device)

        # Act
        x_prev = ddpm.p_sample(model, x_t, t, t_index=500)

        # Assert
        assert x_prev.shape == x_t.shape

    def test_t_index_zero_deterministic(self, device, seed):
        """At t_index=0, output should be deterministic (no noise added)."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        model = self._create_simple_model(device)
        model.eval()
        x_t = torch.randn(4, 1, 28, 28, device=device)
        t = torch.zeros(4, dtype=torch.long, device=device)

        # Act
        x_prev_1 = ddpm.p_sample(model, x_t, t, t_index=0)
        x_prev_2 = ddpm.p_sample(model, x_t, t, t_index=0)

        # Assert
        torch.testing.assert_close(x_prev_1, x_prev_2)

    def test_t_index_nonzero_stochastic(self, device, seed):
        """At t_index>0, output should be stochastic (noise is added)."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        model = self._create_simple_model(device)
        model.eval()
        x_t = torch.randn(4, 1, 28, 28, device=device)
        t = torch.full((4,), 500, dtype=torch.long, device=device)

        # Act
        torch.manual_seed(1)
        x_prev_1 = ddpm.p_sample(model, x_t, t, t_index=500)
        torch.manual_seed(2)
        x_prev_2 = ddpm.p_sample(model, x_t, t, t_index=500)

        # Assert
        assert not torch.allclose(x_prev_1, x_prev_2)

    def test_no_gradient_computation(self, device, seed):
        """p_sample should not compute gradients (uses @torch.no_grad())."""
        # Arrange
        ddpm = DDPM(timesteps=1000).to(device)
        model = self._create_simple_model(device)
        model.eval()
        x_t = torch.randn(4, 1, 28, 28, device=device, requires_grad=True)
        t = torch.full((4,), 500, dtype=torch.long, device=device)

        # Act
        x_prev = ddpm.p_sample(model, x_t, t, t_index=500)

        # Assert
        assert x_prev.grad_fn is None
