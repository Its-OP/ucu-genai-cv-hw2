"""
Unit tests for Rectified Flow.

Tests: interpolation, velocity computation, velocity loss,
       Euler step, Euler sample loop.

All tests follow the AAA pattern (Arrange, Act, Assert).
"""
import pytest
import torch
import torch.nn as nn

from models.rectified_flow import RectifiedFlow, TIMESTEP_SCALE


# ---------------------------------------------------------------------------
#  Simple model fixture for testing (minimal Conv2d)
# ---------------------------------------------------------------------------

class SimpleVelocityModel(nn.Module):
    """Minimal model conforming to model(x, t) -> prediction interface."""

    def __init__(self, channels=1):
        super().__init__()
        self.convolution = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, timestep):
        return self.convolution(x)


# ---------------------------------------------------------------------------
#  TestRectifiedFlowInitialization
# ---------------------------------------------------------------------------

class TestRectifiedFlowInitialization:
    """Tests for RectifiedFlow class initialization."""

    def test_default_sampling_steps(self, device):
        """Default number_of_sampling_steps should be 50."""
        # Arrange & Act
        rectified_flow = RectifiedFlow().to(device)

        # Assert
        assert rectified_flow.number_of_sampling_steps == 50

    def test_custom_sampling_steps(self, device):
        """Custom number_of_sampling_steps should be stored."""
        # Arrange & Act
        rectified_flow = RectifiedFlow(number_of_sampling_steps=100).to(device)

        # Assert
        assert rectified_flow.number_of_sampling_steps == 100

    def test_device_tracking(self, device):
        """Module should track the device it is on."""
        # Arrange & Act
        rectified_flow = RectifiedFlow().to(device)

        # Assert
        assert rectified_flow.device == device


# ---------------------------------------------------------------------------
#  TestRectifiedFlowInterpolate
# ---------------------------------------------------------------------------

class TestRectifiedFlowInterpolate:
    """Tests for the interpolate() method (linear interpolation)."""

    def test_output_shape(self, device, seed, sample_image):
        """Output shape should match x_0 shape."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        noise = torch.randn_like(sample_image)
        t = torch.rand(sample_image.shape[0], device=device)

        # Act
        x_t = rectified_flow.interpolate(sample_image, noise, t)

        # Assert
        assert x_t.shape == sample_image.shape

    def test_t_zero_returns_x0(self, device, seed, sample_image):
        """At t=0, interpolation should return x_0 exactly."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        noise = torch.randn_like(sample_image)
        t = torch.zeros(sample_image.shape[0], device=device)

        # Act
        x_t = rectified_flow.interpolate(sample_image, noise, t)

        # Assert
        assert torch.allclose(x_t, sample_image, atol=1e-6)

    def test_t_one_returns_noise(self, device, seed, sample_image):
        """At t=1, interpolation should return noise exactly."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        noise = torch.randn_like(sample_image)
        t = torch.ones(sample_image.shape[0], device=device)

        # Act
        x_t = rectified_flow.interpolate(sample_image, noise, t)

        # Assert
        assert torch.allclose(x_t, noise, atol=1e-6)

    def test_t_half_is_midpoint(self, device, seed, sample_image):
        """At t=0.5, output should be (x_0 + noise) / 2."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        noise = torch.randn_like(sample_image)
        t = torch.full((sample_image.shape[0],), 0.5, device=device)

        # Act
        x_t = rectified_flow.interpolate(sample_image, noise, t)
        expected = (sample_image + noise) / 2.0

        # Assert
        assert torch.allclose(x_t, expected, atol=1e-6)

    def test_linearity(self, device, seed, sample_image):
        """Interpolation should be linear in t."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        noise = torch.randn_like(sample_image)
        t1 = torch.full((sample_image.shape[0],), 0.3, device=device)
        t2 = torch.full((sample_image.shape[0],), 0.7, device=device)

        # Act
        x_t1 = rectified_flow.interpolate(sample_image, noise, t1)
        x_t2 = rectified_flow.interpolate(sample_image, noise, t2)

        # Assert: x_t = (1-t)*x_0 + t*noise, check manually
        expected_t1 = 0.7 * sample_image + 0.3 * noise
        expected_t2 = 0.3 * sample_image + 0.7 * noise
        assert torch.allclose(x_t1, expected_t1, atol=1e-6)
        assert torch.allclose(x_t2, expected_t2, atol=1e-6)

    def test_per_sample_time(self, device, seed):
        """Different t values per sample should produce different results."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        x_0 = torch.randn(4, 1, 8, 8, device=device)
        noise = torch.randn_like(x_0)
        t = torch.tensor([0.0, 0.33, 0.67, 1.0], device=device)

        # Act
        x_t = rectified_flow.interpolate(x_0, noise, t)

        # Assert: first sample should be x_0, last should be noise
        assert torch.allclose(x_t[0], x_0[0], atol=1e-6)
        assert torch.allclose(x_t[3], noise[3], atol=1e-6)


# ---------------------------------------------------------------------------
#  TestRectifiedFlowComputeVelocity
# ---------------------------------------------------------------------------

class TestRectifiedFlowComputeVelocity:
    """Tests for the compute_velocity() method."""

    def test_output_shape(self, device, seed, sample_image):
        """Velocity shape should match x_0 shape."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        noise = torch.randn_like(sample_image)

        # Act
        velocity = rectified_flow.compute_velocity(sample_image, noise)

        # Assert
        assert velocity.shape == sample_image.shape

    def test_velocity_is_noise_minus_data(self, device, seed, sample_image):
        """Velocity should equal noise - x_0."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        noise = torch.randn_like(sample_image)

        # Act
        velocity = rectified_flow.compute_velocity(sample_image, noise)
        expected = noise - sample_image

        # Assert
        assert torch.allclose(velocity, expected, atol=1e-6)


# ---------------------------------------------------------------------------
#  TestRectifiedFlowVelocityLoss
# ---------------------------------------------------------------------------

class TestRectifiedFlowVelocityLoss:
    """Tests for the velocity_loss() method."""

    def test_output_is_scalar(self, device, seed, sample_image):
        """Loss should be a 0-dimensional scalar tensor."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        t = torch.rand(sample_image.shape[0], device=device)

        # Act
        loss = rectified_flow.velocity_loss(model, sample_image, t)

        # Assert
        assert loss.dim() == 0

    def test_loss_non_negative(self, device, seed, sample_image):
        """MSE loss should always be >= 0."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        t = torch.rand(sample_image.shape[0], device=device)

        # Act
        loss = rectified_flow.velocity_loss(model, sample_image, t)

        # Assert
        assert loss.item() >= 0.0

    def test_loss_differentiable(self, device, seed, sample_image):
        """Loss should have a gradient function for backpropagation."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        t = torch.rand(sample_image.shape[0], device=device)

        # Act
        loss = rectified_flow.velocity_loss(model, sample_image, t)

        # Assert
        assert loss.grad_fn is not None

    def test_deterministic_with_seed(self, device, sample_image):
        """Same seed should produce same loss."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        t = torch.full((sample_image.shape[0],), 0.5, device=device)

        # Act
        torch.manual_seed(123)
        loss_1 = rectified_flow.velocity_loss(model, sample_image, t)
        torch.manual_seed(123)
        loss_2 = rectified_flow.velocity_loss(model, sample_image, t)

        # Assert
        assert torch.allclose(loss_1, loss_2)

    def test_timestep_scaling(self, device, seed, sample_image):
        """Model should receive time values scaled by TIMESTEP_SCALE."""
        # Arrange
        received_timesteps = []

        class TimestepCapturingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.convolution = nn.Conv2d(1, 1, 3, padding=1)

            def forward(self, x, timestep):
                received_timesteps.append(timestep.clone())
                return self.convolution(x)

        rectified_flow = RectifiedFlow().to(device)
        model = TimestepCapturingModel().to(device)
        t = torch.full((sample_image.shape[0],), 0.5, device=device)

        # Act
        rectified_flow.velocity_loss(model, sample_image, t)

        # Assert: model should receive t * TIMESTEP_SCALE = 0.5 * 999.0
        expected_timestep = 0.5 * TIMESTEP_SCALE
        assert torch.allclose(
            received_timesteps[0],
            torch.full_like(received_timesteps[0], expected_timestep),
            atol=1e-3,
        )


# ---------------------------------------------------------------------------
#  TestRectifiedFlowEulerStep
# ---------------------------------------------------------------------------

class TestRectifiedFlowEulerStep:
    """Tests for the euler_step() method."""

    def test_output_shape(self, device, seed, sample_image):
        """Output shape should match input shape."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        t = torch.full((sample_image.shape[0],), 0.5, device=device)

        # Act
        x_next = rectified_flow.euler_step(model, sample_image, t, step_size=0.1)

        # Assert
        assert x_next.shape == sample_image.shape

    def test_no_gradient_computation(self, device, seed, sample_image):
        """Euler step should not track gradients."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        t = torch.full((sample_image.shape[0],), 0.5, device=device)

        # Act
        x_next = rectified_flow.euler_step(model, sample_image, t, step_size=0.1)

        # Assert
        assert x_next.grad_fn is None

    def test_clipping_pixel_space(self, device, seed, sample_image):
        """With clip_denoised=True, output should be in [-1, 1]."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        t = torch.full((sample_image.shape[0],), 0.5, device=device)

        # Act (use large step to potentially exceed bounds)
        x_next = rectified_flow.euler_step(
            model, sample_image, t, step_size=10.0, clip_denoised=True,
        )

        # Assert
        assert x_next.min() >= -1.0
        assert x_next.max() <= 1.0

    def test_no_clipping_latent_space(self, device, seed, sample_image):
        """With clip_denoised=False, output can exceed [-1, 1]."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)

        class LargeOutputModel(nn.Module):
            def forward(self, x, timestep):
                return torch.ones_like(x) * 10.0

        model = LargeOutputModel().to(device)
        t = torch.full((sample_image.shape[0],), 0.5, device=device)

        # Act
        x_next = rectified_flow.euler_step(
            model, sample_image, t, step_size=1.0, clip_denoised=False,
        )

        # Assert: output should exceed [-1, 1] because of large velocity
        assert x_next.abs().max() > 1.0

    def test_numerical_stability(self, device, seed, sample_image):
        """Output should not contain NaN or Inf."""
        # Arrange
        rectified_flow = RectifiedFlow().to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        t = torch.full((sample_image.shape[0],), 0.5, device=device)

        # Act
        x_next = rectified_flow.euler_step(model, sample_image, t, step_size=0.02)

        # Assert
        assert not torch.isnan(x_next).any()
        assert not torch.isinf(x_next).any()


# ---------------------------------------------------------------------------
#  TestRectifiedFlowEulerSampleLoop
# ---------------------------------------------------------------------------

class TestRectifiedFlowEulerSampleLoop:
    """Tests for the euler_sample_loop() method."""

    def test_output_shape(self, device, seed):
        """Output shape should match the requested shape."""
        # Arrange
        rectified_flow = RectifiedFlow(number_of_sampling_steps=10).to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        shape = (2, 1, 8, 8)

        # Act
        samples = rectified_flow.euler_sample_loop(model, shape)

        # Assert
        assert samples.shape == shape

    def test_deterministic_with_seed(self, device):
        """Same seed should produce identical samples."""
        # Arrange
        rectified_flow = RectifiedFlow(number_of_sampling_steps=10).to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        model.eval()
        shape = (2, 1, 8, 8)

        # Act
        torch.manual_seed(42)
        samples_1 = rectified_flow.euler_sample_loop(model, shape)
        torch.manual_seed(42)
        samples_2 = rectified_flow.euler_sample_loop(model, shape)

        # Assert
        assert torch.allclose(samples_1, samples_2)

    def test_return_intermediates_tuple(self, device, seed):
        """With return_intermediates=True, should return (samples, intermediates) tuple."""
        # Arrange
        rectified_flow = RectifiedFlow(number_of_sampling_steps=10).to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        shape = (2, 1, 8, 8)

        # Act
        result = rectified_flow.euler_sample_loop(
            model, shape, return_intermediates=True,
        )

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_intermediates_format(self, device, seed):
        """Each intermediate should be a (step_index, tensor) tuple."""
        # Arrange
        rectified_flow = RectifiedFlow(number_of_sampling_steps=10).to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        shape = (2, 1, 8, 8)

        # Act
        _, intermediates = rectified_flow.euler_sample_loop(
            model, shape, return_intermediates=True,
        )

        # Assert
        assert len(intermediates) > 0
        for step_index, tensor in intermediates:
            assert isinstance(step_index, int)
            assert tensor.shape == shape

    def test_intermediates_includes_final_step(self, device, seed):
        """Intermediates should include step index 0 (the final, clean result)."""
        # Arrange
        rectified_flow = RectifiedFlow(number_of_sampling_steps=10).to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        shape = (2, 1, 8, 8)

        # Act
        _, intermediates = rectified_flow.euler_sample_loop(
            model, shape, return_intermediates=True,
        )

        # Assert
        final_step_indices = [step_index for step_index, _ in intermediates]
        assert 0 in final_step_indices

    def test_various_step_counts(self, device, seed):
        """Should work with various numbers of sampling steps."""
        # Arrange
        model = SimpleVelocityModel(channels=1).to(device)
        shape = (1, 1, 8, 8)

        for steps in [1, 5, 10, 50]:
            # Act
            rectified_flow = RectifiedFlow(number_of_sampling_steps=steps).to(device)
            samples = rectified_flow.euler_sample_loop(model, shape)

            # Assert
            assert samples.shape == shape

    def test_initial_noise_parameter(self, device, seed):
        """When initial_noise is provided, sampling should start from it."""
        # Arrange
        rectified_flow = RectifiedFlow(number_of_sampling_steps=10).to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        model.eval()
        shape = (2, 1, 8, 8)
        initial_noise = torch.randn(shape, device=device)

        # Act: two runs with the same initial noise should produce same result
        samples_1 = rectified_flow.euler_sample_loop(
            model, shape, initial_noise=initial_noise.clone(),
        )
        samples_2 = rectified_flow.euler_sample_loop(
            model, shape, initial_noise=initial_noise.clone(),
        )

        # Assert
        assert torch.allclose(samples_1, samples_2)

    def test_override_number_of_steps(self, device, seed):
        """number_of_steps parameter should override default."""
        # Arrange
        rectified_flow = RectifiedFlow(number_of_sampling_steps=50).to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        shape = (1, 1, 8, 8)

        # Act: explicitly pass a different step count
        samples = rectified_flow.euler_sample_loop(model, shape, number_of_steps=5)

        # Assert: should complete without error and produce correct shape
        assert samples.shape == shape

    def test_no_gradient_computation(self, device, seed):
        """Sample loop should not track gradients."""
        # Arrange
        rectified_flow = RectifiedFlow(number_of_sampling_steps=5).to(device)
        model = SimpleVelocityModel(channels=1).to(device)
        shape = (1, 1, 8, 8)

        # Act
        samples = rectified_flow.euler_sample_loop(model, shape)

        # Assert
        assert samples.grad_fn is None

    def test_latent_space_no_clipping(self, device, seed):
        """With clip_denoised=False, output may exceed [-1, 1]."""
        # Arrange
        class LargeVelocityModel(nn.Module):
            def forward(self, x, timestep):
                return torch.ones_like(x) * 5.0

        rectified_flow = RectifiedFlow(number_of_sampling_steps=5).to(device)
        model = LargeVelocityModel().to(device)
        shape = (1, 1, 8, 8)

        # Act
        samples = rectified_flow.euler_sample_loop(
            model, shape, clip_denoised=False,
        )

        # Assert: with constant large velocity, the result will be offset
        # from the starting noise, possibly outside [-1, 1]
        assert samples.shape == shape
