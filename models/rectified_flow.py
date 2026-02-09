"""
Rectified Flow for diffusion models.

Implements the Rectified Flow framework from Liu et al. 2022:
    "Flow Straight and Fast: Learning to Generate and Transfer Data
     with Rectified Flows" (arXiv:2209.03003)

Replaces the DDPM noise schedule with a simpler linear interpolation
between data and noise, and predicts the velocity (flow) rather than
the noise itself.

Key formulas:
    Forward interpolation:
        x_t = (1 - t) * x_0 + t * epsilon,  where t in [0, 1]

    Target velocity:
        v = epsilon - x_0

    Training loss:
        L = E_{t, x_0, epsilon}[|| v_theta(x_t, t) - v ||^2]

    Euler sampling (reverse ODE integration):
        x_{t - dt} = x_t - dt * v_theta(x_t, t)
        from t=1 (pure noise) to t=0 (clean data)

Components:
    1. RectifiedFlow  — training loss + Euler sampling loop
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# Maximum timestep value for scaling continuous time to the UNet's
# sinusoidal positional embedding range. The UNet expects numeric
# timestep values; we map t in [0, 1] to [0, TIMESTEP_SCALE] so that
# the sinusoidal frequencies cover a useful range.
TIMESTEP_SCALE = 999.0


class RectifiedFlow(nn.Module):
    """
    Rectified Flow framework for training and sampling diffusion models.

    Unlike DDPM, Rectified Flow uses:
        - Linear interpolation between data and noise (no noise schedule)
        - Velocity prediction (v = noise - x_0) instead of noise prediction
        - Euler ODE integration for sampling (no posterior variance needed)
        - Continuous time t in [0, 1] instead of discrete timestep indices

    The same UNet architecture used for DDPM can be reused: only the
    loss function and sampling procedure differ.

    Args:
        number_of_sampling_steps: Default number of Euler steps for
            sampling. Can be overridden per call in euler_sample_loop().
            Typical values: 20–100. Default: 50.
    """

    def __init__(self, number_of_sampling_steps: int = 50):
        super().__init__()
        self.number_of_sampling_steps = number_of_sampling_steps

        # Register a buffer so that .device can be inferred from the module,
        # matching the DDPM pattern where self.betas.device is used.
        self.register_buffer("_device_indicator", torch.tensor(0.0))

    @property
    def device(self) -> torch.device:
        """Return the device this module lives on."""
        return self._device_indicator.device

    def interpolate(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear interpolation between clean data and noise.

        Replaces DDPM's q_sample() forward diffusion process with a simpler
        linear path from data (t=0) to noise (t=1).

        Formula (Liu et al. 2022):
            x_t = (1 - t) * x_0 + t * noise
            where t in [0, 1], x_0 is clean data, noise ~ N(0, I)
            t = 0 -> clean data x_0
            t = 1 -> pure noise

        Args:
            x_0: Clean data samples, shape (batch_size, channels, height, width).
            noise: Gaussian noise, shape (batch_size, channels, height, width).
            t: Continuous time values in [0, 1], shape (batch_size,).

        Returns:
            Interpolated samples x_t, same shape as x_0.
        """
        # Reshape t for broadcasting: (batch_size,) -> (batch_size, 1, 1, 1)
        t_reshaped = t[:, None, None, None]

        # x_t = (1 - t) * x_0 + t * noise
        return (1.0 - t_reshaped) * x_0 + t_reshaped * noise

    def compute_velocity(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the target velocity for training.

        The velocity is the vector field that transports data (at t=0)
        to noise (at t=1) along the linear interpolation path.

        Formula (Liu et al. 2022):
            v = noise - x_0

        Args:
            x_0: Clean data samples, shape (batch_size, channels, height, width).
            noise: Gaussian noise, shape (batch_size, channels, height, width).

        Returns:
            Target velocity, same shape as x_0.
        """
        # v = noise - x_0
        return noise - x_0

    def velocity_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Rectified Flow training loss.

        Formula (Liu et al. 2022):
            L = E_{t, x_0, epsilon}[|| v_theta(x_t, t) - v ||^2]

        where:
            v = epsilon - x_0                      (target velocity)
            x_t = (1 - t) * x_0 + t * epsilon     (linear interpolation)
            v_theta(x_t, t) is the model's velocity prediction

        The continuous time t is scaled to [0, 999] before being passed
        to the model, so the UNet's sinusoidal positional embedding
        operates in the same frequency range as DDPM timestep indices.

        Args:
            model: Velocity prediction network v_theta(x_t, t).
                Must conform to the model(x, t) -> prediction interface.
            x_0: Clean data samples, shape (batch_size, channels, height, width).
            t: Continuous time values in [0, 1], shape (batch_size,).

        Returns:
            Scalar MSE loss averaged over the batch.
        """
        noise = torch.randn_like(x_0)

        # x_t = (1 - t) * x_0 + t * noise
        x_t = self.interpolate(x_0, noise, t)

        # v = noise - x_0
        target_velocity = self.compute_velocity(x_0, noise)

        # Scale continuous time t in [0, 1] to [0, 999] for the UNet's
        # sinusoidal positional embedding (which expects numeric timestep values)
        timestep_for_model = t * TIMESTEP_SCALE

        # v_theta(x_t, t)
        predicted_velocity = model(x_t, timestep_for_model)

        # L = || v_theta(x_t, t) - v ||^2
        return F.mse_loss(predicted_velocity, target_velocity)

    @torch.no_grad()
    def euler_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        step_size: float,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Single Euler integration step for ODE sampling.

        Advances the sample from time t to time (t - step_size) by
        following the predicted velocity field.

        Formula (Liu et al. 2022):
            x_{t - dt} = x_t - dt * v_theta(x_t, t)

        where dt = step_size = 1/N and v_theta is the predicted velocity.
        Integration proceeds from t=1 (noise) toward t=0 (data).

        Args:
            model: Velocity prediction network v_theta(x_t, t).
            x_t: Current sample at time t, shape (batch, channels, height, width).
            t: Current time values in [0, 1], shape (batch_size,).
            step_size: Integration step size dt = 1/N.
            clip_denoised: If True, clip the result to [-1, 1]. Set to True
                for pixel-space (bounded), False for latent-space (unbounded).

        Returns:
            Updated sample at time (t - dt), same shape as x_t.
        """
        # Scale continuous time to UNet's sinusoidal embedding range
        timestep_for_model = t * TIMESTEP_SCALE

        # v_theta(x_t, t)
        predicted_velocity = model(x_t, timestep_for_model)

        # x_{t - dt} = x_t - dt * v_theta(x_t, t)
        x_next = x_t - step_size * predicted_velocity

        if clip_denoised:
            x_next = x_next.clamp(-1.0, 1.0)

        return x_next

    @torch.no_grad()
    def euler_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        number_of_steps: int = None,
        return_intermediates: bool = False,
        intermediate_steps: list = None,
        clip_denoised: bool = True,
        initial_noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Full Euler ODE sampling: generate data from pure noise.

        Algorithm (Liu et al. 2022):
            1. x_1 ~ N(0, I)                          (start from pure noise at t=1)
            2. For i = N-1, N-2, ..., 0:
                   t = (i + 1) / N
                   x_{t - 1/N} = x_t - (1/N) * v_theta(x_t, t)
            3. Return x_0

        Args:
            model: Velocity prediction network v_theta(x_t, t).
            shape: Shape of samples to generate, e.g. (batch, channels, height, width).
            number_of_steps: Number of Euler steps N. If None, uses
                self.number_of_sampling_steps. More steps = higher quality.
            return_intermediates: If True, also return intermediate samples.
            intermediate_steps: Step indices at which to save intermediates.
                Defaults to evenly-spaced indices at ~10% intervals.
            clip_denoised: If True, clip output to [-1, 1] at each step.
                Set to False for latent-space sampling.
            initial_noise: Optional pre-generated noise tensor of shape ``shape``.
                If provided, used as the starting x_1 instead of sampling fresh
                noise. Useful for comparing configurations from identical starts.

        Returns:
            Generated samples tensor of the requested shape.
            If return_intermediates=True, returns (samples, intermediates)
            where intermediates is a list of (step_index, tensor) tuples.
        """
        if number_of_steps is None:
            number_of_steps = self.number_of_sampling_steps

        device = self.device
        batch_size = shape[0]
        step_size = 1.0 / number_of_steps

        # Default intermediate steps: evenly spaced at ~10% intervals
        if intermediate_steps is None:
            intermediate_steps = _compute_intermediate_step_indices(number_of_steps)

        # Start from pure noise: x_1 ~ N(0, I)
        if initial_noise is not None:
            image = initial_noise.to(device)
        else:
            image = torch.randn(shape, device=device)

        intermediates = []

        # Euler integration from t=1 (noise) to t=0 (data)
        # Loop index i goes from (N-1) down to 0
        # At each step: t = (i + 1) / N, so t goes from 1.0 down to 1/N
        for i in tqdm(
            reversed(range(number_of_steps)),
            desc=f"RF Euler Sampling ({number_of_steps} steps)",
            total=number_of_steps,
        ):
            current_t = (i + 1) / number_of_steps
            t_batch = torch.full(
                (batch_size,), current_t, device=device, dtype=torch.float32,
            )

            image = self.euler_step(model, image, t_batch, step_size, clip_denoised)

            if return_intermediates and i in intermediate_steps:
                intermediates.append((i, image.clone()))

        # Always include the final result (step 0) if recording intermediates
        if return_intermediates and (not intermediates or intermediates[-1][0] != 0):
            intermediates.append((0, image.clone()))

        if return_intermediates:
            return image, intermediates
        return image


def _compute_intermediate_step_indices(number_of_steps: int) -> list:
    """
    Compute step indices at approximately every 10% of the Euler sampling process.

    The Euler loop runs from step index (N-1) down to 0. We capture
    snapshots at 0%, 10%, 20%, ..., 100% of progress through the loop.

    Args:
        number_of_steps: Total number of Euler steps N.

    Returns:
        List of unique step indices (integers in [0, N-1]) at which
        to record intermediate samples.
    """
    intermediate_steps = []
    for percentage in range(0, 101, 10):
        # percentage=0 means the very start (step index N-1, pure noise)
        # percentage=100 means the very end (step index 0, clean data)
        step_index = int((number_of_steps - 1) * (1.0 - percentage / 100.0))
        intermediate_steps.append(step_index)

    # Remove duplicates while preserving order
    seen = set()
    unique_steps = []
    for step in intermediate_steps:
        if step not in seen:
            seen.add(step)
            unique_steps.append(step)
    return unique_steps
