"""
DDIM (Denoising Diffusion Implicit Models) sampler.

Implements the sampling algorithm from Song et al. 2020:
    "Denoising Diffusion Implicit Models" (arXiv:2010.02502)

Uses the same trained noise prediction network ε_θ as DDPM, but with a
deterministic (or partially stochastic) sampling formula that supports
far fewer sampling steps (e.g., 50 instead of 1000).

Key formula (Song et al. 2020, Eq. 12):
    x_{τ_{i-1}} = √ᾱ_{τ_{i-1}} · x̂₀
                 + √(1 - ᾱ_{τ_{i-1}} - σ²) · ε_θ
                 + σ · z

where:
    x̂₀ = (x_{τ_i} - √(1 - ᾱ_{τ_i}) · ε_θ) / √ᾱ_{τ_i}
    σ_{τ_i} = η · √((1 - ᾱ_{τ_{i-1}}) / (1 - ᾱ_{τ_i})) · √(1 - ᾱ_{τ_i} / ᾱ_{τ_{i-1}})
    η = 0 → fully deterministic (pure DDIM)
    η = 1 → matches DDPM posterior variance
"""
import torch
import torch.nn as nn
from tqdm import tqdm

from models.ddpm import DDPM


class DDIMSampler(nn.Module):
    """
    DDIM sampler that reuses a trained DDPM's noise prediction network.

    Reads the DDPM's precomputed ᾱ_t buffers directly (no duplication).
    Supports configurable number of sampling steps and stochasticity (η).

    Args:
        ddpm (DDPM): Trained DDPM instance whose alphas_cumprod buffers are used.
        ddim_timesteps (int): Number of DDIM sampling steps (S << T). Default: 50.
        eta (float): Stochasticity parameter.
                     0.0 = fully deterministic (pure DDIM).
                     1.0 = matches DDPM posterior variance.
                     Default: 0.0.
    """

    def __init__(self, ddpm: DDPM, ddim_timesteps: int = 50, eta: float = 0.0):
        super().__init__()
        self.ddpm = ddpm
        self.ddim_timesteps = ddim_timesteps
        self.eta = eta

        # Compute uniformly-spaced subsequence of DDPM timesteps
        timestep_sequence = self._make_ddim_timestep_sequence(
            ddpm_timesteps=ddpm.timesteps,
            ddim_timesteps=ddim_timesteps,
        )
        self.register_buffer('timestep_sequence', timestep_sequence)

    @staticmethod
    def _make_ddim_timestep_sequence(ddpm_timesteps: int, ddim_timesteps: int) -> torch.Tensor:
        """
        Create a uniformly-spaced subsequence of DDPM timesteps for DDIM sampling.

        Selects ddim_timesteps evenly spaced indices from [0, ddpm_timesteps).
        For example, with ddpm_timesteps=1000 and ddim_timesteps=50:
            [0, 20, 40, ..., 960, 980]

        Args:
            ddpm_timesteps: Total number of DDPM diffusion timesteps (T).
            ddim_timesteps: Number of DDIM sampling steps (S).

        Returns:
            Long tensor of shape (ddim_timesteps,) with selected timestep indices.
        """
        step_size = ddpm_timesteps // ddim_timesteps
        timestep_sequence = torch.arange(0, ddpm_timesteps, step_size, dtype=torch.long)
        return timestep_sequence

    def _compute_sigma(
        self,
        alpha_cumprod_current: torch.Tensor,
        alpha_cumprod_previous: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the noise coefficient σ for a DDIM step.

        Formula (Song et al. 2020):
            σ_{τ_i} = η · √((1 - ᾱ_{τ_{i-1}}) / (1 - ᾱ_{τ_i}))
                        · √(1 - ᾱ_{τ_i} / ᾱ_{τ_{i-1}})

        When η = 0, σ = 0 and sampling is fully deterministic.
        When η = 1, σ matches the DDPM posterior variance.

        Args:
            alpha_cumprod_current: ᾱ_{τ_i} at the current timestep.
            alpha_cumprod_previous: ᾱ_{τ_{i-1}} at the target (previous) timestep.

        Returns:
            Noise coefficient σ (same shape as inputs).
        """
        # σ = η · √((1 - ᾱ_{τ_{i-1}}) / (1 - ᾱ_{τ_i})) · √(1 - ᾱ_{τ_i} / ᾱ_{τ_{i-1}})
        sigma = (
            self.eta
            * torch.sqrt((1.0 - alpha_cumprod_previous) / (1.0 - alpha_cumprod_current))
            * torch.sqrt(1.0 - alpha_cumprod_current / alpha_cumprod_previous)
        )
        return sigma

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        x_current: torch.Tensor,
        timestep_current: int,
        timestep_previous: int,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Single DDIM reverse step: from x_{τ_i} to x_{τ_{i-1}}.

        Algorithm (Song et al. 2020, Eq. 12):
            1. Predict noise: ε_θ(x_{τ_i}, τ_i)
            2. Reconstruct x̂₀ = (x_{τ_i} - √(1 - ᾱ_{τ_i}) · ε_θ) / √ᾱ_{τ_i}
            3. Optionally clip x̂₀ to [-1, 1] for numerical stability
               (appropriate for pixel space; disable for latent diffusion)
            4. Compute σ using η parameter
            5. Compute direction pointing to x_t:
                 direction = √(1 - ᾱ_{τ_{i-1}} - σ²) · ε_θ
            6. Combine:
                 x_{τ_{i-1}} = √ᾱ_{τ_{i-1}} · x̂₀ + direction + σ · z

        Args:
            model: Noise prediction network ε_θ(x_t, t).
            x_current: Current noisy sample x_{τ_i}, shape (B, C, H, W).
            timestep_current: Current timestep index τ_i.
            timestep_previous: Target timestep index τ_{i-1}.
                              Use -1 as sentinel for the final step to the clean image.
            clip_denoised: If True, clip predicted x̂₀ to [-1, 1]. Set to True for
                pixel-space diffusion, False for latent-space diffusion where
                latent values are not bounded to [-1, 1].

        Returns:
            Denoised sample x_{τ_{i-1}}, same shape as x_current.
        """
        batch_size = x_current.shape[0]
        device = x_current.device

        # Create batch timestep tensor for the model
        timestep_batch = torch.full(
            (batch_size,), timestep_current, device=device, dtype=torch.long,
        )

        # Step 1: Predict noise ε_θ(x_{τ_i}, τ_i)
        predicted_noise = model(x_current, timestep_batch)

        # Look up ᾱ_{τ_i} for the current timestep
        alpha_cumprod_current = self.ddpm.alphas_cumprod[timestep_current]

        # Look up ᾱ_{τ_{i-1}} for the previous timestep
        # When timestep_previous == -1 (sentinel), we are stepping to the clean image:
        # ᾱ_{-1} = 1.0 (no noise remaining)
        if timestep_previous >= 0:
            alpha_cumprod_previous = self.ddpm.alphas_cumprod[timestep_previous]
        else:
            alpha_cumprod_previous = torch.tensor(1.0, device=device)

        # Step 2: Reconstruct predicted x̂₀ from noise prediction
        # x̂₀ = (x_{τ_i} - √(1 - ᾱ_{τ_i}) · ε_θ) / √ᾱ_{τ_i}
        predicted_original_sample = (
            x_current - torch.sqrt(1.0 - alpha_cumprod_current) * predicted_noise
        ) / torch.sqrt(alpha_cumprod_current)

        # Step 3: Clip x̂₀ to [-1, 1] for numerical stability in pixel space.
        # Disabled for latent-space diffusion where latents are unbounded.
        if clip_denoised:
            predicted_original_sample = predicted_original_sample.clamp(-1.0, 1.0)

        # Step 4: Compute σ using η parameter
        # σ = η · √((1 - ᾱ_{τ_{i-1}}) / (1 - ᾱ_{τ_i})) · √(1 - ᾱ_{τ_i} / ᾱ_{τ_{i-1}})
        sigma = self._compute_sigma(alpha_cumprod_current, alpha_cumprod_previous)

        # Step 5: Compute "direction pointing to x_t" (deterministic component)
        # direction = √(1 - ᾱ_{τ_{i-1}} - σ²) · ε_θ
        direction_to_xt = torch.sqrt(1.0 - alpha_cumprod_previous - sigma ** 2) * predicted_noise

        # Step 6: Combine predicted x̂₀, direction, and optional noise
        # x_{τ_{i-1}} = √ᾱ_{τ_{i-1}} · x̂₀ + direction + σ · z
        x_previous = (
            torch.sqrt(alpha_cumprod_previous) * predicted_original_sample
            + direction_to_xt
        )

        # Add stochastic noise only when σ > 0 (η > 0) and not at the final step
        if sigma > 0 and timestep_previous >= 0:
            noise = torch.randn_like(x_current)
            x_previous = x_previous + sigma * noise

        return x_previous

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        return_intermediates: bool = False,
        intermediate_steps: list = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Full DDIM reverse diffusion: generate images from pure noise.

        Algorithm:
            1. x_{τ_S} ~ N(0, I)       (start from pure noise)
            2. for i = S, S-1, ..., 1:
                   x_{τ_{i-1}} = ddim_sample(x_{τ_i}, τ_i, τ_{i-1})
            3. return x_0

        Uses ddim_timesteps steps instead of the full T DDPM steps.

        Args:
            model: Noise prediction network ε_θ(x_t, t).
            shape: Shape of samples to generate, e.g. (batch, channels, height, width).
            return_intermediates: If True, also return intermediate samples.
            intermediate_steps: Timestep indices at which to save intermediates.
                               Defaults to evenly-spaced steps through the DDIM sequence.
            clip_denoised: If True, clip predicted x̂₀ to [-1, 1] at each reverse step.
                Set to False for latent-space diffusion.

        Returns:
            Generated samples tensor of the requested shape.
            If return_intermediates=True, returns (samples, intermediates) where
            intermediates is a list of (timestep, tensor) tuples.
        """
        # Use the DDPM's device (which is already on the correct device after .to())
        # rather than self.timestep_sequence.device, because the DDIMSampler's own
        # buffer may still be on CPU if .to(device) wasn't called on the sampler.
        device = self.ddpm.betas.device
        batch_size = shape[0]

        # Build the reversed sequence of timestep pairs: (current, previous)
        # E.g., for sequence [0, 20, 40, ..., 980]:
        #   reversed: [980, 960, ..., 20, 0]
        #   pairs: (980, 960), (960, 940), ..., (20, 0), (0, -1)
        # The sentinel -1 means "step to clean image" (ᾱ_{-1} = 1.0)
        reversed_sequence = torch.flip(self.timestep_sequence.to(device), [0])

        # Default intermediate steps: evenly spaced through the DDIM sequence
        if intermediate_steps is None:
            num_intermediates = min(7, len(reversed_sequence))
            intermediate_indices = torch.linspace(
                0, len(reversed_sequence) - 1, num_intermediates, dtype=torch.long,
            )
            intermediate_steps = [reversed_sequence[idx].item() for idx in intermediate_indices]

        # Start from pure noise: x_{τ_S} ~ N(0, I)
        image = torch.randn(shape, device=device)
        intermediates = []

        for step_index in tqdm(
            range(len(reversed_sequence)),
            desc=f'DDIM Sampling ({self.ddim_timesteps} steps)',
            total=len(reversed_sequence),
        ):
            timestep_current = reversed_sequence[step_index].item()

            # Previous timestep is the next element in reversed sequence, or -1 at the end
            if step_index < len(reversed_sequence) - 1:
                timestep_previous = reversed_sequence[step_index + 1].item()
            else:
                timestep_previous = -1  # Sentinel: stepping to clean image

            image = self.ddim_sample(
                model, image, timestep_current, timestep_previous,
                clip_denoised=clip_denoised,
            )

            if return_intermediates and timestep_current in intermediate_steps:
                intermediates.append((timestep_current, image.clone()))

        # Always include the final result at timestep 0 if recording intermediates
        if return_intermediates and (not intermediates or intermediates[-1][0] != 0):
            intermediates.append((0, image.clone()))

        if return_intermediates:
            return image, intermediates
        return image
