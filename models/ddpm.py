import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule from Nichol & Dhariwal 2021.
    Better for low-resolution images.

    Formula:
        ᾱ_t = f(t) / f(0)
        f(t) = cos²((t/T + s) / (1 + s) · π/2)
        β_t = 1 - ᾱ_t / ᾱ_{t-1}
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def extract(values: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Extract values at timestep t and reshape for broadcasting."""
    batch_size = t.shape[0]
    out = values.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1)))


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.

    Implements the forward (noising) and reverse (denoising) diffusion processes.
    """
    def __init__(self, timesteps: int = 1000):
        super().__init__()
        self.timesteps = timesteps

        betas = cosine_beta_schedule(timesteps)

        # α_t = 1 - β_t
        alphas = 1.0 - betas
        # ᾱ_t = ∏_{s=1}^{t} α_s
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # ᾱ_{t-1}
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (will move to device with model)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Precomputed values for q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # Precomputed values for reconstructing x₀ from noise prediction ε_θ:
        #   x₀ = √(1/ᾱ_t) · x_t  -  √(1/ᾱ_t - 1) · ε_θ
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # Posterior variance: β̃_t = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # Posterior mean coefficients (DDPM paper Eq. 7):
        #   μ̃_t(x_t, x₀) = (√ᾱ_{t-1} · β_t)/(1 - ᾱ_t) · x₀  +  (√α_t · (1 - ᾱ_{t-1}))/(1 - ᾱ_t) · x_t
        self.register_buffer(
            'posterior_mean_coeff_x0',
            torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            'posterior_mean_coeff_xt',
            torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Forward diffusion: sample x_t from q(x_t | x_0).

        Formula:
            x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε
            where ε ~ N(0, I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, model: nn.Module, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute simplified training loss.

        Formula:
            L = E_{t, x_0, ε}[||ε - ε_θ(x_t, t)||²]
        """
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = model(x_t, t)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """
        Reverse diffusion step: sample x_{t-1} from p_θ(x_{t-1} | x_t).

        Following the diffusers DDPMScheduler approach:
            1. Predict noise: ε_θ(x_t, t)
            2. Reconstruct x₀: x̂₀ = √(1/ᾱ_t) · x_t  -  √(1/ᾱ_t - 1) · ε_θ
            3. Clip x̂₀ to [-1, 1] for numerical stability
            4. Compute posterior mean (DDPM paper Eq. 7):
               μ̃_t = (√ᾱ_{t-1} · β_t)/(1 - ᾱ_t) · x̂₀  +  (√α_t · (1 - ᾱ_{t-1}))/(1 - ᾱ_t) · x_t
            5. Sample: x_{t-1} = μ̃_t + √β̃_t · z,  where z ~ N(0, I) for t > 0
        """
        # Predict noise ε_θ(x_t, t)
        predicted_noise = model(x_t, t)

        # Reconstruct x₀ from noise prediction:
        #   x̂₀ = (x_t - √(1 - ᾱ_t) · ε_θ) / √ᾱ_t
        #       = √(1/ᾱ_t) · x_t  -  √(1/ᾱ_t - 1) · ε_θ
        predicted_original_sample = (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * predicted_noise
        )

        # Clip predicted x₀ to [-1, 1] for numerical stability
        # (matching diffusers DDPMScheduler clip_sample=True, clip_sample_range=1.0)
        predicted_original_sample = predicted_original_sample.clamp(-1.0, 1.0)

        # Compute posterior mean using clipped x₀ (DDPM paper Eq. 7):
        #   μ̃_t = coeff_x0 · x̂₀ + coeff_xt · x_t
        model_mean = (
            extract(self.posterior_mean_coeff_x0, t, x_t.shape) * predicted_original_sample
            + extract(self.posterior_mean_coeff_xt, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        return_intermediates: bool = False,
        intermediate_steps: list = None,
    ) -> torch.Tensor:
        """
        Full reverse diffusion: generate images from pure noise.

        Algorithm:
            1. x_T ~ N(0, I)
            2. for t = T-1, ..., 0:
                   x_t = p_sample(x_{t+1}, t+1)
            3. return x_0
        """
        device = self.betas.device
        batch_size = shape[0]

        if intermediate_steps is None:
            intermediate_steps = [999, 900, 700, 500, 300, 100, 0]

        # Start from pure noise
        img = torch.randn(shape, device=device)
        intermediates = []

        for i in tqdm(reversed(range(self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)

            if return_intermediates and i in intermediate_steps:
                intermediates.append((i, img.clone()))

        if return_intermediates:
            return img, intermediates
        return img
