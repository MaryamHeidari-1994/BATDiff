from BATDiff.functions import *  
import math

from torch import nn
from einops import rearrange
from functools import partial
import torch.nn.functional as F
from torchvision import utils
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import convolve1d
from BATDiff.functions import lr_consistency_step, atrous_b3_lowpass_2d


try:
    import pywt
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False


class EMA:
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new



class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _to_tensor(x_np: np.ndarray, device, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(x_np).to(device=device).type(dtype)


def _nearest_upsample_2d(img2d: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    Ht, Wt = target_shape
    Hs, Ws = img2d.shape

    if (Hs, Ws) == (Ht, Wt):
        return img2d

    y_idx = np.round(np.linspace(0, Hs - 1, Ht)).astype(int)
    x_idx = np.round(np.linspace(0, Ws - 1, Wt)).astype(int)

    return img2d[np.ix_(y_idx, x_idx)]
def atrous_decompose_batch(
    x: torch.Tensor,
    *,
    level: int = 8,
    wavelet: str = 'b3',
    scale_factor: int = 2,
    return_details: bool = False,
) -> tuple[torch.Tensor, dict] | torch.Tensor:
    b, c, h, w = x.shape
    device = x.device

    res = np.zeros((b, c, h, w), dtype=np.float32)
    x_np = _to_numpy(x)

    for bi in range(b):
        for ch in range(c):
            img = x_np[bi, ch].astype(np.float32)
            cA = atrous_b3_lowpass_2d(img, level=level)
            res[bi, ch] = cA.astype(np.float32)

    coarse = _to_tensor(res, device=device)

    if not return_details:
        return coarse

    detail = np.zeros((b, c, h, w), dtype=np.float32)

    for bi in range(b):
        for ch in range(c):
            img = x_np[bi, ch].astype(np.float32)
            c_prev = img if level == 1 else atrous_b3_lowpass_2d(img, level=level - 1)
            c_cur = atrous_b3_lowpass_2d(img, level=level)
            detail[bi, ch] = (c_prev - c_cur).astype(np.float32)

    details = {
        level: (_to_tensor(detail, device=device),)
    }

    return coarse, details

def atrous_reconstruct_batch(
    coarse: torch.Tensor,
    target_size: tuple[int, int],
    *,
    level: int = 1,
    wavelet: str = 'b3',
    scale_factor: int = 2,
    details: dict | None = None,
    details_gain: float = 1.0,
) -> torch.Tensor:
    b, c, hc, wc = coarse.shape
    Ht, Wt = target_size
    device = coarse.device

    res = np.zeros((b, c, Ht, Wt), dtype=np.float32)
    coarse_np = _to_numpy(coarse)
    details_np = None

    if details is not None and len(details) > 0 and (level in details):
        detail, = details[level]
        details_np = _to_numpy(detail)

    for bi in range(b):
        for ch in range(c):
            base = coarse_np[bi, ch].astype(np.float32)

            if base.shape != (Ht, Wt):
                base = np.array(_nearest_upsample_2d(base, (Ht, Wt)), dtype=np.float32)

            if details_np is None:
                res[bi, ch] = base
            else:
                detail_up = details_np[bi, ch]
                if detail_up.shape != (Ht, Wt):
                    detail_up = _nearest_upsample_2d(detail_up, (Ht, Wt))
                res[bi, ch] = (base + float(details_gain) * detail_up).astype(np.float32)

    return _to_tensor(res, device=device)


class BATDiffConvBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int | None = None, mult: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if exists(time_emb_dim) else None

        self.time_reshape = nn.Conv2d(time_emb_dim, dim, 1)
        self.ds_conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
      h = self.ds_conv(x)

      if exists(self.mlp):
        assert exists(time_emb), 'time emb must be passed in'
        condition = self.mlp(time_emb)
        condition = rearrange(condition, 'b c -> b c 1 1')
        condition = self.time_reshape(condition)
        h = h + condition

      h = self.net(h)
      return h + self.res_conv(x)


class BATDiffModelAtrousWaveletV2(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        *,
        channels: int = 3,
        with_time_emb: bool = True,
        multiscale: bool = False,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device
        self.channels = channels
        self.multiscale = multiscale

        if with_time_emb:
            time_dim = 32

            if multiscale:
                self.SinEmbTime = SinusoidalPosEmb(time_dim)
                self.SinEmbScale = SinusoidalPosEmb(time_dim)
                self.time_mlp = nn.Sequential(
                    nn.Linear(time_dim * 2, time_dim * 4),
                    nn.GELU(),
                    nn.Linear(time_dim * 4, time_dim)
                )
            else:
                self.time_mlp = nn.Sequential(
                    SinusoidalPosEmb(time_dim),
                    nn.Linear(time_dim, time_dim * 4),
                    nn.GELU(),
                    nn.Linear(time_dim * 4, time_dim)
                )
        else:
            time_dim = None
            self.time_mlp = None

        half_dim = int(dim / 2)

        self.l1 = BATDiffConvBlock(channels* 2, half_dim, time_emb_dim=time_dim)
        self.l2 = BATDiffConvBlock(half_dim, dim, time_emb_dim=time_dim)
        self.l3 = BATDiffConvBlock(dim, dim, time_emb_dim=time_dim)
        self.l4 = BATDiffConvBlock(dim, half_dim, time_emb_dim=time_dim)

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            nn.Conv2d(half_dim, out_dim, 1)
        )
        
    def forward(self, x: torch.Tensor, time: torch.Tensor, scale: int | None = None, parent: torch.Tensor | None = None) -> torch.Tensor:
      if parent is None:
          parent = torch.zeros_like(x)

      if parent.shape[-2:] != x.shape[-2:]:
          parent = F.interpolate(parent, size=x.shape[-2:], mode='bilinear', align_corners=False)

      x = torch.cat([x, parent], dim=1)
      if exists(self.multiscale):
          scale_tensor = torch.ones(size=time.shape, device=self.device) * scale
          t = self.SinEmbTime(time)
          s = self.SinEmbScale(scale_tensor)
          t_s_vec = torch.cat((t, s), dim=1)
          cond_vec = self.time_mlp(t_s_vec)
      else:
          t = self.time_mlp(time) if exists(self.time_mlp) else None
          cond_vec = t

      x = self.l1(x, cond_vec)
      x = self.l2(x, cond_vec)
      x = self.l3(x, cond_vec)
      x = self.l4(x, cond_vec)

      return self.final_conv(x)

class MultiScaleGaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        *,
        save_interm: bool = False,
        results_folder: str = '/Results',
        n_scales: int,
        scale_factor: int,
        image_sizes: list[tuple[int, int]],  # as (W,H) in original repo; we’ll store (H,W)
        scale_mul: tuple[float, float] = (1, 1),
        channels: int = 3,
        timesteps: int = 100,
        train_full_t: bool = False,
        scale_losses: list[float] | None = None,
        loss_factor: float = 1,
        loss_type: str = 'l1',
        betas: np.ndarray | torch.Tensor | None = None,
        device: torch.device | None = None,
        reblurring: bool = True,
        sample_limited_t: bool = False,
        omega: float = 0,
        # wavelet / atrous
        use_atrous: bool = False,
        atrous_level: int = 1,
        atrous_wavelet: str = 'b3',
        use_atrous_details: bool = False,
        details_gain: float = 1.0,
    ):
        super().__init__()
        self.lr_observation = None
        self.lr_eta = 0.3
        self.device = device
        self.save_interm = save_interm
        self.results_folder = Path(results_folder)
        self.channels = channels
        self.n_scales = n_scales
        self.scale_factor = scale_factor
        self.image_sizes: tuple[tuple[int, int], ...] = ()
        self.scale_mul = scale_mul

        self.sample_limited_t = sample_limited_t
        self.reblurring = reblurring

        self.img_prev_upsample = None

        
        self.use_atrous = use_atrous
        self.atrous_level = int(max(1, atrous_level))
        self.atrous_wavelet = atrous_wavelet
        self.use_atrous_details = bool(use_atrous_details)
        self.details_gain = float(details_gain)

       
        self._prev_details: dict[int, dict] = {}

        
        self.clip_guided_sampling = False
        self.guidance_sub_iters = None
        self.stop_guidance = None
        self.quantile = 0.8
        self.clip_model = None
        self.clip_strength = None
        self.clip_text = ''
        self.text_embedds = None
        self.text_embedds_hr = None
        self.text_embedds_lr = None
        self.clip_text_features = None
        self.clip_score = []
        self.clip_mask = None
        self.llambda = 0
        self.x_recon_prev = None

        
        self.clip_roi_bb = []

        
        self.omega = omega

       
        self.roi_guided_sampling = False
        self.roi_bbs = []            # list of [y,x,h,w]
        self.roi_bbs_stat = []       # list of [mean_tensor, std_tensor]
        self.roi_target_patch = []

        
        for i in range(n_scales):
            self.image_sizes += ((image_sizes[i][1], image_sizes[i][0]),)

        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.num_timesteps_trained = []
        self.num_timesteps_ideal = []
        self.num_timesteps_trained.append(self.num_timesteps)
        self.num_timesteps_ideal.append(self.num_timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

     
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

       
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        sigma_t = np.sqrt(1. - alphas_cumprod) / np.sqrt(alphas_cumprod)

        
        if scale_losses is not None:
            for i in range(n_scales - 1):
                self.num_timesteps_ideal.append(int(np.argmax(sigma_t > loss_factor * scale_losses[i])))
                if train_full_t:
                    self.num_timesteps_trained.append(int(timesteps))
                else:
                    self.num_timesteps_trained.append(self.num_timesteps_ideal[i + 1])

     
        gammas = torch.zeros(size=(n_scales - 1, self.num_timesteps), device=self.device)
        for i in range(n_scales - 1):
            gammas[i, :] = (torch.tensor(sigma_t, device=self.device) / (loss_factor * scale_losses[i])).clamp(min=0, max=1)
        self.register_buffer('gammas', gammas)

    def roi_patch_modification(self, x_recon, scale=0, eta=0.8):
        x_modified = x_recon
        for bb in self.roi_bbs:  # [y,x,h,w]
            bb = [int(bb_i / np.power(self.scale_factor, self.n_scales - scale - 1)) for bb_i in bb]
            bb_y, bb_x, bb_h, bb_w = bb
            target_patch_resize = F.interpolate(self.roi_target_patch[scale], size=(bb_h, bb_w))
            x_modified[:, :, bb_y:bb_y + bb_h, bb_x:bb_x + bb_w] = (
                eta * target_patch_resize + (1 - eta) * x_modified[:, :, bb_y:bb_y + bb_h, bb_x:bb_x + bb_w]
            )
        return x_modified


    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, s, noise):
        x_recon_ddpm = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

        if not self.reblurring or s == 0:
            return x_recon_ddpm, x_recon_ddpm
        else:
            cur_gammas = self.gammas[s - 1].reshape(-1).clamp(0, 0.55)
            x_tm1_mix = (x_recon_ddpm - extract(cur_gammas, t, x_recon_ddpm.shape) * self.img_prev_upsample) / (
                        1 - extract(cur_gammas, t, x_recon_ddpm.shape))
            x_t_mix = x_recon_ddpm
            return x_tm1_mix, x_t_mix

    def q_posterior(self, x_start, x_t_mix, x_t, t, s):
        if not self.reblurring or s == 0:
            posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
            )
            posterior_variance = extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        elif t[0] > 0:
            x_tm1_mix = x_start
            posterior_variance_low = torch.zeros(x_t.shape, device=self.device)
            posterior_variance_high = 1 - extract(self.alphas_cumprod, t - 1, x_t.shape)
            omega = self.omega
            posterior_variance = (1 - omega) * posterior_variance_low + omega * posterior_variance_high
            posterior_log_variance_clipped = torch.log(posterior_variance.clamp(1e-20, None))

            var_t = posterior_variance

            posterior_mean = extract(self.sqrt_alphas_cumprod, t - 1, x_t.shape) * x_tm1_mix + \
                             torch.sqrt(1 - extract(self.alphas_cumprod, t - 1, x_t.shape) - var_t) * \
                             (x_t - extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t_mix) / \
                             extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        else:
            posterior_mean = x_start
            posterior_variance = extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.enable_grad()
    def p_mean_variance(self, x, t, s, clip_denoised: bool, parent_t=None):
        if int(s) > 0:
            pred_noise = self.denoise_fn(x, t, scale=s, parent=parent_t)
        else:
            pred_noise = self.denoise_fn(x, t, scale=s, parent=None)

        x_recon, x_t_mix = self.predict_start_from_noise(x, t=t, s=s, noise=pred_noise)
        cur_gammas = self.gammas[s - 1].reshape(-1).clamp(0, 0.55) if int(s) > 0 else None

        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (x_recon.clamp(-1., 1.) + 1) * 0.5
            utils.save_image(final_img, str(final_results_folder / f'denoised_t-{t[0]:03}_s-{s}.png'), nrow=4)

        # (CLIP guidance and ROI guidance blocks unchanged)
        if self.clip_guided_sampling and (self.stop_guidance <= t[0] or s < self.n_scales - 1) and self.guidance_sub_iters[s] > 0:
            if clip_denoised:
                x_recon.clamp_(-1., 1.)

            if self.clip_mask is not None:
                x_recon = x_recon * (1 - self.clip_mask) + (
                        (1 - self.llambda) * self.x_recon_prev + self.llambda * x_recon) * self.clip_mask
            x_recon.requires_grad_(True)

            x_recon_renorm = (x_recon + 1) * 0.5
            for i in range(self.guidance_sub_iters[s]):
                self.clip_model.zero_grad()
                if s > 0:
                    score = -self.clip_model.calculate_clip_loss(x_recon_renorm, self.text_embedds_hr)
                else:
                    score = -self.clip_model.calculate_clip_loss(x_recon_renorm, self.text_embedds_lr)

                clip_grad = torch.autograd.grad(score, x_recon, create_graph=False)[0]

                if self.clip_mask is None:
                    clip_grad, clip_mask = thresholded_grad(grad=clip_grad, quantile=self.quantile)
                    self.clip_mask = clip_mask.float()

                if self.save_interm:
                    final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
                    final_results_folder.mkdir(parents=True, exist_ok=True)
                    final_mask = self.clip_mask.type(torch.float64)
                    utils.save_image(final_mask, str(final_results_folder / f'clip_mask_s-{s}.png'), nrow=4)
                    utils.save_image((x_recon.clamp(-1., 1.) + 1) * 0.5,
                                     str(final_results_folder / f'clip_out_s-{s}_t-{t[0]}_subiter_{i}.png'), nrow=4)

                division_norm = torch.linalg.vector_norm(x_recon * self.clip_mask, dim=(1, 2, 3), keepdim=True) / \
                                torch.linalg.vector_norm(clip_grad * self.clip_mask, dim=(1, 2, 3), keepdim=True)

                x_recon += self.clip_strength * division_norm * clip_grad * self.clip_mask
                x_recon.clamp_(-1., 1.)
                x_recon_renorm = (x_recon + 1) * 0.5
                self.clip_score.append(score.detach().cpu())

            self.x_recon_prev = x_recon.detach()
            plt.rcParams['figure.figsize'] = [16, 8]
            plt.plot(self.clip_score)
            plt.grid(True)
            plt.savefig(str(self.results_folder / 'clip_score'))
            plt.clf()

        elif self.roi_guided_sampling and (s < self.n_scales - 1):
            x_recon = self.roi_patch_modification(x_recon, scale=s)

        if int(s) > 0 and t[0] > 0 and self.reblurring:
            x_tm1_mix = extract(cur_gammas, t - 1, x_recon.shape) * self.img_prev_upsample + \
                        (1 - extract(cur_gammas, t - 1, x_recon.shape)) * x_recon
        else:
            x_tm1_mix = x_recon

        if clip_denoised:
            x_tm1_mix.clamp_(-1., 1.)
            x_t_mix.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_tm1_mix, x_t_mix=x_t_mix, x_t=x, t=t, s=s
        )
        return model_mean, posterior_variance, posterior_log_variance

  
    @torch.no_grad()
    def p_sample(self, x, t, s, clip_denoised=True, repeat_noise=False, parent_t=None):
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, s=s, clip_denoised=clip_denoised, parent_t=parent_t
        )
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        nonzero_mask_s = torch.tensor([True], device=self.device).float()

        out = model_mean + nonzero_mask_s * nonzero_mask * (0.5 * model_log_variance).exp() * noise

        if self.lr_observation is not None:
            out = lr_consistency_step(out, self.lr_observation, eta=self.lr_eta)

        return out
        

    @torch.no_grad()
    def p_sample_loop(self, shape, s):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img, str(final_results_folder / f'input_noise_s-{s}.png'), nrow=4)

        if self.sample_limited_t and s < (self.n_scales - 1):
            t_min = self.num_timesteps_ideal[s + 1]
        else:
            t_min = 0

        for i in tqdm(reversed(range(t_min, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t_cur = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t_cur, s, parent_t=None)

            if self.save_interm:
                final_img = (img + 1) * 0.5
                utils.save_image(final_img, str(final_results_folder / f'output_t-{i:03}_s-{s}.png'), nrow=4)

        return img

    @torch.no_grad()
    def sample(self, batch_size=16, scale_0_size=None, s=0):
        image_size = scale_0_size if scale_0_size is not None else self.image_sizes[0]
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size[0], image_size[1]), s=s)

    @torch.no_grad()
                  
    
    def p_sample_via_scale_loop(self, batch_size, img_child, img_parent, s, custom_t=None):
        device = self.betas.device
        if custom_t is None:
            total_t = self.num_timesteps_ideal[min(s, self.n_scales - 1)] - 1
        else:
            total_t = custom_t
        b = batch_size

        self.img_prev_upsample = img_parent

        parent_t = self.q_sample(
          x_start=img_parent,
          t=torch.full((batch_size,), total_t, device=device, dtype=torch.long),
          noise=None
        )

    
        
        if self.save_interm:
          final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
          final_results_folder.mkdir(parents=True, exist_ok=True)
          final_img = (img_child + 1) * 0.5
          utils.save_image(final_img, str(final_results_folder / f'clean_input_s_{s}.png'), nrow=4)
        img = self.q_sample(
          x_start=img_child,
          t=torch.full((batch_size,), total_t, device=device, dtype=torch.long),
          noise=None
        )

        if self.save_interm:
          final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
          final_results_folder.mkdir(parents=True, exist_ok=True)
          final_img = (img + 1) * 0.5
          utils.save_image(final_img, str(final_results_folder / f'noisy_input_s_{s}.png'), nrow=4)
           
        if self.clip_mask is not None:
            if s > 0:
                mul_size = [int(self.image_sizes[s][0] * self.scale_mul[0]), int(self.image_sizes[s][1] * self.scale_mul[1])]
                self.clip_mask = F.interpolate(self.clip_mask, size=mul_size, mode='bilinear')
                self.x_recon_prev = F.interpolate(self.x_recon_prev, size=mul_size, mode='bilinear')
            else:
                self.clip_mask = None

        if self.sample_limited_t and s < (self.n_scales - 1):
          t_min = self.num_timesteps_ideal[s + 1]
        else:
            t_min = 0
        for i in tqdm(reversed(range(t_min, total_t)), desc='sampling loop time step', total=total_t):
            t_cur = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t_cur, s, parent_t=parent_t)

            if i > 0:
              parent_t = self.p_sample(parent_t, t_cur, s - 1, parent_t=None)

            if self.save_interm:
              final_img = (img + 1) * 0.5
              utils.save_image(final_img, str(final_results_folder / f'output_t-{i:03}_s-{s}.png'), nrow=4)

        return img

    @torch.no_grad()
    def sample_via_scale(
        self,
        batch_size,
        img_child,
        img_parent,
        s,
        scale_mul=(1, 1),
        custom_sample=False,
        custom_img_size_idx=0,
        custom_t=None,
        custom_image_size=None,
    ):
       return self.p_sample_via_scale_loop(
        batch_size=batch_size,
        img_child=img_child,
        img_parent=img_parent,
        s=s,
        custom_t=custom_t
    )
    
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, s, noise=None, x_orig=None):
      b, c, h, w = x_start.shape
      noise = default(noise, lambda: torch.randn_like(x_start))
      if int(s) > 0:
        x_child0 = x_start
        x_parent0 = x_orig

        noise_child = default(noise, lambda: torch.randn_like(x_child0))
        noise_parent = torch.randn_like(x_parent0)

        x_child_t = self.q_sample(x_start=x_child0, t=t, noise=noise_child)
        x_parent_t = self.q_sample(x_start=x_parent0, t=t, noise=noise_parent)

        x_recon = self.denoise_fn(x_child_t, t, s, parent=x_parent_t)
        target_noise = noise_child
      else:
          x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
          x_recon = self.denoise_fn(x_noisy, t, s, parent=None)
          target_noise = noise

      if self.loss_type == 'l1':
        loss = (target_noise - x_recon).abs().mean()
      elif self.loss_type == 'l2':
        loss = F.mse_loss(target_noise, x_recon)
      elif self.loss_type == 'l1_pred_img':
          raise NotImplementedError("l1_pred_img is not compatible with the current parent-conditioned setup.")
      else:
          raise NotImplementedError()
      return loss

    def forward(self, x, s, *args, **kwargs):
        if int(s) > 0:
            x_child0 = x[0]
            x_parent0 = x[1]
            b, c, h, w = x_child0.shape
            device = x_child0.device
            img_size = self.image_sizes[s]
            assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
            t = torch.randint(0, self.num_timesteps_trained[s], (b,), device=device).long()
            return self.p_losses(x_child0, t, s, x_orig=x_parent0, *args, **kwargs)
        else:
            b, c, h, w = x[0].shape
            device = x[0].device
            img_size = self.image_sizes[s]
            assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
            t = torch.randint(0, self.num_timesteps_trained[s], (b,), device=device).long()
            return self.p_losses(x[0], t, s, *args, **kwargs)

    
    @torch.no_grad()
    def cache_prev_scale_details(self, prev_img: torch.Tensor, s_minus_1: int):
       
        if not self.use_atrous or not self.use_atrous_details or not _HAS_PYWT:
            return
        try:
            
            _, details = atrous_decompose_batch(
                prev_img, level=self.atrous_level, wavelet=self.atrous_wavelet,
                scale_factor=self.scale_factor, return_details=True
            )
            self._prev_details[s_minus_1] = details
        except Exception as e:
            print("[WARN] cache_prev_scale_details failed:", e)
            self._prev_details.pop(s_minus_1, None)
