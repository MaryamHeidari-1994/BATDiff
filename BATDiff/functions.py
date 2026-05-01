
import torch
from skimage import morphology, filters
from inspect import isfunction
import numpy as np
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
from scipy.ndimage import convolve1d
import os


try:
    import pywt
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

def se(x, level=1):
    x = x.astype(np.float32)
    H, W = x.shape
    out = x.copy()

    for _ in range(level):
        small = np.array(
            Image.fromarray(out).resize((max(1, W // 2), max(1, H // 2)), Image.BILINEAR),
            dtype=np.float32
        )
        out = np.array(
            Image.fromarray(small).resize((W, H), Image.BILINEAR),
            dtype=np.float32
        )

    return out


def dilate_mask(mask, mode):
    if mode == "harmonization":
        element = morphology.disk(radius=7)
    if mode == "editing":
        element = morphology.disk(radius=20)
    mask = mask.permute((1, 2, 0))
    mask = mask[:, :, 0]
    mask = morphology.binary_dilation(mask, selem=element)
    mask = filters.gaussian(mask, sigma=5)
    mask = mask[:, :, None, None]
    mask = mask.transpose(3, 2, 0, 1)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


# for roi_sampling

def stat_from_bbs(image, bb):
    y_bb, x_bb, h_bb, w_bb = bb
    bb_mean = torch.mean(image[:, :,y_bb:y_bb+h_bb, x_bb:x_bb+w_bb], dim=(2,3), keepdim=True)
    bb_std = torch.std(image[:, :, y_bb:y_bb+h_bb, x_bb:x_bb+w_bb], dim=(2,3), keepdim=True)
    return [bb_mean, bb_std]


def extract_patch(image, bb):
    y_bb, x_bb, h_bb, w_bb = bb
    image_patch = image[:, :,y_bb:y_bb+h_bb, x_bb:x_bb+w_bb]
    return image_patch

def lr_consistency_step(x, lr, eta=0.3, mode='bicubic'):
    if lr is None:
        return x

    lr_size = lr.shape[-2:]
    x_down = F.interpolate(x, size=lr_size, mode=mode, align_corners=False)
    diff = x_down - lr
    diff_up = F.interpolate(diff, size=x.shape[-2:], mode=mode, align_corners=False)

    x = x - eta * diff_up
    return x.clamp(-1., 1.)
# for clip sampling
def thresholded_grad(grad, quantile=0.8):
    """
    Receives the calculated CLIP gradients and outputs the soft-tresholded gradients based on the given quantization.
    Also outputs the mask that corresponds to remaining gradients positions.
    """
    grad_energy = torch.norm(grad, dim=1)
    grad_energy_reshape = torch.reshape(grad_energy, (grad_energy.shape[0],-1))
    enery_quant = torch.quantile(grad_energy_reshape, q=quantile, dim=1, interpolation='nearest')[:,None,None] #[batch ,1 ,1]
    gead_energy_minus_energy_quant = grad_energy - enery_quant
    grad_mask = (gead_energy_minus_energy_quant > 0)[:,None,:,:]

    gead_energy_minus_energy_quant_clamp = torch.clamp(gead_energy_minus_energy_quant, min=0)[:,None,:,:]#[b,1,h,w]
    unit_grad_energy = grad / grad_energy[:,None,:,:] #[b,c,h,w]
    unit_grad_energy[torch.isnan(unit_grad_energy)] = 0
    sparse_grad = gead_energy_minus_energy_quant_clamp * unit_grad_energy #[b,c,h,w]
    return sparse_grad, grad_mask

# helper functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def loss_backwards(fp16, loss, optimizer, model=None, **kwargs):
    """Backward pass supporting Apex mixed precision with safe gradient clipping."""
    if fp16 and APEX_AVAILABLE:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    else:
        loss.backward(**kwargs)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()



def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

def _atrous_b3_kernel_1d(level: int):
    base = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    if level <= 1:
        return base
    step = 2 ** (level - 1) - 1
    k = np.zeros((len(base) - 1) * (step + 1) + 1, dtype=np.float32)
    k[::step + 1] = base
    return k
    
def atrous_b3_lowpass_2d(img: np.ndarray, level: int):
    k = _atrous_b3_kernel_1d(level)
    out = convolve1d(img, k, axis=0, mode='mirror')
    out = convolve1d(out, k, axis=1, mode='mirror')
    return out


def atrous_lp_level1(img_np):
  
    img_np = np.clip(img_np.astype(np.float32), 0.0, 1.0)
    H, W, C = img_np.shape
    out = np.zeros_like(img_np, dtype=np.float32)

    for ch in range(C):
        out[:, :, ch] = atrous_b3_lowpass_2d(img_np[:, :, ch], level=1).astype(np.float32)

    return out


def build_atrous_decomposition_b3(img_np, levels=6):
  
    img_np = np.clip(img_np.astype(np.float32), 0.0, 1.0)

    if levels < 1:
        raise ValueError("levels must be >= 1")

    smooths = [img_np.copy()]  

    for _ in range(levels):
        nxt = atrous_lp_level1(smooths[-1])
        smooths.append(nxt)

    details = []
    for s in range(1, levels + 1):
        w_s = (smooths[s - 1] - smooths[s]).astype(np.float32)
        details.append(w_s)

    return smooths, details


def wavelet_plane_to_display(x_ref, w_plane, gain=2.0):
  
    x_ref = np.clip(x_ref.astype(np.float32), 0.0, 1.0)
    w_plane = w_plane.astype(np.float32)

    out = np.zeros_like(w_plane, dtype=np.float32)

    for ch in range(w_plane.shape[2]):
        wc = w_plane[:, :, ch]
        scale = np.percentile(np.abs(wc), 99.0) + 1e-8
        wc = wc / scale
        out[:, :, ch] = np.clip(x_ref[:, :, ch] + 0.22 * gain * wc, 0.0, 1.0)

    return out

def reconstruct_exact_from_smooth(smooth_cur, plane, detail_gain=1.0):

    rec = smooth_cur + detail_gain * plane
    return np.clip(rec, 0.0, 1.0).astype(np.float32)
def create_img_scales(foldername,
                      filename,
                      scale_factor=2,
                      image_size=None,
                      create=False,
                      auto_scale=None,
                      atrous_level=6,
                      sr_factor=8,
                      detail_gain_create=1.0,
                      coarse_base_gain=1.0,
                      recon_blend=0.0,
                      hf_boost=1.0,
                      keep_same_as_input=True):

    img_path = os.path.join(foldername, filename)
    lr_image = Image.open(img_path).convert("RGB")
    save_name = filename.rsplit(".", 1)[0] + ".png"

    if image_size is None:
        if keep_same_as_input:
            image_size = lr_image.size
        else:
            image_size = (lr_image.size[0] * sr_factor,
                          lr_image.size[1] * sr_factor)

    if auto_scale is not None:
        scaler = np.sqrt((image_size[0] * image_size[1]) / auto_scale)
        if scaler > 1:
            image_size = (
                max(1, int(image_size[0] / scaler)),
                max(1, int(image_size[1] / scaler))
            )

    L = int(atrous_level)
    if L < 1:
        raise ValueError("atrous_level must be >= 1")

 
    n_scales = L
    sizes = [image_size for _ in range(n_scales)]
    eff_scale_factor = scale_factor

    x_ref_pil = lr_image.resize(image_size, Image.BICUBIC)
    x_ref = np.asarray(x_ref_pil).astype(np.float32) / 255.0

  
    smooths, details = build_atrous_decomposition_b3(x_ref, levels=L)

  
    planes_coarse_to_fine = list(reversed(details))

   
    smooths_coarse_to_fine = list(reversed(smooths[1:]))  


    stage_recons = []
    for s in range(L):
        plane = detail_gain_create * hf_boost * planes_coarse_to_fine[s]
        smooth_cur = smooths_coarse_to_fine[s]
        rec = reconstruct_exact_from_smooth(smooth_cur, plane, detail_gain=1.0)
        stage_recons.append(rec)

    rescale_losses = []
    prev = smooths[-1].copy()  # cL
    for s in range(L):
        cur = stage_recons[s]
        mse = float(np.mean((cur - prev) ** 2))
        rescale_losses.append(max(mse, 1e-8))
        prev = cur

    if create:
        
        for s in range(L):
            path_scale = os.path.join(foldername, f"scale_{s}")
            Path(path_scale).mkdir(parents=True, exist_ok=True)

            plane = detail_gain_create * hf_boost * planes_coarse_to_fine[s]
            plane_vis = wavelet_plane_to_display(x_ref, plane, gain=2.0)

            plane_img = Image.fromarray((np.clip(plane_vis, 0.0, 1.0) * 255).astype(np.uint8))
            plane_img.save(os.path.join(path_scale, save_name))

      
        for s in range(L - 1):
            plane = detail_gain_create * hf_boost * planes_coarse_to_fine[s]
            smooth_cur = smooths_coarse_to_fine[s]
            rec = reconstruct_exact_from_smooth(smooth_cur, plane, detail_gain=1.0)

            path_recon = os.path.join(foldername, f"scale_{s+1}_recon")
            Path(path_recon).mkdir(parents=True, exist_ok=True)

            recon_img = Image.fromarray((np.clip(rec, 0.0, 1.0) * 255).astype(np.uint8))
            recon_img.save(os.path.join(path_recon, save_name))

    return sizes, rescale_losses, eff_scale_factor, n_scales