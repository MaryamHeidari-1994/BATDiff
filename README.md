# BATDiff: Bivariate À Trous Wavelet Diffusion for Single Image Super-Resolution

Official research implementation of **BATDiff**, a diffusion-based single-image super-resolution framework that combines an undecimated à trous wavelet representation with bivariate cross-scale conditioning.

BATDiff is designed to improve high-frequency reconstruction by explicitly modelling parent–child dependencies across adjacent wavelet scales during the reverse diffusion process.

---

## Overview

Single-image super-resolution is an ill-posed inverse problem where many plausible high-resolution images may correspond to the same low-resolution observation. Existing diffusion-based SR methods can generate visually sharp results, but their high-frequency details may be weakly constrained by the LR input.

BATDiff addresses this issue through two core components:

1. **À trous wavelet decomposition**  
   An undecimated wavelet representation is used to construct spatially aligned multiscale components without destructive downsampling.

2. **Bivariate cross-scale diffusion**  
   During reverse diffusion, each finer-scale state is conditioned on its time-aligned parent state from the adjacent coarser scale.

This design encourages structurally consistent high-frequency recovery while maintaining compatibility with internal-learning single-image SR settings.

---

## Method

Given a low-resolution input image, BATDiff first constructs an HR-grid reference using bicubic upsampling. An à trous wavelet hierarchy is then applied to obtain progressively refined multiscale targets.

The model learns a shared diffusion denoiser across scales. At inference time, reconstruction proceeds from coarse to fine scales, where each scale is guided by the corresponding parent-scale estimate.

Key elements include:

- undecimated à trous wavelet transform;
- progressive coarse-to-fine reconstruction targets;
- shared DDPM-based denoising network;
- bivariate parent–child cross-scale conditioning;
- LR-consistency correction during inference.

---

## Repository Structure

```text
BATDiff/
├── BATDiff/                  # Core BATDiff implementation
├── clip/                     # CLIP-related modules
├── text2live_util/            # Auxiliary utilities
├── main.py                   # Main training and sampling entry point
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
## Installation
git clone https://github.com/MaryamHeidari-1994/BATDiff.git
cd BATDiff
pip install -r requirements.txt

##Usage
python main.py --mode train
Example with custom settings:
python main.py \
  --mode train \
  --dataset_folder ./data/Urban25lr/ \
  --image_name lr.png \
  --results_folder ./results/Urban25lr \
  --use_atrous \
  --atrous_wavelet b3 \
  --atrous_level 6 \
  --sr_factor 8
