# BATDiff: Bivariate À Trous Wavelet Diffusion for Single Image Super-Resolution

Official research implementation of **BATDiff**, a diffusion-based single-image super-resolution framework.

---

## Overview

BATDiff improves high-frequency reconstruction by modelling cross-scale dependencies using à trous wavelet decomposition and diffusion.

---

## Method

- Undecimated à trous wavelet transform  
- Coarse-to-fine reconstruction  
- Bivariate cross-scale conditioning  
- LR-consistency during inference  

---

## Repository Structure

BATDiff/
├── BATDiff/  
├── clip/  
├── text2live_util/  
├── main.py  
├── requirements.txt  
└── README.md  

---

## Installation

git clone https://github.com/MaryamHeidari-1994/BATDiff.git  
cd BATDiff  
pip install -r requirements.txt  

---

## Usage

Train:

python main.py --mode train  

Example:

python main.py \
  --mode train \
  --dataset_folder ./data/Urban25lr/ \
  --image_name lr.png \
  --results_folder ./results/Urban25lr \
  --use_atrous \
  --atrous_wavelet b3 \
  --atrous_level 6 \
  --sr_factor 8  

---

## Notes

This code is provided for research purposes and may require additional configuration.

---

## Author

Maryam Heidari  
University of Bristol  

---

## Citation

@article{heidari2026batdiff,
  title={BATDiff: Bivariate A Trous Wavelet Diffusion for Single Image Super-Resolution},
  author={Heidari, Maryam and Anantrasirichai, Nantheera and Achim, Alin},
  year={2026},
  journal={arXiv preprint arXiv:2603.07234}
}
