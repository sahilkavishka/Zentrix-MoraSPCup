# Zentrix - Mora SP Cup 2026: Low-Light Image Denoising & Enhancement

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Team Name:** Zentrix  
**Competition:** Mora SP Cup 2026  
**Objective:** End-to-end low-light image denoising and perceptual enhancement pipeline on mobile sensor captures.

---

## 🏆 Official Benchmark Performance

| Method / Architecture | Mean PSNR (dB) | Mean SSIM | $\Delta$PSNR (dB) | $\Delta$SSIM | Composite Score | Relative Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Noisy Raw Input** | 19.8257 | 0.4644 | 0.0000 | 0.0000 | 0.0000 | Baseline Reference |
| **Organizer Baseline (NLM)** | 24.3360 | 0.6717 | +4.5103 | +0.2073 | 0.2633 | Baseline |
| **Zentrix ULTRA Hybrid (NAFNet-v3)** | 29.0013 | 0.8817 | +9.1756 | +0.4173 | 0.5339 | +102.8% |
| **Zentrix ULTRA Fine-Tuned (NAFNet-v4)** | 30.4553 | 0.8659 | +10.6296 | +0.4015 | 0.5858 | +122.5% |
| **Zentrix ULTRA Grandmaster (v6)** | **30.8955** | **0.8908** | **+11.0698** | **+0.4264** | **0.6134** | **+132.9%** 🏆 |

$$\text{Composite Score} = 0.6 \cdot \text{clip}\left(\frac{\Delta\text{PSNR}}{15}, 0, 1\right) + 0.4 \cdot \max(\Delta\text{SSIM}, 0)$$

---

## 🔬 Architecture Highlights

1. **Adaptive Defect-Pixel Outlier Suppression**: Dynamic local rank deviation filter ($\tau = 0.18$) removing hot/leakage impulse sensor spikes without blurring true edges.
2. **Multi-Scale Wavelet Denoising**: 2-level 2D Discrete Wavelet Transform (`db2` basis) with MAD noise variance estimation and soft BayesShrinkage.
3. **YCrCb Decoupled Filtering**: Separates high-frequency luminance grain from wide-area chrominance blotches, preserving structural contrast.
4. **Deep Residual Backbone (NAFNet-v3)**: High-capacity architecture (`width=48`, ~4.6M parameters) using SimpleGate and Simplified Channel Attention (SCA).
5. **Frequency-Domain Loss (FFT)**: Trained with compound Charbonnier + FFT frequency loss to reconstruct fine textures and high-frequency structural details.

---

## 📁 Repository Structure

```
Zentrix-MoraSPCup/
├── baseline/                  # Organizer reference baseline code
├── competition_data/          # Dataset directory structure (public & submissions)
├── evaluation/                # Official evaluation scripts (evaluate.py)
├── models/                    # Model checkpoint directory
├── report/                    # Technical report LaTeX/ReportLab generator
├── scripts/
│   ├── models.py              # NAFNet architecture (v1 and v3)
│   ├── dataset.py             # PyTorch dataset with geometric augmentations
│   ├── denoise.py             # Main inference pipeline (100% offline)
│   ├── run_submission.py      # Automated generation, validation & zip packager
│   └── compare_with_baseline.py # Benchmark comparison script
├── Mora_SP_Cup_Colab_Training.ipynb # Complete GPU training notebook
└── README.md
```

---

## 🚀 Quick Start & Inference

### 1. Installation
```bash
pip install -r scripts/requirements.txt
```

### 2. Denoising Images
```bash
python scripts/denoise.py \
    --noise_dir competition_data/submissions/noisy \
    --denoised_dir competition_data/submissions/denoised \
    --tta 1 \
    --sharpness 0.20
```

### 3. Generate Submission Package
```bash
python scripts/run_submission.py --tta 1 --sharpness 0.20
```

---

## 📜 Authors & Citation
- **Team Zentrix** — Mora SP Cup 2026