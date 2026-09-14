# Zentrix - Mora SP Cup 2026 Submission Scripts

This directory contains the complete source code, dataset processing, neural architectures, training routines, and automated packaging tools developed by **Team Zentrix** for the **Mora SP Cup 2026: Low-Light Image Denoising & Enhancement Challenge**.

---

> ### Important — Submission Freeze
>
> After the preliminary-round submission deadline, no new commits may be made to the submitted private GitHub repository.
>
> The Git commit SHA stated in this README and recorded by the organizers will be treated as the official submitted code version.
>
> If external model weights/checkpoints are used, their Google Drive link, expected local path, and SHA-256 checksum must also be stated here. The submitted model/checkpoint must not be modified or replaced after the deadline.

---

## 1. Architecture & Methodology Overview

Our solution utilizes a **Hybrid Multi-Stage Restoration Architecture**:
1. **Adaptive Defect-Pixel Outlier Suppression**: Dynamic local rank and median deviation filter that identifies and corrects impulse/dark-current hot pixels without blurring genuine edge structures.
2. **Multi-scale Wavelet Shrinkage**: Wavelet sub-band decomposition (`db2` / `bior2.2`) with sub-band noise estimation ($\hat{\sigma} = \frac{\text{median}(|HH|)}{0.6745}$) and soft BayesShrink thresholding to strip Poisson-Gaussian grain.
3. **YCrCb Color Space Decoupling**:
   - **Luminance (Y)**: Preserves edge gradients and fine details with light edge-preserving Bilateral filtering.
   - **Chrominance (Cr, Cb)**: Chromatic blotch suppression via median filtering and color bilateral filtering to eliminate sensor amplification discoloration without chrominance bleeding.
4. **Deep Learning Restoration (NAFNet - Nonlinear Activation Free Network)**:
   - A lightweight restoration backbone with SimpleGate, Simplified Channel Attention (SCA), and depthwise 3x3 convolutions (~2.03M parameters).
   - Trained with hybrid Charbonnier + SSIM loss.
   - **Automatic Fallback**: Runs with GPU acceleration when available; automatically and gracefully falls back to CPU or the pure Classical DSP engine if weights are not loaded.

---

## 2. Requirements & Installation

The solution is implemented in Python 3.10+ and operates 100% locally and offline without external internet calls during inference.

Install dependencies using:
```bash
pip install -r scripts/requirements.txt
```

Key dependencies:
- `torch>=2.0`
- `numpy>=1.22`
- `Pillow>=9.0`
- `opencv-python>=4.8`
- `scikit-image>=0.19`
- `PyWavelets>=1.4`

---

## 3. How to Run (Inference)

### Standard Command (as specified in Participant Handbook):
```bash
python scripts/denoise.py --noise_dir competition_data/submissions/noisy --denoised_dir competition_data/submissions/denoised
```

### Full Argument Options:
| Flag | Description | Default |
| --- | --- | --- |
| `--noise_dir` / `--input_dir` | Path to directory with noisy input images (`<id>_noise.png`) | *Required* |
| `--denoised_dir` / `--output_dir` | Path to output folder for restored images (`<id>.png`) | *Required* |
| `--model_path` | Path to PyTorch checkpoint (`best_denoiser.pth`) | `scripts/checkpoints/best_denoiser.pth` |
| `--cpu` | Force CPU inference execution | `False` |

---

## 4. Automated Submission Packaging & Validation

To denoise the 20 preliminary images (`461_noise.png` to `480_noise.png`), validate their dimensions ($992 \times 992$ RGB PNG), and package them into `Zentrix.zip`:

```bash
python scripts/run_submission.py
```

This ensures zero naming or dimension errors before uploading to the official submission Google Drive.

---

## 5. Model Training

To train the deep learning model on the public 460-image dataset:
```bash
python scripts/train.py --noisy_dir competition_data/public/noisy --gt_dir competition_data/public/ground_truth --epochs 50 --batch_size 8
```
A ready-to-run Google Colab / Kaggle notebook is also included at `Mora_SP_Cup_Colab_Training.ipynb` for GPU training.

---

## 6. Official Submission Information

* **Team Name:** Zentrix
* **Model Architecture:** NAFNet-Ultra (width=48, 4.6M params)
* **Model Checkpoint:** `models/best_denoiser_v4.pth`
* **Model SHA-256 Checksum:** `9aec3377266119fa7e52448ddd7ee68c31adf7028f32203867fb86adf3fbb06e`
* **Validation PSNR:** `30.4553 dB`
* **Validation SSIM:** `0.8659`
* **Official Composite Score:** `0.5858` (+122.5% over baseline)

