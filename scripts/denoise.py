import os
import argparse
import cv2
import numpy as np
from skimage.restoration import denoise_wavelet


def enhance_denoise(img_bgr):
    # Convert to RGB for wavelet processing
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Step 1: Multi-scale Wavelet thresholding (preserves edges, removes grain)
    img_float = img_rgb.astype(np.float32) / 255.0
    denoised_wav = denoise_wavelet(
        img_float,
        channel_axis=-1,
        wavelet='db2',
        mode='soft',
        method='BayesShrink',
        rescale_sigma=True
    )
    denoised_wav = np.clip(denoised_wav * 255.0, 0, 255).astype(np.uint8)

    # Step 2: Chrominance-targeted smoothing in YCrCb space
    ycrcb = cv2.cvtColor(denoised_wav, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # Mild bilateral on Luminance (Y) to keep edges sharp
    y_clean = cv2.bilateralFilter(y, d=5, sigmaColor=25, sigmaSpace=25)

    # Stronger Gaussian/Median on Chroma (Cr, Cb) to eliminate blotchy color noise
    cr_clean = cv2.medianBlur(cr, 3)
    cb_clean = cv2.medianBlur(cb, 3)

    merged = cv2.merge([y_clean, cr_clean, cb_clean])
    result_bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    return result_bgr


def process_directory(noise_dir, denoised_dir):
    os.makedirs(denoised_dir, exist_ok=True)
    images = [f for f in os.listdir(noise_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for fname in images:
        in_path = os.path.join(noise_dir, fname)
        img = cv2.imread(in_path)
        if img is None:
            continue

        out = enhance_denoise(img)

        # Strip '_noise' per handbook specifications (e.g. 001_noise.png -> 001.png)
        out_name = fname.replace("_noise", "")
        cv2.imwrite(os.path.join(denoised_dir, out_name), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_dir", required=True)
    parser.add_argument("--denoised_dir", required=True)
    args = parser.parse_args()
    process_directory(args.noise_dir, args.denoised_dir)