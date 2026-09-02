import os
import argparse
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


def compute_metrics(gt_img, test_img):
    """Compute PSNR and multichannel SSIM on RGB images [0, 255]."""
    cur_psnr = psnr_fn(gt_img, test_img, data_range=255)
    cur_ssim = ssim_fn(gt_img, test_img, channel_axis=2, data_range=255)
    return cur_psnr, cur_ssim


def evaluate(gt_dir, noisy_dir, denoised_dir):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    psnr_noisy_list, ssim_noisy_list = [], []
    psnr_denoised_list, ssim_denoised_list = [], []

    print(f"Found {len(gt_files)} ground truth images for evaluation.")

    for gt_name in gt_files:
        base_id = os.path.splitext(gt_name)[0]

        # Ground truth path
        gt_path = os.path.join(gt_dir, gt_name)
        gt_img = cv2.imread(gt_path)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        # Corresponding noisy file (e.g., 001_noise.png)
        noisy_name = f"{base_id}_noise.png"
        noisy_path = os.path.join(noisy_dir, noisy_name)
        if not os.path.exists(noisy_path):
            # Fallback in case noisy file uses the base name directly
            noisy_path = os.path.join(noisy_dir, gt_name)

        noisy_img = cv2.imread(noisy_path)
        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)

        # Denoised output file (e.g., 001.png)
        denoised_path = os.path.join(denoised_dir, gt_name)
        if not os.path.exists(denoised_path):
            print(f"Warning: Missing denoised prediction for {gt_name}. Skipping.")
            continue

        denoised_img = cv2.imread(denoised_path)
        denoised_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)

        # Compute metrics
        p_noisy, s_noisy = compute_metrics(gt_img, noisy_img)
        p_denoised, s_denoised = compute_metrics(gt_img, denoised_img)

        psnr_noisy_list.append(p_noisy)
        ssim_noisy_list.append(s_noisy)
        psnr_denoised_list.append(p_denoised)
        ssim_denoised_list.append(s_denoised)

    # Average metrics
    avg_psnr_noisy = np.mean(psnr_noisy_list)
    avg_ssim_noisy = np.mean(ssim_noisy_list)
    avg_psnr_denoised = np.mean(psnr_denoised_list)
    avg_ssim_denoised = np.mean(ssim_denoised_list)

    # Delta calculations
    delta_psnr = avg_psnr_denoised - avg_psnr_noisy
    delta_ssim = avg_ssim_denoised - avg_ssim_noisy

    # Mora SP Cup scoring equation
    N = np.clip(delta_psnr / 15.0, 0.0, 1.0)
    S = max(delta_ssim, 0.0)
    eval_score = 0.6 * N + 0.4 * S

    print("-" * 50)
    print(f"Noisy PSNR:     {avg_psnr_noisy:.4f} dB | Noisy SSIM:     {avg_ssim_noisy:.4f}")
    print(f"Denoised PSNR:  {avg_psnr_denoised:.4f} dB | Denoised SSIM:  {avg_ssim_denoised:.4f}")
    print(f"Delta PSNR:     {delta_psnr:.4f} dB")
    print(f"Delta SSIM:     {delta_ssim:.4f}")
    print(f"N (PSNR term):  {N:.4f}")
    print(f"S (SSIM term):  {S:.4f}")
    print(f"Evaluation Composite Score: {eval_score:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mora SP Cup 2026 Image Evaluation")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to ground truth folder")
    parser.add_argument("--noisy_dir", type=str, required=True, help="Path to noisy images folder")
    parser.add_argument("--denoised_dir", type=str, required=True, help="Path to denoised images folder")
    args = parser.parse_args()

    evaluate(args.gt_dir, args.noisy_dir, args.denoised_dir)