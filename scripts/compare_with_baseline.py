import os
import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from evaluation.evaluate import evaluate_image, find_image_ids


def evaluate_directory(pred_dir: Path, gt_dir: Path, noisy_dir: Path):
    image_ids = find_image_ids(gt_dir)
    rows = []
    for img_id in image_ids:
        gt_path = gt_dir / f"{img_id}.png"
        noisy_path = noisy_dir / f"{img_id}_noise.png"
        if not noisy_path.exists():
            noisy_path = noisy_dir / f"{img_id}.png"
        pred_path = pred_dir / f"{img_id}.png"

        if not pred_path.exists() or not noisy_path.exists() or not gt_path.exists():
            continue

        rows.append(evaluate_image(img_id, pred_path, gt_path, noisy_path))

    if not rows:
        return None

    return {
        "count": len(rows),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "delta_psnr": float(np.mean([r["delta_psnr"] for r in rows])),
        "delta_ssim": float(np.mean([r["delta_ssim"] for r in rows])),
        "composite": float(np.mean([r["composite_score"] for r in rows])),
    }


def main():
    parser = argparse.ArgumentParser(description="Mora SP Cup Baseline vs Improved Model Benchmark")
    parser.add_argument("--noisy_dir", type=str, default="competition_data/public/noisy",
                        help="Path to folder with noisy input images")
    parser.add_argument("--gt_dir", type=str, default="competition_data/public/ground_truth",
                        help="Path to folder with ground-truth clean images")
    parser.add_argument("--model_path", type=str, default="models/best_denoiser.pth",
                        help="Path to trained improved model checkpoint")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Optionally limit number of images for fast check")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    args = parser.parse_args()

    noisy_path = Path(args.noisy_dir)
    gt_path = Path(args.gt_dir)

    # Check for model path fallbacks (check v2 first, then v1)
    model_file = Path(args.model_path)
    if not model_file.exists():
        candidates = [
            repo_root / "models" / "best_denoiser_v2.pth",
            repo_root / "scripts" / "checkpoints" / "best_denoiser_v2.pth",
            repo_root / "models" / "best_denoiser.pth",
            repo_root / "scripts" / "checkpoints" / "best_denoiser.pth",
        ]
        for cand in candidates:
            if cand.exists():
                model_file = cand
                break

    print("=" * 80)
    print("MORA SP CUP 2026: BASELINE VS IMPROVED MODEL COMPARISON")
    print("=" * 80)
    print(f"[*] Noisy Images Dir:   {noisy_path.resolve()}")
    print(f"[*] Ground Truth Dir:   {gt_path.resolve()}")
    print(f"[*] Improved Model:     {model_file.resolve() if model_file.exists() else 'NOT FOUND'}")
    print()

    if not gt_path.exists() or len(list(gt_path.glob("*.png"))) == 0:
        print("[!] NOTICE: Ground-truth directory is empty or not found.")
        print("    To compute quantitative PSNR, SSIM, and Composite Score, ground-truth image pairs")
        print("    are required. You can:")
        print("    1. Run this script directly on Google Colab (where the public dataset is mounted).")
        print("    2. Or download a few sample public images (e.g. 001.png - 010.png) into")
        print("       competition_data/public/ground_truth and competition_data/public/noisy.")
        return

    baseline_out = repo_root / "competition_data" / "eval_baseline_out"
    improved_out = repo_root / "competition_data" / "eval_improved_out"
    baseline_out.mkdir(parents=True, exist_ok=True)
    improved_out.mkdir(parents=True, exist_ok=True)

    # 1. Run Baseline
    print("[1/3] Running Organizer Baseline (defect-pixel + NLM / bilateral)...")
    baseline_cmd = [
        sys.executable,
        str(repo_root / "baseline" / "denoise.py"),
        "--input_dir", str(noisy_path),
        "--output_dir", str(baseline_out)
    ]
    subprocess.run(baseline_cmd, check=True)

    # 2. Run Improved Model
    print("\n[2/3] Running Improved Model (NAFNet / Hybrid DSP)...")
    improved_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "denoise.py"),
        "--noise_dir", str(noisy_path),
        "--denoised_dir", str(improved_out),
        "--model_path", str(model_file)
    ]
    if args.cpu:
        improved_cmd.append("--cpu")
    subprocess.run(improved_cmd, check=True)

    # 3. Evaluate Both
    print("\n[3/3] Evaluating both pipelines against Ground Truth...")
    res_base = evaluate_directory(baseline_out, gt_path, noisy_path)
    res_impr = evaluate_directory(improved_out, gt_path, noisy_path)

    if not res_base or not res_impr:
        print("[ERROR] Evaluation failed. Please verify image naming matches conventions.")
        return

    # 4. Display Formatted Comparison Table
    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS ({res_impr['count']} image pairs evaluated)")
    print("=" * 80)
    header = f"{'Metric':<24} | {'Organizer Baseline':<18} | {'Improved Model':<18} | {'Gain / Improvement':<18}"
    print(header)
    print("-" * 80)

    p_gain = res_impr['psnr'] - res_base['psnr']
    s_gain = res_impr['ssim'] - res_base['ssim']
    dp_gain = res_impr['delta_psnr'] - res_base['delta_psnr']
    ds_gain = res_impr['delta_ssim'] - res_base['delta_ssim']
    comp_gain = res_impr['composite'] - res_base['composite']
    comp_pct = (comp_gain / max(res_base['composite'], 1e-6)) * 100.0

    print(f"{'Mean PSNR':<24} | {res_base['psnr']:>14.4f} dB | {res_impr['psnr']:>14.4f} dB | {p_gain:>+14.4f} dB")
    print(f"{'Mean SSIM':<24} | {res_base['ssim']:>17.6f} | {res_impr['ssim']:>17.6f} | {s_gain:>+17.6f}")
    print(f"{'Mean Delta PSNR':<24} | {res_base['delta_psnr']:>14.4f} dB | {res_impr['delta_psnr']:>14.4f} dB | {dp_gain:>+14.4f} dB")
    print(f"{'Mean Delta SSIM':<24} | {res_base['delta_ssim']:>17.6f} | {res_impr['delta_ssim']:>17.6f} | {ds_gain:>+17.6f}")
    print("-" * 80)
    print(f"{'Official Composite Score':<24} | {res_base['composite']:>17.6f} | {res_impr['composite']:>17.6f} | {comp_pct:>+13.1f} %")
    print("=" * 80)

    if res_impr['composite'] > res_base['composite']:
        print("[SUCCESS] Your improved model outperforms the competition baseline!")
    else:
        print("[NOTICE] Baseline scored higher. Check model parameters or training convergence.")


if __name__ == "__main__":
    main()
