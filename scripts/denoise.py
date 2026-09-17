"""
scripts/denoise.py
------------------
Flagship Low-Light Image Denoising Pipeline for Mora SP Cup 2026.
Features:
- Adaptive Defect-Pixel & Hot-Pixel Outlier Suppression
- Multi-Scale Discrete Wavelet BayesShrink Denoising (Orthogonal db2 basis)
- YCrCb Chrominance Decoupling & Directional Filtering
- Deep Learning Backbone (NAFNet) with Automatic CPU Fallback
- Test-Time Augmentation (TTA / Geometric Self-Ensembling) for Maximum PSNR/SSIM
- 100% Offline, Zero Network Calls during Inference

USAGE:
    python scripts/denoise.py --noise_dir competition_data/submissions/noisy --denoised_dir competition_data/submissions/denoised
"""

import os
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.restoration import denoise_wavelet

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def correct_defect_pixels(img_rgb_float: np.ndarray, threshold: float = 0.18, kernel_size: int = 3) -> np.ndarray:
    """
    Adaptive defect/hot-pixel outlier suppression using localized median deviation.
    Removes impulse salt-and-pepper artifacts without degrading true edge sharpness.
    """
    corrected = img_rgb_float.copy()
    for c in range(3):
        ch_u8 = (img_rgb_float[..., c] * 255.0).astype(np.uint8)
        med_u8 = cv2.medianBlur(ch_u8, kernel_size)
        med_float = med_u8.astype(np.float32) / 255.0
        deviation = np.abs(img_rgb_float[..., c] - med_float)
        outliers = deviation > threshold
        corrected[..., c] = np.where(outliers, med_float, img_rgb_float[..., c])
    return corrected


def classical_dsp_single(img_rgb_float: np.ndarray) -> np.ndarray:
    """Core classical DSP pass for a single orientation."""
    # 1. Defect pixel repair
    cleaned = correct_defect_pixels(img_rgb_float, threshold=0.18)

    # 2. Multi-scale Wavelet shrinkage on RGB
    wav_denoised = denoise_wavelet(
        cleaned,
        channel_axis=-1,
        wavelet='db2',
        mode='soft',
        method='BayesShrink',
        rescale_sigma=True
    )
    wav_u8 = (np.clip(wav_denoised, 0.0, 1.0) * 255.0).astype(np.uint8)

    # 3. YCrCb Color Space Decoupling
    ycrcb = cv2.cvtColor(wav_u8, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # Luminance: light edge-preserving bilateral filter
    y_clean = cv2.bilateralFilter(y, d=5, sigmaColor=18, sigmaSpace=18)

    # Chrominance: robust noise reduction to eliminate color splotches
    cr_clean = cv2.medianBlur(cr, 3)
    cr_clean = cv2.bilateralFilter(cr_clean, d=7, sigmaColor=32, sigmaSpace=32)

    cb_clean = cv2.medianBlur(cb, 3)
    cb_clean = cv2.bilateralFilter(cb_clean, d=7, sigmaColor=32, sigmaSpace=32)

    merged = cv2.merge([y_clean, cr_clean, cb_clean])
    result_rgb = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0

    # 4. Subtle detail preservation blend
    detail = cleaned - result_rgb
    final_output = result_rgb + 0.04 * detail

    return np.clip(final_output, 0.0, 1.0)


def deep_learning_single(img_rgb_float: np.ndarray, model, device: str) -> np.ndarray:
    """Core deep learning inference pass for a single orientation."""
    import torch
    pre_cleaned = correct_defect_pixels(img_rgb_float, threshold=0.18)
    inp_tensor = torch.from_numpy(pre_cleaned).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out_tensor = model(inp_tensor)
    out_float = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(out_float, 0.0, 1.0)


def apply_tta_ensemble(img_rgb_float: np.ndarray, infer_fn, tta_mode: int = 4) -> np.ndarray:
    """
    Test-Time Augmentation (Geometric Self-Ensemble):
    Evaluates rotated and flipped variants, inverts transformations, and averages.
    Significantly suppresses directional artifacts and boosts PSNR / SSIM.
    """
    if tta_mode <= 1:
        return infer_fn(img_rgb_float)

    accum = np.zeros_like(img_rgb_float, dtype=np.float32)
    weights_total = 0.0

    # 4 Orthogonal Rotations (0°, 90°, 180°, 270°)
    for k in range(min(4, tta_mode)):
        rot_img = np.rot90(img_rgb_float, k)
        denoised_rot = infer_fn(rot_img)
        inv_rot = np.rot90(denoised_rot, -k)
        accum += inv_rot
        weights_total += 1.0

    # Optional 4 Horizontally Flipped Rotations for full 8-fold TTA
    if tta_mode >= 8:
        flipped_base = np.fliplr(img_rgb_float)
        for k in range(4):
            rot_img = np.rot90(flipped_base, k)
            denoised_rot = infer_fn(rot_img)
            inv_rot = np.rot90(denoised_rot, -k)
            accum += np.fliplr(inv_rot)
            weights_total += 1.0

    return accum / max(weights_total, 1.0)


def enhance_sharpness_and_contrast(denoised_rgb_float: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """
    Edge-preserving unsharp detail and micro-contrast enhancement.
    Operates in YCrCb color space to boost structural sharpness in Luminance (Y)
    without corrupting chromatic fidelity. Dramatically improves perceptual clarity
    and structural SSIM score.
    """
    if strength <= 0.0:
        return denoised_rgb_float

    img_u8 = (np.clip(denoised_rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    ycrcb = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # 1. High-frequency edge detail extraction via Gaussian unsharp mask
    y_blur = cv2.GaussianBlur(y, (0, 0), sigmaX=1.5)
    y_detail = cv2.subtract(y, y_blur)
    y_sharp = cv2.addWeighted(y, 1.0, y_detail, strength, 0)

    # 2. Subtle CLAHE to restore micro-contrast in dim structures
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    y_enhanced = cv2.addWeighted(y_sharp, 0.85, clahe.apply(y_sharp), 0.15, 0)

    merged = cv2.merge([y_enhanced, cr, cb])
    sharpened_rgb = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0
    return np.clip(sharpened_rgb, 0.0, 1.0)


def strip_noise_suffix(filename: str) -> str:
    """Strips '_noise' suffix to adhere to submission convention."""
    stem, ext = os.path.splitext(filename)
    if stem.lower().endswith("_noise"):
        stem = stem[:-6]
    return f"{stem}{ext}"


def load_model_if_available(model_path: str, device: str):
    """Attempt to load PyTorch deep learning weights if present."""
    if not model_path:
        return None

    path_obj = Path(model_path)
    # Search common locations if default path not immediately found
    candidates = [
        Path("models/best_denoiser_v10.pth"),
        Path("scripts/checkpoints/best_denoiser_v10.pth"),
        Path(__file__).resolve().parent.parent / "models" / "best_denoiser_v10.pth",
        Path(__file__).resolve().parent / "checkpoints" / "best_denoiser_v10.pth",
        Path("models/best_denoiser_v9.pth"),
        Path("scripts/checkpoints/best_denoiser_v9.pth"),
        Path(__file__).resolve().parent.parent / "models" / "best_denoiser_v9.pth",
        Path(__file__).resolve().parent / "checkpoints" / "best_denoiser_v9.pth",
        Path("models/best_denoiser_v8.pth"),
        Path("scripts/checkpoints/best_denoiser_v8.pth"),
        Path("models/best_denoiser_v7.pth"),
        Path("scripts/checkpoints/best_denoiser_v7.pth"),
        Path(__file__).resolve().parent.parent / "models" / "best_denoiser_v7.pth",
        Path(__file__).resolve().parent / "checkpoints" / "best_denoiser_v7.pth",
        Path("models/best_denoiser_v6.pth"),
        Path("models/best_denoiser_v6_ema.pth"),
        Path("scripts/checkpoints/best_denoiser_v6.pth"),
        Path(__file__).resolve().parent.parent / "models" / "best_denoiser_v6.pth",
        Path(__file__).resolve().parent / "checkpoints" / "best_denoiser_v6.pth",
        Path("models/best_denoiser_v5.pth"),
        Path("scripts/checkpoints/best_denoiser_v5.pth"),
        Path(__file__).resolve().parent.parent / "models" / "best_denoiser_v5.pth",
        Path(__file__).resolve().parent / "checkpoints" / "best_denoiser_v5.pth",
        Path("models/best_denoiser_v4.pth"),
        Path("scripts/checkpoints/best_denoiser_v4.pth"),
        Path(__file__).resolve().parent.parent / "models" / "best_denoiser_v4.pth",
        Path(__file__).resolve().parent / "checkpoints" / "best_denoiser_v4.pth",
        Path("models/best_denoiser_v3.pth"),
        Path("scripts/checkpoints/best_denoiser_v3.pth"),
        Path(__file__).resolve().parent.parent / "models" / "best_denoiser_v3.pth",
        Path(__file__).resolve().parent / "checkpoints" / "best_denoiser_v3.pth",
        Path("models/best_denoiser_v2.pth"),
        Path("scripts/checkpoints/best_denoiser_v2.pth"),
        Path(__file__).resolve().parent.parent / "models" / "best_denoiser_v2.pth",
        Path(__file__).resolve().parent / "checkpoints" / "best_denoiser_v2.pth",
        Path("models/best_denoiser.pth"),
        Path("scripts/checkpoints/best_denoiser.pth"),
        Path(__file__).resolve().parent.parent / "models" / "best_denoiser.pth",
        Path(__file__).resolve().parent / "checkpoints" / "best_denoiser.pth",
    ]
    if not path_obj.is_file():
        for cand in candidates:
            if cand.is_file():
                path_obj = cand
                break
    else:
        # If default requested was best_denoiser.pth but v7, v6, or v4 exists, upgrade automatically
        for cand in candidates[:8]:
            if cand.is_file():
                path_obj = cand
                break

    if not path_obj.is_file():
        return None

    try:
        import torch
        import sys
        
        # Ensure both repository root and scripts/ directory are in sys.path
        scripts_dir = str(Path(__file__).resolve().parent)
        repo_dir = str(Path(__file__).resolve().parent.parent)
        for p in [scripts_dir, repo_dir]:
            if p not in sys.path:
                sys.path.insert(0, p)

        try:
            from scripts.models import build_denoising_model
        except ImportError:
            from models import build_denoising_model

        checkpoint = torch.load(str(path_obj), map_location=device)
        width = checkpoint.get("width", 64 if any(v in str(path_obj) for v in ["v7", "v8", "v9"]) else 48)
        model = build_denoising_model(width=width)
        sd = checkpoint["model_state_dict"]
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items() if k != "n_averaged"}
        model.load_state_dict(clean_sd)
        model.to(device)
        model.eval()
        print(f"[*] Successfully loaded trained checkpoint from {path_obj} (width={width})")
        return model
    except Exception as e:
        print(f"[!] Warning: Checkpoint could not be loaded ({e}). Falling back to Classical DSP.")
        return None


def run_pipeline(noise_dir: str, denoised_dir: str, model_path: str = None, force_cpu: bool = False, tta_mode: int = 4, sharpness: float = 0.35):
    noise_path = Path(noise_dir)
    out_path = Path(denoised_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    if not force_cpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass

    print(f"[*] Inference Hardware: {device.upper()}")
    model = load_model_if_available(model_path, device) if model_path else None

    if model is not None:
        print(f"[*] Active Mode: Deep Neural Network (NAFNet) + TTA-{tta_mode}")
        infer_fn = lambda img: deep_learning_single(img, model, device)
    else:
        print(f"[*] Active Mode: Advanced Classical DSP (Wavelet + YCrCb Bilateral) + TTA-{tta_mode}")
        infer_fn = classical_dsp_single

    files = sorted([f for f in noise_path.iterdir() if f.suffix.lower() in VALID_EXTENSIONS])
    if not files:
        raise SystemExit(f"Error: No image files found in {noise_dir}")

    total_time = 0.0
    processed_count = 0

    for fpath in files:
        out_filename = strip_noise_suffix(fpath.name)
        save_file = out_path / out_filename

        with Image.open(fpath) as img:
            rgb_img = img.convert("RGB")
            arr_float = np.asarray(rgb_img, dtype=np.float32) / 255.0

        t0 = time.time()
        out_float = apply_tta_ensemble(arr_float, infer_fn, tta_mode=tta_mode)
        # Apply edge-preserving sharpness and micro-contrast enhancement
        out_float = enhance_sharpness_and_contrast(out_float, strength=sharpness)
        t_cost = time.time() - t0
        total_time += t_cost
        processed_count += 1

        out_uint8 = (out_float * 255.0).round().astype(np.uint8)
        Image.fromarray(out_uint8).save(save_file, format="PNG")

    avg_ms = (total_time / processed_count) * 1000.0 if processed_count else 0.0
    print(f"\n[+] Successfully denoised {processed_count} images in {total_time:.2f}s ({avg_ms:.1f} ms/image)")
    print(f"[+] Output directory: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Mora SP Cup 2026 Denoising Pipeline")
    parser.add_argument("--noise_dir", "--input_dir", dest="noise_dir", required=True,
                        help="Path to folder containing noisy images")
    parser.add_argument("--denoised_dir", "--output_dir", dest="denoised_dir", required=True,
                        help="Path to save denoised images")
    parser.add_argument("--model_path", type=str, default="models/best_denoiser.pth",
                        help="Path to model weights (optional)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference fallback")
    parser.add_argument("--tta", type=int, default=4, choices=[1, 4, 8],
                        help="Test-Time Augmentation ensemble passes (1=off, 4=rotations, 8=rot+flip)")
    parser.add_argument("--sharpness", type=float, default=0.35,
                        help="Edge-preserving sharpness strength (default: 0.35)")

    args = parser.parse_args()
    run_pipeline(args.noise_dir, args.denoised_dir, args.model_path, args.cpu, args.tta, args.sharpness)


if __name__ == "__main__":
    main()