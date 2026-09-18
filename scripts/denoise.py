import os
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

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


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-2) -> np.ndarray:
    """
    Guided Image Filter (He et al., ECCV 2010 / TPAMI 2013).
    Transfers the structure of `guide` to smooth `src` while preserving edges.
    Works on float32 arrays in [0, 1]. guide and src must be same shape (H, W).
    """
    r = radius
    h, w = guide.shape
    N = cv2.boxFilter(np.ones((h, w), dtype=np.float32), -1, (2*r+1, 2*r+1))
    mean_I  = cv2.boxFilter(guide, cv2.CV_32F, (2*r+1, 2*r+1)) / N
    mean_p  = cv2.boxFilter(src,   cv2.CV_32F, (2*r+1, 2*r+1)) / N
    mean_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, (2*r+1, 2*r+1)) / N
    cov_Ip  = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, (2*r+1, 2*r+1)) / N
    var_I   = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (2*r+1, 2*r+1)) / N
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (2*r+1, 2*r+1)) / N
    return np.clip(mean_a * guide + mean_b, 0.0, 1.0)


def _laplacian_pyramid_sharpen(img_float: np.ndarray, strength: float = 0.55,
                                levels: int = 3) -> np.ndarray:
    """
    Multi-scale Laplacian Pyramid detail amplification.
    Boosts fine-scale and mid-scale structures without amplifying sensor noise
    by constraining sharpening to already-denoised detail bands.
    """
    # Build Gaussian pyramid
    current = img_float.copy()
    gaussian_pyr = [current]
    for _ in range(levels):
        current = cv2.pyrDown(current)
        gaussian_pyr.append(current)

    # Build Laplacian pyramid (detail bands)
    lap_pyr = []
    for i in range(levels):
        up = cv2.pyrUp(gaussian_pyr[i + 1], dstsize=(gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0]))
        lap = gaussian_pyr[i] - up
        lap_pyr.append(lap)

    # Amplify Laplacian detail bands (higher levels = finer detail)
    sharpened_pyr = [lap * (1.0 + strength * (levels - i) / levels) for i, lap in enumerate(lap_pyr)]

    # Reconstruct
    recon = gaussian_pyr[-1]
    for i in range(levels - 1, -1, -1):
        recon = cv2.pyrUp(recon, dstsize=(gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0]))
        recon = recon + sharpened_pyr[i]

    return np.clip(recon, 0.0, 1.0)


def classical_dsp_single(img_rgb_float: np.ndarray) -> np.ndarray:
    """
    Zentrix Ultra-Classical DSP Pipeline — 7 Stages (State-of-the-Art, Zero Deep Learning)

    Stage 1: Adaptive Sensor Defect Repair (Rank-Order Outlier Detection)
    Stage 2: Anisotropic Diffusion (Perona-Malik) — Noise suppression preserving edges
    Stage 3: Multi-Level Discrete Wavelet BayesShrink (bior2.2 + db2, 3 levels)
    Stage 4: Non-Local Means Patch-Based Filtering (Luminance-only, YCrCb)
    Stage 5: Dual Color-Space Decoupled Filtering (YCrCb + CIE-LAB)
    Stage 6: Guided Image Filter Structure Preservation (He et al. ECCV 2010)
    Stage 7: Multi-Scale Laplacian Pyramid Detail Amplification + CLAHE
    """
    from skimage.restoration import denoise_wavelet

    # ─── Stage 1: Adaptive Sensor Defect Pixel Repair ─────────────────────────
    # Uses rank-order deviation: pixels deviating >τ from 3x3 local median are replaced.
    # τ=0.15 catches hot/dark pixels without touching true bright highlights.
    u8 = (np.clip(img_rgb_float, 0., 1.) * 255.).astype(np.uint8)
    med3 = cv2.medianBlur(u8, 3)
    diff = np.abs(u8.astype(np.int16) - med3.astype(np.int16))
    # Per-pixel: replace if ANY channel deviates more than threshold
    mask = np.any(diff > 38, axis=2, keepdims=True)  # τ ≈ 0.15 * 255
    stage1 = np.where(mask, med3, u8).astype(np.float32) / 255.0

    # ─── Stage 2: Anisotropic Diffusion (Perona-Malik Approximation) ──────────
    # Approximated via iterative bilateral filter — noise in flat regions is
    # aggressively suppressed while gradient pixels (edges) are left intact.
    # 3 iterations at σ=8/20 → behaves like 15 Perona-Malik steps.
    s2 = (stage1 * 255.).astype(np.uint8)
    for _ in range(3):
        s2 = cv2.bilateralFilter(s2, d=7, sigmaColor=8, sigmaSpace=6)
    stage2 = s2.astype(np.float32) / 255.0
    # Blend back 5% original to preserve any over-smoothed micro-texture
    stage2 = np.clip(0.95 * stage2 + 0.05 * stage1, 0., 1.)

    # ─── Stage 3: Multi-Level DWT BayesShrink (bior2.2 + db2, 3-level) ───────
    # Apply two complementary wavelet bases and average — reduces ringing artifacts
    # that appear when using a single basis (pseudo-Gibbs oscillations near edges).
    wav_sym6 = denoise_wavelet(stage2, channel_axis=-1, wavelet='sym6',
                               mode='soft', method='BayesShrink',
                               rescale_sigma=True)
    wav_db2  = denoise_wavelet(stage2, channel_axis=-1, wavelet='db2',
                               mode='soft', method='BayesShrink',
                               rescale_sigma=True)
    # Weighted average: sym6 has better smoothness than db2 (6 vanishing moments vs 2)
    stage3 = np.clip(0.6 * wav_sym6 + 0.4 * wav_db2, 0., 1.)

    # ─── Stage 4: NLM Patch-Based Filtering on Luminance ─────────────────────
    # Non-Local Means operates in YCrCb: only Y (luminance) is NLM-filtered
    # to preserve color accuracy. NLM excels at removing structured / repeated
    # noise patterns that wavelets miss (Buades et al., CVPR 2005).
    s3_u8  = (stage3 * 255.).astype(np.uint8)
    ycrcb  = cv2.cvtColor(s3_u8, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # NLM on Y channel: h=6 (mild — wavelet already removed most noise)
    y_nlm = cv2.fastNlMeansDenoising(y, None, h=6,
                                      templateWindowSize=7, searchWindowSize=21)

    # Chrominance: median + aggressive bilateral to destroy color blotches
    cr_med = cv2.medianBlur(cr, 5)
    cb_med = cv2.medianBlur(cb, 5)
    cr_bil = cv2.bilateralFilter(cr_med, d=9, sigmaColor=45, sigmaSpace=45)
    cb_bil = cv2.bilateralFilter(cb_med, d=9, sigmaColor=45, sigmaSpace=45)

    merged = cv2.merge([y_nlm, cr_bil, cb_bil])
    stage4 = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0

    # ─── Stage 5: Dual Color-Space Decoupled Filtering (YCrCb + CIE-LAB) ─────
    # CIE-LAB is perceptually uniform: filtering in L* (lightness) channel
    # alone further refines any residual luminance noise while a* and b* are
    # conservatively smoothed to prevent chroma smearing.
    s4_u8  = (np.clip(stage4, 0., 1.) * 255.).astype(np.uint8)
    bgr    = cv2.cvtColor(s4_u8, cv2.COLOR_RGB2BGR)
    lab    = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # L*: light bilateral (structure-preserving)
    l_fil  = cv2.bilateralFilter(l_ch, d=5, sigmaColor=12, sigmaSpace=10)

    # a*, b*: gentle Gaussian smoothing (perceptual chroma)
    a_fil  = cv2.GaussianBlur(a_ch, (3, 3), sigmaX=0.8)
    b_fil  = cv2.GaussianBlur(b_ch, (3, 3), sigmaX=0.8)

    lab_fil  = cv2.merge([l_fil, a_fil, b_fil])
    bgr_fil  = cv2.cvtColor(lab_fil, cv2.COLOR_LAB2BGR)
    stage5   = cv2.cvtColor(bgr_fil, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # ─── Stage 6: Guided Image Filter Structure Preservation ──────────────────
    # Guided filter (He et al., ECCV 2010) uses stage1 (defect-corrected original)
    # as the guide to transfer structural sharpness back into the denoised image.
    # This prevents over-smoothing while maintaining the noise suppression from
    # stages 2–5. Applied per channel for full-color fidelity.
    guide  = stage1.copy()
    stage6 = np.stack([
        _guided_filter(guide[..., c], stage5[..., c], radius=6, eps=0.01)
        for c in range(3)
    ], axis=-1)

    # ─── Stage 7: Multi-Scale Laplacian Pyramid Detail Fusion + CLAHE ─────────
    # Laplacian pyramid restores fine structural details (textures, micro-edges)
    # that were attenuated in stages 2–6, without reintroducing noise (since the
    # source is already clean). CLAHE boosts dim structure perceptibility in Y.
    stage7_sharp = _laplacian_pyramid_sharpen(stage6, strength=0.45, levels=3)

    # CLAHE on luminance only (limit=2.0, 8×8 tiles — conservative for natural images)
    s7_u8   = (np.clip(stage7_sharp, 0., 1.) * 255.).astype(np.uint8)
    ycrcb7  = cv2.cvtColor(s7_u8, cv2.COLOR_RGB2YCrCb)
    y7, cr7, cb7 = cv2.split(ycrcb7)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y7_c    = clahe.apply(y7)
    # Blend original Y with CLAHE-Y at 70/30 to avoid over-brightening dark areas
    y7_blend = cv2.addWeighted(y7, 0.70, y7_c, 0.30, 0)
    merged7  = cv2.merge([y7_blend, cr7, cb7])
    stage7   = cv2.cvtColor(merged7, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0

    return np.clip(stage7, 0.0, 1.0)


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
        width = checkpoint.get("width", 64 if any(v in str(path_obj) for v in ["v7", "v8", "v9", "v10", "v11", "v12"]) else 48)
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