"""
scripts/run_submission.py
-------------------------
Automated submission generator and validator for Mora SP Cup 2026.

Actions performed:
1. Runs `scripts/denoise.py` on the preliminary noisy set (461_noise.png - 480_noise.png).
2. Verifies that all 20 images exist in `competition_data/submissions/denoised/` named 461.png to 480.png.
3. Strictly validates that every image is:
   - 992 x 992 resolution
   - 3-channel RGB uint8 format
   - Uncorrupted valid PNG
4. Archives the 20 denoised PNG files into `Zentrix.zip` at repository root ready for Google Drive submission.
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path
from PIL import Image

EXPECTED_COUNT = 20
EXPECTED_START = 461
EXPECTED_END = 480
EXPECTED_DIMS = (992, 992)
TEAM_NAME = "Zentrix"


def validate_image(filepath: Path):
    if not filepath.exists():
        return False, f"Missing file: {filepath}"
    try:
        with Image.open(filepath) as img:
            if img.format != "PNG":
                return False, f"{filepath.name}: Invalid format {img.format} (Expected: PNG)"
            if img.size != EXPECTED_DIMS:
                return False, f"{filepath.name}: Invalid size {img.size} (Expected: {EXPECTED_DIMS})"
            if img.mode != "RGB":
                return False, f"{filepath.name}: Invalid mode {img.mode} (Expected: RGB)"
    except Exception as e:
        return False, f"Failed to open {filepath}: {e}"
    return True, "OK"


def package_submission(tta_mode: int = 1, sharpness: float = 0.35):
    repo_root = Path(__file__).resolve().parent.parent
    noisy_dir = repo_root / "competition_data" / "submissions" / "noisy"
    denoised_dir = repo_root / "competition_data" / "submissions" / "denoised"
    zip_path = repo_root / f"{TEAM_NAME}.zip"

    print("=" * 60)
    print(f"Mora SP Cup 2026 - Preliminary Submission Packaging: {TEAM_NAME}")
    print(f"[*] Settings: TTA={tta_mode} | Edge Sharpness Strength={sharpness}")
    print("=" * 60)

    if not noisy_dir.exists():
        print(f"[ERROR] Noisy directory not found: {noisy_dir}")
        sys.exit(1)

    denoised_dir.mkdir(parents=True, exist_ok=True)

    # 1. Execute denoiser
    print("\n[Step 1] Running denoising pipeline...")
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "denoise.py"),
        "--noise_dir", str(noisy_dir),
        "--denoised_dir", str(denoised_dir),
        "--tta", str(tta_mode),
        "--sharpness", str(sharpness)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("[ERROR] Denoising failed:")
        print(res.stderr)
        sys.exit(res.returncode)

    # 2. Strict Verification
    print("\n[Step 2] Validating denoised images...")
    all_valid = True
    validated_files = []

    for img_id in range(EXPECTED_START, EXPECTED_END + 1):
        filename = f"{img_id}.png"
        filepath = denoised_dir / filename
        valid, msg = validate_image(filepath)
        if not valid:
            print(f"  [FAIL] {msg}")
            all_valid = False
        else:
            validated_files.append(filepath)

    if not all_valid or len(validated_files) != EXPECTED_COUNT:
        print(f"\n[ERROR] Validation failed! Valid images: {len(validated_files)}/{EXPECTED_COUNT}")
        sys.exit(1)

    print(f"  [PASS] All {EXPECTED_COUNT} images strictly validated (992x992 RGB PNG).")

    # 3. Create ZIP Archive
    print(f"\n[Step 3] Packaging into {zip_path.name}...")
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in validated_files:
            # Place directly in root of zip as required by handbook
            zf.write(f, arcname=f.name)

    print(f"  [SUCCESS] Created {zip_path.name} (Size: {zip_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Contents: {len(validated_files)} files ({EXPECTED_START}.png - {EXPECTED_END}.png)")
    print("\n" + "=" * 60)
    print("SUBMISSION PACKAGE READY FOR UPLOAD TO GOOGLE DRIVE")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mora SP Cup Submission Packaging")
    parser.add_argument("--tta", type=int, default=1, choices=[1, 4, 8],
                        help="TTA passes (1=fast CPU ~5min, 4=high accuracy ~20min)")
    parser.add_argument("--sharpness", type=float, default=0.35,
                        help="Edge-preserving sharpness strength (default: 0.35)")
    args = parser.parse_args()
    package_submission(tta_mode=args.tta, sharpness=args.sharpness)
