import os
import argparse
import cv2


def run_baseline_denoise(noise_dir, denoised_dir):
    os.makedirs(denoised_dir, exist_ok=True)
    image_files = [f for f in os.listdir(noise_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Running baseline denoising on {len(image_files)} images...")

    for filename in image_files:
        input_path = os.path.join(noise_dir, filename)
        img = cv2.imread(input_path)
        if img is None:
            continue

        # Baseline starter method: Simple Fast Non-Local Means / Bilateral
        denoised = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)

        # Mora SP Cup renaming rule: strip '_noise' if present (e.g., 461_noise.png -> 461.png)
        output_name = filename.replace("_noise", "")
        output_path = os.path.join(denoised_dir, output_name)

        cv2.imwrite(output_path, denoised)

    print(f"Saved denoised outputs to: {denoised_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Denoising Script")
    parser.add_argument("--noise_dir", type=str, required=True, help="Directory with input noisy images")
    parser.add_argument("--denoised_dir", type=str, required=True, help="Directory to save denoised images")
    args = parser.parse_args()

    run_baseline_denoise(args.noise_dir, args.denoised_dir)