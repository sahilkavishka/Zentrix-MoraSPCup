import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class DenoisingPairDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir, patch_size=256, augment=True):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.patch_size = patch_size
        self.augment = augment

        # Find all ground-truth images
        self.gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
        if len(self.gt_files) == 0:
            raise ValueError(f"No ground-truth images found in {gt_dir}")

    def __len__(self):
        return len(self.gt_files)

    def _apply_augmentations(self, noisy, gt):
        # 1. Random Crop (H, W, C) -> (patch_size, patch_size, C)
        h, w, _ = noisy.shape
        if self.patch_size and h >= self.patch_size and w >= self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            noisy = noisy[top:top + self.patch_size, left:left + self.patch_size]
            gt = gt[top:top + self.patch_size, left:left + self.patch_size]

        # 2. Synchronized Horizontal Flip
        if random.random() > 0.5:
            noisy = np.fliplr(noisy)
            gt = np.fliplr(gt)

        # 3. Synchronized Vertical Flip
        if random.random() > 0.5:
            noisy = np.flipud(noisy)
            gt = np.flipud(gt)

        # 4. Synchronized Random 90-degree Rotation
        rot_k = random.choice([0, 1, 2, 3])
        if rot_k > 0:
            noisy = np.rot90(noisy, rot_k)
            gt = np.rot90(gt, rot_k)

        return np.ascontiguousarray(noisy), np.ascontiguousarray(gt)

    def __getitem__(self, idx):
        gt_path = self.gt_files[idx]
        base_id = os.path.splitext(os.path.basename(gt_path))[0]

        # Build path to corresponding noisy image (001.png -> 001_noise.png)
        noisy_name = f"{base_id}_noise.png"
        noisy_path = os.path.join(self.noisy_dir, noisy_name)
        if not os.path.exists(noisy_path):
            noisy_path = os.path.join(self.noisy_dir, f"{base_id}.png")

        # Load RGB images
        gt_bgr = cv2.imread(gt_path)
        noisy_bgr = cv2.imread(noisy_path)

        if gt_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {gt_path}")
        if noisy_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {noisy_path}")

        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        noisy_rgb = cv2.cvtColor(noisy_bgr, cv2.COLOR_BGR2RGB)

        # Apply synchronized transformations
        if self.augment:
            noisy_rgb, gt_rgb = self._apply_augmentations(noisy_rgb, gt_rgb)

        # Convert [0, 255] uint8 -> [0.0, 1.0] float32 PyTorch Tensor (C, H, W)
        noisy_tensor = torch.from_numpy(noisy_rgb).permute(2, 0, 1).float() / 255.0
        gt_tensor = torch.from_numpy(gt_rgb).permute(2, 0, 1).float() / 255.0

        return noisy_tensor, gt_tensor