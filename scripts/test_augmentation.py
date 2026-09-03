import os
import torch
from torch.utils.data import DataLoader
from dataset import DenoisingPairDataset

def verify_pipeline():
    noisy_dir = "competition_data/public/noisy"
    gt_dir = "competition_data/public/ground_truth"

    if not os.path.exists(noisy_dir) or not os.path.exists(gt_dir):
        print(f"Error: Missing directories. Please verify {noisy_dir} and {gt_dir} exist.")
        return

    # Initialize Dataset with 256x256 patch size
    train_dataset = DenoisingPairDataset(
        noisy_dir=noisy_dir,
        gt_dir=gt_dir,
        patch_size=256,
        augment=True
    )

    # Create PyTorch DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    print(f"Total image pairs indexed: {len(train_dataset)}")

    # Fetch a single batch
    noisy_batch, gt_batch = next(iter(train_loader))

    print(f"Noisy Batch Shape: {noisy_batch.shape} (Expected: [8, 3, 256, 256])")
    print(f"GT Batch Shape:    {gt_batch.shape} (Expected: [8, 3, 256, 256])")
    print(f"Tensor Value Range: [{noisy_batch.min():.2f}, {noisy_batch.max():.2f}]")
    print("Augmentation pipeline verified successfully!")

if __name__ == "__main__":
    verify_pipeline()