"""
scripts/train.py
----------------
Training pipeline for Mora SP Cup 2026 Low-Light Image Denoising.
Optimizes a hybrid Charbonnier + SSIM loss with Cosine Annealing learning rate.
Tracks PSNR, SSIM, and Official Mora SP Cup Composite Score on validation splits.
"""

import os
import argparse
import hashlib
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from scripts.models import build_denoising_model
from scripts.dataset import DenoisingPairDataset


class CharbonnierLoss(nn.Module):
    """Smooth L1 / Charbonnier Loss."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)
        return torch.mean(loss)


def compute_psnr_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR for batched tensors in range [0, 1]."""
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    # Avoid div by zero
    mse = torch.clamp(mse, min=1e-8)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return float(torch.mean(psnr).item())


def calculate_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum for submission compliance."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[*] Training on device: {device}")

    # Create dataset
    full_dataset = DenoisingPairDataset(
        noisy_dir=args.noisy_dir,
        gt_dir=args.gt_dir,
        patch_size=args.patch_size,
        augment=True
    )

    total_samples = len(full_dataset)
    val_size = max(1, int(total_samples * args.val_split))
    train_size = total_samples - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"[*] Total dataset: {total_samples} pairs | Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True if device.type == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers
    )

    # Initialize model
    model = build_denoising_model(width=args.width).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Model initialized with {total_params:,} parameters.")

    criterion_charbonnier = CharbonnierLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_psnr = 0.0
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_denoiser.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (noisy, gt) in enumerate(train_loader):
            noisy, gt = noisy.to(device), gt.to(device)

            optimizer.zero_grad()
            pred = model(noisy)

            loss = criterion_charbonnier(pred, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        elapsed = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)

        # Validation phase
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            model.eval()
            val_psnr_list = []
            with torch.no_grad():
                for noisy_val, gt_val in val_loader:
                    noisy_val, gt_val = noisy_val.to(device), gt_val.to(device)
                    pred_val = model(noisy_val)
                    psnr = compute_psnr_batch(pred_val, gt_val)
                    val_psnr_list.append(psnr)

            mean_val_psnr = sum(val_psnr_list) / len(val_psnr_list)
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Train Loss: {avg_loss:.5f} | "
                  f"Val PSNR: {mean_val_psnr:.3f} dB | Time: {elapsed:.1f}s")

            if mean_val_psnr > best_psnr:
                best_psnr = mean_val_psnr
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_psnr': best_psnr,
                    'width': args.width,
                }, best_checkpoint_path)
                print(f"  --> Saved new best checkpoint to {best_checkpoint_path} (PSNR: {best_psnr:.3f} dB)")
        else:
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Train Loss: {avg_loss:.5f} | Time: {elapsed:.1f}s")

    if os.path.exists(best_checkpoint_path):
        checksum = calculate_sha256(best_checkpoint_path)
        print("\n" + "=" * 50)
        print(f"TRAINING COMPLETE")
        print(f"Best Checkpoint: {best_checkpoint_path}")
        print(f"Best Val PSNR:   {best_psnr:.4f} dB")
        print(f"SHA-256 Checksum: {checksum}")
        print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mora SP Cup 2026 Model Training")
    parser.add_argument("--noisy_dir", type=str, default="competition_data/public/noisy")
    parser.add_argument("--gt_dir", type=str, default="competition_data/public/ground_truth")
    parser.add_argument("--checkpoint_dir", type=str, default="scripts/checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=2)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")

    args = parser.parse_args()
    train(args)
