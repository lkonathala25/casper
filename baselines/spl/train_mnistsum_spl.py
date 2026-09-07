"""Train MNIST-Sum with pre-compiled SDD constraint via SPL.

Uses a LeNet-style CNN backbone to classify pairs of MNIST digits,
with an SDD-based semantic loss that enforces the addition constraint.
"""

import argparse
import os
import sys
import time

# Use grids pypsdd (supports log_space in generate_tf_ac)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "grids", "pypsdd"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "grids"))

from pathlib import Path as _Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from compute_mpe import CircuitMPE

LOG_ZERO = -300.0


# ── Dataset ──────────────────────────────────────────────────────────

class MNISTSumDataset(Dataset):
    """Pairs of MNIST digits with sum labels.

    Each sample: (image1, image2, digit1_label, digit2_label, sum_label).
    Pairs are formed by sampling indices with a fixed seed for reproducibility.
    """

    def __init__(self, train=True, num_samples=5000, seed=42):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.mnist = datasets.MNIST(
            root=str(_Path(__file__).resolve().parents[2] / "data"), train=train, download=True, transform=transform
        )
        rng = torch.Generator().manual_seed(seed)
        n = len(self.mnist)
        self.idx1 = torch.randint(0, n, (num_samples,), generator=rng)
        self.idx2 = torch.randint(0, n, (num_samples,), generator=rng)

    def __len__(self):
        return len(self.idx1)

    def __getitem__(self, i):
        img1, d1 = self.mnist[self.idx1[i].item()]
        img2, d2 = self.mnist[self.idx2[i].item()]
        return img1, img2, d1, d2, d1 + d2


# ── Model ────────────────────────────────────────────────────────────

class LeNet(nn.Module):
    """Shared LeNet backbone: 2 conv + 2 fc → 10 logits per image."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MNISTSumModel(nn.Module):
    """Two-digit model: shared LeNet → 20 logits (2 digits x 10 classes)."""

    def __init__(self):
        super().__init__()
        self.backbone = LeNet()

    def forward(self, img1, img2):
        logits1 = self.backbone(img1)   # (B, 10)
        logits2 = self.backbone(img2)   # (B, 10)
        return torch.stack([logits1, logits2], dim=1)  # (B, 2, 10)


# ── Training ─────────────────────────────────────────────────────────

def build_sum_litleaves(sum_labels, batch_size, clamped=True):
    """Build litleaves for the 19 sum indicator variables.

    Args:
        sum_labels: (B,) tensor of sum values 0-18
        batch_size: int
        clamped: if True, clamp to observed sum; if False, all free
    Returns:
        list of 19 [neg_weight, pos_weight] pairs, each shape (B, 1)
    """
    device = sum_labels.device
    litleaves = []
    for s in range(19):
        if not clamped:
            litleaves.append([
                torch.zeros(batch_size, 1, device=device),
                torch.zeros(batch_size, 1, device=device),
            ])
        else:
            # s == sum_labels → [0, 0]; else → [0, LOG_ZERO]
            is_match = (sum_labels == s).float()  # (B,)
            pos_w = torch.where(
                sum_labels == s,
                torch.tensor(0.0, device=device),
                torch.tensor(LOG_ZERO, device=device),
            ).unsqueeze(1)  # (B, 1)
            neg_w = torch.zeros(batch_size, 1, device=device)
            litleaves.append([neg_w, pos_w])
    return litleaves


def train_epoch(model, cmpe, loader, optimizer, digit_loss_weight, device):
    model.train()
    total_loss = 0.0
    total_digit_correct = 0
    total_sum_correct = 0
    total_samples = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for img1, img2, d1, d2, s in loader:
        img1, img2 = img1.to(device), img2.to(device)
        d1, d2, s = d1.to(device), d2.to(device), s.to(device)
        digit_labels = torch.stack([d1, d2], dim=1)  # (B, 2)
        B = img1.size(0)

        logits = model(img1, img2)  # (B, 2, 10)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, 2, 10)

        # ── digit litleaves: 20 variables, each (B, 1) ──
        digit_litleaves = []
        for i in range(2):      # digit index
            for k in range(10):  # class index
                digit_litleaves.append([
                    torch.zeros(B, 1, device=device),
                    log_probs[:, i, k].unsqueeze(1),
                ])

        # ── sum litleaves ──
        sum_litleaves_clamped = build_sum_litleaves(s, B, clamped=True)
        sum_litleaves_free = build_sum_litleaves(s, B, clamped=False)

        # ── WMC in log-space ──
        log_num = cmpe.beta.generate_tf_ac(
            digit_litleaves + sum_litleaves_clamped, log_space=True
        )
        log_den = cmpe.beta.generate_tf_ac(
            digit_litleaves + sum_litleaves_free, log_space=True
        )

        # squeeze out num_reps dim → (B,)
        log_num = log_num.squeeze(-1)
        log_den = log_den.squeeze(-1)

        sum_nll = -(log_num - log_den).mean()

        # ── digit cross-entropy ──
        digit_ce = F.nll_loss(log_probs.reshape(-1, 10), digit_labels.reshape(-1))

        loss = sum_nll + digit_loss_weight * digit_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── metrics ──
        total_loss += loss.item() * B
        pred_digits = log_probs.argmax(dim=-1)  # (B, 2)
        total_digit_correct += (pred_digits == digit_labels).sum().item()
        pred_sums = pred_digits.sum(dim=1)
        total_sum_correct += (pred_sums == s).sum().item()
        total_samples += B

    n = total_samples
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )
    return {
        "loss": total_loss / n,
        "digit_acc": total_digit_correct / (2 * n),
        "sum_acc": total_sum_correct / n,
        "violation_rate": 0.0,  # 0 by construction with SDD constraint
        "peak_mem_mb": peak_mem_mb,
    }


def eval_epoch(model, cmpe, loader, digit_loss_weight, device):
    model.eval()
    total_loss = 0.0
    total_digit_correct = 0
    total_sum_correct = 0
    total_samples = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for img1, img2, d1, d2, s in loader:
            img1, img2 = img1.to(device), img2.to(device)
            d1, d2, s = d1.to(device), d2.to(device), s.to(device)
            digit_labels = torch.stack([d1, d2], dim=1)
            B = img1.size(0)

            logits = model(img1, img2)
            log_probs = F.log_softmax(logits, dim=-1)

            digit_litleaves = []
            for i in range(2):
                for k in range(10):
                    digit_litleaves.append([
                        torch.zeros(B, 1, device=device),
                        log_probs[:, i, k].unsqueeze(1),
                    ])

            sum_litleaves_clamped = build_sum_litleaves(s, B, clamped=True)
            sum_litleaves_free = build_sum_litleaves(s, B, clamped=False)

            log_num = cmpe.beta.generate_tf_ac(
                digit_litleaves + sum_litleaves_clamped, log_space=True
            ).squeeze(-1)
            log_den = cmpe.beta.generate_tf_ac(
                digit_litleaves + sum_litleaves_free, log_space=True
            ).squeeze(-1)

            sum_nll = -(log_num - log_den).mean()
            digit_ce = F.nll_loss(log_probs.reshape(-1, 10), digit_labels.reshape(-1))
            loss = sum_nll + digit_loss_weight * digit_ce

            total_loss += loss.item() * B
            pred_digits = log_probs.argmax(dim=-1)
            total_digit_correct += (pred_digits == digit_labels).sum().item()
            pred_sums = pred_digits.sum(dim=1)
            total_sum_correct += (pred_sums == s).sum().item()
            total_samples += B

    n = total_samples
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )
    return {
        "loss": total_loss / n,
        "digit_acc": total_digit_correct / (2 * n),
        "sum_acc": total_sum_correct / n,
        "violation_rate": 0.0,
        "peak_mem_mb": peak_mem_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Train MNIST-Sum with SPL")
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--supervised", action="store_true", default=True)
    parser.add_argument("--digit-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load SDD ──
    vtree_path = f"sdd_files/mnistsum_N{args.N}.vtree"
    sdd_path = f"sdd_files/mnistsum_N{args.N}.sdd"
    assert os.path.exists(sdd_path), f"Missing {sdd_path} — run compile_mnistsum_sdd.py first"

    cmpe = CircuitMPE(vtree_path, sdd_path)
    cmpe.beta.num_reps = 1
    print(f"Loaded SDD from {sdd_path}")

    # ── Data ──
    train_ds = MNISTSumDataset(train=True, num_samples=args.train_samples, seed=42)
    test_ds = MNISTSumDataset(train=False, num_samples=args.test_samples, seed=99)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    print(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")

    # ── Model ──
    model = MNISTSumModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Training loop ──
    print(
        f"\n{'Epoch':>5} {'Loss':>8} {'DigAcc':>8} {'SumAcc':>8} "
        f"{'Viol':>6} {'Time':>7} {'PeakMem(MB)':>12}"
    )
    print("-" * 62)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(
            model, cmpe, train_loader, optimizer, args.digit_loss_weight, device
        )
        elapsed = time.time() - t0

        print(
            f"{epoch:5d} "
            f"{train_metrics['loss']:8.4f} "
            f"{train_metrics['digit_acc']:8.4f} "
            f"{train_metrics['sum_acc']:8.4f} "
            f"{train_metrics['violation_rate']:6.3f} "
            f"{elapsed:6.1f}s "
            f"{train_metrics['peak_mem_mb']:12.1f}"
        )

    # ── Final evaluation ──
    print("\n--- Test set evaluation ---")
    test_metrics = eval_epoch(model, cmpe, test_loader, args.digit_loss_weight, device)
    print(f"  Loss:           {test_metrics['loss']:.4f}")
    print(f"  Digit accuracy: {test_metrics['digit_acc']:.4f}")
    print(f"  Sum accuracy:   {test_metrics['sum_acc']:.4f}")
    print(f"  Violation rate: {test_metrics['violation_rate']:.4f}")
    print(f"  Peak GPU mem:   {test_metrics['peak_mem_mb']:.1f} MB")


if __name__ == "__main__":
    main()
