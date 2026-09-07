"""Train MNIST-Add (N-digit operands) with pre-compiled SDD constraint via SPL.

Each sample has 2N MNIST images forming two N-digit numbers a and b
(MSB-first per operand). The model predicts each digit independently
via a shared LeNet, and is trained with:

    loss = sum_nll + digit_loss_weight * digit_ce

where sum_nll is the SDD-based semantic loss (negative log WMC ratio
under the addition constraint) and digit_ce is per-digit cross-entropy
on the true digit labels — i.e. fully supervised at the digit level.
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

class MNISTAddDataset(Dataset):
    """Pairs of N-digit MNIST numbers with sum labels.

    Each sample yields:
        images: tensor of shape (2N, 1, 28, 28) — operand a's digits
                MSB→LSB, then operand b's digits.
        digits: tensor of shape (2N,) — true digit labels.
        s:      scalar long — true sum a + b ∈ {0..2*(10^N - 1)}.
    """

    def __init__(self, N=2, train=True, num_samples=5000, seed=42):
        self.N = N
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.mnist = datasets.MNIST(
            root=str(_Path(__file__).resolve().parents[2] / "data"), train=train, download=True, transform=transform
        )
        rng = torch.Generator().manual_seed(seed)
        n = len(self.mnist)
        self.idx = torch.randint(0, n, (num_samples, 2 * N), generator=rng)

    def __len__(self):
        return self.idx.size(0)

    def __getitem__(self, i):
        ids = self.idx[i]
        imgs, digits = [], []
        for j in range(2 * self.N):
            img, d = self.mnist[ids[j].item()]
            imgs.append(img)
            digits.append(d)
        images = torch.stack(imgs, dim=0)
        digits_t = torch.tensor(digits, dtype=torch.long)
        a = 0
        b = 0
        for k in range(self.N):
            a = a * 10 + int(digits[k])
            b = b * 10 + int(digits[self.N + k])
        return images, digits_t, a + b


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


class MNISTAddModel(nn.Module):
    """Shared LeNet over flattened (B*2N, 1, 28, 28) → (B, 2N, 10) logits."""

    def __init__(self, N=2):
        super().__init__()
        self.N = N
        self.backbone = LeNet()

    def forward(self, images):
        B, K = images.shape[0], images.shape[1]
        flat = images.view(B * K, 1, 28, 28)
        return self.backbone(flat).view(B, K, 10)


# ── Litleaves ────────────────────────────────────────────────────────

def build_sum_litleaves(sum_labels, batch_size, num_sum_vars, clamped=True):
    """Build litleaves for the sum indicator variables.

    Args:
        sum_labels: (B,) tensor of sum values 0..num_sum_vars-1
        batch_size: int
        num_sum_vars: 2*(10^N - 1) + 1
        clamped: if True, clamp to observed sum; if False, all free
    Returns:
        list of num_sum_vars [neg_weight, pos_weight] pairs, each (B, 1)
    """
    device = sum_labels.device
    litleaves = []
    zero_b1 = torch.zeros(batch_size, 1, device=device)
    log_zero_t = torch.tensor(LOG_ZERO, device=device)
    zero_t = torch.tensor(0.0, device=device)

    for s in range(num_sum_vars):
        if not clamped:
            litleaves.append([zero_b1, zero_b1])
        else:
            pos_w = torch.where(sum_labels == s, zero_t, log_zero_t).unsqueeze(1)
            litleaves.append([zero_b1, pos_w])
    return litleaves


# ── Epoch loops ──────────────────────────────────────────────────────

def _decode_numbers(digits, N):
    """Decode (B, 2N) digits into (a, b) tensors of shape (B,) MSB-first."""
    powers = 10 ** torch.arange(N - 1, -1, -1, device=digits.device)
    a = (digits[:, :N] * powers).sum(dim=1)
    b = (digits[:, N:] * powers).sum(dim=1)
    return a, b


def _run_epoch(model, cmpe, loader, optimizer, digit_loss_weight, device, N, train):
    if train:
        model.train()
    else:
        model.eval()

    num_digits = 2 * N
    num_sum_vars = 2 * (10 ** N - 1) + 1

    total_loss = 0.0
    total_digit = 0
    total_num = 0
    total_sum = 0
    total_samples = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for images, digit_labels, s in loader:
            images = images.to(device, non_blocking=True)
            digit_labels = digit_labels.to(device, non_blocking=True)
            s = s.to(device, non_blocking=True)
            B = images.size(0)

            logits = model(images)                             # (B, 2N, 10)
            log_probs = F.log_softmax(logits, dim=-1)

            # ── digit litleaves: 2N*10 vars ──
            digit_litleaves = []
            zero_b1 = torch.zeros(B, 1, device=device)
            for j in range(num_digits):
                for k in range(10):
                    digit_litleaves.append([zero_b1, log_probs[:, j, k].unsqueeze(1)])

            # ── sum litleaves ──
            sum_clamped = build_sum_litleaves(s, B, num_sum_vars, clamped=True)
            sum_free = build_sum_litleaves(s, B, num_sum_vars, clamped=False)

            log_num = cmpe.beta.generate_tf_ac(
                digit_litleaves + sum_clamped, log_space=True
            ).squeeze(-1)
            log_den = cmpe.beta.generate_tf_ac(
                digit_litleaves + sum_free, log_space=True
            ).squeeze(-1)

            sum_nll = -(log_num - log_den).mean()
            digit_ce = F.nll_loss(log_probs.reshape(-1, 10), digit_labels.reshape(-1))
            loss = sum_nll + digit_loss_weight * digit_ce

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * B
            pred = log_probs.argmax(dim=-1)                    # (B, 2N)
            total_digit += (pred == digit_labels).sum().item()
            pred_a, pred_b = _decode_numbers(pred, N)
            true_a, true_b = _decode_numbers(digit_labels, N)
            total_num += ((pred_a == true_a).sum() + (pred_b == true_b).sum()).item()
            total_sum += ((pred_a + pred_b) == s).sum().item()
            total_samples += B

    n = total_samples
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )
    return {
        "loss": total_loss / n,
        "digit_acc": total_digit / (num_digits * n),
        "num_acc": total_num / (2 * n),
        "sum_acc": total_sum / n,
        "violation_rate": 0.0,  # 0 by construction with SDD constraint
        "peak_mem_mb": peak_mem_mb,
    }


def train_epoch(model, cmpe, loader, optimizer, digit_loss_weight, device, N):
    return _run_epoch(model, cmpe, loader, optimizer, digit_loss_weight, device, N, train=True)


def eval_epoch(model, cmpe, loader, digit_loss_weight, device, N):
    return _run_epoch(model, cmpe, loader, None, digit_loss_weight, device, N, train=False)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MNIST-Add with SPL")
    parser.add_argument("--N", type=int, default=2,
                        help="digits per operand (N=2 → 53+44)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--digit-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  N={args.N} digits per operand")

    # ── Load SDD ──
    vtree_path = f"sdd_files/mnistadd_N{args.N}.vtree"
    sdd_path = f"sdd_files/mnistadd_N{args.N}.sdd"
    assert os.path.exists(sdd_path), f"Missing {sdd_path} — run compile_mnistadd_sdd.py first"

    cmpe = CircuitMPE(vtree_path, sdd_path)
    cmpe.beta.num_reps = 1
    print(f"Loaded SDD from {sdd_path}")

    # ── Data ──
    train_ds = MNISTAddDataset(
        N=args.N, train=True, num_samples=args.train_samples, seed=args.seed
    )
    test_ds = MNISTAddDataset(
        N=args.N, train=False, num_samples=args.test_samples, seed=args.seed + 57
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")

    # ── Model ──
    model = MNISTAddModel(N=args.N).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Training loop ──
    print(
        f"\n{'Epoch':>5} {'Loss':>8} {'DigAcc':>8} {'NumAcc':>8} "
        f"{'SumAcc':>8} {'Viol':>6} {'Time':>7} {'PeakMem(MB)':>12}"
    )
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        m = train_epoch(model, cmpe, train_loader, optimizer,
                        args.digit_loss_weight, device, args.N)
        elapsed = time.time() - t0

        print(
            f"{epoch:5d} "
            f"{m['loss']:8.4f} "
            f"{m['digit_acc']:8.4f} "
            f"{m['num_acc']:8.4f} "
            f"{m['sum_acc']:8.4f} "
            f"{m['violation_rate']:6.3f} "
            f"{elapsed:6.1f}s "
            f"{m['peak_mem_mb']:12.1f}"
        )

    # ── Final evaluation ──
    print("\n--- Test set evaluation ---")
    tm = eval_epoch(model, cmpe, test_loader, args.digit_loss_weight, device, args.N)
    print(f"  Loss:            {tm['loss']:.4f}")
    print(f"  Digit accuracy:  {tm['digit_acc']:.4f}")
    print(f"  Number accuracy: {tm['num_acc']:.4f}")
    print(f"  Sum accuracy:    {tm['sum_acc']:.4f}   (consistency)")
    print(f"  Violation rate:  {tm['violation_rate']:.4f}")
    print(f"  Peak GPU mem:    {tm['peak_mem_mb']:.1f} MB")


if __name__ == "__main__":
    main()
