#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import MNISTAddDataset, mnist_add_num_samples  # noqa: E402
from model import MNISTConstraintNet  # noqa: E402
from train import get_config, split_logits  # noqa: E402


DEFAULT_N = (1, 2, 4, 15, 100, 256)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(model, loader, optimizer, variable_ranges, n_inputs, device):
    model.train()
    n_output = len(variable_ranges) - n_inputs
    total_loss = 0.0
    batches = 0
    sync(device)
    start = time.perf_counter()
    for images, digit_labels, output_labels, _ in loader:
        images = images.to(device, non_blocking=True)
        digit_labels = digit_labels.to(device, non_blocking=True)
        output_labels = output_labels.to(device, non_blocking=True)

        logits = model.get_logits(images)
        slices = split_logits(logits, variable_ranges)
        input_ce = sum(
            F.cross_entropy(slices[i], digit_labels[:, i])
            for i in range(n_inputs)
        )
        output_ce = sum(
            F.cross_entropy(slices[n_inputs + j], output_labels[:, j])
            for j in range(n_output)
        )
        probabilities = torch.cat(
            [F.softmax(part, dim=1) for part in slices], dim=1
        )
        _, projection_info = model.constraint_layer(probabilities)
        loss = input_ce + output_ce + projection_info["projection_loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item())
        batches += 1
    sync(device)
    return time.perf_counter() - start, total_loss / max(batches, 1), batches


@torch.no_grad()
def inference_pass(model, loader, variable_ranges, n_inputs, device):
    model.eval()
    n_output = len(variable_ranges) - n_inputs
    samples = 0
    batches = 0
    exact = 0
    digit_correct = 0
    sync(device)
    start = time.perf_counter()
    for images, _, output_labels, _ in loader:
        images = images.to(device, non_blocking=True)
        output_labels = output_labels.to(device, non_blocking=True)
        _, projected, _ = model.forward_diagnostic(images)
        slices = split_logits(projected, variable_ranges)
        predictions = torch.stack(
            [
                slices[n_inputs + j].argmax(dim=1)
                for j in range(n_output)
            ],
            dim=1,
        )
        exact += int((predictions == output_labels).all(dim=1).sum().item())
        digit_correct += int((predictions == output_labels).sum().item())
        samples += images.shape[0]
        batches += 1
    sync(device)
    elapsed = time.perf_counter() - start
    return {
        "elapsed_s": elapsed,
        "batches": batches,
        "samples": samples,
        "exact_accuracy_pct": 100 * exact / max(samples, 1),
        "digit_accuracy_pct": (
            100 * digit_correct / max(samples * n_output, 1)
        ),
    }


def warm_up(model, loader, variable_ranges, n_inputs, device) -> None:
    model.train()
    batch = next(iter(loader))
    images, digit_labels, output_labels, _ = batch
    images = images.to(device)
    digit_labels = digit_labels.to(device)
    output_labels = output_labels.to(device)
    n_output = len(variable_ranges) - n_inputs
    logits = model.get_logits(images)
    slices = split_logits(logits, variable_ranges)
    loss = sum(
        F.cross_entropy(slices[i], digit_labels[:, i])
        for i in range(n_inputs)
    ) + sum(
        F.cross_entropy(slices[n_inputs + j], output_labels[:, j])
        for j in range(n_output)
    )
    loss.backward()
    model.zero_grad(set_to_none=True)
    sync(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", nargs="+", type=int, default=list(DEFAULT_N))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--inference-repetitions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=(
            ROOT / "benchmark_results"
            / "mnist_add_general_compiler_functional_fastpath"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "mnist_add_training_inference",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    print(f"Device: {torch.cuda.get_device_name(device)}", flush=True)

    transform = transforms.ToTensor()
    train_base = torchvision.datasets.MNIST(
        args.data_root, train=True, download=True, transform=transform
    )
    test_base = torchvision.datasets.MNIST(
        args.data_root, train=False, download=True, transform=transform
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for n_digits in args.n:
        print(f"Benchmarking N={n_digits}...", flush=True)
        variable_ranges, n_inputs, _ = get_config(n_digits, "add")
        artifact = args.artifact_dir / f"mnist_add_n{n_digits}_compiled.txt"
        if not artifact.exists():
            raise FileNotFoundError(artifact)

        train_dataset = MNISTAddDataset(
            train_base,
            n_digits,
            mnist_add_num_samples(n_digits, "train"),
            data_seed=args.seed,
        )
        test_dataset = MNISTAddDataset(
            test_base,
            n_digits,
            mnist_add_num_samples(n_digits, "test"),
            data_seed=args.seed + 10_000,
        )
        loader_args = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

        model = MNISTConstraintNet(
            task="add",
            variable_ranges=variable_ranges,
            n_inputs=n_inputs,
            projection_backend="compiled",
            compiled_file=artifact,
        ).to(device)
        learning_rate = 3e-4 if n_digits in (100, 256) else 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        warm_up(model, train_loader, variable_ranges, n_inputs, device)

        epoch_times = []
        epoch_losses = []
        train_batches = 0
        for epoch in range(args.epochs):
            elapsed, loss, train_batches = train_epoch(
                model, train_loader, optimizer, variable_ranges,
                n_inputs, device,
            )
            epoch_times.append(elapsed)
            epoch_losses.append(loss)
            print(
                f"  epoch {epoch + 1}/{args.epochs}: "
                f"{elapsed:.3f}s loss={loss:.4f}",
                flush=True,
            )

        # One untimed full pass warms data-loader workers and inference paths.
        inference_pass(model, test_loader, variable_ranges, n_inputs, device)
        inference_results = [
            inference_pass(
                model, test_loader, variable_ranges, n_inputs, device
            )
            for _ in range(args.inference_repetitions)
        ]
        inference_times = [result["elapsed_s"] for result in inference_results]
        reference = inference_results[-1]

        row = {
            "n_digits": n_digits,
            "device": torch.cuda.get_device_name(device),
            "batch_size": args.batch_size,
            "train_samples": len(train_dataset),
            "train_batches": train_batches,
            "measured_epochs": args.epochs,
            "avg_epoch_s": statistics.fmean(epoch_times),
            "std_epoch_s": (
                statistics.stdev(epoch_times) if len(epoch_times) > 1 else 0.0
            ),
            "avg_train_loss": statistics.fmean(epoch_losses),
            "test_samples": reference["samples"],
            "test_batches": reference["batches"],
            "inference_repetitions": args.inference_repetitions,
            "avg_inference_pass_s": statistics.fmean(inference_times),
            "std_inference_pass_s": (
                statistics.stdev(inference_times)
                if len(inference_times) > 1 else 0.0
            ),
            "avg_inference_batch_s": (
                statistics.fmean(inference_times) / reference["batches"]
            ),
            "avg_inference_sample_s": (
                statistics.fmean(inference_times) / reference["samples"]
            ),
            "exact_accuracy_pct": reference["exact_accuracy_pct"],
            "digit_accuracy_pct": reference["digit_accuracy_pct"],
        }
        rows.append(row)
        print(
            f"  avg epoch={row['avg_epoch_s']:.3f}s; "
            f"inference={row['avg_inference_batch_s']:.6f}s/batch",
            flush=True,
        )
        del model, optimizer, train_loader, test_loader
        torch.cuda.empty_cache()

    csv_path = args.output_dir / "timings.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
