import argparse
import csv
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    MNISTSumDataset,
    MNISTAddDataset,
    mnist_sum_num_samples,
    mnist_add_num_samples,
)
from loss import MaskedLoss, PlainSumLoss
from model import MNISTConstraintNet
from carry_chain import compute_carry_chain


class IterTimeoutError(Exception):
    pass


@contextmanager
def _iter_timeout(seconds):
    """SIGALRM-based wall-clock cap on a single training step. No-op if seconds <= 0."""
    if seconds is None or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise IterTimeoutError(f"training step exceeded {seconds:.1f}s")

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


def _make_addn_constraint(n_digits):
    """Build a place-value addition constraint function for two N-digit numbers."""
    def _addn_constraint(digit_preds):
        n = n_digits
        first, second = digit_preds[:n], digit_preds[n:]
        first_num = sum(d * (10 ** (n - 1 - i)) for i, d in enumerate(first))
        second_num = sum(d * (10 ** (n - 1 - i)) for i, d in enumerate(second))
        return first_num + second_num
    return _addn_constraint


def get_config(n_digits, task):
    """Return (variable_ranges, n_inputs, constraint_fn) for the chosen task."""
    if task == 'sum':
        max_sum = 9 * n_digits
        return [10] * n_digits + [max_sum + 1], n_digits, None
    if task == 'add':
        n_inputs = 2 * n_digits
        n_output_digits = n_digits + 1
        return [10] * (n_inputs + n_output_digits), n_inputs, _make_addn_constraint(n_digits)
    raise ValueError(f"Unknown task: {task!r}")


def resolve_compiled_file(task, n_digits, explicit=None):
    """Path to the compiled constraint artifact for (task, n_digits).

    MNIST-Add's carry-chain DSL is its own compile fixpoint at every N, so a
    missing artifact is generated rather than compiled. MNIST-Sum's flat
    encoding gives S a domain of 9N+1 and only N=2 is tractable; larger N needs
    the digit-tree encoding, which has a different variable layout than
    get_config() builds, so it is refused here rather than silently downgraded.
    """
    if explicit is not None:
        return explicit

    examples = Path(__file__).resolve().parents[1] / "examples"
    if task == 'add':
        dsl = examples / f"mnist_add_n{n_digits}.txt"
        if not dsl.exists():
            sys.path.append(str(examples))
            from generate_mnist_add import generate_mnist_add_dsl
            dsl.write_text(generate_mnist_add_dsl(n_digits))
        return str(dsl)

    artifact = examples / f"mnist_sum_n{n_digits}_compiled.txt"
    if artifact.exists():
        return str(artifact)
    raise SystemExit(
        f"no compiled artifact for task=sum N={n_digits} "
        f"({artifact} not found). The flat sum encoding is only tractable at "
        f"N=2; for larger N use the digit-tree encoding, or pass "
        f"--projection-backend closed_form to run the archived layer."
    )


def split_logits(logits, variable_ranges):
    out, offset = [], 0
    for n_cats in variable_ranges:
        out.append(logits[:, offset:offset + n_cats])
        offset += n_cats
    return out


CSV_HEADER = [
    'task', 'n_digits', 'projection_backend', 'constraint_mode', 'run_id', 'epoch', 'loss',
    'neural_acc', 'symbolic_acc', 'constraint_sat',
    'avg_digit_acc', 'avg_l2_edit', 'grad_norm',
    'gpu_mem_mb', 'epoch_time_s',
]


def _nan_row(task, n_digits, backend, constraint_mode, run_id, epoch, gpu_mem_mb, epoch_time):
    nan = float('nan')
    return [task, n_digits, backend, constraint_mode, run_id, epoch, nan,
            nan, nan, nan, nan, nan, nan,
            gpu_mem_mb, epoch_time]


def train_epoch_sum(model, loader, optimizer, criterion, variable_ranges,
                    n_digits, device, per_iter_timeout_s):
    model.train()
    total_loss, total_l2, n_batches = 0.0, 0.0, 0
    total_grad_norm, grad_norm_count = 0.0, 0
    timed_out = False

    pbar = tqdm(loader, leave=False, desc="train")
    for batch_idx, (imgs, digit_labels, sum_label) in enumerate(pbar):
        imgs = imgs.to(device)
        digit_labels = digit_labels.to(device)
        sum_label = sum_label.to(device)

        try:
            with _iter_timeout(per_iter_timeout_s):
                logits = model.get_logits(imgs)
                logit_slices = split_logits(logits, variable_ranges)
                target_list = [digit_labels[:, i] for i in range(n_digits)] + [sum_label]
                loss, info = criterion(*logit_slices, *target_list)

                optimizer.zero_grad()
                loss.backward()

                grad_params = [p for p in model.digit_encoder.parameters() if p.grad is not None]
                if grad_params:
                    g = torch.sqrt(sum((p.grad ** 2).sum() for p in grad_params)).item()
                    total_grad_norm += g
                    grad_norm_count += 1

                optimizer.step()
        except IterTimeoutError as e:
            print(f"  [TIMEOUT] sum step exceeded budget at iter {batch_idx}: {e}")
            timed_out = True
            break

        total_loss += loss.item()
        total_l2 += info['mean_l2_edit']
        n_batches += 1

    denom = max(1, n_batches)
    avg_grad_norm = total_grad_norm / grad_norm_count if grad_norm_count > 0 else float('nan')
    return {
        'loss': total_loss / denom,
        'avg_l2': total_l2 / denom,
        'grad_norm': avg_grad_norm,
        'timed_out': timed_out,
    }


def evaluate_sum(model, loader, variable_ranges, n_digits, device, apply_constraint=True):
    model.eval()
    n_vars = len(variable_ranges)
    correct = {f'a{i+1}': 0 for i in range(n_vars)}
    correct['constraint'] = 0
    correct['symbolic_sum'] = 0
    total = 0
    total_l2 = 0.0

    with torch.no_grad():
        for imgs, digit_labels, sum_label in tqdm(loader, leave=False, desc="eval"):
            imgs = imgs.to(device)
            digit_labels = digit_labels.to(device)
            sum_label = sum_label.to(device)

            if apply_constraint:
                raw_probs, eval_probs, _info = model.forward_diagnostic(imgs)
                l2_edit = torch.norm(eval_probs - raw_probs, p=2, dim=1).sum().item()
            else:
                eval_probs = model.forward_raw(imgs)
                l2_edit = 0.0

            proj_slices = split_logits(eval_probs, variable_ranges)
            preds = [s.argmax(dim=1) for s in proj_slices]

            target_list = [digit_labels[:, i] for i in range(n_digits)] + [sum_label]
            for i in range(n_vars):
                correct[f'a{i+1}'] += (preds[i] == target_list[i]).sum().item()

            symbolic_result = sum(preds[:-1])
            correct['constraint'] += (symbolic_result == preds[-1]).sum().item()
            correct['symbolic_sum'] += (symbolic_result == sum_label).sum().item()

            total_l2 += l2_edit
            total += imgs.shape[0]

    return {
        'avg_digit_acc': 100 * sum(correct[f'a{i+1}'] for i in range(n_digits)) / (total * n_digits),
        'neural_acc': 100 * correct[f'a{n_vars}'] / total,
        'symbolic_acc': 100 * correct['symbolic_sum'] / total,
        'constraint_sat': 100 * correct['constraint'] / total,
        'avg_l2_edit': total_l2 / total,
    }


def train_epoch_add(model, loader, optimizer, variable_ranges, n_inputs,
                    device, lambda_proj, per_iter_timeout_s, apply_constraint=True):
    model.train()
    n_output = len(variable_ranges) - n_inputs
    total_loss, total_proj = 0.0, 0.0
    total_grad_norm, n_batches = 0.0, 0
    timed_out = False

    pbar = tqdm(loader, leave=False, desc="train")
    for batch_idx, (imgs, digit_labels, output_digit_labels, _sum_label) in enumerate(pbar):
        imgs = imgs.to(device)
        digit_labels = digit_labels.to(device)
        output_digit_labels = output_digit_labels.to(device)

        try:
            with _iter_timeout(per_iter_timeout_s):
                logits = model.get_logits(imgs)
                logit_slices = split_logits(logits, variable_ranges)

                input_ce = sum(F.cross_entropy(logit_slices[i], digit_labels[:, i])
                               for i in range(n_inputs))
                output_ce = sum(F.cross_entropy(logit_slices[n_inputs + j],
                                                output_digit_labels[:, j])
                                for j in range(n_output))

                if apply_constraint:
                    probs = torch.cat([F.softmax(s, dim=1) for s in logit_slices], dim=1)
                    _proj_probs, proj_info = model.constraint_layer(probs)
                    proj_loss = proj_info['projection_loss']
                    loss = input_ce + output_ce + lambda_proj * proj_loss
                else:
                    proj_loss = torch.zeros((), device=device)
                    loss = input_ce + output_ce

                optimizer.zero_grad()
                loss.backward()

                grad_params = [p for p in model.digit_encoder.parameters() if p.grad is not None]
                if grad_params:
                    g = torch.sqrt(sum((p.grad ** 2).sum() for p in grad_params)).item()
                    total_grad_norm += g

                optimizer.step()
        except IterTimeoutError as e:
            print(f"  [TIMEOUT] add step exceeded budget at iter {batch_idx}: {e}")
            timed_out = True
            break

        total_loss += loss.item()
        total_proj += proj_loss.item()
        n_batches += 1

    n = max(1, n_batches)
    return {
        'loss': total_loss / n,
        'proj_loss': total_proj / n,
        'grad_norm': total_grad_norm / n,
        'timed_out': timed_out,
    }


def evaluate_add(model, loader, variable_ranges, n_inputs, n_digits_per_num, device,
                 apply_constraint=True):
    model.eval()
    n_output = n_digits_per_num + 1

    input_correct = 0
    output_correct = 0
    sum_correct = 0
    chain_satisfied = 0
    total_l2 = 0.0
    total = 0

    with torch.no_grad():
        for imgs, digit_labels, output_digit_labels, _sum_label in tqdm(loader, leave=False, desc="eval"):
            imgs = imgs.to(device)
            digit_labels = digit_labels.to(device)
            output_digit_labels = output_digit_labels.to(device)
            B = imgs.shape[0]

            if apply_constraint:
                raw_probs, eval_probs, _info = model.forward_diagnostic(imgs)
                l2_edit = torch.norm(eval_probs - raw_probs, p=2, dim=1).sum().item()
            else:
                raw_probs = model.forward_raw(imgs)
                eval_probs = raw_probs
                l2_edit = 0.0

            raw_slices = split_logits(raw_probs, variable_ranges)
            proj_slices = split_logits(eval_probs, variable_ranges)

            for i in range(n_inputs):
                input_correct += (raw_slices[i].argmax(dim=1) == digit_labels[:, i]).sum().item()

            proj_output_preds = torch.stack(
                [proj_slices[n_inputs + j].argmax(dim=1) for j in range(n_output)], dim=1)
            output_correct += (proj_output_preds == output_digit_labels).sum().item()
            sum_correct += (proj_output_preds == output_digit_labels).all(dim=1).sum().item()

            input_preds = torch.stack(
                [raw_slices[i].argmax(dim=1) for i in range(n_inputs)], dim=1)
            a_preds = input_preds[:, :n_digits_per_num]
            b_preds = input_preds[:, n_digits_per_num:]
            s_consistent = compute_carry_chain(a_preds, b_preds)
            chain_satisfied += (proj_output_preds == s_consistent).all(dim=1).sum().item()

            total_l2 += l2_edit
            total += B

    return {
        'avg_digit_acc': 100 * input_correct / (total * n_inputs),
        'neural_acc': 100 * output_correct / (total * n_output),
        'symbolic_acc': 100 * sum_correct / total,
        'constraint_sat': 100 * chain_satisfied / total,
        'avg_l2_edit': total_l2 / total,
    }


def run_one(task, n_digits, seed, epochs, lr, batch_size, device,
            train_loader, test_loader, csv_writer, csv_file,
            projection_backend, solver_time_limit,
            per_iter_timeout_s, max_run_time_s, constraint_mode='train_infer',
            compiled_file=None):
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    train_constraint = constraint_mode in ('train_infer', 'train_only')
    infer_constraint = constraint_mode in ('train_infer', 'infer_only')

    variable_ranges, n_inputs, constraint_fn = get_config(n_digits, task)
    n_vars = len(variable_ranges)

    model = MNISTConstraintNet(
        task=task,
        variable_ranges=variable_ranges,
        n_inputs=n_inputs,
        constraint_fn=constraint_fn,
        projection_backend=projection_backend,
        solver_time_limit=solver_time_limit,
        compiled_file=compiled_file,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    if task == 'sum':
        if train_constraint:
            n_sum_categories = variable_ranges[-1]
            criterion = MaskedLoss(n_sum_categories=n_sum_categories,
                                   constraint_fn=constraint_fn)
        else:
            criterion = PlainSumLoss()

    print(f"\n{'=' * 60}")
    print(f"task={task}  N={n_digits}  seed={seed}  backend={projection_backend}  "
          f"mode={constraint_mode}")
    print(f"epochs={epochs}  lr={lr}  batch_size={batch_size}")
    print(f"{'=' * 60}")

    run_start = time.time()
    aborted = False

    for epoch in range(epochs):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.time()

        if task == 'sum':
            train_info = train_epoch_sum(
                model, train_loader, optimizer, criterion, variable_ranges,
                n_digits, device, per_iter_timeout_s)
        else:
            train_info = train_epoch_add(
                model, train_loader, optimizer, variable_ranges, n_inputs,
                device, lambda_proj=1.0, per_iter_timeout_s=per_iter_timeout_s,
                apply_constraint=train_constraint)

        if train_info['timed_out']:
            epoch_time = time.time() - epoch_start
            gpu_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                          if device.type == 'cuda' else float('nan'))
            csv_writer.writerow(_nan_row(task, n_digits, projection_backend,
                                         constraint_mode, seed, epoch + 1,
                                         gpu_mem_mb, epoch_time))
            csv_file.flush()
            print(f"  Aborting after epoch {epoch+1} timeout marker.")
            aborted = True
            break

        if task == 'sum':
            metrics = evaluate_sum(model, test_loader, variable_ranges, n_digits, device,
                                   apply_constraint=infer_constraint)
        else:
            metrics = evaluate_add(model, test_loader, variable_ranges, n_inputs,
                                   n_digits, device, apply_constraint=infer_constraint)

        epoch_time = time.time() - epoch_start
        gpu_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                      if device.type == 'cuda' else float('nan'))

        csv_writer.writerow([
            task, n_digits, projection_backend, constraint_mode, seed, epoch + 1,
            train_info['loss'],
            metrics['neural_acc'],
            metrics['symbolic_acc'],
            metrics['constraint_sat'],
            metrics['avg_digit_acc'],
            metrics['avg_l2_edit'],
            train_info['grad_norm'],
            gpu_mem_mb, epoch_time,
        ])
        csv_file.flush()

        print(f"Epoch {epoch+1:3d} | loss={train_info['loss']:.4f} | "
              f"digit={metrics['avg_digit_acc']:.1f}% | "
              f"neural={metrics['neural_acc']:.1f}% | "
              f"symbolic={metrics['symbolic_acc']:.1f}% | "
              f"sat={metrics['constraint_sat']:.1f}% | "
              f"L2={metrics['avg_l2_edit']:.4f} | {epoch_time:.1f}s")

        scheduler.step()

        if max_run_time_s is not None and (time.time() - run_start) > max_run_time_s:
            elapsed = time.time() - run_start
            csv_writer.writerow(_nan_row(task, n_digits, projection_backend,
                                         constraint_mode, seed, epoch + 1,
                                         gpu_mem_mb, elapsed))
            csv_file.flush()
            print(f"  Per-run cap hit ({elapsed:.1f}s > {max_run_time_s}s). Aborting.")
            aborted = True
            break

    return aborted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['sum', 'add'])
    parser.add_argument('--n-digits', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--projection-backend', type=str, default='compiled',
                        choices=['compiled', 'closed_form', 'cvxpy'],
                        help="'compiled' (default) projects against the "
                             "compiled clause artifact; 'closed_form' and "
                             "'cvxpy' run the archived on-the-fly layers.")
    parser.add_argument('--compiled-file', type=str, default=None,
                        help="Compiled constraint artifact; required by "
                             "--projection-backend compiled.")
    parser.add_argument('--constraint-mode', type=str, default='train_infer',
                        choices=['train_infer', 'train_only', 'infer_only', 'none'],
                        help='When the constraint is applied: at train+inference, '
                             'train only, inference only, or never (unconstrained).')
    parser.add_argument('--solver-time-limit', type=float, default=None)
    parser.add_argument('--per-iter-timeout-s', type=float, default=None)
    parser.add_argument('--max-run-time-s', type=float, default=None)
    parser.add_argument('--data-root', type=str, default='./data')
    args = parser.parse_args()

    if args.epochs is None:
        args.epochs = 200 if args.task == 'sum' else 100
    if args.batch_size is None:
        args.batch_size = 16 if args.task == 'sum' else 64
    if args.lr is None:
        if args.task == 'sum':
            args.lr = 1e-3
        else:
            args.lr = 3e-4 if args.n_digits in (100, 256) else 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(
        args.data_root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(
        args.data_root, train=False, download=True, transform=transform)

    if args.task == 'sum':
        train_n = mnist_sum_num_samples('train')
        test_n = mnist_sum_num_samples('test')
        train_ds = MNISTSumDataset(mnist_train, n_digits=args.n_digits,
                                   num_samples=train_n, data_seed=args.seed)
        test_ds = MNISTSumDataset(mnist_test, n_digits=args.n_digits,
                                  num_samples=test_n, data_seed=args.seed + 10000)
    else:
        train_n = mnist_add_num_samples(args.n_digits, 'train')
        test_n = mnist_add_num_samples(args.n_digits, 'test')
        train_ds = MNISTAddDataset(mnist_train, n_digits=args.n_digits,
                                   num_samples=train_n, data_seed=args.seed)
        test_ds = MNISTAddDataset(mnist_test, n_digits=args.n_digits,
                                  num_samples=test_n, data_seed=args.seed + 10000)

    print(f"  train={train_n}  test={test_n}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    write_header = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
    with open(args.csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
            f.flush()
        run_one(
            task=args.task,
            n_digits=args.n_digits,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            csv_writer=writer,
            csv_file=f,
            projection_backend=args.projection_backend,
            solver_time_limit=args.solver_time_limit,
            compiled_file=(
                resolve_compiled_file(args.task, args.n_digits, args.compiled_file)
                if args.projection_backend == 'compiled' else None
            ),
            per_iter_timeout_s=args.per_iter_timeout_s,
            max_run_time_s=args.max_run_time_s,
            constraint_mode=args.constraint_mode,
        )


if __name__ == "__main__":
    main()
