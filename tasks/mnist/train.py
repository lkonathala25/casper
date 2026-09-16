import sys as _sys
from pathlib import Path as _Path

# Runnable entry point: put the shared compiler and this task's own directory on
# the path. Sibling task directories are deliberately not added -- they share
# module names (data, model, train, minedit).
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "casper", _Path(__file__).resolve().parent):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

import argparse
import hashlib
import json
import socket
import subprocess
import os
import resource
import signal
import sys
import time
from collections import defaultdict
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
from loss import PlainSumLoss
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


def n_sum_digits(n_digits):
    """Answer digits in the balanced decimal sum tree: one per level, plus one."""
    if n_digits < 2 or n_digits & (n_digits - 1):
        raise ValueError(f"digit-tree MNIST-Sum needs a power-of-two N, got {n_digits}")
    return n_digits.bit_length()


def sum_to_digits(totals, k):
    """(B,) integer totals -> (B, k) decimal digits, most significant first."""
    out = []
    x = totals
    for _ in range(k):
        out.append(x % 10)
        x = torch.div(x, 10, rounding_mode='floor')
    return torch.stack(out[::-1], dim=1)


def digits_to_sum(digit_preds):
    """List of k (B,) digit predictions, most significant first -> (B,) integers."""
    total = torch.zeros_like(digit_preds[0])
    for d in digit_preds:
        total = total * 10 + d
    return total


def _sum_targets(digit_labels, sum_label, n_digits, n_vars):
    """Per-variable targets for MNIST-Sum under either output encoding."""
    targets = [digit_labels[:, i] for i in range(n_digits)]
    n_out = n_vars - n_digits
    if n_out == 1:
        return targets + [sum_label]
    digits = sum_to_digits(sum_label, n_out)
    return targets + [digits[:, j] for j in range(n_out)]


def get_config(n_digits, task):
    """Return (variable_ranges, n_inputs) for the chosen task.

    MNIST-Sum's answer is the balanced decimal digit tree's output digits; the
    flat single-sum-variable encoding needs one clause per combination of all N
    digits and lives with the archived on-the-fly layers.
    """
    if task == 'sum':
        return [10] * (n_digits + n_sum_digits(n_digits)), n_digits
    if task == 'add':
        n_inputs = 2 * n_digits
        return [10] * (n_inputs + n_digits + 1), n_inputs
    raise ValueError(f"Unknown task: {task!r}")


def resolve_compiled_file(task, n_digits, explicit=None):
    """Path to the constraint file for (task, n_digits), generating it if absent.

    Both tasks' constraint files are their own compile fixpoints -- every bucket
    is a deterministic function of earlier variables, so eliminating anything
    yields only tautologies. They are therefore generated and loaded directly,
    never compiled.
    """
    if explicit is not None:
        return explicit

    root = Path(__file__).resolve().parents[2]
    out = root / "constraints" / "mnist"
    out.mkdir(parents=True, exist_ok=True)
    sys.path.append(str(Path(__file__).resolve().parent))
    if task == 'add':
        dsl = out / f"mnist_add_n{n_digits}.txt"
        if not dsl.exists():
            from generate_mnist_add import generate_mnist_add_dsl
            dsl.write_text(generate_mnist_add_dsl(n_digits))
        return str(dsl)

    dsl = out / f"mnist_sum_digit_tree_n{n_digits}.txt"
    if not dsl.exists():
        from generate_mnist_sum_digit_tree import generate_mnist_sum_digit_tree_dsl
        dsl.write_text(generate_mnist_sum_digit_tree_dsl(n_digits))
    return str(dsl)


def check_fixpoint(path):
    """Refuse to train unless a generated constraint file is its own compile.

    The layer loads the file with Constraints.from_compiled_dsl, i.e. as already
    compiled, which is only sound if eliminating any variable derives nothing new.
    compile_functional_dag verifies that (every bucket a total function of earlier
    variables) in linear time and raises NotFunctionalDAG otherwise. Returns the
    check time and the constraint set's size for the run config.
    """
    from functional_dag_compile import NotFunctionalDAG, compile_functional_dag
    from signed_clauses import parse_constraints
    start = time.perf_counter()
    cs = parse_constraints(Path(path).read_text())
    try:
        compile_functional_dag(cs)
    except NotFunctionalDAG as err:
        raise SystemExit(
            f"{path} is not a functional-DAG fixpoint ({err}); loading it as compiled "
            f"would enforce an incomplete constraint. Compile it and pass --compiled-file.")
    elapsed = time.perf_counter() - start
    print(f"Fixpoint check: {Path(path).name} is a functional DAG, loaded as compiled "
          f"({elapsed:.1f}s)", flush=True)
    return {'fixpoint_check_s': elapsed, 'n_variables': len(cs.var_names),
            'n_clauses': len(cs.clauses)}


def split_logits(logits, variable_ranges):
    out, offset = [], 0
    for n_cats in variable_ranges:
        out.append(logits[:, offset:offset + n_cats])
        offset += n_cats
    return out


METRIC_KEYS = [
    'digit_exact', 'digit_macro', 'operand1_exact', 'operand1_macro', 'operand2_exact', 'operand2_macro',
    'sum_exact', 'sum_macro', 'raw_sum_exact', 'raw_sum_macro', 'symbolic_sum_exact',
    'constraint_sat', 'avg_l2_edit',
]


def _sync(device):
    """Wait for queued CUDA work so wall-clock timings are accurate."""
    if device.type == 'cuda':
        torch.cuda.synchronize()


class _LayerTimer:
    """Accumulates wall time spent inside the constraint layer, synchronising CUDA around each call."""

    def __init__(self, device):
        self.device = device
        self.total = 0.0

    @contextmanager
    def __call__(self):
        _sync(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.device)
            self.total += time.perf_counter() - start


def _argmax_slices(probs, variable_ranges):
    """(B, sum(variable_ranges)) probabilities -> (B, n_vars) argmax per variable."""
    return torch.stack([sl.argmax(dim=1) for sl in split_logits(probs, variable_ranges)], dim=1)


def _exact_macro(pred, target):
    """Row sums of (every position right, fraction of positions right) for (B, P) tensors."""
    ok = pred == target
    return ok.all(dim=1).sum().item(), ok.float().mean(dim=1).sum().item()


def _git_describe():
    """Commit of the code being run, marked -dirty when the working tree has changes."""
    try:
        out = subprocess.run(
            ['git', '-C', str(Path(__file__).resolve().parents[2]), 'describe', '--always', '--dirty'],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _json_safe(obj):
    """Replace NaN/inf with None, recursively, so the file is strict JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_run_json(path, record):
    """Atomically rewrite the run's JSON (config, epochs, summary) so a crash never leaves it half-written."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(_json_safe(record), indent=2, default=str))
    os.replace(tmp, path)


def save_checkpoint(path, model, optimizer, scheduler, epoch, row):
    """Atomically save model, optimizer and scheduler state after an epoch, with that epoch's metrics."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'metrics': _json_safe(row)}, tmp)
    os.replace(tmp, path)


def summarise_run(history, status, wall_time_s):
    """Final and best-epoch metrics plus total cost, for the run JSON's summary."""
    done = [r for r in history if not r['timed_out']]
    total = lambda rows, k: float(sum(r[k] for r in rows))
    examples = sum(r['n_train'] for r in done)
    return {
        'status': status,
        'epochs_completed': len(done),
        'wall_time_s': wall_time_s,
        'train_time_s': total(history, 'train_time_s'),
        'train_layer_time_s': total(history, 'train_layer_time_s'),
        'eval_time_s': total(history, 'eval_time_s'),
        'eval_layer_time_s': total(history, 'eval_layer_time_s'),
        'gpu_mem_peak_mb': max((r['gpu_mem_peak_mb'] for r in history), default=float('nan')),
        'gpu_mem_reserved_peak_mb': max((r['gpu_mem_reserved_peak_mb'] for r in history), default=float('nan')),
        'cpu_rss_peak_mb': max((r['cpu_rss_peak_mb'] for r in history), default=float('nan')),
        'mean_s_per_train_example': total(done, 'train_time_s') / examples if examples else float('nan'),
        'final': done[-1] if done else None,
        'best_by_sum_exact': max(done, key=lambda r: r['sum_exact']) if done else None,
    }


def train_epoch_sum(model, loader, optimizer, criterion, variable_ranges,
                    n_digits, device, per_iter_timeout_s,
                    apply_constraint=False, lambda_proj=1.0, layer_timer=None):
    model.train()
    total_loss, total_l2, total_proj, n_batches = 0.0, 0.0, 0.0, 0
    layer_timer = layer_timer or _LayerTimer(device)
    total_grad_norm, grad_norm_count = 0.0, 0
    timed_out = False

    pbar = tqdm(loader, leave=False, desc="train", disable=None)
    for batch_idx, (imgs, digit_labels, sum_label) in enumerate(pbar):
        imgs = imgs.to(device)
        digit_labels = digit_labels.to(device)
        sum_label = sum_label.to(device)
        batch_proj = 0.0

        try:
            with _iter_timeout(per_iter_timeout_s):
                logits = model.get_logits(imgs)
                logit_slices = split_logits(logits, variable_ranges)
                target_list = _sum_targets(digit_labels, sum_label,
                                           n_digits, len(variable_ranges))
                loss, info = criterion(*logit_slices, *target_list)

                if apply_constraint:
                    probs = torch.cat(
                        [F.softmax(sl, dim=1) for sl in logit_slices], dim=1)
                    with layer_timer():
                        _proj_probs, proj_info = model.constraint_layer(probs)
                    loss = loss + lambda_proj * proj_info['projection_loss']
                    batch_proj = float(proj_info['projection_loss'].detach())
                    info['mean_l2_edit'] = proj_info['avg_l2_per_sample']

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
        total_proj += batch_proj
        n_batches += 1

    denom = max(1, n_batches)
    avg_grad_norm = total_grad_norm / grad_norm_count if grad_norm_count > 0 else float('nan')
    return {
        'loss': total_loss / denom,
        'avg_l2': total_l2 / denom,
        'proj_loss': total_proj / denom,
        'grad_norm': avg_grad_norm,
        'timed_out': timed_out,
    }


def evaluate_sum(model, loader, variable_ranges, n_digits, device, apply_constraint=True,
                 layer_timer=None):
    """Test metrics for MNIST-Sum (exact = every digit right, macro = per-digit accuracy).

    digit_exact/digit_macro read the raw network's N input digits (all right / share right). sum_* read the answer digits
    after the layer when it is applied at inference; raw_sum_* the network's own
    answer head; symbolic_sum_exact the sum of the raw input digits. Sum has no
    two operands, so the operand columns are NaN.
    """
    model.eval()
    layer_timer = layer_timer or _LayerTimer(device)
    n_vars = len(variable_ranges)
    c = defaultdict(float)
    total = 0
    with torch.no_grad():
        for imgs, digit_labels, sum_label in tqdm(loader, leave=False, desc="eval", disable=None):
            imgs = imgs.to(device)
            digit_labels = digit_labels.to(device)
            sum_label = sum_label.to(device)

            raw_probs = model.forward_raw(imgs)
            eval_probs = raw_probs
            if apply_constraint:
                with layer_timer():
                    eval_probs, _info = model.constraint_layer(raw_probs.clone())
                c['l2'] += torch.norm(eval_probs - raw_probs, p=2, dim=1).sum().item()

            raw_preds = _argmax_slices(raw_probs, variable_ranges)
            eval_preds = _argmax_slices(eval_probs, variable_ranges)
            targets = torch.stack(_sum_targets(digit_labels, sum_label, n_digits, n_vars), dim=1)

            e, m = _exact_macro(raw_preds[:, :n_digits], targets[:, :n_digits])
            c['digit_exact'] += e
            c['digit'] += m
            e, m = _exact_macro(eval_preds[:, n_digits:], targets[:, n_digits:])
            c['sum_exact'] += e
            c['sum_macro'] += m
            e, m = _exact_macro(raw_preds[:, n_digits:], targets[:, n_digits:])
            c['raw_sum_exact'] += e
            c['raw_sum_macro'] += m
            c['symbolic'] += (raw_preds[:, :n_digits].sum(dim=1) == sum_label).sum().item()
            eval_total = (eval_preds[:, -1] if n_vars - n_digits == 1
                          else digits_to_sum(list(eval_preds[:, n_digits:].unbind(dim=1))))
            c['constraint'] += (eval_preds[:, :n_digits].sum(dim=1) == eval_total).sum().item()
            total += imgs.shape[0]

    t, nan = max(total, 1), float('nan')
    return {
        'digit_exact': 100 * c['digit_exact'] / t,
        'digit_macro': 100 * c['digit'] / t,
        'operand1_exact': nan, 'operand1_macro': nan,
        'operand2_exact': nan, 'operand2_macro': nan,
        'sum_exact': 100 * c['sum_exact'] / t,
        'sum_macro': 100 * c['sum_macro'] / t,
        'raw_sum_exact': 100 * c['raw_sum_exact'] / t,
        'raw_sum_macro': 100 * c['raw_sum_macro'] / t,
        'symbolic_sum_exact': 100 * c['symbolic'] / t,
        'constraint_sat': 100 * c['constraint'] / t,
        'avg_l2_edit': c['l2'] / t,
    }


def train_epoch_add(model, loader, optimizer, variable_ranges, n_inputs,
                    device, lambda_proj, per_iter_timeout_s, apply_constraint=True,
                    layer_timer=None):
    model.train()
    n_output = len(variable_ranges) - n_inputs
    total_loss, total_proj, total_l2 = 0.0, 0.0, 0.0
    layer_timer = layer_timer or _LayerTimer(device)
    total_grad_norm, n_batches = 0.0, 0
    timed_out = False

    pbar = tqdm(loader, leave=False, desc="train", disable=None)
    for batch_idx, (imgs, digit_labels, output_digit_labels, _sum_label) in enumerate(pbar):
        imgs = imgs.to(device)
        digit_labels = digit_labels.to(device)
        output_digit_labels = output_digit_labels.to(device)
        batch_l2 = 0.0

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
                    with layer_timer():
                        _proj_probs, proj_info = model.constraint_layer(probs)
                    proj_loss = proj_info['projection_loss']
                    batch_l2 = proj_info['avg_l2_per_sample']
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
        total_l2 += batch_l2
        n_batches += 1

    n = max(1, n_batches)
    return {
        'loss': total_loss / n,
        'proj_loss': total_proj / n,
        'avg_l2': total_l2 / n,
        'grad_norm': total_grad_norm / n,
        'timed_out': timed_out,
    }


def evaluate_add(model, loader, variable_ranges, n_inputs, n_digits_per_num, device,
                 apply_constraint=True, layer_timer=None):
    """Test metrics for MNIST-Add, with the operand/sum breakdown's definitions.

    Exact = every digit right; macro = per-digit accuracy, both on the hard argmax.
    operand{1,2}_* and digit_exact/digit_macro (all 2N digits) read the raw network's
    digits for the two N-digit operands (independent of the layer). sum_* read the N+1 answer digits after the
    layer when it is applied at inference; raw_sum_* the network's own answer head;
    symbolic_sum_exact the carry chain of the raw operand digits. constraint_sat is
    the share of answers that equal that carry chain.
    """
    model.eval()
    layer_timer = layer_timer or _LayerTimer(device)
    N = n_digits_per_num
    c = defaultdict(float)
    total = 0
    with torch.no_grad():
        for imgs, digit_labels, output_digit_labels, _sum_label in tqdm(loader, leave=False, desc="eval", disable=None):
            imgs = imgs.to(device)
            digit_labels = digit_labels.to(device)
            output_digit_labels = output_digit_labels.to(device)

            raw_probs = model.forward_raw(imgs)
            eval_probs = raw_probs
            if apply_constraint:
                with layer_timer():
                    eval_probs, _info = model.constraint_layer(raw_probs.clone())
                c['l2'] += torch.norm(eval_probs - raw_probs, p=2, dim=1).sum().item()

            raw_preds = _argmax_slices(raw_probs, variable_ranges)
            eval_preds = _argmax_slices(eval_probs, variable_ranges)
            operands = raw_preds[:, :n_inputs]

            e, m = _exact_macro(operands, digit_labels)
            c['digit_exact'] += e
            c['digit'] += m
            for k, cols in ((1, slice(0, N)), (2, slice(N, 2 * N))):
                e, m = _exact_macro(operands[:, cols], digit_labels[:, cols])
                c[f'op{k}_exact'] += e
                c[f'op{k}_macro'] += m
            e, m = _exact_macro(eval_preds[:, n_inputs:], output_digit_labels)
            c['sum_exact'] += e
            c['sum_macro'] += m
            e, m = _exact_macro(raw_preds[:, n_inputs:], output_digit_labels)
            c['raw_sum_exact'] += e
            c['raw_sum_macro'] += m
            chain = compute_carry_chain(operands[:, :N], operands[:, N:])
            c['symbolic'] += (chain == output_digit_labels).all(dim=1).sum().item()
            c['constraint'] += (eval_preds[:, n_inputs:] == chain).all(dim=1).sum().item()
            total += imgs.shape[0]

    t = max(total, 1)
    return {
        'digit_exact': 100 * c['digit_exact'] / t,
        'digit_macro': 100 * c['digit'] / t,
        'operand1_exact': 100 * c['op1_exact'] / t,
        'operand1_macro': 100 * c['op1_macro'] / t,
        'operand2_exact': 100 * c['op2_exact'] / t,
        'operand2_macro': 100 * c['op2_macro'] / t,
        'sum_exact': 100 * c['sum_exact'] / t,
        'sum_macro': 100 * c['sum_macro'] / t,
        'raw_sum_exact': 100 * c['raw_sum_exact'] / t,
        'raw_sum_macro': 100 * c['raw_sum_macro'] / t,
        'symbolic_sum_exact': 100 * c['symbolic'] / t,
        'constraint_sat': 100 * c['constraint'] / t,
        'avg_l2_edit': c['l2'] / t,
    }


def run_one(task, n_digits, seed, epochs, lr, batch_size, device,
            train_loader, test_loader, out_path, record,
            per_iter_timeout_s, max_run_time_s, constraint_mode='train_infer',
            compiled_file=None, projection_backend='closed_form',
            solver_time_limit=None):
    """Train and evaluate one run, appending each epoch to record['epochs'] and
    rewriting the run JSON after every epoch. checkpoint_last.pt (every epoch) and
    checkpoint_best.pt (best sum_exact) go next to the JSON; returns the run summary."""
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    train_constraint = constraint_mode in ('train_infer', 'train_only')
    infer_constraint = constraint_mode in ('train_infer', 'infer_only')

    variable_ranges, n_inputs = get_config(n_digits, task)

    model = MNISTConstraintNet(
        task=task,
        variable_ranges=variable_ranges,
        n_inputs=n_inputs,
        compiled_file=compiled_file,
        projection_backend=projection_backend,
        solver_time_limit=solver_time_limit,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    n_sum_out = len(variable_ranges) - n_inputs if task == 'sum' else 0
    if task == 'sum':
        criterion = PlainSumLoss()

    n_train, n_test = len(train_loader.dataset), len(test_loader.dataset)
    print(f"\n{'=' * 60}")
    print(f"task={task}  N={n_digits}  seed={seed}  "
          f"projection={projection_backend}  "
          f"mode={constraint_mode}")
    print(f"epochs={epochs}  lr={lr}  batch_size={batch_size}  train={n_train}  test={n_test}")
    print(f"{'=' * 60}")

    history, status = [], 'completed'
    run_start = time.time()
    run_dir = Path(out_path).parent
    best_sum_exact = -1.0

    for epoch in range(epochs):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        lr_now = optimizer.param_groups[0]['lr']
        train_timer, eval_timer = _LayerTimer(device), _LayerTimer(device)
        epoch_start = time.time()

        _sync(device)
        t0 = time.perf_counter()
        if task == 'sum':
            train_info = train_epoch_sum(
                model, train_loader, optimizer, criterion, variable_ranges,
                n_digits, device, per_iter_timeout_s,
                apply_constraint=(train_constraint and n_sum_out > 1),
                layer_timer=train_timer)
        else:
            train_info = train_epoch_add(
                model, train_loader, optimizer, variable_ranges, n_inputs,
                device, lambda_proj=1.0, per_iter_timeout_s=per_iter_timeout_s,
                apply_constraint=train_constraint, layer_timer=train_timer)
        _sync(device)
        train_time = time.perf_counter() - t0

        if train_info['timed_out']:
            metrics, eval_time = {k: float('nan') for k in METRIC_KEYS}, 0.0
        else:
            t1 = time.perf_counter()
            if task == 'sum':
                metrics = evaluate_sum(model, test_loader, variable_ranges, n_digits, device,
                                       apply_constraint=infer_constraint, layer_timer=eval_timer)
            else:
                metrics = evaluate_add(model, test_loader, variable_ranges, n_inputs,
                                       n_digits, device, apply_constraint=infer_constraint,
                                       layer_timer=eval_timer)
            _sync(device)
            eval_time = time.perf_counter() - t1

        row = {
            'task': task, 'n_digits': n_digits, 'projection_backend': projection_backend,
            'constraint_mode': constraint_mode, 'seed': seed, 'epoch': epoch + 1, 'lr': lr_now,
            'train_loss': train_info['loss'],
            'train_proj_loss': train_info.get('proj_loss', float('nan')),
            'grad_norm': train_info['grad_norm'],
            **metrics,
            'n_train': n_train, 'n_test': n_test,
            'train_time_s': train_time, 'train_layer_time_s': train_timer.total,
            'eval_time_s': eval_time, 'eval_layer_time_s': eval_timer.total,
            's_per_train_example': train_time / max(n_train, 1),
            'epoch_time_s': time.time() - epoch_start,
            'gpu_mem_peak_mb': (torch.cuda.max_memory_allocated() / (1024 ** 2)
                                if device.type == 'cuda' else float('nan')),
            'gpu_mem_reserved_peak_mb': (torch.cuda.max_memory_reserved() / (1024 ** 2)
                                         if device.type == 'cuda' else float('nan')),
            'cpu_rss_peak_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            'timed_out': int(train_info['timed_out']),
        }
        record['epochs'].append(row)
        history.append(row)
        record['summary'] = summarise_run(history, 'running', time.time() - run_start)
        write_run_json(out_path, record)

        if train_info['timed_out']:
            status = 'timed_out'
            print(f"  Aborting after epoch {epoch+1}: a training step exceeded its time budget.")
            break

        operands = (f"op exact={row['operand1_exact']:.1f}/{row['operand2_exact']:.1f}% | "
                    if task == 'add' else "")
        print(f"Epoch {epoch+1:3d} | loss={row['train_loss']:.4f} | digit={row['digit_macro']:.1f}% | "
              f"{operands}sum exact={row['sum_exact']:.1f}% macro={row['sum_macro']:.1f}% "
              f"(raw {row['raw_sum_exact']:.1f}%) | sat={row['constraint_sat']:.1f}% | "
              f"L2={row['avg_l2_edit']:.4f} | train {train_time:.1f}s (layer {train_timer.total:.1f}s) "
              f"eval {eval_time:.1f}s")

        scheduler.step()
        save_checkpoint(run_dir / 'checkpoint_last.pt', model, optimizer, scheduler, epoch + 1, row)
        if row['sum_exact'] > best_sum_exact:
            best_sum_exact = row['sum_exact']
            save_checkpoint(run_dir / 'checkpoint_best.pt', model, optimizer, scheduler, epoch + 1, row)

        if max_run_time_s is not None and (time.time() - run_start) > max_run_time_s:
            status = 'run_time_cap'
            print(f"  Per-run cap hit ({time.time() - run_start:.1f}s > {max_run_time_s}s). Aborting.")
            break

    return summarise_run(history, status, time.time() - run_start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['sum', 'add'])
    parser.add_argument('--n-digits', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--out', type=str, required=True,
                        help='Run JSON: config, per-epoch metrics and summary.')
    parser.add_argument('--projection-backend', type=str, default='closed_form',
                        choices=['closed_form', 'cvxpy'],
                        help="Solver for the MinEdit projection. The constraint "
                             "set is identical either way; 'cvxpy' solves the "
                             "same projection as a QP via SCS.")
    parser.add_argument('--solver-time-limit', type=float, default=None,
                        help="Per-solve budget for --projection-backend cvxpy.")
    parser.add_argument('--compiled-file', type=str, default=None,
                        help="Constraint file to load. Defaults to the "
                             "generated one for (task, N).")
    parser.add_argument('--constraint-mode', type=str, default='train_infer',
                        choices=['train_infer', 'train_only', 'infer_only', 'none'],
                        help='When the constraint is applied: at train+inference, '
                             'train only, inference only, or never (unconstrained).')
    parser.add_argument('--per-iter-timeout-s', type=float, default=None)
    parser.add_argument('--max-run-time-s', type=float, default=None)
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--per-example-timeout-s', type=float, default=None,
                        help='Time-out per training example (the paper uses 5 s); sets '
                             '--per-iter-timeout-s to this times the batch size when that is unset.')
    parser.add_argument('--num-train-samples', type=int, default=None,
                        help='Training examples, drawn with replacement so MNIST images are '
                             'reused. Default: 5000 for sum, 60000 // (2 * n_digits) for add.')
    parser.add_argument('--num-test-samples', type=int, default=None,
                        help='Test examples, drawn with replacement. Default: 1000 for sum, '
                             '10000 // (2 * n_digits) for add.')
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

    if args.per_example_timeout_s is not None and args.per_iter_timeout_s is None:
        args.per_iter_timeout_s = args.per_example_timeout_s * args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(
        args.data_root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(
        args.data_root, train=False, download=True, transform=transform)

    if args.task == 'sum':
        train_n = args.num_train_samples or mnist_sum_num_samples('train')
        test_n = args.num_test_samples or mnist_sum_num_samples('test')
        train_ds = MNISTSumDataset(mnist_train, n_digits=args.n_digits,
                                   num_samples=train_n, data_seed=args.seed)
        test_ds = MNISTSumDataset(mnist_test, n_digits=args.n_digits,
                                  num_samples=test_n, data_seed=args.seed + 10000)
    else:
        train_n = args.num_train_samples or mnist_add_num_samples(args.n_digits, 'train')
        test_n = args.num_test_samples or mnist_add_num_samples(args.n_digits, 'test')
        train_ds = MNISTAddDataset(mnist_train, n_digits=args.n_digits,
                                   num_samples=train_n, data_seed=args.seed)
        test_ds = MNISTAddDataset(mnist_test, n_digits=args.n_digits,
                                  num_samples=test_n, data_seed=args.seed + 10000)

    print(f"  train={train_n}  test={test_n}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    compiled_file = resolve_compiled_file(args.task, args.n_digits, args.compiled_file)
    check_info = {}
    if args.compiled_file is None:
        check_info = check_fixpoint(compiled_file)
    else:
        print(f"Using --compiled-file {compiled_file} as a compile output (not re-checked)")

    blob = Path(compiled_file).read_bytes()
    run_config = {
        **vars(args), 'n_train': train_n, 'n_test': test_n,
        'compiled_file': str(compiled_file), 'compiled_file_bytes': len(blob),
        'compiled_file_sha256': hashlib.sha256(blob).hexdigest(), **check_info,
        'commit': _git_describe(), 'host': socket.gethostname(),
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    record = {'config': run_config, 'epochs': [], 'summary': None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_run_json(args.out, record)

    summary = run_one(
        task=args.task,
        n_digits=args.n_digits,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        out_path=args.out,
        record=record,
        compiled_file=compiled_file,
        projection_backend=args.projection_backend,
        solver_time_limit=args.solver_time_limit,
        per_iter_timeout_s=args.per_iter_timeout_s,
        max_run_time_s=args.max_run_time_s,
        constraint_mode=args.constraint_mode,
    )
    record['summary'] = summary
    record['config']['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    write_run_json(args.out, record)
    print(f"Run {summary['status']}: {summary['epochs_completed']} epochs -> {args.out}")


if __name__ == "__main__":
    main()
