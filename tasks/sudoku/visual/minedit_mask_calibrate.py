"""Mask-only calibration for the MinEdit pipeline.

Only mask_nn is trained; perception + nn_solver are frozen. The mask head
is fed the PROJECTED solver distribution (output of SudokuConstraintLayer)
so it can learn to predict per-cell correctness in the space it will gate
at eval time. Target at cell j is 1 iff argmax(projected)[j] equals the
ground-truth digit.

KEEP=1 convention (shared with minedit_sudoku.py and rl_eval_sudoku.py):
mask=1 keeps the projected digit, mask=0 blanks for Prolog.

Stability design:
  * Per-cell weighted BCE with reduction='none' and w_neg weighting the
    rare wrong-projected cells (avoids the near-zero pos_weight regime).
  * Gradient clipping on mask_nn parameters.
  * Early stopping on full-board correct count with min-delta patience.
  * Optional eval-time mask-logit bias sweep; checkpoint metadata records
    the best bias alongside the weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import os
import json
import random
import time
from argparse import ArgumentParser

import numpy as np

import paths
from datasets import SudokuDataset_RL
from models.transformer_sudoku import get_model
from casper_sudoku_layer import add_edit_layer_args, build_edit_layer
from group_order_utils import parse_group_order, format_group_order
from sudoku_solver.board import Board

import pathlib
PROLOG_RULES_PATH = pathlib.Path(__file__).resolve().parent / 'sudoku_solver' / 'sudoku_prolog.pl'

try:
    from pyswip import Prolog
except Exception:
    print('-->> Prolog not installed')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-10


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_biases(s):
    if not s:
        return [0.0]
    return [float(x.strip()) for x in s.split(',') if x.strip() != '']


class MaskCalibrator:
    def __init__(self, model, train_loader, test_loader, model_dir,
                 learning_rate, batch_size, seed, args):
        self.model_dir = model_dir
        self.network = model
        self.constraint_layer = build_edit_layer(args, device)

        # Freeze perception + nn_solver; only mask_nn receives gradients.
        for p in self.network.perception.parameters():
            p.requires_grad = False
        for p in self.network.nn_solver.parameters():
            p.requires_grad = False
        for p in self.network.mask_nn.parameters():
            p.requires_grad = True

        self.optimizer = torch.optim.AdamW(
            self.network.mask_nn.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
        )
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_board_correct = None
        self.best_bias = 0.0
        self.best_epoch = -1
        self.epochs_since_improve = 0
        self.batch_size = batch_size
        self.args = args
        self.seed = seed
        self.biases = _parse_biases(args.eval_mask_biases)

    def adjust_learning_rate(self, epoch):
        lr = self.args.lr
        if self.args.warmup > 0 and epoch < self.args.warmup:
            lr = lr * (epoch + 1) / self.args.warmup
        elif not self.args.disable_cos:
            denom = max(self.args.epochs - self.args.warmup, 1)
            lr *= 0.5 * (1. + math.cos(
                math.pi * (epoch - self.args.warmup) / denom
            ))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _forward_solver(self, data):
        """perception + nn_solver forward (both frozen)."""
        x0 = self.network.perception(data)
        x1 = torch.exp(x0)
        x2 = self.network.nn_solver(x1)
        return F.softmax(x2, dim=2)  # (B, 81, 9)

    def _mask_logits(self, projected):
        out = self.network.mask_nn(projected)
        return out.squeeze(-1) if out.dim() == 3 else out  # (B, 81)

    def train_epoch(self, epoch, current_lr):
        self.network.perception.eval()
        self.network.nn_solver.eval()
        self.network.mask_nn.train()

        w_neg = float(self.args.neg_weight)
        clip_norm = float(self.args.clip_norm)
        print(
            f"[mask-calib] Epoch {epoch} start | "
            f"lr={current_lr:.2e} neg_weight={w_neg:.2f} "
            f"clip_norm={clip_norm:.2f} "
            f"patience={self.epochs_since_improve}/{self.args.patience}"
        )

        loss_sum = 0.0
        n_batches = 0
        pos_total = 0
        neg_total = 0
        pos_contrib_sum = 0.0
        neg_contrib_sum = 0.0
        grad_norm_sum = 0.0

        for data, target in self.train_loader:
            self.optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)

            # Frozen pipeline — no activation storage.
            with torch.no_grad():
                solution_boards = self._forward_solver(data)
                projected, _ = self.constraint_layer(
                    solution_boards,
                    max_iterations=self.args.minedit_iters_train,
                )

            # mask_nn trained on the PROJECTED distribution it will gate.
            masking_logits = self._mask_logits(projected)  # (B, 81)

            gt_digits = target.argmax(dim=-1) - 1
            projected_pred = projected.argmax(dim=2)
            mask_target = (projected_pred == gt_digits).float()  # (B, 81)

            # Per-cell weighted BCE. Positives get weight 1, negatives
            # (rare wrong-projected cells that should be blanked) get w_neg.
            # Overall loss stays on the standard BCE scale — no near-zero
            # pos_weight regime.
            bce = F.binary_cross_entropy_with_logits(
                masking_logits, mask_target, reduction='none',
            )
            weights = torch.where(
                mask_target == 0,
                torch.full_like(mask_target, w_neg),
                torch.ones_like(mask_target),
            )
            loss = (bce * weights).mean()
            loss.backward()

            # Gradient clipping for optimisation stability.
            g_norm = torch.nn.utils.clip_grad_norm_(
                self.network.mask_nn.parameters(), clip_norm,
            )
            self.optimizer.step()

            with torch.no_grad():
                is_pos = (mask_target > 0.5)
                weighted = bce * weights
                pos_contrib_sum += float((weighted * is_pos).sum().item())
                neg_contrib_sum += float((weighted * (~is_pos)).sum().item())
                pos_total += int(is_pos.sum().item())
                neg_total += int((~is_pos).sum().item())

            loss_sum += float(loss.item())
            n_batches += 1
            grad_norm_sum += float(g_norm)

        total_cells = max(pos_total + neg_total, 1)
        pos_frac = pos_total / total_cells
        neg_frac = neg_total / total_cells
        pos_per_cell = pos_contrib_sum / max(pos_total, 1)
        neg_per_cell = neg_contrib_sum / max(neg_total, 1)
        avg_gn = grad_norm_sum / max(n_batches, 1)
        print(
            f"[mask-calib] Epoch {epoch} train "
            f"loss={loss_sum / max(n_batches, 1):.4f} "
            f"pos_frac={pos_frac:.4f} neg_frac={neg_frac:.4f} "
            f"w_pos/cell={pos_per_cell:.4f} w_neg/cell={neg_per_cell:.4f} "
            f"avg_grad_norm={avg_gn:.4f}"
        )

    def _eval_biases(self):
        """Single eval pass that scores every bias in self.biases.

        Solver + constraint-layer + mask_nn forward once per batch; for
        each bias we recompute the gate and run Prolog per-sample.
        Returns (per_bias_stats, avg_sat)."""
        self.network.eval()

        prolog_instance = Prolog()
        prolog_instance.consult(str(PROLOG_RULES_PATH))

        biases = self.biases
        per_bias = {
            b: dict(
                total_cells=0, correct_cells=0, fully_correct=0,
                neuro_correct=0, wrong_corrected=0, correct_corrupted=0,
                blank_cells=0, wrong_projected_cells=0,
                wrong_projected_kept=0, n_samples=0,
            )
            for b in biases
        }
        sat_weighted = 0.0
        sat_n = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(device)
                target = target.to(device)
                bsz = data.shape[0]

                solution_boards = self._forward_solver(data)
                projected, _ = self.constraint_layer(
                    solution_boards,
                    max_iterations=self.args.minedit_iters_eval,
                )
                sat_weighted += (
                    self.constraint_layer.constraint_satisfaction_rate(projected) * bsz
                )
                sat_n += bsz

                masking_logits = self._mask_logits(projected)  # (B, 81)
                gt_boards = target.argmax(dim=-1)              # (B, 81) 1..9
                projected_1to9 = projected.argmax(dim=2) + 1   # (B, 81) 1..9
                wrong_proj = (projected_1to9 != gt_boards)

                for b_val in biases:
                    prob = (masking_logits + b_val).sigmoid().unsqueeze(-1)
                    gate = torch.cat(
                        ((1 - prob), projected * prob), dim=-1,
                    )
                    cleaned = gate.argmax(dim=-1)              # (B, 81) 0..9
                    kept_mask = (cleaned != 0)

                    st = per_bias[b_val]
                    st['blank_cells'] += int((~kept_mask).sum().item())
                    st['wrong_projected_cells'] += int(wrong_proj.sum().item())
                    st['wrong_projected_kept'] += int(
                        (wrong_proj & kept_mask).sum().item()
                    )

                    cleaned_cpu = cleaned.cpu()
                    for s_idx in range(bsz):
                        board = Board(
                            cleaned_cpu[s_idx].reshape((9, 9)).int()
                        )
                        try:
                            ok = board.solve(
                                solver='prolog',
                                prolog_instance=prolog_instance,
                            )
                        except StopIteration:
                            ok = False
                        if ok:
                            final_sol = torch.from_numpy(
                                board.board.reshape(81,)
                            ).to(gt_boards.device)
                        else:
                            final_sol = projected_1to9[s_idx]

                        matches = int(
                            (final_sol == gt_boards[s_idx]).sum().item()
                        )
                        board_full = (matches == 81)
                        st['correct_cells'] += matches
                        st['total_cells'] += 81
                        if board_full:
                            st['fully_correct'] += 1

                        neuro_full = bool(
                            (projected_1to9[s_idx] == gt_boards[s_idx]).all().item()
                        )
                        if neuro_full:
                            st['neuro_correct'] += 1
                        if not neuro_full and board_full:
                            st['wrong_corrected'] += 1
                        if neuro_full and not board_full:
                            st['correct_corrupted'] += 1
                        st['n_samples'] += 1

                torch.cuda.empty_cache()

        avg_sat = sat_weighted / max(sat_n, 1)
        return per_bias, avg_sat

    def test_epoch(self, epoch):
        per_bias, avg_sat = self._eval_biases()
        biases = self.biases

        summaries = {}
        for b_val in biases:
            st = per_bias[b_val]
            n = max(st['n_samples'], 1)
            tc = max(st['total_cells'], 1)
            n_wrong = max(st['n_samples'] - st['neuro_correct'], 1)
            summaries[b_val] = dict(
                board_acc=st['fully_correct'] / n,
                cell_acc=st['correct_cells'] / tc,
                wrong_corrected_rate=st['wrong_corrected'] / n_wrong,
                corrupted_rate=(
                    st['correct_corrupted'] / max(st['neuro_correct'], 1)
                ),
                blank_rate=st['blank_cells'] / tc,
                keep_rate_wrong_proj=(
                    st['wrong_projected_kept']
                    / max(st['wrong_projected_cells'], 1)
                ),
                fully_correct=st['fully_correct'],
                n_samples=st['n_samples'],
                neuro_correct=st['neuro_correct'],
                wrong_corrected=st['wrong_corrected'],
                correct_corrupted=st['correct_corrupted'],
            )

        if len(biases) > 1:
            print(f"[mask-calib][Epoch {epoch}] bias sweep:")
            print(
                f"  {'bias':>8} {'board_acc':>10} "
                f"{'cor_wrong':>10} {'cor_correct':>12} {'blank_rt':>10}"
            )
            for b_val in biases:
                s = summaries[b_val]
                print(
                    f"  {b_val:>+8.3f} {s['board_acc']:>10.4f} "
                    f"{s['wrong_corrected_rate']:>10.4f} "
                    f"{s['corrupted_rate']:>12.4f} "
                    f"{s['blank_rate']:>10.4f}"
                )

        best_b = max(biases, key=lambda b: summaries[b]['fully_correct'])
        best_s = summaries[best_b]
        print(
            f"----[mask-calib][Epoch {epoch}] "
            f"best_bias={best_b:+.3f} "
            f"board_acc={best_s['board_acc']:.4f} "
            f"({best_s['fully_correct']}/{best_s['n_samples']}) "
            f"cell_acc={best_s['cell_acc']:.4f} "
            f"| neuro_board={best_s['neuro_correct']}/{best_s['n_samples']} "
            f"wrong_corrected={best_s['wrong_corrected']}/"
            f"{best_s['n_samples'] - best_s['neuro_correct']} "
            f"({best_s['wrong_corrected_rate']:.4f}) "
            f"corrupted_correct={best_s['correct_corrupted']}/"
            f"{best_s['neuro_correct']} ({best_s['corrupted_rate']:.4f}) "
            f"| blank_rate={best_s['blank_rate']:.4f} "
            f"keep_rate_wrong_proj={best_s['keep_rate_wrong_proj']:.4f} "
            f"| alldiff_sat={avg_sat:.4f}"
        )

        improved = (
            self.best_board_correct is None
            or best_s['fully_correct']
            >= (self.best_board_correct + self.args.min_delta)
        )
        if improved:
            self.best_board_correct = best_s['fully_correct']
            self.best_bias = best_b
            self.best_epoch = epoch
            self.epochs_since_improve = 0
            path = os.path.join(
                self.model_dir, f'checkpoint_best_R_{self.seed}.pth'
            )
            torch.save(self.network.state_dict(), path)
            meta_path = os.path.join(
                self.model_dir, f'checkpoint_best_R_{self.seed}.json'
            )
            with open(meta_path, 'w') as f:
                json.dump(dict(
                    seed=self.seed,
                    epoch=epoch,
                    best_bias=best_b,
                    best_board_correct=best_s['fully_correct'],
                    n_samples=best_s['n_samples'],
                    board_acc=best_s['board_acc'],
                    edit_layer=self.args.edit_layer,
                    constraints_file=(
                        self.args.constraints_file
                        if self.args.edit_layer == 'casper' else None
                    ),
                    constraints_format=(
                        self.args.constraints_format
                        if self.args.edit_layer == 'casper' else None
                    ),
                    casper_cell_order=(
                        self.args.casper_cell_order
                        if self.args.edit_layer == 'casper' else None
                    ),
                ), f, indent=2)
            print(
                f"[mask-calib] saved best to {path} "
                f"(board_acc={best_s['board_acc']:.4f}, bias={best_b:+.3f})"
            )
        else:
            self.epochs_since_improve += 1

        return best_s['fully_correct'], self.epochs_since_improve

    def train(self, n_epochs):
        # Baseline with pretrained mask; always writes initial checkpoint
        # so a --epochs 0 run still produces something evaluable.
        self.test_epoch(0)
        for epoch in range(1, n_epochs + 1):
            lr = self.adjust_learning_rate(epoch)
            t0 = time.time()
            self.train_epoch(epoch, lr)
            t1 = time.time()
            print(f"[mask-calib] train time: {t1 - t0:.1f}s")
            _, patience_ct = self.test_epoch(epoch)
            if patience_ct >= self.args.patience:
                print(
                    f"[mask-calib] early stopping at epoch {epoch} "
                    f"(no improvement for {patience_ct} epochs; "
                    f"best_epoch={self.best_epoch} "
                    f"best_board_correct={self.best_board_correct} "
                    f"best_bias={self.best_bias:+.3f})"
                )
                break


if __name__ == "__main__":
    parser = ArgumentParser('sudoku_minedit_mask_calibrate')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--data', type=str, default='satnet')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--warmup', default=0, type=int)
    parser.add_argument('-b', '--batch-size', default=200, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight-decay', default=3e-1, type=float)
    parser.add_argument('--disable-cos', action='store_true')
    parser.add_argument('--block-len', default=81, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--model-dir',
                        default=str(paths.CKPT_DIR / 'minedit' / 'mask_calibrated'),
                        type=str)
    parser.add_argument('--pairwise', action='store_true',
                        help='Use historical pairwise constraint-layer path '
                             '(--edit-layer minedit only).')
    parser.add_argument('--constraint-group-order', default='default', type=str,
                        help='Processing order for the 27 Sudoku groups inside '
                             'SudokuConstraintLayer. "default" (default) keeps the '
                             'canonical 0..26 order. Otherwise a comma-separated '
                             'permutation of 0..26.')
    add_edit_layer_args(parser)
    parser.add_argument('--minedit-iters-train', default=3, type=int,
                        help='Edit-layer iterations at train time (rounds for casper).')
    parser.add_argument('--minedit-iters-eval', default=10, type=int,
                        help='Edit-layer iterations at eval time (rounds for casper).')
    # Stability / imbalance handling
    parser.add_argument('--neg-weight', default=20.0, type=float,
                        help='Per-cell BCE weight for negative (blank-target) cells.')
    parser.add_argument('--clip-norm', default=1.0, type=float,
                        help='Gradient clipping norm on mask_nn parameters.')
    # Early stopping
    parser.add_argument('--patience', default=5, type=int,
                        help='Epochs without improvement before stopping.')
    parser.add_argument('--min-delta', default=1, type=int,
                        help='Minimum improvement in full-board correct count '
                             'to reset patience.')
    # Eval-time bias sweep
    parser.add_argument('--eval-mask-biases', default='0.0', type=str,
                        help='Comma-separated additive biases on mask logits '
                             'to evaluate each epoch. Best bias is recorded '
                             'in checkpoint metadata.')
    args = parser.parse_args()

    args.constraint_group_order = parse_group_order(args.constraint_group_order)
    print(f'[mask-calib] constraint-group-order = '
          f'{format_group_order(args.constraint_group_order)}')

    set_deterministic(args.seed)

    train_dataset = SudokuDataset_RL(args.data, '-train')
    test_dataset = SudokuDataset_RL(args.data, '-valid')

    model = get_model(block_len=args.block_len)
    model.load_pretrained_models(args.data)
    model.to(device)

    os.makedirs(args.model_dir, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    calibrator = MaskCalibrator(
        model, train_loader, test_loader, args.model_dir,
        args.lr, args.batch_size, args.seed, args,
    )
    calibrator.train(args.epochs)
