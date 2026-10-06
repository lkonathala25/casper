"""LLM + frozen solver + perception fusion calibration for MinEdit.

Pipeline:
  CNN (frozen)  ->  [LLM (frozen) + nn_solver (frozen) + perception (frozen)]
                ->  FusionHead / temperature_head (frozen by default)
                ->  MinEdit (frozen)  ->  mask_nn (trainable)  ->  Prolog

In the casper port the fusion head is FROZEN by default (--freeze-fusion-head
is on; --train-fusion-head opts out with a warning): training it jointly
degraded it (CASPER_MASK_TRAINING_PORT.md §2.3). Frozen at its identity init
the head outputs the solver's probabilities exactly, so only mask_nn trains.

The FusionHead recovers useful downstream uncertainty by fusing:
  - cached LLM solved-board logits (one-hot-like, no alternative uncertainty)
  - frozen nn_solver soft probabilities (genuine uncertainty, key signal)
  - perception log-probs (CNN confidence per digit/cell)

At init FusionHead outputs solver_probs exactly, recovering the historical
MinEdit/mask_nn operating regime and giving alldiff_sat ≈ 97% from epoch 0.

Training is optionally staged (--stage-mask-freeze-epochs N):
  Stage 1 (epochs 1..N)  : train FusionHead only, mask_nn frozen.
  Stage 2 (epochs N+1..) : unfreeze mask_nn, joint fine-tuning.

Loss composition (per batch):
  L = lambda_mask * L_mask  +  lambda_temp * L_temp
  - L_mask : per-cell weighted BCE on projected-distribution correctness.
  - L_temp : label-smoothed NLL on fused calibrated probs vs GT digits.

Checkpoint format mirrors minedit_mask_calibrate.py:
  checkpoint_best_R_{seed}.pth   — LLMTransformerSudoku.state_dict()
  checkpoint_best_R_{seed}.json  — sidecar with best_bias / metadata

KEEP=1 convention, early stopping, cosine LR, gradient clipping.
"""
import math
import os
import json
import random
import time
import gc
from argparse import ArgumentParser
from pathlib import Path

# Point pyswip at the sudoku env's SWI-Prolog installation.
# This avoids installing swi-prolog via conda-forge in sudoku-fine (which would
# downgrade Python) while keeping torch/unsloth untouched.
# The two vars must be set before `from pyswip import Prolog` is executed.
import paths
_SWIPL_ENV = str(paths.SWIPL_ENV)
_SWIPL_LIB = f"{_SWIPL_ENV}/lib/swipl/lib/x86_64-linux/libswipl.so"
_SWIPL_HOME = f"{_SWIPL_ENV}/lib/swipl"
if Path(_SWIPL_LIB).exists():
    os.environ.setdefault("LIBSWIPL_PATH", _SWIPL_LIB)
    os.environ.setdefault("SWI_HOME_DIR", _SWIPL_HOME)

import numpy as np
import torch
import torch.nn.functional as F
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from datasets import SudokuDataset_RL
from models.transformer_sudoku import get_llm_model
from models.llm_solver_adapter import LLMSolverAdapter, DEFAULT_FINETUNED_MODEL_PATH
from casper_sudoku_layer import add_edit_layer_args, build_edit_layer
from group_order_utils import parse_group_order, format_group_order
from sudoku_solver.board import Board

PROLOG_RULES_PATH = Path(__file__).resolve().parent / "sudoku_solver" / "sudoku_prolog.pl"

try:
    from pyswip import Prolog
except Exception:
    print("-->> Prolog not installed")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


class LLMCalibrator:
    """Mirror of MaskCalibrator but with LLM backbone + temperature head.

    Trainable: temperature_head, mask_nn.
    Frozen:    perception, nn_solver, LLM adapter, MinEdit, Prolog.
    """

    def __init__(self, model, llm_adapter, train_loader, test_loader,
                 model_dir, learning_rate, batch_size, seed, args):
        self.model_dir = model_dir
        self.network = model
        self.llm_adapter = llm_adapter
        self.constraint_layer = build_edit_layer(args, device)

        # Freeze perception and nn_solver.
        for p in self.network.perception.parameters():
            p.requires_grad = False
        for p in self.network.nn_solver.parameters():
            p.requires_grad = False
        # Train mask_nn, and temperature_head unless --freeze-fusion-head.
        # A frozen head at its identity init outputs solver_probs exactly, so
        # the LLM then has no effect on the projected board.
        self.freeze_fusion_head = bool(getattr(args, "freeze_fusion_head", False))
        for p in self.network.temperature_head.parameters():
            p.requires_grad = not self.freeze_fusion_head
        for p in self.network.mask_nn.parameters():
            p.requires_grad = True

        self.trainable_modules = [self.network.mask_nn]
        if not self.freeze_fusion_head:
            self.trainable_modules.insert(0, self.network.temperature_head)
        trainable = [p for mod in self.trainable_modules for p in mod.parameters()]
        print(f"[llm-calib] fusion head {'FROZEN' if self.freeze_fusion_head else 'trainable'}; "
              f"{sum(p.numel() for p in trainable)} trainable params")
        self.optimizer = torch.optim.AdamW(
            trainable,
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

    # ------------------------------------------------------------------ LR

    def adjust_learning_rate(self, epoch):
        lr = self.args.lr
        if self.args.warmup > 0 and epoch < self.args.warmup:
            lr = lr * (epoch + 1) / self.args.warmup
        elif not self.args.disable_cos:
            denom = max(self.args.epochs - self.args.warmup, 1)
            lr *= 0.5 * (
                1.0 + math.cos(math.pi * (epoch - self.args.warmup) / denom)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    # ------------------------------------------------------------------ forward helpers

    def _forward_llm(self, data, perc_out=None):
        """Perception (frozen) + nn_solver (frozen) + LLM (frozen) + FusionHead (trainable).

        Computes all three frozen signals under no_grad, then runs the
        trainable FusionHead which is the only part that needs gradients.

        perc_out: pre-computed perception output (B, 81, 10); if None it is
        computed here. Pass it in when you've already run perception (e.g. to
        check cache hits) to avoid a second forward pass.

        Returns:
            calibrated:   (B, 81, 9) fused probabilities (grad on FusionHead)
            solver_probs: (B, 81, 9) frozen nn_solver soft probs (no grad)
        """
        with torch.no_grad():
            if perc_out is None:
                perc_out = self.network.perception(data)   # (B, 81, 10) log-probs
            # Frozen nn_solver — genuine soft uncertainty (key MinEdit signal).
            solver_probs = F.softmax(
                self.network.nn_solver(torch.exp(perc_out)), dim=-1
            )  # (B, 81, 9)
            llm_logits = self.llm_adapter.solve_batch(perc_out).to(device)  # (B, 81, 9)

        # FusionHead is trainable; frozen inputs are detached.
        calibrated = self.network.temperature_head(
            llm_logits, solver_probs, perc_out.detach()
        )  # (B, 81, 9) fused probs
        return calibrated, solver_probs

    def _mask_logits(self, projected):
        out = self.network.mask_nn(projected)
        return out.squeeze(-1) if out.dim() == 3 else out  # (B, 81)

    # ------------------------------------------------------------------ train

    def train_epoch(self, epoch, current_lr):
        self.network.perception.eval()
        self.network.nn_solver.eval()
        if self.freeze_fusion_head:
            self.network.temperature_head.eval()
        else:
            self.network.temperature_head.train()
        # During Stage 1, mask_nn is frozen: keep it in eval mode so any
        # stochastic behavior (BN, dropout) is deterministic and the stage is
        # a clean fusion-only training phase.
        mask_trainable = any(
            p.requires_grad for p in self.network.mask_nn.parameters()
        )
        if mask_trainable:
            self.network.mask_nn.train()
        else:
            self.network.mask_nn.eval()

        w_neg = float(self.args.neg_weight)
        clip_norm = float(self.args.clip_norm)
        lambda_mask = float(self.args.lambda_mask)
        lambda_temp = float(self.args.lambda_temp)

        print(
            f"[llm-calib] Epoch {epoch} start | "
            f"lr={current_lr:.2e} neg_weight={w_neg:.2f} "
            f"clip_norm={clip_norm:.2f} "
            f"patience={self.epochs_since_improve}/{self.args.patience}"
        )

        loss_sum = 0.0
        mask_loss_sum = 0.0
        temp_loss_sum = 0.0
        n_batches = 0
        pos_total = 0
        neg_total = 0
        pos_contrib_sum = 0.0
        neg_contrib_sum = 0.0
        grad_norm_sum = 0.0
        solver_cell_correct = 0
        solver_cell_total = 0

        self.llm_adapter.reset_counters()

        pbar = tqdm(
            self.train_loader,
            desc=f"Ep{epoch:02d}",
            ncols=120,
            leave=False,
            unit="batch",
        )
        for data, target in pbar:
            self.optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)

            # Optionally skip boards that would trigger an LLM call (cache miss).
            # Perception is cheap (frozen CNN); the LLM is the expensive part.
            perc_out = None
            if self.args.skip_cache_miss:
                with torch.no_grad():
                    perc_out = self.network.perception(data)
                hit_idx = self.llm_adapter.get_cached_indices(perc_out)
                if not hit_idx:
                    continue
                if len(hit_idx) < data.shape[0]:
                    idx_t = torch.tensor(hit_idx, device=device)
                    data = data[idx_t]
                    target = target[idx_t]
                    perc_out = perc_out[idx_t]

            # Frozen signals + FusionHead (trainable).
            calibrated, solver_probs = self._forward_llm(data, perc_out=perc_out)

            # Constraint projection with grad enabled.
            # SudokuConstraintLayer is differentiable (all ops are torch.where /
            # index_put / minedit L2 projection). Gradient from L_mask flows back
            # through projected → calibrated → temperature_head, giving the
            # fusion head a signal aligned with projected-board quality.
            projected, _ = self.constraint_layer(
                calibrated,
                max_iterations=self.args.minedit_iters_train,
            )

            # Mask head trained on projected distribution it will gate.
            masking_logits = self._mask_logits(projected)  # (B, 81)

            gt_digits = target.argmax(dim=-1) - 1   # (B, 81) 0..8
            projected_pred = projected.argmax(dim=2)
            mask_target = (projected_pred == gt_digits).float()  # (B, 81)

            with torch.no_grad():
                solver_cell_correct += int((solver_probs.argmax(-1) == gt_digits).sum().item())
                solver_cell_total   += gt_digits.numel()

            # Weighted BCE mask loss (positives weight=1, negatives weight=w_neg).
            bce = F.binary_cross_entropy_with_logits(
                masking_logits, mask_target, reduction="none",
            )
            weights = torch.where(
                mask_target == 0,
                torch.full_like(mask_target, w_neg),
                torch.ones_like(mask_target),
            )
            L_mask = (bce * weights).mean()

            # Calibration loss: label-smoothed NLL.
            # Plain NLL drives T→0 (sharpening) because 90%+ of cells are
            # already correct; label smoothing forces the analytical optimum
            # to T≈3 (α=0.3 → target p_gt=0.73 for LLM_CONF=10).
            smooth = float(self.args.label_smoothing)
            log_probs = calibrated.clamp(min=EPS).log().view(-1, 9)
            flat_gt = gt_digits.view(-1)
            one_hot = F.one_hot(flat_gt, 9).float()
            smooth_target = smooth / 9 + (1.0 - smooth) * one_hot
            L_temp = -(smooth_target * log_probs).sum(-1).mean()

            # With a frozen head L_temp has no trainable inputs; it is still
            # computed above for logging but left out of the objective.
            if self.freeze_fusion_head:
                loss = lambda_mask * L_mask
            else:
                loss = lambda_mask * L_mask + lambda_temp * L_temp
            loss.backward()

            trainable_params = [
                p for mod in self.trainable_modules for p in mod.parameters()
            ]
            g_norm = torch.nn.utils.clip_grad_norm_(trainable_params, clip_norm)
            self.optimizer.step()

            with torch.no_grad():
                is_pos = mask_target > 0.5
                weighted = bce * weights
                pos_contrib_sum += float((weighted * is_pos).sum().item())
                neg_contrib_sum += float((weighted * (~is_pos)).sum().item())
                pos_total += int(is_pos.sum().item())
                neg_total += int((~is_pos).sum().item())

            loss_sum += float(loss.item())
            mask_loss_sum += float(L_mask.item())
            temp_loss_sum += float(L_temp.item())
            n_batches += 1
            grad_norm_sum += float(g_norm)
            pbar.set_postfix(
                loss=f"{loss_sum / n_batches:.4f}",
                mask=f"{mask_loss_sum / n_batches:.4f}",
                temp=f"{temp_loss_sum / n_batches:.4f}",
            )

        nb = max(n_batches, 1)
        total_cells = max(pos_total + neg_total, 1)
        print(
            f"[llm-calib] Epoch {epoch} train "
            f"loss={loss_sum / nb:.4f} "
            f"mask_loss={mask_loss_sum / nb:.4f} "
            f"temp_loss={temp_loss_sum / nb:.4f} "
            f"solver_cell_acc={solver_cell_correct / max(solver_cell_total, 1):.4f} "
            f"pos_frac={pos_total / total_cells:.4f} "
            f"neg_frac={neg_total / total_cells:.4f} "
            f"pos/cell={pos_contrib_sum / max(pos_total, 1):.4f} "
            f"neg/cell={neg_contrib_sum / max(neg_total, 1):.4f} "
            f"avg_grad_norm={grad_norm_sum / nb:.4f}"
        )
        self.llm_adapter.log_counters(prefix=f"[llm-calib] Epoch {epoch} adapter stats: ")

    # ------------------------------------------------------------------ eval

    def _eval_biases(self):
        """Single eval pass scoring every bias in self.biases.

        Computes three forward paths per batch:
          - nn_solver through MinEdit  → solver_neuro baseline (upper bound)
          - fused head through MinEdit → fused_neuro (stage b)
          - per-bias Prolog cleanup    → board_acc (stage c)

        Returns (per_bias, avg_sat, llm_raw_correct, llm_raw_n,
                 solver_neuro_correct, solver_neuro_n).
        """
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
        llm_raw_correct = 0
        llm_raw_n = 0
        solver_neuro_correct = 0
        solver_neuro_n = 0

        self.llm_adapter.reset_counters()

        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(device)
                target = target.to(device)

                # Perception is cheap (frozen CNN); run it first so we can
                # probe the cache before potentially invoking the LLM.
                perc_out = self.network.perception(data)

                if self.args.skip_cache_miss:
                    hit_idx = self.llm_adapter.get_cached_indices(perc_out)
                    if not hit_idx:
                        continue
                    if len(hit_idx) < data.shape[0]:
                        idx_t = torch.tensor(hit_idx, device=device)
                        data = data[idx_t]
                        target = target[idx_t]
                        perc_out = perc_out[idx_t]

                bsz = data.shape[0]
                gt_boards = target.argmax(dim=-1)              # (B, 81) 1..9

                # nn_solver baseline: the distribution FusionHead starts from.
                solver_probs = F.softmax(
                    self.network.nn_solver(torch.exp(perc_out)), dim=-1
                )  # (B, 81, 9)
                solver_proj, _ = self.constraint_layer(
                    solver_probs,
                    max_iterations=self.args.minedit_iters_eval,
                )
                solver_pred_1to9 = solver_proj.argmax(dim=-1) + 1  # (B, 81) 1..9
                solver_neuro_correct += int(
                    (solver_pred_1to9 == gt_boards).all(dim=1).sum().item()
                )
                solver_neuro_n += bsz

                # Fused head path.
                llm_logits = self.llm_adapter.solve_batch(perc_out).to(device)
                llm_pred_1to9 = llm_logits.argmax(dim=-1) + 1  # (B, 81) 1..9
                calibrated = self.network.temperature_head(
                    llm_logits, solver_probs, perc_out
                )  # (B, 81, 9)

                projected, _ = self.constraint_layer(
                    calibrated,
                    max_iterations=self.args.minedit_iters_eval,
                )
                sat_weighted += (
                    self.constraint_layer.constraint_satisfaction_rate(projected) * bsz
                )
                sat_n += bsz

                masking_logits = self._mask_logits(projected)  # (B, 81)
                gt_boards = target.argmax(dim=-1)              # (B, 81) 1..9
                llm_raw_correct += int((llm_pred_1to9 == gt_boards).all(dim=1).sum().item())
                llm_raw_n += bsz
                projected_1to9 = projected.argmax(dim=2) + 1   # (B, 81) 1..9
                wrong_proj = projected_1to9 != gt_boards

                for b_val in biases:
                    prob = (masking_logits + b_val).sigmoid().unsqueeze(-1)
                    gate = torch.cat(((1 - prob), projected * prob), dim=-1)
                    cleaned = gate.argmax(dim=-1)   # (B, 81) 0..9
                    kept_mask = cleaned != 0

                    st = per_bias[b_val]
                    st["blank_cells"] += int((~kept_mask).sum().item())
                    st["wrong_projected_cells"] += int(wrong_proj.sum().item())
                    st["wrong_projected_kept"] += int(
                        (wrong_proj & kept_mask).sum().item()
                    )

                    cleaned_cpu = cleaned.cpu()
                    for s_idx in range(bsz):
                        board = Board(cleaned_cpu[s_idx].reshape((9, 9)).int())
                        try:
                            ok = board.solve(
                                solver="prolog",
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
                        board_full = matches == 81
                        st["correct_cells"] += matches
                        st["total_cells"] += 81
                        if board_full:
                            st["fully_correct"] += 1

                        neuro_full = bool(
                            (projected_1to9[s_idx] == gt_boards[s_idx]).all().item()
                        )
                        if neuro_full:
                            st["neuro_correct"] += 1
                        if not neuro_full and board_full:
                            st["wrong_corrected"] += 1
                        if neuro_full and not board_full:
                            st["correct_corrupted"] += 1
                        st["n_samples"] += 1

                torch.cuda.empty_cache()

        avg_sat = sat_weighted / max(sat_n, 1)
        self.llm_adapter.log_counters(prefix="[llm-calib] eval adapter stats: ")
        return (per_bias, avg_sat, llm_raw_correct, llm_raw_n,
                solver_neuro_correct, solver_neuro_n)

    def test_epoch(self, epoch):
        (per_bias, avg_sat, llm_raw_correct, llm_raw_n,
         solver_neuro_correct, solver_neuro_n) = self._eval_biases()
        biases = self.biases

        summaries = {}
        for b_val in biases:
            st = per_bias[b_val]
            n = max(st["n_samples"], 1)
            tc = max(st["total_cells"], 1)
            n_wrong = max(st["n_samples"] - st["neuro_correct"], 1)
            summaries[b_val] = dict(
                board_acc=st["fully_correct"] / n,
                cell_acc=st["correct_cells"] / tc,
                wrong_corrected_rate=st["wrong_corrected"] / n_wrong,
                corrupted_rate=(
                    st["correct_corrupted"] / max(st["neuro_correct"], 1)
                ),
                blank_rate=st["blank_cells"] / tc,
                keep_rate_wrong_proj=(
                    st["wrong_projected_kept"]
                    / max(st["wrong_projected_cells"], 1)
                ),
                fully_correct=st["fully_correct"],
                n_samples=st["n_samples"],
                neuro_correct=st["neuro_correct"],
                wrong_corrected=st["wrong_corrected"],
                correct_corrupted=st["correct_corrupted"],
            )

        # (s), (a), (b) are bias-independent; pull (b) from any bias entry.
        any_s = summaries[biases[0]]
        n_eval = max(any_s["n_samples"], 1)
        llm_raw_acc = llm_raw_correct / max(llm_raw_n, 1)
        neuro_correct = any_s["neuro_correct"]
        neuro_acc = neuro_correct / n_eval
        solver_neuro_acc = solver_neuro_correct / max(solver_neuro_n, 1)

        # Per-stage accuracy header (printed once, before the bias sweep).
        print(
            f"[llm-calib][Epoch {epoch}] stage accuracy:"
            f"\n  (s) nn_solver → MinEdit (target)   : "
            f"{solver_neuro_acc:.4f}  ({solver_neuro_correct}/{solver_neuro_n})"
            f"\n  (a) LLM raw                        : "
            f"{llm_raw_acc:.4f}  ({llm_raw_correct}/{llm_raw_n})"
            f"\n  (b) fused → MinEdit (neuro)        : "
            f"{neuro_acc:.4f}  ({neuro_correct}/{n_eval})"
        )

        if len(biases) > 1:
            print(
                f"[llm-calib][Epoch {epoch}] "
                f"bias sweep — (c) +Prolog board_acc:"
            )
            print(
                f"  {'bias':>8} {'(c)board_acc':>13} "
                f"{'cor_wrong':>10} {'cor_correct':>12} {'blank_rt':>10}"
            )
            for b_val in biases:
                s = summaries[b_val]
                print(
                    f"  {b_val:>+8.3f} {s['board_acc']:>13.4f} "
                    f"{s['wrong_corrected_rate']:>10.4f} "
                    f"{s['corrupted_rate']:>12.4f} "
                    f"{s['blank_rate']:>10.4f}"
                )

        best_b = max(biases, key=lambda b: summaries[b]["fully_correct"])
        best_s = summaries[best_b]
        print(
            f"----[llm-calib][Epoch {epoch}] "
            f"(s) solver_neuro={solver_neuro_acc:.4f} ({solver_neuro_correct}/{solver_neuro_n}) "
            f"| (a) llm_raw={llm_raw_acc:.4f} ({llm_raw_correct}/{llm_raw_n}) "
            f"| (b) fused_neuro={neuro_acc:.4f} ({neuro_correct}/{n_eval}) "
            f"| (c) board_acc={best_s['board_acc']:.4f} "
            f"({best_s['fully_correct']}/{best_s['n_samples']}) @ bias={best_b:+.3f} "
            f"cell_acc={best_s['cell_acc']:.4f} "
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
            or best_s["fully_correct"]
            >= (self.best_board_correct + self.args.min_delta)
        )
        if improved:
            self.best_board_correct = best_s["fully_correct"]
            self.best_bias = best_b
            self.best_epoch = epoch
            self.epochs_since_improve = 0
            path = os.path.join(
                self.model_dir, f"checkpoint_best_R_{self.seed}.pth"
            )
            torch.save(self.network.state_dict(), path)
            meta_path = os.path.join(
                self.model_dir, f"checkpoint_best_R_{self.seed}.json"
            )
            with open(meta_path, "w") as f:
                json.dump(
                    dict(
                        seed=self.seed,
                        epoch=epoch,
                        best_bias=best_b,
                        best_board_correct=best_s["fully_correct"],
                        n_samples=best_s["n_samples"],
                        board_acc=best_s["board_acc"],
                        pipeline="llm",
                        use_perception_features=self.args.use_perception_features,
                        freeze_fusion_head=self.freeze_fusion_head,
                        edit_layer=self.args.edit_layer,
                        constraints_file=(
                            self.args.constraints_file
                            if self.args.edit_layer == "casper" else None
                        ),
                        constraints_format=(
                            self.args.constraints_format
                            if self.args.edit_layer == "casper" else None
                        ),
                        casper_cell_order=(
                            self.args.casper_cell_order
                            if self.args.edit_layer == "casper" else None
                        ),
                    ),
                    f,
                    indent=2,
                )
            print(
                f"[llm-calib] saved best to {path} "
                f"(board_acc={best_s['board_acc']:.4f}, bias={best_b:+.3f})"
            )
        else:
            self.epochs_since_improve += 1

        return best_s["fully_correct"], self.epochs_since_improve

    def _set_mask_trainable(self, trainable: bool):
        for p in self.network.mask_nn.parameters():
            p.requires_grad = trainable

    def train(self, n_epochs):
        stage_freeze = getattr(self.args, "stage_mask_freeze_epochs", 0)
        if stage_freeze > 0:
            self._set_mask_trainable(False)
            print(
                f"[llm-calib] Stage 1: fusion head training for {stage_freeze} epoch(s); "
                f"mask_nn frozen."
            )

        # Baseline with pretrained weights; always writes initial checkpoint
        # so a --epochs 0 run still produces something evaluable.
        self.test_epoch(0)
        for epoch in range(1, n_epochs + 1):
            if stage_freeze > 0 and epoch == stage_freeze + 1:
                self._set_mask_trainable(True)
                print(
                    f"[llm-calib] Stage 2 (epoch {epoch}): unfreezing mask_nn "
                    f"for joint fine-tuning."
                )

            lr = self.adjust_learning_rate(epoch)
            t0 = time.time()
            self.train_epoch(epoch, lr)
            t1 = time.time()
            print(f"[llm-calib] train time: {t1 - t0:.1f}s")
            _, patience_ct = self.test_epoch(epoch)
            if patience_ct >= self.args.patience:
                print(
                    f"[llm-calib] early stopping at epoch {epoch} "
                    f"(no improvement for {patience_ct} epochs; "
                    f"best_epoch={self.best_epoch} "
                    f"best_board_correct={self.best_board_correct} "
                    f"best_bias={self.best_bias:+.3f})"
                )
                break


if __name__ == "__main__":
    parser = ArgumentParser("sudoku_minedit_llm_mask_calibrate")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--data", type=str, default="satnet")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--warmup", default=0, type=int)
    parser.add_argument("-b", "--batch-size", default=200, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=3e-1, type=float)
    parser.add_argument("--disable-cos", action="store_true")
    parser.add_argument("--block-len", default=81, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument(
        "--model-dir",
        default=str(paths.CKPT_DIR / "minedit" / "llm_mask_calibrated" / "seed0"),
        type=str,
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Use historical pairwise constraint-layer path.",
    )
    parser.add_argument(
        "--constraint-group-order", default="default", type=str,
        help="Processing order for the 27 Sudoku groups inside "
             "SudokuConstraintLayer. \"default\" (default) keeps the "
             "canonical 0..26 order. Otherwise a comma-separated "
             "permutation of 0..26.",
    )
    add_edit_layer_args(parser)
    parser.add_argument("--minedit-iters-train", default=10, type=int)
    parser.add_argument("--minedit-iters-eval", default=10, type=int)
    # Loss weights
    parser.add_argument(
        "--lambda-mask",
        default=1.0,
        type=float,
        help="Weight for mask BCE loss.",
    )
    parser.add_argument(
        "--lambda-temp",
        default=0.2,
        type=float,
        help="Weight for temperature-head calibration NLL loss.",
    )
    parser.add_argument(
        "--label-smoothing",
        default=0.3,
        type=float,
        help=(
            "Label smoothing α for the calibration NLL loss. "
            "With LLM_CONF=10 and α=0.3 the analytical optimum is T≈3.3, "
            "preventing NLL from driving T→0 (default 0.3)."
        ),
    )
    # Stability / imbalance
    parser.add_argument(
        "--neg-weight",
        default=20.0,
        type=float,
        help="Per-cell BCE weight for negative (blank-target) cells.",
    )
    parser.add_argument(
        "--clip-norm",
        default=1.0,
        type=float,
        help="Gradient clipping norm on trainable heads.",
    )
    # Staged training
    parser.add_argument(
        "--stage-mask-freeze-epochs",
        default=0,
        type=int,
        help=(
            "Train fusion head only for this many epochs with mask_nn frozen "
            "(Stage 1), then unfreeze mask_nn for joint fine-tuning (Stage 2). "
            "0 = no staging, both trained from epoch 1 (default)."
        ),
    )
    parser.add_argument(
        "--freeze-fusion-head",
        dest="freeze_fusion_head",
        action="store_true",
        default=True,
        help=(
            "Freeze temperature_head (fusion head) at its loaded weights and "
            "train mask_nn only (default: on). At the identity init the head "
            "outputs the frozen solver's probabilities, so the LLM has no effect."
        ),
    )
    parser.add_argument(
        "--train-fusion-head",
        dest="freeze_fusion_head",
        action="store_false",
        help=(
            "Opt out of --freeze-fusion-head and train temperature_head jointly "
            "with mask_nn. Not recommended: it degraded the head (spec §2.3)."
        ),
    )
    # Early stopping
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument(
        "--min-delta",
        default=1,
        type=int,
        help="Minimum improvement in full-board correct count to reset patience.",
    )
    # Eval-time bias sweep
    parser.add_argument(
        "--eval-mask-biases",
        default="0.0",
        type=str,
        help="Comma-separated additive biases on mask logits to evaluate each epoch.",
    )
    # LLM options
    parser.add_argument(
        "--llm-cache-dir",
        default=str(paths.LLM_CACHE_DIR),
        type=str,
        help="Directory for caching LLM outputs (default: paths.LLM_CACHE_DIR, "
             "env SUDOKU_LLM_CACHE_DIR).",
    )
    parser.add_argument(
        "--finetuned-model-path",
        default=None,
        type=str,
        help="Path to fine-tuned LoRA checkpoint. Defaults to the SATNet SFT checkpoint.",
    )
    parser.add_argument(
        "--generation-batch-size",
        default=4,
        type=int,
        help="Boards per model.generate() call in the LLM adapter.",
    )
    parser.add_argument(
        "--skip-cache-miss",
        dest="skip_cache_miss",
        action="store_true",
        default=True,
        help=(
            "During training, skip any board in a batch that would require "
            "an LLM call (i.e. is not in the disk/memory cache). "
            "Useful when the cache is complete and the LLM consistently "
            "fails on the uncached boards anyway (default: on)."
        ),
    )
    parser.add_argument(
        "--no-skip-cache-miss",
        dest="skip_cache_miss",
        action="store_false",
        help="Generate uncached boards with the LLM instead of skipping them "
             "(needs a GPU; writes to --llm-cache-dir).",
    )
    # TemperatureHead v2 options
    parser.add_argument(
        "--use-perception-features",
        dest="use_perception_features",
        action="store_true",
        default=True,
        help=(
            "Pass perception log-probs to TemperatureHead so it can model "
            "CNN uncertainty per cell (default: on). Recorded in checkpoint "
            "sidecar; use --no-perception-features at eval time to match."
        ),
    )
    parser.add_argument(
        "--no-perception-features",
        dest="use_perception_features",
        action="store_false",
        help="Disable perception conditioning in TemperatureHead.",
    )
    parser.add_argument(
        "--temp-hidden-dim",
        default=128,
        type=int,
        help="Hidden dimension of the TemperatureHead residual MLP (default 128).",
    )
    args = parser.parse_args()
    if not args.freeze_fusion_head:
        print(
            "[llm-calib] WARNING: --train-fusion-head: the fusion head will be "
            "trained jointly with mask_nn. With this recipe that degraded the "
            "head (fused->CASPER valid acc 58.0% -> 27.3% over 9 epochs, "
            "PipeAcc 79.6; CASPER_MASK_TRAINING_PORT.md §2.3).",
            flush=True,
        )

    args.constraint_group_order = parse_group_order(args.constraint_group_order)
    print(f"[llm-calib] constraint-group-order = "
          f"{format_group_order(args.constraint_group_order)}")

    set_deterministic(args.seed)

    train_dataset = SudokuDataset_RL(args.data, "-train")
    test_dataset = SudokuDataset_RL(args.data, "-valid")

    model = get_llm_model(
        block_len=args.block_len,
        use_perception_features=args.use_perception_features,
        temp_hidden_dim=args.temp_hidden_dim,
    )
    model.load_pretrained_models(args.data)
    model.to(device)

    llm_adapter = LLMSolverAdapter(
        model_path=args.finetuned_model_path or DEFAULT_FINETUNED_MODEL_PATH,
        cache_dir=args.llm_cache_dir,
        generation_batch_size=args.generation_batch_size,
    )

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

    calibrator = LLMCalibrator(
        model, llm_adapter, train_loader, test_loader,
        args.model_dir, args.lr, args.batch_size, args.seed, args,
    )

    try:
        calibrator.train(args.epochs)
    except KeyboardInterrupt:
        print("[llm-calib] KeyboardInterrupt received; shutting down cleanly.", flush=True)
        raise
    finally:
        llm_adapter.close()
        del calibrator
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
