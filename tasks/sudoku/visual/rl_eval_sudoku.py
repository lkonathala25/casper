import os
import json
import argparse
from pathlib import Path
from time import time
import math
import numpy as np
import torch
import torch.nn as nn
import statistics
from models.transformer_sudoku import get_model
from datasets import SudokuDataset_RL
import random
from nasr import compute_reward
from sudoku_solver.board import Board
import paths
_SWIPL_ENV = str(paths.SWIPL_ENV)
_SWIPL_LIB = f"{_SWIPL_ENV}/lib/swipl/lib/x86_64-linux/libswipl.so"
_SWIPL_HOME = f"{_SWIPL_ENV}/lib/swipl"
if Path(_SWIPL_LIB).exists():
    os.environ.setdefault("LIBSWIPL_PATH", _SWIPL_LIB)
    os.environ.setdefault("SWI_HOME_DIR", _SWIPL_HOME)
try:
    from pyswip import Prolog
except Exception:
    print('-->> Prolog not installed')
from sudoku_solver.board import check_input_board,check_consistency_board
from models.perception import SequentialPerception
from casper_sudoku_layer import add_edit_layer_args, build_edit_layer
from group_order_utils import parse_group_order, format_group_order
from tqdm import tqdm
from utils.utils import retrieve_hints_from_solution

SRC_DIR = Path(__file__).resolve().parent
PROLOG_RULES_PATH = SRC_DIR / 'sudoku_solver' / 'sudoku_prolog.pl'

def init_parser():
    parser = argparse.ArgumentParser(description='Quick testing script')

    # General args
    parser.add_argument('--gpu-id', default=0, type=int)
    parser.add_argument('-j', '--workers', default=0, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='log frequency (by iteration)')
    parser.add_argument('--batch-size', default=100, type=int,
                        help='Batch size')
    parser.add_argument('--nasr', type=str, default='rl',
                        help='choice of nasr with nasr_rl or pretrained (without rl)')
    parser.add_argument('--data', type=str, default='satnet',
                        help='dataset name between big_kaggle, minimal_17, multiple_sol and satnet')
    parser.add_argument('--noise-setting', default='xxx/yyy.json', type=str,
                        help='Json file of noise setting (dict)')
    parser.add_argument('--solver', type=str, default='prolog',
                        help='symbolic solver to use. available options prolog and backtrack')
    parser.add_argument('--transform-data', type=str, default=None,
                        help='noise type to add to the images of sudoku digits. available options are blur and rotation')
    parser.add_argument('--transform-data-param', default=0.0, type=float,
                        help='noise value. angle degrees for rotation and sigma for blur.')
    # Model args
    parser.add_argument('--block-len', default=81, type=int,
                        help='board size')
    parser.add_argument('--code-rate', default=2, type=int,
                        help='Code rate')
    parser.add_argument('--ckpt-path', type=str, default='checkpoint/tensor/checkpoint_best_R_1234.pth',
                        help='RL checkpoint path (absolute, or relative to sudoku/src).')
    parser.add_argument('--use-minedit', action='store_true',
                        help='Project solver output through SudokuConstraintLayer '
                            'before handing it to the mask + Prolog. Uses the '
                            'KEEP=1 mask convention (matching NASR pretraining and '
                            'minedit_sudoku.py training).')
    parser.add_argument('--cts-masking', action='store_true',
                        help='Use the CTS keep/blank conversion without MinEdit '
                            'projection: cleaned = argmax([1-mask, P*mask]). '
                            'This matches cts_sudoku.py and cts_llm_sudoku.py. '
                            'Do not combine with legacy NASR round(mask)*argmax(P).')
    parser.add_argument('--minedit-iters', default=10, type=int,
                        help='Max outer iterations for the constraint-layer projection at eval.')
    parser.add_argument('--pairwise', action='store_true',
                        help='Use the historical pairwise per-digit conflict repair path '
                             'inside SudokuConstraintLayer. Default (unset) uses the joint '
                             'k-ary AllDifferent projection via Hungarian assignment.')
    parser.add_argument('--constraint-group-order', default='default', type=str,
                        help='Processing order for the 27 Sudoku groups inside '
                             'SudokuConstraintLayer (0-8 rows, 9-17 columns, 18-26 boxes). '
                             '"default" (default) keeps the canonical 0..26 order. '
                             'Otherwise a comma-separated permutation of 0..26, e.g. '
                             '"26,25,...,0". Reorders only the processing sequence, not '
                             'group membership or the Hungarian assignment logic. Only '
                             'meaningfully affects behaviour when --pairwise is not set '
                             '(joint AllDifferent path).')
    add_edit_layer_args(parser)
    parser.add_argument('--oracle-blanking', type=str, default='none',
                        choices=['none', 'wrong-projected', 'all-non-givens'],
                        help='Optional sanity upper-bound check. '
                            'wrong-projected: blank projected cells whose argmax is wrong '
                            '(uses target, analysis only). '
                            'all-non-givens: keep only puzzle givens and blank the rest '
                            '(uses target+dataset clues, analysis only).')
    parser.add_argument('--mask-verbose', action='store_true',
                        help='Run full mask diagnostics in one pass: standard metrics '
                             '+ both oracle blanking checks (wrong-projected and '
                             'all-non-givens).')
    parser.add_argument('--mask-logit-bias', default=0.0, type=float,
                    help='Additive bias applied to mask logits before sigmoid. '
                        'Negative values make masking more aggressive '
                        '(more blanking), positive values less aggressive.')

    # ---- LLM backbone (opt-in) ----
    parser.add_argument(
        '--solver-backbone', default='transformer',
        choices=['transformer', 'llm'],
        help='Solver backbone. "llm" loads a LLMTransformerSudoku checkpoint '
             '(minedit_llm_mask_calibrate.py, cts_llm_sudoku.py, or '
             'train_anesi_llm_sudoku.py). Add --use-minedit explicitly when '
             'the checkpoint was trained with MinEdit in the loop (minedit '
             'baseline). Omit --use-minedit for CTS-LLM and ANESI-LLM.',
    )
    parser.add_argument(
        '--llm-cache-dir', default=str(paths.LLM_CACHE_DIR), type=str,
        help='Directory for LLM output disk cache (default: paths.LLM_CACHE_DIR, '
             'env SUDOKU_LLM_CACHE_DIR).',
    )
    parser.add_argument(
        '--finetuned-model-path', default=None, type=str,
        help='Path to fine-tuned LoRA checkpoint for the LLM adapter. '
             'Defaults to paths.LLM_SFT_PATH (the SATNet SFT checkpoint). '
             'Only used with --solver-backbone llm.',
    )
    parser.add_argument(
        '--generation-batch-size', default=4, type=int,
        help='Boards per model.generate() call in the LLM adapter. '
             'Only used with --solver-backbone llm.',
    )
    parser.add_argument(
        '--skip-cache-miss',
        action='store_true',
        help='LLM mode only: use cache-only evaluation (solve_batch_cached_only). '
             'Boards not found in the cache receive zero logits instead of '
             'triggering live LLM generation. Use on subsequent eval runs after '
             'the cache has been populated by the first run.',
    )
    # TemperatureHead v2 options — must match the training-time flags.
    parser.add_argument(
        '--use-perception-features',
        dest='use_perception_features',
        action='store_true',
        default=True,
        help='Pass perception log-probs to TemperatureHead (default: on). '
             'Must match the flag used during training.',
    )
    parser.add_argument(
        '--no-perception-features',
        dest='use_perception_features',
        action='store_false',
        help='Disable perception conditioning in TemperatureHead.',
    )
    parser.add_argument(
        '--temp-hidden-dim', default=128, type=int,
        help='Hidden dimension of the TemperatureHead residual MLP (default 128). '
             'Must match the value used during training.',
    )

    return parser


def final_output(model, ground_truth_sol, solution_boards, masking_boards, args,
                 constraint_layer=None):
    """Build a partial board, pass it to Prolog, return completed boards.

    Mask convention (KEEP=1 — shared between both paths; matches NASR
    pretraining and minedit_sudoku.py):
      use_minedit=False and cts_masking=False (legacy NASR RL):
          cleaned = argmax(solution)+1 * round(mask)    -> mask=1 KEEPS cell
          Mask source: learned mask head applied to the raw solver output
          (unchanged from NASR).
      use_minedit=True OR cts_masking=True (same KEEP=1 polarity):
          cleaned = cat(((1-mask_prob), projected*mask_prob)).argmax
          -> mask=1 KEEPS cell (argmax picks 1..9)
          -> mask=0 BLANKS cell (argmax picks 0)
          With MinEdit, mask source is re-run on the PROJECTED solver
          distribution so the mask is predicted from the same tensor it
          will gate. With CTS masking and no MinEdit, the original mask
          logits are used directly on the unprojected board tensor.

    If a constraint_layer is provided, solution_boards is projected through
    it before the argmax so the partial board is AllDifferent-consistent,
    and masking_boards is replaced with the mask head's prediction on that
    projected tensor.
    """
    ground_truth_boards = torch.argmax(ground_truth_sol, dim=2)

    if constraint_layer is not None:
        projected, _ = constraint_layer(solution_boards, max_iterations=args.minedit_iters)
        # Align mask with the tensor it will gate: re-run the learned
        # mask head on the projected distribution (matches the input
        # format that each nasr mode originally feeds mask_nn).
        if args.nasr == 'pretrained':
            proj_idx = projected.argmax(dim=2) + 1
            mask_input = nn.functional.one_hot(proj_idx, num_classes=10)[:, :, 1:].to(torch.float32)
        else:
            mask_input = projected
        masking_boards = model.mask_nn(mask_input)
    else:
        projected = solution_boards

    masking_prob = (masking_boards + args.mask_logit_bias).sigmoid()

    if args.use_minedit or args.cts_masking:
        mask_prob_e = masking_prob.unsqueeze(-1) if masking_prob.dim() == 2 else masking_prob
        logits = torch.cat(
            ((1 - mask_prob_e), projected * mask_prob_e), dim=-1
        )
        cleaned_boards = logits.argmax(dim=-1).cpu()   # 0 where blank, 1..9 otherwise
        solution_boards_new = projected.argmax(dim=2) + 1
    else:
        solution_boards_new = projected.argmax(dim=2) + 1
        mask_boards = torch.round(masking_prob).squeeze(-1) if masking_prob.dim() == 3 else torch.round(masking_prob)
        cleaned_boards = np.multiply(solution_boards_new.cpu(), mask_boards.cpu())

    final_boards = []
    if args.solver == "prolog":
        prolog_instance = Prolog()
        prolog_instance.consult(str(PROLOG_RULES_PATH))

    time_solver = 0.0
    for i in range(len(cleaned_boards)):
        board_to_solver = Board(cleaned_boards[i].reshape((9, 9)).int())
        time_begin = time()
        try:
            if args.solver == "prolog":
                solver_success = board_to_solver.solve(
                    solver='prolog', prolog_instance=prolog_instance
                )
            else:
                solver_success = board_to_solver.solve(solver='backtrack')
        except StopIteration:
            solver_success = False
        time_solver = (time() - time_begin)
        final_solution = board_to_solver.board.reshape(81,)
        if not solver_success:
            final_solution = solution_boards_new[i].cpu()
        reward = compute_reward(
            solution_boards_new[i].cpu(), final_solution, ground_truth_boards[i]
        )
        model.rewards.append(reward)
        final_boards.append(final_solution)
    # Effective per-cell mask actually applied by the pipeline (1=kept,
    # 0=blanked), derived from cleaned_boards in BOTH branches so it
    # always matches what Prolog actually saw:
    #   * minedit path: gate is argmax((1-p_m, projected*p_m)) — a cell
    #     with p_m>0.5 can still be blanked when projected confidence
    #     is low, so round(sigmoid(mask)) does not match.
    #   * non-minedit path: gate uses round(sigmoid(mask + bias)), so
    #     a non-zero --mask-logit-bias makes round(sigmoid(mask)) (no
    #     bias) disagree with the actual gating.
    # Using cleaned_boards != 0 makes intuition_eval's nep invariant
    # hold under any bias and any mode.
    kept = (cleaned_boards != 0)
    if not torch.is_tensor(kept):
        kept = torch.from_numpy(np.asarray(kept))
    applied_mask = kept.to(torch.float32).unsqueeze(-1)
    return final_boards, time_solver, applied_mask


def validate(val_loader, model, args, constraint_layer=None, epoch=None, time_begin=None,
             llm_adapter=None):
    model.eval()
    n = 0
    reward_sum = 0.0
    sat_weighted = 0.0
    casper_certified = 0
    casper_seen = 0
    preds = []
    labels = []
    inputs = []
    solutions = []     # post-projection (so downstream eval_improvement sees
                       # what Prolog actually received)
    mask = []
    total_cells = 0
    blank_cells = 0
    wrong_projected_cells = 0
    wrong_projected_kept = 0
    llm_raw_correct = 0
    llm_raw_n = 0
    oracle_modes = []
    if args.mask_verbose:
        oracle_modes = ['wrong-projected', 'all-non-givens']
    elif args.oracle_blanking != 'none':
        oracle_modes = [args.oracle_blanking]
    oracle_stats = {
        mode: {'correct': 0, 'total': 0}
        for mode in oracle_modes
    }

    oracle_prolog = None
    if len(oracle_modes) > 0 and args.solver == 'prolog':
        oracle_prolog = Prolog()
        oracle_prolog.consult(str(PROLOG_RULES_PATH))

    time_begin = time()
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(val_loader, desc='eval', unit='batch', ncols=80, leave=True)):

            images = images.to(args.gpu_id)
            target = target.to(args.gpu_id)

            if llm_adapter is not None:
                # Fusion path: perception + nn_solver (frozen) + LLM (frozen)
                # -> FusionHead -> MinEdit.
                # masking_boards is a placeholder; final_output will re-run
                # mask_nn on the projected distribution (use-minedit path).
                perc_out = model.perception(images)           # (B,81,10) log-probs
                solver_probs = torch.nn.functional.softmax(
                    model.nn_solver(torch.exp(perc_out)), dim=-1
                )  # (B, 81, 9) frozen soft uncertainty
                if getattr(args, 'skip_cache_miss', False):
                    llm_logits = llm_adapter.solve_batch_cached_only(perc_out)
                else:
                    llm_logits = llm_adapter.solve_batch(perc_out)
                llm_logits = llm_logits.to(args.gpu_id)
                solution_boards = model.temperature_head(
                    llm_logits, solver_probs, perc_out
                )  # (B, 81, 9)
                masking_boards = model.mask_nn(solution_boards)
                # overridden in final_output when constraint_layer is not None (MinEdit path)
                _llm_pred = llm_logits.argmax(dim=-1).cpu() + 1   # (B,81) 1..9
                _gt       = target.argmax(dim=2).cpu()             # (B,81) 1..9
                llm_raw_correct += int((_llm_pred == _gt).all(dim=1).sum().item())
                llm_raw_n += images.size(0)
            else:
                solution_boards, masking_boards = model(images, nasr=args.nasr)

            if constraint_layer is not None:
                projected, proj_info = constraint_layer(
                    solution_boards, max_iterations=args.minedit_iters
                )
                if 'n_certified' in proj_info:
                    casper_certified += proj_info['n_certified']
                    casper_seen += images.size(0)
                sat_weighted += (
                    constraint_layer.constraint_satisfaction_rate(projected)
                    * images.size(0)
                )
            else:
                projected = solution_boards

            # Mask prediction is realigned with the projected distribution
            # inside final_output (use-minedit path): the learned mask head
            # is re-run on the projected tensor so that mask logits describe
            # the same digits that get gated. In the non-minedit path the
            # legacy mask computed from raw solver output is used as before.
            # final_output returns the mask it actually applied (realigned
            # in the minedit path, raw otherwise); we record THAT one so
            # downstream diagnostics (intuition_eval) see the mask used by
            # the pipeline rather than a stale raw prediction.
            pred_final, _, applied_mask = final_output(
                model, target, solution_boards, masking_boards, args,
                constraint_layer=constraint_layer,
            )

            # Diagnostics for the effective keep/blank policy after projection.
            gt_digits = target.argmax(dim=2).cpu()                      # (B, 81), 1..9
            projected_digits = projected.argmax(dim=2).cpu() + 1        # (B, 81), 1..9
            keep_mask = applied_mask.squeeze(-1).cpu().to(torch.bool)   # (B, 81)
            wrong_proj = (projected_digits != gt_digits)

            total_cells += int(keep_mask.numel())
            blank_cells += int((~keep_mask).sum().item())
            wrong_projected_cells += int(wrong_proj.sum().item())
            wrong_projected_kept += int((wrong_proj & keep_mask).sum().item())

            # Optional oracle sanity check (analysis only): run the same symbolic
            # solver from oracle-blanked partial boards and report board accuracy.
            if len(oracle_modes) > 0:
                gt_np = gt_digits.numpy()
                oracle_inputs = {}
                for mode in oracle_modes:
                    if mode == 'wrong-projected':
                        oracle_cleaned = projected_digits.clone()
                        oracle_cleaned[wrong_proj] = 0
                    else:
                        oracle_cleaned_np = []
                        for b in range(gt_np.shape[0]):
                            hints = retrieve_hints_from_solution(
                                gt_np[b], args.data,
                            ).reshape(81)
                            oracle_cleaned_np.append(hints)
                        oracle_cleaned = torch.from_numpy(
                            np.asarray(oracle_cleaned_np, dtype=np.int64)
                        )
                    oracle_inputs[mode] = oracle_cleaned

                for mode in oracle_modes:
                    oracle_cleaned = oracle_inputs[mode]
                    for b in range(oracle_cleaned.shape[0]):
                        board_to_solver = Board(oracle_cleaned[b].reshape((9, 9)).int())
                        try:
                            if args.solver == 'prolog':
                                ok = board_to_solver.solve(
                                    solver='prolog', prolog_instance=oracle_prolog,
                                )
                            else:
                                ok = board_to_solver.solve(solver='backtrack')
                        except StopIteration:
                            ok = False

                        final_solution = torch.from_numpy(
                            board_to_solver.board.reshape(81,)
                        )
                        if not ok:
                            final_solution = oracle_cleaned[b]

                        oracle_stats[mode]['correct'] += int(
                            (final_solution == gt_digits[b]).all().item()
                        )
                        oracle_stats[mode]['total'] += 1

            mask.append(applied_mask)
            preds.append(pred_final)
            labels.append(target)
            inputs.append(images)
            solutions.append(projected)

            reward_sum += float(sum(model.rewards))
            n += images.size(0)
            model.rewards = []
            torch.cuda.empty_cache()

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_r = reward_sum / max(n, 1)
                tqdm.write(f'[Test][{i}/{len(val_loader)}] avg_cell_reward: {avg_r:.4f}')
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    avg_r = reward_sum / max(n, 1)
    print(f'avg_cell_reward {avg_r:.4f} \t \t Time: {total_mins:.2f}')
    if constraint_layer is not None:
        print(f'AllDifferent sat rate: {sat_weighted / max(n, 1):.4f}')
    if casper_seen > 0:
        print(f'CASPER certified: {casper_certified}/{casper_seen}')
    blank_rate = blank_cells / max(total_cells, 1)
    keep_rate_wrong_proj = 0.0
    if wrong_projected_cells > 0:
        keep_rate_wrong_proj = wrong_projected_kept / wrong_projected_cells
    print(f'Mask blank rate: {blank_rate:.4f}')
    print(f'Mask keep-rate on wrong projected cells: {keep_rate_wrong_proj:.4f}')
    if len(oracle_modes) > 0:
        for mode in oracle_modes:
            correct = oracle_stats[mode]['correct']
            total = oracle_stats[mode]['total']
            oracle_acc = correct / max(total, 1)
            print(f'Oracle blanking ({mode}) board_acc: {oracle_acc:.4f} '
                  f'({correct}/{total})')

    pred_boards_b = torch.cat(solutions, 0).sigmoid().cpu().numpy()
    y_pred_b = []
    for idx in range(len(pred_boards_b)):
        r_pred_board = np.zeros((81), dtype=int)
        pred_board = pred_boards_b[idx]
        for k in range(81):
            pred_cell = pred_board[k].argmax()+1
            r_pred_board[k] = float(pred_cell)
        y_pred_b.append(r_pred_board)

    y_pred = []
    for j in preds:
        for i in j:
            y_pred.append(i) 
    y_targ = torch.cat(labels, 0).cpu().numpy()
    input_b = torch.cat(inputs, 0).cpu().numpy()
    mask_b = torch.cat(mask, 0).cpu().numpy()
    if llm_raw_n > 0:
        llm_raw_pct = llm_raw_correct * 100.0 / llm_raw_n
        print(f'* Num. of correct solution boards from LLM (raw): {llm_raw_correct}/{llm_raw_n}-- ({llm_raw_pct:.2f}%)')
    print("-------------------------------")
    print("Improvement:")
    print("-----------")
    eval_improvement(y_pred,y_pred_b,y_targ,args.data)
    print("-------------------------------")
    intuition_eval(input_b,y_pred,y_pred_b,mask_b,y_targ,args)
    print("-------------------------------")


def intuition_eval(input_boards,y_pred,y_pred_b,mask_b,y_targ,args):
    perception_path = paths.PRETRAINED_DIR / 'perception' / args.data / 'checkpoint_best.pth'
    if not perception_path.exists():
        raise FileNotFoundError(f'Missing perception checkpoint: {perception_path}')
    perception = SequentialPerception()
    perception.to(args.gpu_id)
    perception.load_state_dict(torch.load(perception_path, map_location='cpu')) 
    input_boards = (torch.from_numpy(input_boards)).to(torch.float32).to(args.gpu_id)
    perception_boards = None
    with torch.no_grad():
        for i in input_boards:
            result_tensor = perception(i.unsqueeze(dim=0))
            if perception_boards is None:
                perception_boards = result_tensor
            else:
                perception_boards = torch.cat((perception_boards,result_tensor))
    perception_boards = perception_boards.clone().detach()
    perception_boards = torch.exp(perception_boards)
    perception_boards = perception_boards.argmax(dim=2)
    perception_boards = perception_boards.cpu().detach().numpy()
    n_errors_p_corrected_by_ns = []
    n_error_perception_masked = []
    n_error_ns_masked_i = []
    n_error_ns_masked_s = []
    
    for i in tqdm(range(len(y_pred))):
        perception_board = perception_boards[i].reshape(81)
        neuro_solver_board = np.array(y_pred_b[i]).reshape(81)
        mask_board = np.array(mask_b[i]).astype(int)
        pipeline_board = np.array(y_pred[i]).reshape(81)
        gt_board = np.array(y_targ[i]).reshape((81,10))
        gt_board = gt_board.argmax(axis=1)

        input_board = retrieve_hints_from_solution(gt_board,args.data).reshape(81)
        
        nep = 0 # number error perception
        nep_m = 0 # number error perception corrected by the mask
        nep_cnp = 0 # number error perception corrected by solverNN

        nens_s = 0 # number error neuro solver, in the solutions
        nens_i = 0 # number error neuro solver, in the inputs
        nens_m_s = 0 # number error solverNN, in the solutions, that are found by the mask
        nens_m_i = 0 # number error solverNN, in the inputs, that are found by the mask

        for j in range(81):
        
            if input_board[j]!= 0 and input_board[j]!=perception_board[j]: # error perception
                if pipeline_board[j] == gt_board[j]: # has been corrected
                    nep+=1
                    if neuro_solver_board[j] == gt_board[j]:
                        nep_cnp += 1
                    else:
                        if mask_board[j] == 0:
                            nep_m += 1
            
            assert nep == nep_cnp + nep_m

            if neuro_solver_board[j]!=gt_board[j]: # error solverNN 
                if input_board[j]== 0: # error in the solutions
                    nens_s+=1
                    if pipeline_board[j] == gt_board[j]: # has been corrected
                        nens_m_s+=1
                else: # error in the input
                    nens_i+=1
                    if pipeline_board[j] == gt_board[j]: # has been corrected
                        nens_m_i+=1

        if nep!=0:
            n_errors_p_corrected_by_ns.append(nep_cnp/nep)
            n_error_perception_masked.append(nep_m/nep)
        if nens_i!=0:
            n_error_ns_masked_i.append(nens_m_i/nens_i)
        if nens_s!=0:
            n_error_ns_masked_s.append(nens_m_s/nens_s)
        
    print(f"Num errors of the perception corrected by the SolverNN (avg): {statistics.mean(n_errors_p_corrected_by_ns)*100}%)")   
    print(f"Num errors of the perception corrected by the Mask-Predictor (avg): {statistics.mean(n_error_perception_masked)*100}%)")   
    print("\n---\n")
    print(f"Num errors of the solverNN in the input corrected (avg): {statistics.mean(n_error_ns_masked_i)*100}%)")   
    print(f"Num errors of the solverNN in the solutions corrected(avg): {statistics.mean(n_error_ns_masked_s)*100}%)")   


def eval_improvement(y_pred,y_pred_b,y_target,dataset_name=None):
    time_solutions = []
    y_pred = [l.tolist()for l in y_pred]
    y_pred_b = [l.tolist()for l in y_pred_b]
    y_pred = [list(map(int,i)) for i in y_pred]
    y_pred_b = [list(map(int,i)) for i in y_pred_b]
    y_targ = []
    for board in y_target:
        tmp_board = []
        for cell in board:
            tmp_board.append(cell.argmax())
        y_targ.append(tmp_board)
    y_targ = [list(map(int,i)) for i in y_targ]
    num_total = len(y_pred)
    num_correct_b =0
    num_correct = 0
    num_corrected_boards = 0
    num_wrong_boards = 0
    average_correct_cells = []
    for i in range(len(y_pred)):
        correct_cells=0.0
        if y_pred_b[i] == y_targ[i]:
            num_correct_b += 1
        else:
            if dataset_name == 'multiple_sol':
                input_board = retrieve_hints_from_solution(np.array(y_targ[i]),dataset_name)
                check_input = check_input_board(input_board,np.array(y_pred_b[i]))
                consistent = check_consistency_board(np.array(y_pred_b[i]))
                if check_input and consistent:
                    num_correct_b += 1
                    #print('alternative solution found')
                else:
                  if y_pred[i] == y_targ[i]:
                    num_corrected_boards += 1  
            else:
                if y_pred[i] == y_targ[i]:
                    num_corrected_boards += 1
        
        if y_pred[i] == y_targ[i]:
            num_correct += 1
            time_solutions.append(1)
        else:
            time_solutions.append(0)
            if y_pred_b[i] == y_targ[i]:
                num_wrong_boards += 1
        for j,k in zip(y_pred[i],y_targ[i]):
            if j==k:
                correct_cells+=1
        correct_cells/=81
        average_correct_cells.append(correct_cells)
    print(f"* Num. of correct solution boards from Neuro-Solver: {num_correct_b}/{num_total}-- ({(num_correct_b*100.)/(num_total):.2f}%)")
    print(f"* Num. of correct solution boards from NASR: {num_correct}/{num_total}-- ({((num_correct)*100.)/(num_total):.2f}%)")
    if (num_total-num_correct_b) >0 :
        print(f"* Num. of wrong solution boards (from Neuro-solver) \n  that have been corrected with NASR: {num_corrected_boards}/{num_total-num_correct_b} -- ({(num_corrected_boards*100.)/(num_total-num_correct_b):.2f}%)")
    if num_correct_b>0 :
        print(f"* Num. of correct solution boards (from Neuro-solver) \n  that have been corrupted with NASR: {num_wrong_boards}/{num_correct_b} -- ({num_wrong_boards*100./num_correct_b:.2f}%)")
    print(f"* Num correct cells (avg): {statistics.mean(average_correct_cells)*100}%")   
    return time_solutions


def main():
    parser = init_parser()
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = SRC_DIR / ckpt_path

    # Hard-refuse the shipped NASR RL fallback. Silently swapping in
    # outputs/rl/<data>/checkpoint_best_R.pth hid experiments that were
    # accidentally evaluating the shipped baseline instead of the
    # user-specified calibrated / trained checkpoint. Always exit
    # loudly if the requested file isn't present.
    if args.nasr == 'rl' and not ckpt_path.exists():
        fallback_ckpt = paths.PRETRAINED_DIR / 'rl' / args.data / 'checkpoint_best_R.pth'
        raise FileNotFoundError(
            'Refusing to auto-load shipped NASR RL checkpoint as a fallback.\n'
            f'  requested --ckpt-path : {ckpt_path}  (does not exist)\n'
            f'  shipped fallback      : {fallback_ckpt}  (BLOCKED)\n'
            'Fix the --ckpt-path argument instead of silently using the shipped NASR.'
        )

    if os.path.isfile(args.noise_setting):
        with open(args.noise_setting) as f:
            noise_setting = json.load(f)
    else:
        noise_setting = {"noise_type": "awgn", "snr": -0.5}
    noise_setting = str(noise_setting).replace(' ', '').replace("'", "")

    if args.transform_data:
        test_dataset = SudokuDataset_RL(args.data,'-test',transform=args.transform_data,t_param=args.transform_data_param)
    else:
        test_dataset = SudokuDataset_RL(args.data,'-test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)

    # ---- Model loading ----
    llm_adapter = None
    if args.solver_backbone == 'llm':
        from models.transformer_sudoku import get_llm_model
        from models.llm_solver_adapter import LLMSolverAdapter, DEFAULT_FINETUNED_MODEL_PATH
        model = get_llm_model(
            block_len=args.block_len,
            use_perception_features=args.use_perception_features,
            temp_hidden_dim=args.temp_hidden_dim,
        )
        print(f'[rl_eval_sudoku] LLM mode: loading LLMTransformerSudoku checkpoint: {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.to(args.gpu_id)
        _llm_model_path = args.finetuned_model_path or DEFAULT_FINETUNED_MODEL_PATH
        print(f'[rl_eval_sudoku] LLM adapter: fine-tuned checkpoint: {_llm_model_path}')
        llm_adapter = LLMSolverAdapter(
            model_path=_llm_model_path,
            cache_dir=args.llm_cache_dir,
            generation_batch_size=args.generation_batch_size,
        )
    else:
        model = get_model(block_len=args.block_len)
        if args.nasr == 'rl':
            # Print the exact file we load so downstream aggregation scripts
            # can grep for this line and confirm the intended checkpoint.
            print(f'[rl_eval_sudoku] loading checkpoint: {ckpt_path}')
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))  #to check preformance of the RL
        else:
            assert args.nasr == 'pretrained', f'{args.nasr} not supported, choose between pretrained and rl'
            model.load_pretrained_models(args.data) #to check preformance of the pretrained pipeline without RL
        model.to(args.gpu_id)

    constraint_layer = None
    if args.use_minedit:
        group_order = parse_group_order(args.constraint_group_order)
        args.constraint_group_order = group_order
        constraint_layer = build_edit_layer(args, args.gpu_id)
        print(f'[use-minedit] constraint-layer projection enabled '
              f'(edit_layer={args.edit_layer}, max_iterations={args.minedit_iters}, '
              f'mask convention: 1->KEEP, '
              f'group_order={format_group_order(group_order)})')

    # Main loop
    print("Beginning testing")
    validate(test_loader, model, args, constraint_layer=constraint_layer,
             llm_adapter=llm_adapter)
    if llm_adapter is not None:
        llm_adapter.log_counters(prefix="[rl_eval_sudoku] final LLM adapter stats: ")
 

if __name__ == '__main__':

    main()
