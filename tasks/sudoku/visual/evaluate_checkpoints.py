"""Systematically evaluate all seed checkpoints in a folder via rl_eval_sudoku.py.

Checkpoint layout
-----------------
Two layouts are supported via --seed-subdir-pattern:

  Flat (default, --seed-subdir-pattern ""):
    <folder>/checkpoint_best_R_<seed>.pth

  Per-seed subdirectory (e.g. --seed-subdir-pattern "seed{seed}_v3"):
    <folder>/seed0_v3/checkpoint_best_R_0.pth
    <folder>/seed1_v3/checkpoint_best_R_1.pth
    ...

Model families
--------------
  --minedit  :  minedit baseline — pass --use-minedit to rl_eval_sudoku.py;
                each seed's sidecar .json must contain a best_bias field.
  --llm      :  LLM backbone (--solver-backbone llm).  Does NOT imply --minedit;
                set --minedit explicitly for the minedit-LLM family.
                Omit --minedit for CTS-LLM and ANESI-LLM.

Bias selection
--------------
  minedit mode  : bias read from per-seed sidecar JSON (best_bias field).
  non-minedit   : every seed is evaluated once per bias in --bias-grid and the
                  run with the highest NASR-correct-boards count is kept.

LLM caching
-----------
First eval run  : the LLM test-board cache does not yet exist; live LLM
                  generation populates it.  After all seeds complete a sentinel
                  file .eval_cache_built is written inside --llm-cache-dir.

Subsequent runs : the sentinel is detected automatically → --skip-cache-miss is
                  forwarded to rl_eval_sudoku.py so boards absent from the cache
                  receive zero logits instead of triggering new generation.

Pass --skip-cache-miss explicitly to force this behaviour on a single run
regardless of the sentinel state.

Outputs (all written inside --folder):
  evaluation_summary.log   human-readable table (Metric / Mean / Std / N)
  evaluation_per_seed.csv  one row per seed, every parsed metric + chosen bias
  evaluation_raw.log       full command lines + stdout/stderr per invocation

Pass --output-tag <tag> to write disambiguated filenames instead (e.g.
evaluation_summary__<tag>.log), so repeated sweeps against the same --folder
(e.g. one run per --constraint-group-order permutation) don't overwrite each
other's results. Default "" preserves the legacy filenames exactly.

--constraint-group-order controls the processing order of the 27 Sudoku
groups inside SudokuConstraintLayer; it is forwarded verbatim to
rl_eval_sudoku.py and recorded in the CSV/raw log/summary. "default" (the
default) keeps the historical behaviour unchanged.

The script never trusts rl_eval_sudoku.py's "fallback checkpoint" path:
each required file is verified beforehand and missing files raise.
"""
import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
from pathlib import Path

import paths
from group_order_utils import parse_group_order, format_group_order


SRC_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = 'rl_eval_sudoku.py'
_EVAL_CACHE_SENTINEL = '.eval_cache_built'


# (key, label, regex, converter). Patterns match the relevant line in
# rl_eval_sudoku.py stdout. Percentages are captured without the % sign.
PATTERNS = [
    ('casper_certified', 'CASPER certified boards',
     r'CASPER certified:\s*(\d+)/\d+', int),
    ('avg_cell_reward', 'avg_cell_reward',
     r'avg_cell_reward\s+([\d.eE+\-]+)', float),
    ('alldiff_sat', 'AllDifferent sat rate',
     r'AllDifferent sat rate:\s*([\d.eE+\-]+)', float),
    ('blank_rate', 'Mask blank rate',
     r'Mask blank rate:\s*([\d.eE+\-]+)', float),
    ('keep_rate_wrong_proj', 'Mask keep-rate on wrong projected cells',
     r'Mask keep-rate on wrong projected cells:\s*([\d.eE+\-]+)', float),
    ('oracle_wrong_projected', 'Oracle wrong-projected board_acc',
     r'Oracle blanking \(wrong-projected\) board_acc:\s*([\d.eE+\-]+)', float),
    ('oracle_all_non_givens', 'Oracle all-non-givens board_acc',
     r'Oracle blanking \(all-non-givens\) board_acc:\s*([\d.eE+\-]+)', float),
    ('llm_raw_board_acc_pct', 'LLM raw board acc (%)',
     r'correct solution boards from LLM \(raw\):[^(]*\(([\d.]+)%\)', float),
    ('neuro_solver_board_acc_pct', 'Neuro-Solver board acc (%)',
     r'correct solution boards from Neuro-Solver:[^(]*\(([\d.]+)%\)', float),
    ('nasr_board_acc_pct', 'NASR board acc (%)',
     r'correct solution boards from NASR:[^(]*\(([\d.]+)%\)', float),
    ('wrong_corrected_by_nasr_pct', 'Wrong boards corrected by NASR (%)',
     r'corrected with NASR:\s*\d+/\d+\s*--\s*\(([\d.]+)%\)', float),
    ('correct_corrupted_by_nasr_pct', 'Correct boards corrupted by NASR (%)',
     r'corrupted with NASR:\s*\d+/\d+\s*--\s*\(([\d.]+)%\)', float),
    ('avg_cell_acc_pct', 'Avg cell accuracy (%)',
     r'Num correct cells \(avg\):\s*([\d.]+)%', float),
    ('perception_corr_solver_pct', 'Perception errors corrected by SolverNN (%)',
     r'perception corrected by the SolverNN[^:]*:\s*([\d.]+)%', float),
    ('perception_corr_mask_pct', 'Perception errors corrected by Mask-Predictor (%)',
     r'perception corrected by the Mask-Predictor[^:]*:\s*([\d.]+)%', float),
    ('solver_input_corr_pct', 'SolverNN input errors corrected (%)',
     r'solverNN in the input corrected[^:]*:\s*([\d.]+)%', float),
    ('solver_solutions_corr_pct', 'SolverNN solution errors corrected (%)',
     r'solverNN in the solutions corrected[^:]*:\s*([\d.]+)%', float),
]

NASR_COUNT_RE = re.compile(
    r'correct solution boards from NASR:\s*(\d+)/(\d+)',
)


def extract_metrics(stdout):
    out = {}
    for key, _label, pat, conv in PATTERNS:
        m = re.search(pat, stdout, flags=re.MULTILINE)
        if m is None:
            out[key] = float('nan')
            continue
        try:
            out[key] = conv(m.group(1))
        except Exception:
            out[key] = float('nan')
    return out


def extract_nasr_count(stdout):
    m = NASR_COUNT_RE.search(stdout)
    if m is None:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _pipe_reader(pipe, lines: list):
    """Read all lines from pipe, appending to lines. Used in a thread."""
    for line in pipe:
        lines.append(line)


def run_eval(ckpt_path, bias, use_minedit, python_bin,
             llm_mode=False, llm_cache_dir=None, prompt_template=None,
             llm_batch_size=8, finetuned_model_path=None,
             generation_batch_size=4, skip_cache_miss=False, cts_masking=False,
             data='satnet', constraint_group_order='default',
             edit_layer_args=(), verbose=True):
    # -u: force unbuffered stdout/stderr in the subprocess so lines arrive
    # immediately rather than in large chunks when the pipe buffer fills.
    cmd = [
        python_bin, '-u', EVAL_SCRIPT,
        '--ckpt-path', str(ckpt_path),
        '--solver', 'prolog',
        '--nasr', 'rl',
        '--data', data,
        '--mask-logit-bias', f'{bias:.6f}',
        '--mask-verbose',
        '--constraint-group-order', str(constraint_group_order),
    ]
    if use_minedit:
        cmd.append('--use-minedit')
        cmd.extend(edit_layer_args)
    if cts_masking:
        cmd.append('--cts-masking')
    if llm_mode:
        cmd.extend(['--solver-backbone', 'llm'])
        if llm_cache_dir:
            cmd.extend(['--llm-cache-dir', str(llm_cache_dir)])
        if finetuned_model_path:
            cmd.extend(['--finetuned-model-path', str(finetuned_model_path)])
        cmd.extend(['--generation-batch-size', str(generation_batch_size)])
        if skip_cache_miss:
            cmd.append('--skip-cache-miss')
        # prompt_template and llm_batch_size are kept for any callers that
        # still pass them, but are deprecated and not forwarded.

    proc = subprocess.Popen(
        cmd, cwd=str(SRC_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    out_lines: list[str] = []
    err_lines: list[str] = []

    if verbose:
        # Stream both pipes live to terminal while also capturing in memory.
        def _live_reader(pipe, lines, stream):
            for line in pipe:
                lines.append(line)
                stream.write(line)
                stream.flush()

        t_out = threading.Thread(target=_live_reader, args=(proc.stdout, out_lines, sys.stdout))
        t_err = threading.Thread(target=_live_reader, args=(proc.stderr, err_lines, sys.stderr))
    else:
        t_out = threading.Thread(target=_pipe_reader, args=(proc.stdout, out_lines))
        t_err = threading.Thread(target=_pipe_reader, args=(proc.stderr, err_lines))

    t_out.start()
    t_err.start()
    t_out.join()
    t_err.join()
    proc.wait()

    return cmd, ''.join(out_lines), ''.join(err_lines), proc.returncode


def nan_safe_stats(values):
    clean = [
        v for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    n = len(clean)
    if n == 0:
        return float('nan'), float('nan'), 0
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if n >= 2 else 0.0
    return mean, std, n


def resolve_bias_from_sidecar(ckpt_path):
    """Read best_bias from <ckpt>.json (strip .pth, append .json)."""
    meta_path = ckpt_path.with_suffix('.json')
    if not meta_path.is_file():
        raise FileNotFoundError(
            f'Missing sidecar JSON for minedit bias: {meta_path}'
        )
    with open(meta_path) as f:
        meta = json.load(f)
    if 'best_bias' not in meta:
        raise ValueError(
            f"'best_bias' missing from sidecar {meta_path}"
        )
    return float(meta['best_bias'])


def evaluate_one(ckpt, fixed_bias, use_minedit, bias_grid, python_bin,
                 llm_mode=False, llm_cache_dir=None, prompt_template=None,
                 llm_batch_size=8, finetuned_model_path=None,
                 generation_batch_size=4, skip_cache_miss=False,
                 cts_masking=False, data='satnet',
                 constraint_group_order='default', edit_layer_args=()):
    """Evaluate one checkpoint. If fixed_bias is not None, use exactly that bias;
    otherwise sweep bias_grid and keep the run with the highest NASR-correct count.
    Returns (selected_bias, metrics, (num, den), invocations)."""
    invocations = []
    if fixed_bias is not None:
        cmd, stdout, stderr, rc = run_eval(
            ckpt, fixed_bias, use_minedit, python_bin,
            llm_mode=llm_mode, llm_cache_dir=llm_cache_dir,
            finetuned_model_path=finetuned_model_path,
            generation_batch_size=generation_batch_size,
            skip_cache_miss=skip_cache_miss,
            cts_masking=cts_masking,
            data=data,
            constraint_group_order=constraint_group_order,
            edit_layer_args=edit_layer_args,
        )
        invocations.append((fixed_bias, cmd, stdout, stderr, rc))
        if rc != 0:
            raise RuntimeError(
                f'rl_eval_sudoku.py failed (rc={rc}).\nstderr:\n{stderr}'
            )
        metrics = extract_metrics(stdout)
        num, den = extract_nasr_count(stdout)
        return fixed_bias, metrics, (num, den), invocations

    best = None
    for bias in bias_grid:
        cmd, stdout, stderr, rc = run_eval(
            ckpt, bias, use_minedit, python_bin,
            llm_mode=llm_mode, llm_cache_dir=llm_cache_dir,
            finetuned_model_path=finetuned_model_path,
            generation_batch_size=generation_batch_size,
            skip_cache_miss=skip_cache_miss,
            cts_masking=cts_masking,
            data=data,
            constraint_group_order=constraint_group_order,
            edit_layer_args=edit_layer_args,
        )
        invocations.append((bias, cmd, stdout, stderr, rc))
        if rc != 0:
            raise RuntimeError(
                f'rl_eval_sudoku.py failed for bias={bias} '
                f'(rc={rc}).\nstderr:\n{stderr}'
            )
        num, den = extract_nasr_count(stdout)
        if num is None:
            raise RuntimeError(
                f'Could not parse NASR count for bias={bias}.'
            )
        if best is None or num > best['num']:
            best = dict(
                bias=bias, num=num, den=den,
                metrics=extract_metrics(stdout),
            )
    return best['bias'], best['metrics'], (best['num'], best['den']), invocations


def _llm_eval_cache_populated(llm_cache_dir):
    """True when the sentinel file is present (first-run cache build completed)."""
    if not llm_cache_dir:
        return False
    return (Path(llm_cache_dir) / _EVAL_CACHE_SENTINEL).is_file()


def _mark_llm_eval_cache_built(llm_cache_dir):
    """Create sentinel after a successful eval run that populated the cache."""
    if not llm_cache_dir:
        return
    (Path(llm_cache_dir) / _EVAL_CACHE_SENTINEL).touch()


def _ckpt_path(folder, seed, seed_subdir_pattern):
    """Resolve checkpoint path for a given seed and optional subdir pattern."""
    if seed_subdir_pattern:
        subdir = seed_subdir_pattern.format(seed=seed)
        return folder / subdir / f'checkpoint_best_R_{seed}.pth'
    return folder / f'checkpoint_best_R_{seed}.pth'


def preflight(folder, seeds, minedit_mode, seed_subdir_pattern=''):
    """Verify every required checkpoint (and sidecar in minedit mode) exists.

    Never rely on rl_eval_sudoku.py's fallback checkpoint path — we want
    this script to fail fast and loudly on missing files."""
    ckpts = {}
    biases = {}
    for seed in seeds:
        ckpt = _ckpt_path(folder, seed, seed_subdir_pattern)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f'Missing required checkpoint: {ckpt}'
            )
        ckpts[seed] = ckpt
        if minedit_mode:
            meta_path = ckpt.with_suffix('.json')
            if not meta_path.is_file():
                raise FileNotFoundError(
                    f'Missing sidecar JSON (minedit mode requires best_bias): '
                    f'{meta_path}'
                )
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception as exc:
                raise RuntimeError(
                    f'Failed to load {meta_path}: {exc}'
                )
            if 'best_bias' not in meta:
                raise ValueError(
                    f"'best_bias' missing from sidecar {meta_path}"
                )
            biases[seed] = float(meta['best_bias'])
    return ckpts, biases


def _banner(msg: str):
    """Print a visible separator before each subprocess run."""
    line = '=' * 72
    print(f'\n{line}\n[eval] {msg}\n{line}', flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate all seed checkpoints in a folder.',
    )
    parser.add_argument('--folder', required=True, type=str,
                        help='Checkpoint root folder (absolute or relative).')
    parser.add_argument('--seeds', default='0,1,2,3,4', type=str,
                        help='Comma-separated seed indices to evaluate (default: 0,1,2,3,4).')
    parser.add_argument('--seed-subdir-pattern', default='', type=str,
                        help='Per-seed subdirectory pattern relative to --folder. '
                             'Use {seed} as placeholder, e.g. "seed{seed}_v3". '
                             'Default "" means checkpoints live directly in --folder.')
    parser.add_argument('--minedit', action='store_true',
                        help='Evaluate as use-minedit pipeline (reads per-seed best_bias '
                             'from sidecar JSON and passes --use-minedit to rl_eval_sudoku.py).')
    parser.add_argument('--bias-grid', default='-1.0,0.0,1.0', type=str,
                        help='Comma-separated biases for non-minedit mode.')
    parser.add_argument('--python-bin', default=sys.executable, type=str,
                        help='Python executable used to launch rl_eval_sudoku.py.')
    parser.add_argument('--data', default='satnet', type=str,
                        help='Dataset name forwarded to rl_eval_sudoku.py (default: satnet).')
    parser.add_argument('--constraint-group-order', default='default', type=str,
                        help='Processing order for the 27 Sudoku groups, forwarded to '
                             'rl_eval_sudoku.py as --constraint-group-order. "default" '
                             '(default) keeps the canonical 0..26 order. Otherwise a '
                             'comma-separated permutation of 0..26. Validated eagerly '
                             'here before any subprocess is launched.')
    # Edit layer (forwarded to rl_eval_sudoku.py with --use-minedit). Kept
    # torch-free here: values are passed through, rl_eval validates them.
    parser.add_argument('--edit-layer', default='casper', choices=['casper', 'minedit'],
                        help='Edit layer forwarded to rl_eval_sudoku.py (minedit mode only).')
    parser.add_argument('--constraints-file', default=None, type=str,
                        help='Forwarded as --constraints-file (casper). Default: '
                             'rl_eval_sudoku.py default.')
    parser.add_argument('--constraints-format', default='auto',
                        choices=['auto', 'compiled-txt', 'json', 'dsl'])
    parser.add_argument('--casper-cell-order', default='confidence',
                        choices=['confidence', 'canonical'])
    parser.add_argument('--pairwise', action='store_true',
                        help='Forwarded as --pairwise (edit-layer minedit only).')
    parser.add_argument('--output-tag', default='', type=str,
                        help='Suffix distinguishing this run\'s output files so repeated '
                             'sweeps (e.g. one per group ordering) do not overwrite each '
                             'other. Default "" preserves the legacy filenames exactly '
                             '(evaluation_summary.log, evaluation_per_seed.csv, '
                             'evaluation_raw.log). A non-empty tag, e.g. "order_00", '
                             'writes evaluation_summary__order_00.log etc.')
    # LLM pipeline options (opt-in; does NOT imply --minedit)
    parser.add_argument('--llm', action='store_true',
                        help='Evaluate as LLM backbone pipeline '
                             '(passes --solver-backbone llm to rl_eval_sudoku.py). '
                             'Does NOT imply --minedit: pass --minedit explicitly for the '
                             'minedit-LLM family; omit it for CTS-LLM and ANESI-LLM.')
    parser.add_argument('--llm-cache-dir', default=None, type=str,
                        help='Directory for LLM output cache (passed through). '
                             'Default with --llm: paths.LLM_CACHE_DIR '
                             '(env SUDOKU_LLM_CACHE_DIR).')
    parser.add_argument('--finetuned-model-path', default=None, type=str,
                        help='Path to fine-tuned LoRA checkpoint for the LLM adapter '
                             '(passed through to rl_eval_sudoku.py). Defaults to the '
                             'SATNet SFT checkpoint.')
    parser.add_argument('--generation-batch-size', default=4, type=int,
                        help='Boards per model.generate() call in the LLM adapter '
                             '(passed through to rl_eval_sudoku.py).')
    parser.add_argument('--skip-cache-miss', action='store_true',
                        help='LLM mode: forward --skip-cache-miss to rl_eval_sudoku.py '
                             '(use solve_batch_cached_only; skip boards absent from cache). '
                             'Auto-enabled when the eval-cache sentinel is present inside '
                             '--llm-cache-dir. Pass explicitly to force on a single run.')
    parser.add_argument('--cts-masking', action='store_true',
                        help='Forward --cts-masking to rl_eval_sudoku.py. Use this '
                            'for CTS and CTS-LLM checkpoints, whose training-time '
                            'masking uses argmax([1-mask, P*mask]) without MinEdit.')
    parser.add_argument('--prompt-template', default=None, type=str,
                        help='[DEPRECATED — ignored by the migrated adapter] '
                             'Kept for backward compatibility only.')
    parser.add_argument('--llm-batch-size', default=8, type=int,
                        help='[DEPRECATED — use --generation-batch-size instead] '
                             'Kept for backward compatibility only.')
    # Sanity-check anchor: evaluate a specific reference checkpoint first,
    # compare against a known NASR board_acc, fail loudly on mismatch. This
    # detects (a) fallback-checkpoint regressions, (b) broken env, (c) stale
    # checkpoints, BEFORE burning time on the full seed sweep. Opt-in
    # via --sanity-check; the other --sanity-* args only take effect when
    # --sanity-check is passed.
    parser.add_argument('--sanity-check', action='store_true',
                        help='Run the sanity-check evaluation before the seed sweep.')
    parser.add_argument('--sanity-ckpt', type=str, default=None,
                        help='Sanity-check checkpoint path. Default: '
                             '<folder>/checkpoint_best_R_1234.pth.')
    parser.add_argument('--sanity-expected-board-acc', type=float, default=85.60,
                        help='Expected NASR board acc (percent) for the sanity ckpt.')
    parser.add_argument('--sanity-tolerance-pct', type=float, default=0.2,
                        help='Allowed absolute difference in NASR board acc '
                             '(percentage points). Default 0.2 (~ 1 board at 500).')
    args = parser.parse_args()
    if args.llm and args.llm_cache_dir is None:
        args.llm_cache_dir = str(paths.LLM_CACHE_DIR)

    # Validate the group ordering eagerly, before touching the filesystem or
    # launching any subprocess, so a malformed --constraint-group-order fails
    # loudly and immediately rather than after burning time on preflight/eval.
    group_order = parse_group_order(args.constraint_group_order)
    print(f'[eval] constraint-group-order = {format_group_order(group_order)}',
          flush=True)

    if args.pairwise and args.edit_layer != 'minedit':
        raise SystemExit('--pairwise only applies to --edit-layer minedit')
    edit_layer_args = ['--edit-layer', args.edit_layer]
    if args.edit_layer == 'casper':
        if args.constraints_file:
            edit_layer_args += ['--constraints-file', args.constraints_file]
        edit_layer_args += ['--constraints-format', args.constraints_format,
                            '--casper-cell-order', args.casper_cell_order]
    elif args.pairwise:
        edit_layer_args.append('--pairwise')
    edit_layer_desc = ' '.join(edit_layer_args)
    print(f'[eval] edit layer: {edit_layer_desc}', flush=True)

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f'Folder not found: {folder}')

    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    if not seeds:
        raise SystemExit('--seeds must be a non-empty comma-separated list.')

    ckpts, minedit_biases = preflight(
        folder, seeds, args.minedit, args.seed_subdir_pattern,
    )
    if args.minedit:
        # Sidecars written before the edit-layer flag have no 'edit_layer'
        # key; those checkpoints were trained with the legacy minedit layer.
        for seed, ckpt in ckpts.items():
            with open(ckpt.with_suffix('.json')) as f:
                trained_with = json.load(f).get('edit_layer', 'minedit')
            if trained_with != args.edit_layer:
                print(f'[eval] WARNING: seed {seed} was trained with '
                      f'edit_layer={trained_with} but is evaluated with '
                      f'--edit-layer {args.edit_layer}', flush=True)

    bias_grid = None
    if not args.minedit:
        bias_grid = [
            float(x) for x in args.bias_grid.split(',') if x.strip() != ''
        ]
        if not bias_grid:
            raise SystemExit('--bias-grid must be non-empty in non-minedit mode.')

    # Auto-detect LLM cache state: if the sentinel is present, the cache was
    # populated by a previous run → use cache-only to skip regenerating boards
    # that failed to parse (they won't be in the cache).
    skip_cache_miss = args.skip_cache_miss
    if args.llm and args.llm_cache_dir and not skip_cache_miss:
        if _llm_eval_cache_populated(args.llm_cache_dir):
            skip_cache_miss = True
            print(
                f'[llm-cache] Sentinel found in {args.llm_cache_dir}: '
                f'using cache-only evaluation (--skip-cache-miss).',
                flush=True,
            )

    # Empty --output-tag preserves the legacy fixed filenames exactly; a
    # non-empty tag (e.g. "order_00") disambiguates multi-order sweeps so
    # repeated invocations against the same --folder don't overwrite results.
    tag_suffix = f'__{args.output_tag}' if args.output_tag else ''
    raw_log_path = folder / f'evaluation_raw{tag_suffix}.log'
    csv_path = folder / f'evaluation_per_seed{tag_suffix}.csv'
    summary_path = folder / f'evaluation_summary{tag_suffix}.log'

    per_seed_rows = []
    raw_chunks = []

    # ---- Sanity check (opt-in via --sanity-check) --------------------
    if args.sanity_check:
        sanity_ckpt = (
            Path(args.sanity_ckpt).expanduser().resolve()
            if args.sanity_ckpt
            else folder / 'checkpoint_best_R_1234.pth'
        )
        if not sanity_ckpt.is_file():
            raise FileNotFoundError(
                f'Sanity checkpoint missing: {sanity_ckpt}. '
                f'Pass --sanity-ckpt <path> to point elsewhere, or drop '
                f'--sanity-check to skip the sanity step entirely.'
            )
        fixed_bias = (
            resolve_bias_from_sidecar(sanity_ckpt) if args.minedit else None
        )
        bias_str = f'{fixed_bias:+.3f}' if fixed_bias is not None else 'SWEEP'
        print(
            f'[sanity] ckpt={sanity_ckpt} bias={bias_str} '
            f'(expected board_acc={args.sanity_expected_board_acc:.2f}% '
            f'+/- {args.sanity_tolerance_pct:.2f} pp)',
            flush=True,
        )

        selected_bias, metrics, (num, den), invocations = evaluate_one(
            sanity_ckpt, fixed_bias, args.minedit, bias_grid,
            args.python_bin,
            llm_mode=args.llm, llm_cache_dir=args.llm_cache_dir,
            finetuned_model_path=args.finetuned_model_path,
            generation_batch_size=args.generation_batch_size,
            skip_cache_miss=skip_cache_miss,
            cts_masking=args.cts_masking,
            data=args.data,
            constraint_group_order=args.constraint_group_order,
            edit_layer_args=edit_layer_args,
        )
        for bias, cmd, stdout, stderr, rc in invocations:
            raw_chunks.append(
                f'===== SANITY | ckpt={sanity_ckpt} | bias {bias:+.3f} | '
                f'order={format_group_order(group_order)} | tag={args.output_tag or "-"} | '
                f'rc={rc} =====\n'
                f'CMD: {" ".join(cmd)}\n'
                f'-- stdout --\n{stdout}\n'
                f'-- stderr --\n{stderr}\n'
            )

        board_acc = metrics.get('nasr_board_acc_pct', float('nan'))
        delta = abs(board_acc - args.sanity_expected_board_acc)
        ok = (not math.isnan(board_acc)) and (delta <= args.sanity_tolerance_pct)
        verdict = 'OK' if ok else 'FAIL'
        print(
            f'[sanity]  -> {verdict} bias={selected_bias:+.3f} '
            f'nasr={num}/{den} board_acc={board_acc:.2f}% '
            f'(expected {args.sanity_expected_board_acc:.2f}%, '
            f'delta={delta:.2f} pp, tol={args.sanity_tolerance_pct:.2f} pp)',
            flush=True,
        )
        if not ok:
            raise RuntimeError(
                f'Sanity check FAILED for {sanity_ckpt}:\n'
                f'  measured NASR board_acc = {board_acc:.2f}%\n'
                f'  expected                = {args.sanity_expected_board_acc:.2f}%\n'
                f'  delta                   = {delta:.2f} pp '
                f'(tolerance {args.sanity_tolerance_pct:.2f} pp)\n'
                f'This usually means the pipeline is loading the wrong '
                f'checkpoint, or the environment drifted. Inspect '
                f'evaluation_raw.log for the loaded checkpoint banner.'
            )
    # ------------------------------------------------------------------

    n_seeds = len(seeds)
    for seed_idx, seed in enumerate(seeds):
        ckpt = ckpts[seed]
        if args.minedit:
            bias = minedit_biases[seed]
            _banner(f'seed {seed}/{n_seeds-1} | bias {bias:+.3f} | ckpt {ckpt.name} | '
                    f'order={format_group_order(group_order)} | tag={args.output_tag or "-"}')
            cmd, stdout, stderr, rc = run_eval(
                ckpt, bias, use_minedit=True, python_bin=args.python_bin,
                llm_mode=args.llm, llm_cache_dir=args.llm_cache_dir,
                finetuned_model_path=args.finetuned_model_path,
                generation_batch_size=args.generation_batch_size,
                skip_cache_miss=skip_cache_miss,
                cts_masking=args.cts_masking,
                data=args.data,
                constraint_group_order=args.constraint_group_order,
                edit_layer_args=edit_layer_args,
            )
            raw_chunks.append(
                f'===== seed {seed} | bias {bias:+.3f} | '
                f'order={format_group_order(group_order)} | tag={args.output_tag or "-"} | '
                f'rc={rc} =====\n'
                f'CMD: {" ".join(cmd)}\n'
                f'-- stdout --\n{stdout}\n'
                f'-- stderr --\n{stderr}\n'
            )
            if rc != 0:
                raise RuntimeError(
                    f'rl_eval_sudoku.py failed for seed={seed} (rc={rc}).\n'
                    f'stderr:\n{stderr}'
                )
            metrics = extract_metrics(stdout)
            num, den = extract_nasr_count(stdout)
            selected_bias = bias
            print(
                f'\n[RESULT] seed={seed} nasr={num}/{den} '
                f'board_acc={metrics.get("nasr_board_acc_pct", float("nan")):.2f}%'
                f'  ({seed_idx+1}/{n_seeds} seeds done)\n',
                flush=True,
            )
        else:
            best = None
            for bias_idx, bias in enumerate(bias_grid):
                _banner(
                    f'seed {seed}/{n_seeds-1} | bias {bias:+.3f} '
                    f'({bias_idx+1}/{len(bias_grid)}) | ckpt {ckpt.name} | '
                    f'order={format_group_order(group_order)} | tag={args.output_tag or "-"}'
                )
                cmd, stdout, stderr, rc = run_eval(
                    ckpt, bias, use_minedit=False, python_bin=args.python_bin,
                    llm_mode=args.llm, llm_cache_dir=args.llm_cache_dir,
                    finetuned_model_path=args.finetuned_model_path,
                    generation_batch_size=args.generation_batch_size,
                    skip_cache_miss=skip_cache_miss,
                    cts_masking=args.cts_masking,
                    data=args.data,
                    constraint_group_order=args.constraint_group_order,
                    edit_layer_args=edit_layer_args,
                )
                raw_chunks.append(
                    f'===== seed {seed} | bias {bias:+.3f} | '
                    f'order={format_group_order(group_order)} | tag={args.output_tag or "-"} | '
                    f'rc={rc} =====\n'
                    f'CMD: {" ".join(cmd)}\n'
                    f'-- stdout --\n{stdout}\n'
                    f'-- stderr --\n{stderr}\n'
                )
                if rc != 0:
                    raise RuntimeError(
                        f'rl_eval_sudoku.py failed for seed={seed}, '
                        f'bias={bias} (rc={rc}).\nstderr:\n{stderr}'
                    )
                num, den = extract_nasr_count(stdout)
                if num is None:
                    raise RuntimeError(
                        f'Could not parse NASR count for seed={seed}, '
                        f'bias={bias}.'
                    )
                print(
                    f'\n[bias result] seed={seed} bias={bias:+.3f} nasr={num}/{den}\n',
                    flush=True,
                )
                if best is None or num > best['num']:
                    best = dict(
                        bias=bias, num=num, den=den,
                        metrics=extract_metrics(stdout),
                    )
            metrics = best['metrics']
            selected_bias = best['bias']
            print(
                f'\n[RESULT] seed={seed} best_bias={selected_bias:+.3f} '
                f'nasr={best["num"]}/{best["den"]} '
                f'board_acc={metrics.get("nasr_board_acc_pct", float("nan")):.2f}%'
                f'  ({seed_idx+1}/{n_seeds} seeds done)\n',
                flush=True,
            )

        row = {
            'seed': seed, 'selected_bias': selected_bias,
            'output_tag': args.output_tag,
            'constraint_group_order': format_group_order(group_order),
            'edit_layer': edit_layer_desc,
            **metrics,
        }
        per_seed_rows.append(row)

    # After a successful first-run eval with the LLM, mark the cache as built
    # so subsequent runs auto-enable --skip-cache-miss.
    if args.llm and args.llm_cache_dir and not args.skip_cache_miss and not skip_cache_miss:
        _mark_llm_eval_cache_built(args.llm_cache_dir)
        print(
            f'[llm-cache] Cache sentinel written to {args.llm_cache_dir}. '
            f'Subsequent runs will use cache-only evaluation.',
            flush=True,
        )

    # Per-seed CSV
    metric_keys = [k for k, _, _, _ in PATTERNS]
    fieldnames = ['seed', 'selected_bias', 'output_tag', 'constraint_group_order',
                  'edit_layer'] + metric_keys
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_seed_rows:
            writer.writerow(row)

    # Raw log
    with open(raw_log_path, 'w') as f:
        f.write('\n'.join(raw_chunks))

    # Summary table
    lines = []
    lines.append(f'Checkpoint folder: {folder}')
    mode_str = 'minedit' if args.minedit else 'non-minedit'
    if args.llm:
        mode_str = f'llm+{mode_str}'
    lines.append(f'Mode: {mode_str}')
    if not args.minedit:
        lines.append(f'Bias grid: {args.bias_grid}')
    lines.append(f'Seeds: {seeds}')
    lines.append(f'Output tag: {args.output_tag or "(none)"}')
    lines.append(f'Constraint group order: {format_group_order(group_order)}')
    lines.append(f'Edit layer: {edit_layer_desc}')
    lines.append('')
    lines.append('| Metric | Mean | Std | N |')
    lines.append('|---|---|---|---|')

    # Selected bias stats (per-seed)
    bmean, bstd, bn = nan_safe_stats([r['selected_bias'] for r in per_seed_rows])
    lines.append(f'| selected_bias | {bmean:.4f} | {bstd:.4f} | {bn} |')

    for key, label, _pat, _conv in PATTERNS:
        vals = [r.get(key) for r in per_seed_rows]
        mean, std, n = nan_safe_stats(vals)
        if math.isnan(mean):
            lines.append(f'| {label} | NaN | NaN | 0 |')
        else:
            lines.append(f'| {label} | {mean:.4f} | {std:.4f} | {n} |')

    summary = '\n'.join(lines) + '\n'
    with open(summary_path, 'w') as f:
        f.write(summary)

    print()
    print(summary)
    print(f'Wrote:\n  {summary_path}\n  {csv_path}\n  {raw_log_path}')


if __name__ == '__main__':
    main()
