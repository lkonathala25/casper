import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("casper", "tasks/warcraft"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from compute_cost_ratio import _is_valid_src_sink_path  # noqa: E402

from steps_walk import StepsWalkDecoder, edge_endpoints_array  # noqa: E402


def synthetic_scores(rng, g, horizon, mode):
    """Stand-in tick scores for Phase 3; the real bridge arrives in Phase 4.

    "smooth" draws independent gaussian scores — an unguided walk that wanders
    the full horizon. "guided" tilts them towards the target so the walk
    arrives early, which is the regime where Improvement 2's early cut bites.
    "ties" quantises hard, so feasible candidates collide exactly or within
    eps — the regime where argmax and the min-edit choice could disagree.
    """
    n = g * g
    s = rng.normal(size=(horizon + 1, n)).astype(np.float64)
    if mode == "ties":
        return np.round(s * 2.0) / 2.0
    if mode == "guided":
        rows, cols = np.divmod(np.arange(n), g)
        to_t = (g - 1 - rows) + (g - 1 - cols)
        return s - 1.5 * to_t[None, :]
    return s


def classify_divergence(d, tol=1e-6):
    """True iff a divergence is a real disagreement rather than a tie break.

    A tie break is when the two candidates' probabilities are equal to float32
    resolution, so their min-edit distances differ only by rounding and either
    choice is equally the argmax.
    """
    probs = d["probs"]
    dists = d["distances"]
    cands = d["candidates"]
    i_arg = cands.index(d["argmax"])
    i_min = cands.index(d["minedit"])
    return (
        probs[i_arg] - probs[i_min] > tol and dists[i_arg] - dists[i_min] > tol
    )


def main() -> None:
    """Run the guided walk over synthetic scores and report the Phase 3 gates."""
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=12)
    p.add_argument("--boards", type=int, default=200)
    p.add_argument("--cut-boards", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--tie-break", choices=["no-backtrack", "index"],
                   default="no-backtrack")
    p.add_argument("--artifact", type=Path, default=None)
    p.add_argument(
        "--out", type=Path, default=ROOT / "logs" / "steps_phase3_results.json"
    )
    args = p.parse_args()

    import torch

    dec = StepsWalkDecoder(
        args.size, eps=args.eps, dtype=getattr(torch, args.dtype)
    )
    L, t = dec.enc.horizon, dec.enc.target
    endpoints = edge_endpoints_array(args.size)
    src_flat, sink_flat = 0, (args.size - 1) * args.size + (args.size - 1)

    artifact_check = None
    if args.artifact is not None:
        from resolution import from_json

        artifact_check = dec.feasible_matches_artifact(
            from_json(json.loads(args.artifact.read_text()))
        )
        print(f"feasible set == compiled artifact: {artifact_check}", flush=True)

    rng = np.random.default_rng(args.seed)
    modes = ["smooth", "guided", "ties"]
    per_mode = {}

    for mode in modes:
        dead_ends = 0
        decisions = 0
        step_divergences = 0
        strict_divergences = 0
        board_divergences = 0
        divergence_examples = []
        cut_mismatches = []
        n_erased_boards = 0
        snipped_total = 0
        snipped_boards = []
        valid = 0
        arrivals = []
        ms = []

        for b in range(args.boards):
            scores = synthetic_scores(rng, args.size, L, mode)

            t0 = time.perf_counter()
            seq_a, stats_a = dec.walk(scores, selection="argmax", early_cut=True,
                                     tie_break=args.tie_break)
            ms.append((time.perf_counter() - t0) * 1e3)
            dead_ends += stats_a.dead_ends
            arrivals.append(stats_a.arrival)
            decisions += stats_a.decisions

            # Improvement 1 gate: full per-candidate distances at the real eps.
            seq_m, stats_m = dec.walk(scores, selection="minedit", early_cut=True,
                                      tie_break=args.tie_break)
            if stats_m.divergences:
                step_divergences += len(stats_m.divergences)
                strict = [d for d in stats_m.divergences if classify_divergence(d)]
                strict_divergences += len(strict)
                board_divergences += 1
                pool = strict or stats_m.divergences
                if len(divergence_examples) < 5:
                    divergence_examples.append(
                        {
                            "board": b,
                            "mode": mode,
                            "strict": bool(strict),
                            "first": pool[0],
                        }
                    )

            # Improvement 2 gate: early cut must change no output.
            if b < args.cut_boards:
                seq_full, _ = dec.walk(scores, selection="argmax", early_cut=False,
                                       tie_break=args.tie_break)
                if seq_full != seq_a:
                    cut_mismatches.append(b)

            erased, snipped, edge_vec = dec.decode(seq_a)
            if snipped:
                n_erased_boards += 1
                snipped_total += snipped
                snipped_boards.append(snipped)
            if _is_valid_src_sink_path(
                edge_vec.astype(bool), endpoints, src_flat, sink_flat
            ):
                valid += 1
            else:
                print(f"  INVALID certificate: mode={mode} board={b}", flush=True)

        per_mode[mode] = {
            "boards": args.boards,
            "dead_ends": dead_ends,
            "decision_steps": decisions,
            "argmax_vs_minedit_step_divergences": step_divergences,
            "argmax_vs_minedit_strict_divergences": strict_divergences,
            "argmax_vs_minedit_board_divergences": board_divergences,
            "divergence_rate_boards": board_divergences / args.boards,
            "divergence_rate_steps": (
                step_divergences / decisions if decisions else 0.0
            ),
            "divergence_examples": divergence_examples,
            "early_cut_boards_checked": args.cut_boards,
            "early_cut_mismatches": cut_mismatches,
            "boards_needing_erasure": n_erased_boards,
            "pct_boards_needing_erasure": 100.0 * n_erased_boards / args.boards,
            "mean_cells_snipped_all_boards": snipped_total / args.boards,
            "mean_cells_snipped_when_erased": (
                float(np.mean(snipped_boards)) if snipped_boards else 0.0
            ),
            "valid_certificates": valid,
            "certificate_failures": args.boards - valid,
            "mean_arrival_step": float(np.mean(arrivals)),
            "horizon": L,
            "ms_per_board_argmax": float(np.mean(ms)),
        }
        print(json.dumps({mode: per_mode[mode]}, indent=2), flush=True)

    summary = {
        "grid": args.size,
        "eps": args.eps,
        "dtype": args.dtype,
        "tie_break": args.tie_break,
        "seed": args.seed,
        "target_cell": t,
        "feasible_matches_artifact": artifact_check,
        "modes": per_mode,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
