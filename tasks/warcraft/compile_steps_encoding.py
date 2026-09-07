import argparse
import io
import json
import re
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("casper", "tasks/warcraft"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from resolution import _is_tautology, compile_constraints, to_json  # noqa: E402
from signed_clauses import parse_constraints  # noqa: E402

_STEP_RE = re.compile(
    r"^step=(\d+) var=(\S+) pi_in=(\d+) .*?n_resolvents_kept=(\d+)"
)


class _Tee(io.TextIOBase):
    """Forward compile chatter to a log file and to the real stdout."""

    def __init__(self, log_handle):
        self.log = log_handle
        self.real = sys.__stdout__

    def write(self, s):
        self.log.write(s)
        self.real.write(s)
        return len(s)

    def flush(self):
        self.log.flush()
        self.real.flush()


def main() -> None:
    """Compile one steps-encoding DSL file and report the tex table's columns."""
    p = argparse.ArgumentParser()
    p.add_argument("dsl", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--cap", type=int, default=None)
    p.add_argument("--no-verbose-pass", action="store_true",
                   help="Skip the second, instrumented compile. It exists only "
                        "to count resolvents per step and costs a full extra "
                        "compile plus a log proportional to the clause count, "
                        "which is prohibitive at 30x30.")
    args = p.parse_args()

    out = args.out or args.dsl.with_name(args.dsl.stem + "_compiled.json")
    log = args.log or ROOT / "logs" / (args.dsl.stem + "_compile.log")
    log.parent.mkdir(parents=True, exist_ok=True)

    text = args.dsl.read_text()
    t_parse = time.time()
    cs = parse_constraints(text)
    parse_elapsed = time.time() - t_parse

    kept_in = [
        c for c in cs.clauses if c.literals and not _is_tautology(c, cs.var_domains)
    ]
    print(
        f"dsl={args.dsl.name} input_size_mb={args.dsl.stat().st_size / 1e6:.2f} "
        f"vars={len(cs.var_names)} clauses_emitted={len(cs.clauses)} "
        f"clauses_in={len(kept_in)} parse_s={parse_elapsed:.2f}",
        flush=True,
    )

    # Timing pass: verbose chatter runs to hundreds of thousands of lines and
    # would dominate the wall clock, so the reported time comes from a quiet
    # run and the resolvent accounting from a separate instrumented one.
    stream = out.with_name(out.stem + ".stream.txt")
    t0 = time.time()
    cc = compile_constraints(cs, verbose=False, cap=args.cap, stream_to=stream)
    elapsed = time.time() - t0
    print(f"quiet compile_s={elapsed:.2f}", flush=True)

    if args.no_verbose_pass:
        # The per-step resolvent tally is only observable from the instrumented
        # run. Report it as unknown rather than inferring it from pile sizes,
        # which subsumption makes non-comparable. The independent evidence that
        # the artifact is correct is StepsWalkDecoder.feasible_matches_artifact.
        verbose_elapsed = None
        steps = []
        resolvents = None
    else:
        with open(log, "w") as fh:
            t_v = time.time()
            with redirect_stdout(_Tee(fh)):
                compile_constraints(parse_constraints(text), verbose=True,
                                    cap=args.cap)
            verbose_elapsed = time.time() - t_v

        steps = [
            (int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)))
            for m in (_STEP_RE.match(line) for line in log.read_text().splitlines())
            if m
        ]
        resolvents = sum(s[3] for s in steps)

    out.write_text(json.dumps(to_json(cc)))
    summary = {
        "dsl": str(args.dsl),
        "input_size_mb": round(args.dsl.stat().st_size / 1e6, 3),
        "vars": len(cs.var_names),
        "clauses_emitted": len(cs.clauses),
        "clauses_in": len(kept_in),
        "resolvents": resolvents,
        "steps_with_resolvents": [s[:2] + (s[3],) for s in steps if s[3]],
        "compile_s": round(elapsed, 2),
        "compile_s_verbose": (round(verbose_elapsed, 2)
                              if verbose_elapsed is not None else None),
        "artifact": str(out),
        "artifact_mb": round(out.stat().st_size / 1e6, 3),
        "artifact_stream_mb": round(stream.stat().st_size / 1e6, 3),
    }
    (log.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
