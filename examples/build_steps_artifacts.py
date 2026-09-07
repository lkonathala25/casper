import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tasks" / "warcraft") not in sys.path:
    sys.path.insert(0, str(ROOT / "tasks" / "warcraft"))

from steps_encoding import build_steps_encoding, write_artifacts  # noqa: E402


def main() -> None:
    """Emit the steps-encoding DSL file (plus domain sidecar) for one grid size."""
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=12)
    p.add_argument("--untrimmed", action="store_true")
    p.add_argument(
        "--outdir", type=Path, default=ROOT / "constraints" / "warcraft" / "steps_artifacts"
    )
    args = p.parse_args()

    enc = build_steps_encoding(args.size, trimmed=not args.untrimmed)
    tag = "untrimmed" if args.untrimmed else "trimmed"
    dsl, sidecar = write_artifacts(
        enc, args.outdir / f"steps_{args.size}_{tag}.txt"
    )
    print(
        f"g={enc.g} L={enc.horizon} variant={tag} "
        f"clauses={len(enc.clauses)} "
        f"domain_values={sum(len(d) for d in enc.domains)}"
    )
    print(f"  dsl     {dsl}  ({dsl.stat().st_size / 1e6:.2f} MB)")
    print(f"  domains {sidecar}")


if __name__ == "__main__":
    main()
