from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "casper"))

from casper_layer import CasperLayer
from constraints import Constraints


def rss_kb() -> int:
    """Peak resident set size in kilobytes (Linux: ru_maxrss is in kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def fmt(kb: int) -> str:
    """Format kB as a human-readable MiB string."""
    return f"{kb / 1024:.1f} MiB ({kb} kB)"


def main() -> None:
    compiled_path = ROOT / "examples" / "sudoku_4x4_compiled.txt"

    rss0 = rss_kb()
    print(f"[t=0]   after Python startup + imports : {fmt(rss0)}")

    t0 = time.time()
    constraints = Constraints.from_compiled_dsl(compiled_path)
    t_load = time.time() - t0
    rss1 = rss_kb()
    print(
        f"[load]  after Constraints.from_compiled_dsl ({t_load:.2f}s) : "
        f"{fmt(rss1)}  (+{fmt(rss1 - rss0)})"
    )

    layer = CasperLayer(constraints)
    rss2 = rss_kb()
    print(f"[layer] after CasperLayer construction       : {fmt(rss2)}  (+{fmt(rss2 - rss1)})")

    torch.manual_seed(0)
    B = 8
    logits = torch.randn(B, 16 * 4)
    probs = torch.softmax(logits.view(B, 16, 4), dim=-1).view(B, -1)
    rss3 = rss_kb()
    print(f"[probs] after random batch (B={B}) built      : {fmt(rss3)}  (+{fmt(rss3 - rss2)})")

    t0 = time.time()
    adjusted, info = layer(probs)
    t_fwd = time.time() - t0
    rss4 = rss_kb()
    print(
        f"[fwd]   after forward pass ({t_fwd*1000:.1f} ms, "
        f"{t_fwd/B*1000:.1f} ms/sample) : {fmt(rss4)}  (+{fmt(rss4 - rss3)})"
    )

    for i in range(5):
        adjusted, info = layer(probs)
    rss5 = rss_kb()
    print(f"[fwd*5] after 5 more forward passes           : {fmt(rss5)}  (+{fmt(rss5 - rss4)})")

    print(f"\nPeak RSS over the whole run                   : {fmt(rss5)}")
    print(f"Memory attributable to the compiled artifact  : {fmt(rss1 - rss0)}")
    print(f"Memory growth per inference batch (B={B})      : {fmt(max(rss4 - rss3, rss5 - rss4))}")


if __name__ == "__main__":
    main()
