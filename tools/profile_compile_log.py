#!/usr/bin/env python3
"""Parse a `compile_constraints` verbose log and chart per-variable timing.

The compile already emits one `step=...` summary line per eliminated variable
with `elapsed_s` (total wall time for the step) plus the dominant sub-phase
timers `expand_elapsed_s` (recursive resolvent enumeration) and
`sub_elapsed_s` (forward-subsumption dedup of the new clause set). This reads
those lines, writes a CSV, prints a text table + ASCII bars, and saves a PNG
breaking down where the time goes per variable in elimination order.
"""
import argparse
import csv
import re
from pathlib import Path

# Numeric fields parsed off each `step=...` summary line.
FIELDS = [
    "step", "pi_in", "n_minimal_J", "n_resolvents_raw", "n_resolvents_kept",
    "max_J_size", "pi_out_before_sub", "pi_out_after_sub", "sub_elim",
    "sub_elapsed_s", "n_distinct_sets", "n_taut", "fwd_sub_amb",
    "fwd_sub_intra", "prefix_taut", "lookahead_taut", "prefix_sub",
    "pruned_leaves", "dup", "expand_elapsed_s", "elapsed_s",
]


def parse_log(path: Path) -> list[dict]:
    """Return one dict of parsed metrics per `step=` summary line, in log order."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("step="):
            continue
        var_m = re.search(r"var=(\S+)", line)
        row = {"var": var_m.group(1) if var_m else "?"}
        for f in FIELDS:
            m = re.search(rf"\b{f}=([0-9.]+)", line)
            if m:
                row[f] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
        rows.append(row)
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    """Write all parsed metrics to CSV, ordered by elimination step (0..n-1)."""
    cols = ["step", "var"] + [f for f in FIELDS if f != "step"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["step"]):
            w.writerow({c: r.get(c, "") for c in cols})


def fmt_hms(seconds: float) -> str:
    """Format a duration in seconds as compact h/m/s."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{seconds:.2f}s"


def write_markdown(rows: list[dict], out: Path) -> None:
    """Write a per-variable timing summary as a Markdown table."""
    order = sorted(rows, key=lambda r: r["step"], reverse=True)  # elimination order
    total = sum(r.get("elapsed_s", 0.0) for r in rows)
    lines = [
        "# 4x4 sudoku compile — per-variable timing",
        "",
        f"Total compile wall time: **{fmt_hms(total)}** ({total:,.1f}s), "
        f"min-degree elimination ordering, 16 variables.",
        "",
        "| elim# | var | clauses in (`pi_in`) | raw resolvents | "
        "expand | subsum | total | % |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(order):
        el = r.get("elapsed_s", 0.0)
        lines.append(
            f"| {i} | {r['var']} | {r.get('pi_in', 0):,} | "
            f"{r.get('n_resolvents_raw', 0):,} | "
            f"{fmt_hms(r.get('expand_elapsed_s', 0.0))} | "
            f"{fmt_hms(r.get('sub_elapsed_s', 0.0))} | "
            f"{fmt_hms(el)} | {100 * el / total if total else 0:.1f}% |"
        )
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote Markdown table -> {out}")


def print_table(rows: list[dict]) -> None:
    """Print a per-variable timing table + ASCII bar chart, slowest annotated."""
    order = sorted(rows, key=lambda r: r["step"], reverse=True)  # elimination order
    total = sum(r.get("elapsed_s", 0.0) for r in rows)
    peak = max(r.get("elapsed_s", 0.0) for r in rows)
    print(f"\nTotal compile wall time: {fmt_hms(total)}  ({total:.1f}s)\n")
    hdr = f"{'elim#':>5} {'var':>5} {'pi_in':>7} {'raw_resolv':>13} " \
          f"{'expand':>10} {'subsum':>9} {'total':>10} {'%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(order):
        el = r.get("elapsed_s", 0.0)
        bar = "#" * int(round(40 * el / peak)) if peak else ""
        print(
            f"{i:>5} {r['var']:>5} {r.get('pi_in', 0):>7} "
            f"{r.get('n_resolvents_raw', 0):>13,} "
            f"{fmt_hms(r.get('expand_elapsed_s', 0.0)):>10} "
            f"{fmt_hms(r.get('sub_elapsed_s', 0.0)):>9} "
            f"{fmt_hms(el):>10} {100 * el / total if total else 0:>5.1f}% {bar}"
        )


def make_table_png(
    rows: list[dict], out: Path, baseline: dict | None = None
) -> None:
    """Render the per-variable timing summary as a standalone PNG table.

    If `baseline` (a {var: serial_elapsed_s} map) is given, append a speedup
    column (baseline_total / this_total per variable)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(rows, key=lambda r: r["step"], reverse=True)
    total = sum(r.get("elapsed_s", 0.0) for r in rows)
    peak = max(r.get("elapsed_s", 0.0) for r in rows)
    base_total = sum(baseline.values()) if baseline else 0.0
    headers = ["elim#", "var", "clauses in", "raw resolvents",
               "expand", "subsum", "total", "%"]
    if baseline:
        headers += ["serial", "speedup"]
    cells, pcts = [], []
    for i, r in enumerate(order):
        el = r.get("elapsed_s", 0.0)
        pcts.append(100 * el / total if total else 0)
        row = [
            str(i), r["var"], f"{r.get('pi_in', 0):,}",
            f"{r.get('n_resolvents_raw', 0):,}",
            fmt_hms(r.get("expand_elapsed_s", 0.0)),
            fmt_hms(r.get("sub_elapsed_s", 0.0)),
            fmt_hms(el), f"{pcts[-1]:.1f}%",
        ]
        if baseline:
            b = baseline.get(r["var"], 0.0)
            sp = f"{b / el:.1f}x" if el > 0 and b > 0 else "-"
            row += [fmt_hms(b), sp]
        cells.append(row)

    title = (
        f"4x4 sudoku compile — per-variable timing\n"
        f"total {fmt_hms(total)} ({total:,.0f}s), min-degree ordering, "
        f"{len(rows)} variables"
    )
    if baseline:
        title += (
            f"\nvs serial {fmt_hms(base_total)} → "
            f"overall {base_total / total:.1f}x speedup"
        )

    fig, ax = plt.subplots(
        figsize=(11 + (2.4 if baseline else 0), 0.45 * len(cells) + 1.5)
    )
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=14)
    tbl = ax.table(cellText=cells, colLabels=headers,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    sp_col = len(headers) - 1
    # Header styling.
    for c in range(len(headers)):
        cell = tbl[0, c]
        cell.set_facecolor("#34495e")
        cell.set_text_props(color="white", fontweight="bold")
    # Shade each total/% row by its share of wall time so the hotspots pop;
    # bold the speedup cell so the win per variable reads at a glance.
    for i, pct in enumerate(pcts, start=1):
        shade = (1.0, 1.0 - min(pct, 100) / 130, 1.0 - min(pct, 100) / 130)
        for c in (6, 7):
            tbl[i, c].set_facecolor(shade)
        if baseline:
            tbl[i, sp_col].set_facecolor("#d5f5e3")
            tbl[i, sp_col].set_text_props(fontweight="bold")
        if order[i - 1].get("elapsed_s", 0.0) == peak:
            for c in range(len(headers)):
                tbl[i, c].set_text_props(fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved table PNG -> {out}")


def make_plot(rows: list[dict], out: Path) -> None:
    """Save a stacked bar chart of expand/subsumption/other time per variable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(rows, key=lambda r: r["step"], reverse=True)
    labels = [f"{i}\n{r['var']}" for i, r in enumerate(order)]
    expand = [r.get("expand_elapsed_s", 0.0) for r in order]
    sub = [r.get("sub_elapsed_s", 0.0) for r in order]
    other = [
        max(0.0, r.get("elapsed_s", 0.0)
            - r.get("expand_elapsed_s", 0.0)
            - r.get("sub_elapsed_s", 0.0))
        for r in order
    ]
    x = range(len(order))

    fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(13, 9))
    for ax in (ax_lin, ax_log):
        b = ax.bar(x, expand, label="expand (enumeration)", color="#c0392b")
        ax.bar(x, sub, bottom=expand, label="forward-subsumption", color="#2980b9")
        ax.bar(x, other, bottom=[e + s for e, s in zip(expand, sub)],
               label="other", color="#95a5a6")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("seconds")
        ax.grid(axis="y", alpha=0.3)
    ax_lin.set_title("4x4 compile: per-variable wall time (elimination order) — linear")
    ax_lin.legend()
    ax_log.set_yscale("log")
    ax_log.set_title("same, log scale")
    ax_log.set_xlabel("elimination index / variable")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nSaved plot -> {out}")


def main() -> None:
    """Parse the log path from argv and emit CSV, table, and PNG."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, nargs="?",
                    default=Path("sudoku_4x4_compile.log"))
    ap.add_argument("--csv", type=Path, default=Path("compile_4x4_timing.csv"))
    ap.add_argument("--png", type=Path, default=Path("compile_4x4_timing.png"))
    ap.add_argument("--md", type=Path, default=Path("compile_4x4_timing.md"))
    ap.add_argument("--table-png", type=Path,
                    default=Path("compile_4x4_timing_table.png"))
    ap.add_argument("--baseline", type=Path, default=None,
                    help="Serial timing CSV; adds a per-var speedup column.")
    args = ap.parse_args()

    baseline = None
    if args.baseline and args.baseline.exists():
        baseline = {}
        with args.baseline.open() as fh:
            for row in csv.DictReader(fh):
                try:
                    baseline[row["var"]] = float(row["elapsed_s"])
                except (KeyError, ValueError):
                    pass

    rows = parse_log(args.log)
    if not rows:
        raise SystemExit(f"no `step=` lines found in {args.log}")
    write_csv(rows, args.csv)
    print_table(rows)
    print(f"\nWrote CSV -> {args.csv}")
    write_markdown(rows, args.md)
    make_table_png(rows, args.table_png, baseline=baseline)
    make_plot(rows, args.png)


if __name__ == "__main__":
    main()
