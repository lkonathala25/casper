from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
RULE = "#d8d7d2"
HEADER_BG = "#1a1a19"

BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
    "#0d366b",
]
LIGHT_TEXT_FROM = 8


def load_rows(path: Path) -> list[dict]:
    """Read the sweep CSV, coercing the numeric columns."""
    rows = []
    for r in csv.DictReader(path.open()):
        for k in ("m", "band", "trials", "clauses_in", "clauses_out",
                  "resolvents", "max_bucket"):
            if r.get(k):
                r[k] = int(r[k])
        for k in ("induced_width", "compile_med", "compile_p25", "compile_p75",
                  "compile_s", "compile_std"):
            if r.get(k):
                r[k] = float(r[k])
        rows.append(r)
    return rows


def shade(value: float, lo: float, hi: float) -> tuple[str, str]:
    """Map a magnitude to a (fill, text) pair off the sequential blue ramp."""
    if hi <= lo:
        idx = 0
    else:
        frac = (math.log10(value) - lo) / (hi - lo)
        idx = round(frac * (len(BLUE_RAMP) - 1))
    idx = max(0, min(len(BLUE_RAMP) - 1, idx))
    return BLUE_RAMP[idx], ("#ffffff" if idx >= LIGHT_TEXT_FROM else INK)


def render(rows: list[dict], out: Path, title: str, subtitle: str) -> None:
    """Draw the sweep as a publication-style table image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cols = [
        ("topology", 0.055, "left"), ("b", 0.038, "right"),
        ("m", 0.050, "right"), ("width", 0.058, "right"),
        ("seeds", 0.052, "right"), ("clauses in", 0.082, "right"),
        ("clauses out", 0.090, "right"), ("resolvents", 0.098, "right"),
        ("median", 0.080, "right"), ("IQR", 0.150, "right"),
        ("per var", 0.078, "right"),
    ]
    x_edges, acc = [], 0.045
    for _n, w, _a in cols:
        x_edges.append(acc)
        acc += w
    total_w = acc + 0.02

    times = [r["compile_med"] for r in rows]
    lo, hi = math.log10(min(times)), math.log10(max(times))

    row_h = 0.30
    fig_h = 1.55 + row_h * len(rows)
    fig, ax = plt.subplots(figsize=(12.2, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)

    y = fig_h - 0.30
    ax.text(0.045, y, title, fontsize=14, fontweight="bold", color=INK,
            va="center")
    y -= 0.30
    ax.text(0.045, y, subtitle, fontsize=8.5, color=INK_MUTED, va="center")

    y -= 0.42
    ax.add_patch(Rectangle((0.035, y - 0.135), total_w, 0.30,
                           facecolor=HEADER_BG, edgecolor="none"))
    for (name, w, align), x in zip(cols, x_edges):
        hx = x if align == "left" else x + w - 0.006
        ax.text(hx, y, name, fontsize=8.5, color="#ffffff", va="center",
                ha=align, fontweight="semibold")

    y -= 0.24
    prev_group = None
    for r in rows:
        y -= row_h
        group = (r["graph"], r["band"])
        if prev_group is not None and group != prev_group:
            ax.plot([0.035, 0.035 + total_w], [y + row_h / 2, y + row_h / 2],
                    color=RULE, lw=0.9)
        prev_group = group

        band = str(r["band"]) if r["graph"] == "exp" else "–"
        per_var = r["compile_med"] / r["m"] * 1000
        cells = [
            r["graph"], band, f"{r['m']}", f"{r['induced_width']:g}",
            f"{r['trials']}", f"{r['clauses_in']:,}", f"{r['clauses_out']:,}",
            f"{r['resolvents']:,}", f"{r['compile_med']:.4f}s",
            f"{r['compile_p25']:.4f} – {r['compile_p75']:.4f}",
            f"{per_var:.2f} ms",
        ]
        for i, ((name, w, align), x, text) in enumerate(zip(cols, x_edges, cells)):
            color, weight = INK, "normal"
            if name == "median":
                fill, color = shade(r["compile_med"], lo, hi)
                ax.add_patch(Rectangle(
                    (x - 0.004, y - 0.115), w, 0.235,
                    facecolor=fill, edgecolor="none",
                ))
                weight = "semibold"
            elif name in ("IQR", "clauses in", "seeds"):
                color = INK_MUTED
            elif name == "topology":
                weight = "semibold"
            tx = x if align == "left" else x + w - 0.006
            ax.text(tx, y, text, fontsize=8.5, color=color, va="center",
                    ha=align, fontweight=weight,
                    family="monospace" if align == "right" else "sans-serif")

    y -= 0.34
    ax.plot([0.035, 0.035 + total_w], [y + 0.10, y + 0.10], color=RULE, lw=0.9)
    ax.text(
        0.045, y - 0.08,
        "width = induced width of the primal graph under the elimination "
        "ordering actually used. Shading on `median` is log-scaled compile "
        "time. `poly` is a random spanning tree; `exp` is banded "
        "(edge iff 0 < |i-j| ≤ b).",
        fontsize=7.6, color=INK_MUTED, va="center",
    )

    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote table -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render a sweep CSV as a table image."
    )
    p.add_argument("csv_path", type=Path)
    p.add_argument("out_path", type=Path)
    p.add_argument("--title", default="Compilation time vs. constraint-graph topology")
    p.add_argument(
        "--subtitle",
        default="Signed-resolution compile, min-fill elimination ordering. "
                "K=6, tau=0.5, 2 clauses/edge. Median over independently "
                "generated instances.",
    )
    args = p.parse_args()
    render(load_rows(args.csv_path), args.out_path, args.title, args.subtitle)


if __name__ == "__main__":
    main()
