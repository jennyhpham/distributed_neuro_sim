"""
RQ1 plots: latency distribution, infrastructure overhead, throughput, per-digit heatmap.
Only baseline and 0.5cpu scenarios (0.25cpu OOM-killed — discussed separately).

Usage:
    python scripts/plot_rq1.py
    python scripts/plot_rq1.py --output-dir results/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

SCENARIOS = {
    "Baseline\n(no limits)": "results/baseline.json",
    "0.5 CPU\n512 MB": "results/constrained_0.5cpu.json",
}
COLORS = ["#4C72B0", "#DD8452"]   # blue, orange
NEUROMORPHIC_FLOOR = 100          # configured simulation window (ms)

# empirical floor = min latency observed across all runs
def empirical_floor(datasets: list[dict]) -> float:
    return min(min(d["raw_latencies_ms"]) for d in datasets)


def load_data(base_dir: Path) -> dict[str, dict]:
    result = {}
    for label, rel_path in SCENARIOS.items():
        path = base_dir / rel_path
        with open(path) as f:
            result[label] = json.load(f)
    return result


# ── Plot 1: latency distribution — two-panel, each axis scaled to its data ───
def plot_latency_distribution(data: dict[str, dict], out: Path) -> None:
    labels = list(data.keys())
    raw    = [data[l]["raw_latencies_ms"] for l in labels]
    colors = COLORS[: len(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Fig 1 — Latency Distribution Under Resource Constraints", fontsize=13)

    def _boxplot_on(ax, values, color, label):
        bp = ax.boxplot(
            [values],
            patch_artist=True,
            widths=0.45,
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(linewidth=1.4),
            capprops=dict(linewidth=1.4),
            flierprops=dict(marker="o", markersize=5, alpha=0.6,
                            markerfacecolor=color, markeredgecolor=color),
        )
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.82)

        q1, med, q3 = np.percentile(values, [25, 50, 75])
        lo, hi = min(values), max(values)

        # annotate key percentiles
        stats = {
            "min": lo, "p50": med, "p75": q3,
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
            "max": hi,
        }
        x_right = 1.32
        for name, val in stats.items():
            ax.annotate(
                f"{name} = {val:,.0f} ms",
                xy=(1, val), xytext=(x_right, val),
                fontsize=8, va="center", color="#333333",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8),
            )

        # neuromorphic floor
        ax.axhline(NEUROMORPHIC_FLOOR, color="grey", linestyle=":", linewidth=1.4,
                   label=f"floor ({NEUROMORPHIC_FLOOR} ms)")

        padding = (hi - lo) * 0.25
        ax.set_ylim(lo - padding, hi + padding * 2.5)
        ax.set_xlim(0.5, 2.0)          # leave room for annotations
        ax.set_xticks([1])
        ax.set_xticklabels([label], fontsize=11)
        ax.set_ylabel("End-to-end latency (ms)")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v:,.0f}"
        ))
        ax.legend(fontsize=8, loc="upper left")

    for ax, vals, color, label in zip(axes, raw, colors, labels):
        _boxplot_on(ax, vals, color, label)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved → {out}")


# ── Plot 2: infrastructure overhead bar chart ─────────────────────────────────
def plot_overhead(data: dict[str, dict], out: Path) -> None:
    e_floor = empirical_floor(list(data.values()))

    labels = list(data.keys())
    overheads = [data[l]["latency_ms"]["p50"] - e_floor for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, overheads, color=COLORS[: len(labels)], alpha=0.82, width=0.45)

    for bar, val in zip(bars, overheads):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + max(overheads) * 0.015,
                f"{val:,.1f} ms",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Infrastructure overhead — p50 − floor (ms)")
    ax.set_title(
        f"Fig 2 — Infrastructure Overhead (empirical floor = {e_floor:.0f} ms)"
    )
    ax.set_ylim(0, max(overheads) * 1.18)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved → {out}")


# ── Plot 3: throughput bar chart ──────────────────────────────────────────────
def plot_throughput(data: dict[str, dict], out: Path) -> None:
    labels = list(data.keys())
    values = [data[l]["throughput_req_per_sec"] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, values, color=COLORS[: len(labels)], alpha=0.82, width=0.45)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + max(values) * 0.015,
                f"{val:.3f} req/s",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Throughput (requests / second)")
    ax.set_title("Fig 3 — Throughput Under Resource Constraints")
    ax.set_ylim(0, max(values) * 1.18)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved → {out}")


# ── Plot 4: per-digit latency heatmap ────────────────────────────────────────
def plot_per_digit_heatmap(data: dict[str, dict], out: Path) -> None:
    labels = list(data.keys())
    digits = list(range(10))

    # matrix: rows=digits, cols=scenarios
    matrix = np.zeros((10, len(labels)))
    for col, label in enumerate(labels):
        per_digit = {d["digit"]: d["latency_ms"]["mean"] for d in data[label]["per_digit"]}
        for row, digit in enumerate(digits):
            matrix[row, col] = per_digit[digit]

    # normalise each column (scenario) to [0,1] so colour shows within-scenario spread
    col_min = matrix.min(axis=0, keepdims=True)
    col_max = matrix.max(axis=0, keepdims=True)
    norm_matrix = (matrix - col_min) / (col_max - col_min + 1e-9)

    # clean labels (remove newlines for axis)
    clean_labels = [l.replace("\n", " ") for l in labels]

    fig, ax = ax_main = plt.subplots(figsize=(6, 6))
    im = ax.imshow(norm_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    # annotate cells with absolute ms values
    for row in range(10):
        for col in range(len(labels)):
            val = matrix[row, col]
            text_color = "white" if norm_matrix[row, col] > 0.6 else "black"
            ax.text(col, row, f"{val:,.0f}", ha="center", va="center",
                    fontsize=9, color=text_color)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(clean_labels)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"Digit {d}" for d in digits])
    ax.set_title("Fig 4 — Per-Digit Mean Latency (ms)\n(colour = within-scenario relative rank)")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Relative latency (within scenario)")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results…")
    data = load_data(base_dir)

    print("Rendering plots…")
    plot_latency_distribution(data, out_dir / "rq1_latency_distribution.png")
    plot_overhead(data, out_dir / "rq1_overhead.png")
    plot_throughput(data, out_dir / "rq1_throughput.png")
    plot_per_digit_heatmap(data, out_dir / "rq1_per_digit_heatmap.png")

    print(f"\nDone. All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
