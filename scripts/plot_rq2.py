"""
RQ2 plots: K3d orchestration overhead and replica scaling behaviour.

Reads results from:
  - results/baseline.json              (RQ1 — Docker no limits)
  - results/constrained_0.5cpu.json    (RQ1 — Docker 0.5 CPU)
  - results/k3d_1replica.json          (RQ2 — K3d 1 pod)
  - results/k3d_3replicas.json         (RQ2 — K3d 3 pods)

Missing files are reported and the affected plots are skipped.

Usage:
    python scripts/plot_rq2.py
    python scripts/plot_rq2.py --output-dir results/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── shared style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

NEUROMORPHIC_FLOOR_MS = 100

# colours: baseline=blue, constrained=orange, k3d-1=green, k3d-3=purple
PALETTE = {
    "baseline":       "#4C72B0",
    "constrained":    "#DD8452",
    "k3d_1replica":   "#55A868",
    "k3d_3replicas":  "#8172B2",
}


# ── data loading ──────────────────────────────────────────────────────────────

def _load(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [MISSING] {path} — run scripts/run_k3d_benchmark.py first")
        return None
    with open(path) as f:
        return json.load(f)


def load_all(base: Path) -> dict[str, dict | None]:
    return {
        "baseline":       _load(base / "results/baseline.json"),
        "constrained":    _load(base / "results/constrained_0.5cpu.json"),
        "k3d_1replica":   _load(base / "results/k3d_1replica.json"),
        "k3d_3replicas":  _load(base / "results/k3d_3replicas.json"),
    }


def empirical_floor(datasets: list[dict]) -> float:
    return min(min(d["raw_latencies_ms"]) for d in datasets)


# ── Plot 1: latency comparison — baseline vs K3d 1r vs K3d 3r ────────────────

def plot_latency_comparison(data: dict, out: Path) -> None:
    scenarios = {
        "Baseline\n(Docker, no limits)": ("baseline",      PALETTE["baseline"]),
        "K3d\n1 replica":                ("k3d_1replica",  PALETTE["k3d_1replica"]),
        "K3d\n3 replicas":               ("k3d_3replicas", PALETTE["k3d_3replicas"]),
    }
    available = {k: v for k, v in scenarios.items() if data[v[0]] is not None}
    if not available:
        print("  [SKIP] plot_latency_comparison — no K3d results yet")
        return

    labels  = list(available.keys())
    colors  = [v[1] for v in available.values()]
    raw     = [data[v[0]]["raw_latencies_ms"] for v in available.values()]

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("Fig 5 — Latency Distribution: Baseline vs K3d", fontsize=13)

    def _draw(ax, values, color, label):
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

        lo, hi = min(values), max(values)
        stats = {
            "min": lo,
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": hi,
        }
        for name, val in stats.items():
            ax.annotate(
                f"{name} = {val:,.0f} ms",
                xy=(1, val), xytext=(1.35, val),
                fontsize=8, va="center", color="#333333",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8),
            )

        ax.axhline(NEUROMORPHIC_FLOOR_MS, color="grey", linestyle=":",
                   linewidth=1.4, label=f"floor ({NEUROMORPHIC_FLOOR_MS} ms)")
        padding = (hi - lo) * 0.25
        ax.set_ylim(lo - padding, hi + padding * 2.5)
        ax.set_xlim(0.5, 2.0)
        ax.set_xticks([1])
        ax.set_xticklabels([label], fontsize=11)
        ax.set_ylabel("End-to-end latency (ms)")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.legend(fontsize=8, loc="upper left")

    for ax, vals, color, label in zip(axes, raw, colors, labels):
        _draw(ax, vals, color, label)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved → {out}")


# ── Plot 2: unified overhead bar chart (all scenarios) ────────────────────────

def plot_unified_overhead(data: dict, out: Path) -> None:
    # Order: baseline → constrained-0.5 → K3d-1r → K3d-3r
    order = [
        ("Baseline\n(no limits)",    "baseline",      PALETTE["baseline"]),
        ("0.5 CPU\n512 MB",          "constrained",   PALETTE["constrained"]),
        ("K3d\n1 replica",           "k3d_1replica",  PALETTE["k3d_1replica"]),
        ("K3d\n3 replicas",          "k3d_3replicas", PALETTE["k3d_3replicas"]),
    ]
    present = [(l, k, c) for l, k, c in order if data[k] is not None]
    if len(present) < 2:
        print("  [SKIP] plot_unified_overhead — need at least 2 scenarios")
        return

    # empirical floor from all available datasets
    e_floor = empirical_floor([data[k] for _, k, _ in present])
    labels   = [l for l, _, _ in present]
    overheads = [data[k]["latency_ms"]["p50"] - e_floor for _, k, _ in present]
    colors   = [c for _, _, c in present]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, overheads, color=colors, alpha=0.82, width=0.5)

    for bar, val in zip(bars, overheads):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + max(overheads) * 0.01,
                f"{val:,.0f} ms",
                ha="center", va="bottom", fontsize=9)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("Infrastructure overhead — p50 − floor (ms, log scale)")
    ax.set_title(
        f"Fig 6 — Unified Infrastructure Overhead\n"
        f"(empirical floor = {e_floor:.0f} ms across all scenarios)"
    )

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved → {out}")


# ── Plot 3: K3d replica scaling — throughput bar chart ────────────────────────

def plot_replica_scaling(data: dict, out: Path) -> None:
    needed = ["k3d_1replica", "k3d_3replicas"]
    if any(data[k] is None for k in needed):
        print("  [SKIP] plot_replica_scaling — K3d results missing")
        return

    labels    = ["1 replica", "3 replicas"]
    colors    = [PALETTE["k3d_1replica"], PALETTE["k3d_3replicas"]]
    throughputs = [data[k]["throughput_req_per_sec"] for k in needed]
    p50s        = [data[k]["latency_ms"]["p50"] for k in needed]

    fig, (ax_tp, ax_lat) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Fig 7 — K3d Replica Scaling", fontsize=13)

    # throughput
    bars = ax_tp.bar(labels, throughputs, color=colors, alpha=0.82, width=0.45)
    for bar, val in zip(bars, throughputs):
        ax_tp.text(bar.get_x() + bar.get_width() / 2,
                   val + max(throughputs) * 0.015,
                   f"{val:.3f} req/s",
                   ha="center", va="bottom", fontsize=10)
    scale_factor = throughputs[1] / throughputs[0] if throughputs[0] else float("nan")
    ax_tp.set_title(f"Throughput  (×{scale_factor:.1f} scaling)", fontsize=11)
    ax_tp.set_ylabel("Requests / second")
    ax_tp.set_ylim(0, max(throughputs) * 1.2)

    # p50 latency (should stay flat — confirms requests are not serialised)
    bars2 = ax_lat.bar(labels, p50s, color=colors, alpha=0.82, width=0.45)
    for bar, val in zip(bars2, p50s):
        ax_lat.text(bar.get_x() + bar.get_width() / 2,
                    val + max(p50s) * 0.015,
                    f"{val:,.0f} ms",
                    ha="center", va="bottom", fontsize=10)
    ax_lat.axhline(NEUROMORPHIC_FLOOR_MS, color="grey", linestyle=":",
                   linewidth=1.4, label=f"floor ({NEUROMORPHIC_FLOOR_MS} ms)")
    ax_lat.set_title("p50 Latency  (should stay flat)", fontsize=11)
    ax_lat.set_ylabel("Median latency (ms)")
    ax_lat.set_ylim(0, max(p50s) * 1.2)
    ax_lat.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved → {out}")


# ── Plot 4: concurrent load balancing — the real scaling story ────────────────

def plot_concurrent_scaling(base: Path, out: Path) -> None:
    path = base / "results/k3d_concurrent.json"
    if not path.exists():
        print("  [SKIP] plot_concurrent_scaling — run scripts/run_concurrent_benchmark.py first")
        return

    with open(path) as f:
        scenarios = json.load(f)

    labels      = [s["label"] for s in scenarios]
    p50s        = [s["latency_ms"]["p50"] for s in scenarios]
    p99s        = [s["latency_ms"]["p99"] for s in scenarios]
    throughputs = [s["throughput_req_per_sec"] for s in scenarios]

    # colours: baseline=blue, queuing=orange, scaling=green
    colors = [PALETTE["baseline"], PALETTE["constrained"], PALETTE["k3d_3replicas"]]

    fig, (ax_lat, ax_tp) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fig 8 — K3d Load Balancing Under Concurrent Requests", fontsize=13)

    x = np.arange(len(labels))
    w = 0.35

    # latency: grouped bars p50 + p99
    bars_p50 = ax_lat.bar(x - w/2, p50s, w, label="p50", color=colors, alpha=0.82)
    bars_p99 = ax_lat.bar(x + w/2, p99s, w, label="p99", color=colors, alpha=0.45,
                           edgecolor=colors, linewidth=1.2)
    ax_lat.axhline(NEUROMORPHIC_FLOOR_MS, color="grey", linestyle=":",
                   linewidth=1.4, label=f"floor ({NEUROMORPHIC_FLOOR_MS} ms)")
    for bar, val in zip(bars_p50, p50s):
        ax_lat.text(bar.get_x() + bar.get_width() / 2, val + 5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(labels, fontsize=9)
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title("Latency  (p50 solid / p99 faded)")
    ax_lat.legend(fontsize=9)

    # throughput bars
    bars_tp = ax_tp.bar(labels, throughputs, color=colors, alpha=0.82, width=0.45)
    for bar, val in zip(bars_tp, throughputs):
        ax_tp.text(bar.get_x() + bar.get_width() / 2,
                   val + max(throughputs) * 0.015,
                   f"{val:.3f} req/s", ha="center", va="bottom", fontsize=9)
    ax_tp.set_xticklabels(labels, fontsize=9)
    ax_tp.set_ylabel("Throughput (requests / second)")
    ax_tp.set_title("Throughput")
    ax_tp.set_ylim(0, max(throughputs) * 1.2)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved → {out}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--base-dir",   default=".")
    args = parser.parse_args()

    base    = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results…")
    data = load_all(base)

    present  = sum(1 for v in data.values() if v is not None)
    missing  = sum(1 for v in data.values() if v is None)
    print(f"  {present} scenario(s) loaded, {missing} missing\n")

    print("Rendering plots…")
    plot_latency_comparison(data, out_dir / "rq2_latency_comparison.png")
    plot_unified_overhead(  data, out_dir / "rq2_unified_overhead.png")
    plot_replica_scaling(   data, out_dir / "rq2_replica_scaling.png")
    plot_concurrent_scaling(base, out_dir / "rq2_concurrent_scaling.png")

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
