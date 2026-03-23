"""Benchmark logic for the neuro-sim inference API.

Sends MNIST test images to a running neuro-sim server and collects
latency, accuracy, and spike-count metrics. Designed to be re-run at
different resource constraints to produce thesis comparison data.
"""

import io
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import requests
import yaml
from torchvision.datasets import MNIST
from torchvision import transforms


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "url": "http://localhost:8000",
    "n_runs": 10,
    "timeout": 30,
    "data_dir": "/tmp/mnist_benchmark",
    "output_json": None,
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> dict:
    """Load benchmark config from YAML, falling back to defaults."""
    cfg = DEFAULTS.copy()
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(path) as f:
            file_cfg = yaml.safe_load(f) or {}
        # Config file may have a top-level 'benchmark' key or be flat
        cfg.update(file_cfg.get("benchmark", file_cfg))
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_n_per_digit(data_dir: str, n: int) -> dict[int, list[bytes]]:
    """Return up to n image_bytes per digit class 0–9 from MNIST test set."""
    dataset = MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    samples: dict[int, list[bytes]] = {i: [] for i in range(10)}
    for img_tensor, label in dataset:
        if len(samples[label]) < n:
            pil_img = transforms.ToPILImage()(img_tensor)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            samples[label].append(buf.getvalue())
        if all(len(v) == n for v in samples.values()):
            break

    return samples


def check_health(url: str) -> dict:
    resp = requests.get(f"{url}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def predict(url: str, image_bytes: bytes, timeout: int = 30) -> dict:
    resp = requests.post(
        f"{url}/predict",
        files={"file": ("image.png", image_bytes, "image/png")},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def percentile(data: list[float], p: int) -> float:
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    url: str,
    n_runs: int,
    output_json: Optional[str],
    timeout: int = 30,
    data_dir: str = "/tmp/mnist_benchmark",
) -> dict:
    print(f"  Target : {url}")
    print(f"  Runs   : {n_runs} images per digit × 10 digits = {n_runs * 10} total requests")
    print(f"  Timeout: {timeout}s per request")

    # Health check
    print("\n[1/3] Health check...", end=" ", flush=True)
    try:
        health = check_health(url)
    except Exception as e:
        print(f"FAILED\n  {e}")
        print("  Is the container running? docker run --rm -p 8000:8000 neuro-sim:latest")
        sys.exit(1)

    print("OK")
    print(f"       Model    : {health['model']}")
    print(f"       Neurons  : {health['n_neurons']}")
    print(f"       Sim window: {health['simulation_time_ms']} ms (neuromorphic floor latency)")
    print(f"       Device   : {health['device']}")

    # Load MNIST samples
    print("\n[2/3] Loading MNIST test samples...", end=" ", flush=True)
    samples_by_digit = load_n_per_digit(data_dir, n_runs)
    actual_n = min(len(v) for v in samples_by_digit.values())
    print(f"OK ({actual_n} images per digit × 10 classes = {actual_n * 10} total)")

    # Run benchmark
    print(f"\n[3/3] Running {actual_n * 10} requests...\n")

    results = []
    all_latencies = []
    correct_proportion = 0
    correct_all_activity = 0
    total = 0

    wall_start = time.perf_counter()

    for label in range(10):
        latencies = []
        spike_counts = []
        digit_correct_proportion = 0
        digit_correct_all_activity = 0

        for image_bytes in samples_by_digit[label]:
            result = predict(url, image_bytes, timeout=timeout)
            latencies.append(result["inference_time_ms"])
            spike_counts.append(result["spike_count"])
            all_latencies.append(result["inference_time_ms"])

            if result["digit"] == label:
                correct_proportion += 1
                digit_correct_proportion += 1
            if result["all_activity_digit"] == label:
                correct_all_activity += 1
                digit_correct_all_activity += 1
            total += 1

        n_digit = len(latencies)
        results.append({
            "digit": label,
            "n_images": n_digit,
            "accuracy_proportion": round(digit_correct_proportion / n_digit, 4),
            "accuracy_all_activity": round(digit_correct_all_activity / n_digit, 4),
            "latency_ms": {
                "min": round(min(latencies), 2),
                "mean": round(statistics.mean(latencies), 2),
                "max": round(max(latencies), 2),
            },
            "mean_spike_count": round(statistics.mean(spike_counts), 1),
        })

        print(
            f"  Digit {label}: acc={digit_correct_proportion}/{n_digit}"
            f" | latency mean={statistics.mean(latencies):.1f}ms"
            f" | spikes mean={statistics.mean(spike_counts):.1f}"
        )

    wall_elapsed = time.perf_counter() - wall_start

    # Summary
    sim_time_ms = health["simulation_time_ms"]
    p50 = percentile(all_latencies, 50)
    p95 = percentile(all_latencies, 95)
    p99 = percentile(all_latencies, 99)
    overhead = p50 - sim_time_ms

    print(f"\n  {'─'*56}")
    print(f"  RESULTS SUMMARY")
    print(f"  {'─'*56}")
    print(f"  Accuracy (proportion-weighting) : {correct_proportion}/{total} = {correct_proportion/total*100:.0f}%")
    print(f"  Accuracy (all-activity)         : {correct_all_activity}/{total} = {correct_all_activity/total*100:.0f}%")
    print(f"\n  Latency (wall-clock per request):")
    print(f"    min  : {min(all_latencies):.1f} ms")
    print(f"    p50  : {p50:.1f} ms")
    print(f"    p95  : {p95:.1f} ms")
    print(f"    p99  : {p99:.1f} ms")
    print(f"    max  : {max(all_latencies):.1f} ms")
    print(f"\n  Neuromorphic floor (sim window) : {sim_time_ms} ms (fixed by SNN)")
    print(f"  Orchestration overhead (p50)    : {overhead:.1f} ms  [= p50 - floor]")
    print(f"\n  Total wall time : {wall_elapsed:.1f}s for {len(all_latencies)} requests")
    print(f"  Throughput      : {len(all_latencies)/wall_elapsed:.2f} req/s")
    print(f"  {'─'*56}\n")

    output = {
        "config": {
            "url": url,
            "n_images_per_digit": actual_n,
            "total_requests": len(all_latencies),
            "model": health["model"],
            "n_neurons": health["n_neurons"],
            "simulation_time_ms": sim_time_ms,
        },
        "accuracy": {
            "proportion_weighting": round(correct_proportion / total, 4),
            "all_activity": round(correct_all_activity / total, 4),
        },
        "latency_ms": {
            "min": round(min(all_latencies), 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "max": round(max(all_latencies), 2),
            "neuromorphic_floor": sim_time_ms,
            "overhead_p50": round(overhead, 2),
        },
        "throughput_req_per_sec": round(len(all_latencies) / wall_elapsed, 3),
        "per_digit": results,
        "raw_latencies_ms": [round(l, 2) for l in all_latencies],
    }

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"  Raw results saved to: {output_json}")

    return output
