"""
Concurrent load balancing benchmark for K3d RQ2.

Sends requests from N_CLIENTS simultaneous workers to demonstrate that
K3d's Service load balancing distributes SNN inference across replicas.

Test matrix:
  - 1 replica  + 1 client  → baseline (no concurrency, no scaling)
  - 1 replica  + 3 clients → queuing  (clients block each other on one pod)
  - 3 replicas + 3 clients → scaling  (each client gets its own pod, no queuing)

The key metric: latency under concurrent load.
  - 1r/3c latency ≈ 3× the 1r/1c latency  → requests queue behind one pod
  - 3r/3c latency ≈ 1× the 1r/1c latency  → each pod serves one client independently

Usage (requires server running and accessible):
    python scripts/run_concurrent_benchmark.py --url http://host.docker.internal:9090

Before running:
    # Apply ingress so traffic routes through Traefik → Service → pods:
    kubectl apply -f k3d/ingress.yaml

    # Scale to 1 replica first (host PowerShell):
    kubectl scale deployment neuro-sim --replicas=1 -n neuro-sim
    kubectl wait --for=condition=available --timeout=60s deployment/neuro-sim -n neuro-sim

    # Then run this script — it steps through the test matrix automatically,
    # prompting you to change replica count between scenarios.
"""

import argparse
import io
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from torchvision.datasets import MNIST
from torchvision import transforms


# ── MNIST loader ──────────────────────────────────────────────────────────────

def load_mnist_images(n_per_digit: int, data_dir: str) -> list[tuple[int, bytes]]:
    """Return list of (digit, png_bytes) pairs, n_per_digit per class."""
    dataset = MNIST(
        root=data_dir, train=False, download=True,
        transform=transforms.ToTensor(),
    )
    counts = {d: 0 for d in range(10)}
    images = []
    for img_tensor, label in dataset:
        if counts[label] >= n_per_digit:
            continue
        buf = io.BytesIO()
        transforms.ToPILImage()(img_tensor).save(buf, format="PNG")
        images.append((label, buf.getvalue()))
        counts[label] += 1
        if all(v >= n_per_digit for v in counts.values()):
            break
    return images


# ── single request ────────────────────────────────────────────────────────────

def infer(url: str, digit: int, png_bytes: bytes, timeout: int) -> dict:
    t0 = time.monotonic()
    resp = requests.post(
        f"{url}/predict",
        files={"file": ("img.png", png_bytes, "image/png")},
        timeout=timeout,
    )
    latency_ms = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    return {"digit": digit, "latency_ms": latency_ms}


# ── worker: one client's share of images ─────────────────────────────────────

def worker(url: str, images: list[tuple[int, bytes]], timeout: int) -> list[float]:
    """Send assigned images sequentially, return list of latencies."""
    latencies = []
    for digit, png in images:
        result = infer(url, digit, png, timeout)
        latencies.append(result["latency_ms"])
    return latencies


# ── concurrent scenario runner ────────────────────────────────────────────────

def run_scenario(
    label: str,
    url: str,
    n_clients: int,
    images: list[tuple[int, bytes]],
    timeout: int,
) -> dict:
    """
    Split images across n_clients workers, fire them all simultaneously,
    collect latencies and wall-clock throughput.
    """
    print(f"\n  {label}")
    print(f"  {n_clients} concurrent client(s) × {len(images) // n_clients} requests each"
          f" = {len(images)} total requests")

    # split images evenly across clients
    chunks = [images[i::n_clients] for i in range(n_clients)]

    t_wall_start = time.monotonic()
    all_latencies = []

    with ThreadPoolExecutor(max_workers=n_clients) as pool:
        futures = [pool.submit(worker, url, chunk, timeout) for chunk in chunks]
        for f in as_completed(futures):
            all_latencies.extend(f.result())

    wall_s = time.monotonic() - t_wall_start
    throughput = len(all_latencies) / wall_s

    all_latencies.sort()
    n = len(all_latencies)

    result = {
        "label": label,
        "n_clients": n_clients,
        "n_requests": n,
        "wall_time_s": round(wall_s, 3),
        "throughput_req_per_sec": round(throughput, 3),
        "latency_ms": {
            "min":  round(min(all_latencies), 2),
            "mean": round(statistics.mean(all_latencies), 2),
            "p50":  round(all_latencies[int(n * 0.50)], 2),
            "p95":  round(all_latencies[int(n * 0.95)], 2),
            "p99":  round(all_latencies[int(n * 0.99)], 2),
            "max":  round(max(all_latencies), 2),
        },
        "raw_latencies_ms": [round(l, 2) for l in all_latencies],
    }

    print(f"  wall time  : {wall_s:.1f}s")
    print(f"  throughput : {throughput:.3f} req/s")
    print(f"  p50 latency: {result['latency_ms']['p50']} ms")
    print(f"  p99 latency: {result['latency_ms']['p99']} ms")
    return result


# ── health check ─────────────────────────────────────────────────────────────

def wait_healthy(url: str, timeout: int = 60) -> None:
    print(f"  Checking {url}/health …", end="", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.ok:
                print(" OK")
                return
        except requests.RequestException:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    raise RuntimeError(f"Server at {url} not healthy after {timeout}s")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concurrent K3d load balancing benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default="http://host.docker.internal:9090")
    parser.add_argument("--n-images", type=int, default=5,
                        help="Images per digit per client (default 5)")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--data-dir", default="/tmp/mnist_benchmark")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wait_healthy(args.url)

    # load images once — 3 clients × n_images × 10 digits
    n_clients = 3
    total_per_digit = n_clients * args.n_images
    print(f"\nLoading {total_per_digit} images per digit from MNIST…")
    images = load_mnist_images(total_per_digit, args.data_dir)
    print(f"  {len(images)} images loaded")

    results = []

    # ── Scenario 1: 1 replica, 1 client (baseline) ───────────────────────────
    input("\n[Step 1/3] Ensure 1 replica is running, then press Enter…\n"
          "  kubectl scale deployment neuro-sim --replicas=1 -n neuro-sim\n"
          "  kubectl wait --for=condition=available --timeout=60s deployment/neuro-sim -n neuro-sim\n")
    r = run_scenario(
        label="1 replica / 1 client",
        url=args.url,
        n_clients=1,
        images=images[: args.n_images * 10],   # only 1 client's worth
        timeout=args.timeout,
    )
    results.append(r)

    # ── Scenario 2: 1 replica, 3 clients (queuing) ───────────────────────────
    input("\n[Step 2/3] Still 1 replica — press Enter to fire 3 concurrent clients…\n")
    r = run_scenario(
        label="1 replica / 3 clients",
        url=args.url,
        n_clients=3,
        images=images,
        timeout=args.timeout,
    )
    results.append(r)

    # ── Scenario 3: 3 replicas, 3 clients (scaling) ──────────────────────────
    input("\n[Step 3/3] Scale to 3 replicas, then press Enter…\n"
          "  kubectl scale deployment neuro-sim --replicas=3 -n neuro-sim\n"
          "  kubectl wait --for=condition=available --timeout=120s deployment/neuro-sim -n neuro-sim\n")
    r = run_scenario(
        label="3 replicas / 3 clients",
        url=args.url,
        n_clients=3,
        images=images,
        timeout=args.timeout,
    )
    results.append(r)

    # ── save ──────────────────────────────────────────────────────────────────
    out = out_dir / "k3d_concurrent.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved → {out}")

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  {'Scenario':<28} {'p50 (ms)':>10} {'p99 (ms)':>10} {'req/s':>10}")
    print(f"{'─'*65}")
    baseline_p50 = results[0]["latency_ms"]["p50"]
    for r in results:
        p50  = r["latency_ms"]["p50"]
        p99  = r["latency_ms"]["p99"]
        tput = r["throughput_req_per_sec"]
        factor = f"  (×{p50/baseline_p50:.1f})" if r != results[0] else ""
        print(f"  {r['label']:<28} {p50:>10.0f} {p99:>10.0f} {tput:>10.3f}{factor}")
    print(f"{'─'*65}")


if __name__ == "__main__":
    main()
