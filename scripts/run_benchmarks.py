"""Run all benchmark scenarios defined in benchmark.yaml against the neuro-sim container.

Each scenario starts a Docker container with the specified resource limits, waits for
it to become healthy, runs the benchmark, then removes the container.

The container's bridge IP is resolved via `docker inspect` so this works correctly
inside a devcontainer, where localhost does not reach sibling Docker containers.

Usage:
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --config configs/benchmark.yaml
    python scripts/run_benchmarks.py --n-runs 50
    python scripts/run_benchmarks.py --output-dir results/run1
    python scripts/run_benchmarks.py --timeout 120 --n-runs 20
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml
from neuro_sim.benchmark import run_benchmark


HEALTH_POLL_INTERVAL = 2   # seconds between health check attempts
HEALTH_MAX_ATTEMPTS = 30   # max attempts before giving up (~60 s)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("benchmark", raw)


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def docker_run(image: str, docker_args: dict) -> str:
    """Start a detached container and return its full container ID."""
    cmd = ["docker", "run", "-d"]
    if "cpus" in docker_args:
        cmd += [f"--cpus={docker_args['cpus']}"]
    if "memory" in docker_args:
        cmd += [f"--memory={docker_args['memory']}"]
    if "memory_swap" in docker_args:
        cmd += [f"--memory-swap={docker_args['memory_swap']}"]
    cmd.append(image)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def get_container_ip(container_id: str) -> str:
    """Return the container's Docker bridge IP address."""
    result = subprocess.run(
        [
            "docker", "inspect",
            "--format", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            container_id,
        ],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def is_container_running(container_id: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", container_id],
        capture_output=True, text=True,
    )
    return result.stdout.strip() == "true"


def is_oom_killed(container_id: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.OOMKilled}}", container_id],
        capture_output=True, text=True,
    )
    return result.stdout.strip() == "true"


def remove_container(container_id: str) -> None:
    subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def wait_healthy(container_id: str, url: str) -> bool:
    """Poll /health until the container responds OK or the wait window expires."""
    print("    Waiting for container to be ready", end="", flush=True)
    for _ in range(HEALTH_MAX_ATTEMPTS):
        if not is_container_running(container_id):
            print(" EXITED")
            logs = subprocess.run(
                ["docker", "logs", container_id],
                capture_output=True, text=True,
            )
            print(logs.stdout[-2000:] or logs.stderr[-2000:], file=sys.stderr)
            return False
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.ok:
                print(" OK")
                return True
        except requests.RequestException:
            pass
        print(".", end="", flush=True)
        time.sleep(HEALTH_POLL_INTERVAL)
    print(" TIMEOUT")
    return False


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(scenario: dict, cfg: dict, output_dir: Path, n_runs: int, timeout: int, index: int, total: int) -> None:
    name = scenario["name"]
    output = output_dir / scenario["output"]
    docker_args = scenario.get("docker", {})

    print()
    print("=" * 60)
    print(f"  Scenario {index}/{total}: {name}")
    print("=" * 60)

    # Start container — no host-port binding; connect via bridge IP instead
    container_id = docker_run(cfg["image"], docker_args)
    print(f"  Container : {container_id[:12]}")

    # Resolve bridge IP (required in devcontainer — localhost won't reach sibling containers)
    container_ip = get_container_ip(container_id)
    url = f"http://{container_ip}:{cfg['port']}"

    # Wait for healthy or record failure and skip
    if not wait_healthy(container_id, url):
        if is_oom_killed(container_id):
            reason = "OOM-killed — memory limit too low for this workload"
            error = {"error": "oom_killed", "message": "Container OOM-killed before becoming healthy"}
        else:
            reason = "container did not become healthy in time"
            error = {"error": "startup_timeout", "message": "Container did not become healthy within the wait window"}
        output.write_text(json.dumps(error))
        remove_container(container_id)
        print(f"  [ SKIPPED: {reason} ]")
        return

    # Run benchmark against the container
    run_benchmark(
        url=url,
        n_runs=n_runs,
        output_json=str(output),
        timeout=timeout,
        data_dir=cfg.get("data_dir", "/tmp/mnist_benchmark"),
    )

    # Stop and remove container
    remove_container(container_id)
    print(f"  [ Scenario {index}/{total} complete — results: {output} ]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all neuro-sim benchmark scenarios from benchmark.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/run_benchmarks.py\n"
            "  python scripts/run_benchmarks.py --config configs/benchmark.yaml\n"
            "  python scripts/run_benchmarks.py --n-runs 50\n"
            "  python scripts/run_benchmarks.py --output-dir results/run1\n"
            "  python scripts/run_benchmarks.py --timeout 120 --n-runs 20"
        ),
    )
    parser.add_argument(
        "--config", default="configs/benchmark.yaml",
        help="Path to benchmark YAML config (default: configs/benchmark.yaml)",
    )
    parser.add_argument(
        "--n-runs", type=int, dest="n_runs", default=None,
        help="Images per digit per scenario (overrides config)",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Per-request HTTP timeout in seconds (overrides config)",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Directory for result JSON files (overrides config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_runs = args.n_runs or cfg.get("n_runs", 20)
    timeout = args.timeout or cfg.get("timeout", 30)
    output_dir = Path(args.output_dir or cfg.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = cfg.get("scenarios", [])
    if not scenarios:
        print("No scenarios defined under 'benchmark.scenarios' in config.", file=sys.stderr)
        sys.exit(1)

    print()
    print("neuro-sim Benchmark Suite")
    print(f"Config     : {args.config}")
    print(f"Image      : {cfg['image']}")
    print(f"Port       : {cfg['port']}")
    print(f"Runs       : {n_runs} images per digit × 10 digits = {n_runs * 10} requests per scenario")
    print(f"Timeout    : {timeout}s per request")
    print(f"Output dir : {output_dir}/")
    print(f"Scenarios  : {len(scenarios)}")
    print("Network    : direct container IP (devcontainer-safe)")

    for i, scenario in enumerate(scenarios, start=1):
        run_scenario(scenario, cfg, output_dir, n_runs, timeout, index=i, total=len(scenarios))

    print()
    print(f"All scenarios complete. Results saved to {output_dir}/")
    for scenario in scenarios:
        print(f"  {output_dir}/{scenario['output']}")


if __name__ == "__main__":
    main()
