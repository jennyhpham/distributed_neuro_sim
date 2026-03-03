#!/usr/bin/env bash
# Run all three benchmark scenarios against the neuro-sim container.
# Each scenario starts a container with different resource limits, waits
# for it to be healthy, runs the benchmark, then stops it.
#
# Usage:
#   ./scripts/run_benchmarks.sh
#   ./scripts/run_benchmarks.sh --n-runs 50   # more requests per digit

set -euo pipefail

IMAGE="neuro-sim:latest"
PORT=8000
URL="http://localhost:${PORT}"
N_RUNS=${2:-20}  # default 20 runs per digit

# Parse --n-runs flag
for i in "$@"; do
  case $i in
    --n-runs=*) N_RUNS="${i#*=}" ;;
    --n-runs)   N_RUNS="${2}"; shift ;;
  esac
done

mkdir -p results

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

wait_healthy() {
  echo -n "    Waiting for container to be ready"
  for i in $(seq 1 30); do
    if curl -sf "${URL}/health" > /dev/null 2>&1; then
      echo " OK"
      return 0
    fi
    echo -n "."
    sleep 2
  done
  echo " TIMEOUT"
  return 1
}

run_scenario() {
  local name="$1"
  local output="$2"
  shift 2
  local docker_args=("$@")

  echo ""
  echo "============================================================"
  echo "  Scenario: ${name}"
  echo "============================================================"

  # Start container in background
  CONTAINER_ID=$(docker run --rm -d \
    -p "${PORT}:${PORT}" \
    "${docker_args[@]}" \
    "${IMAGE}")
  echo "  Container: ${CONTAINER_ID:0:12}"

  # Wait for healthy or abort
  if ! wait_healthy; then
    echo "  ERROR: Container did not become healthy in time."
    docker stop "${CONTAINER_ID}" > /dev/null 2>&1 || true
    exit 1
  fi

  # Run benchmark
  python scripts/benchmark.py \
    --url "${URL}" \
    --n-runs "${N_RUNS}" \
    --output-json "${output}"

  # Stop container
  echo "  Stopping container..."
  docker stop "${CONTAINER_ID}" > /dev/null 2>&1 || true
  # Brief pause to ensure port is freed before next scenario
  sleep 2
}

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

echo ""
echo "neuro-sim Benchmark Suite"
echo "Image  : ${IMAGE}"
echo "Runs   : ${N_RUNS} per digit × 10 digits = $((N_RUNS * 10)) requests per scenario"
echo "Output : results/"

run_scenario \
  "Baseline — no resource limits" \
  "results/baseline.json"

run_scenario \
  "Edge scenario 1 — SpiNNaker-class (0.25 CPU / 256MB)" \
  "results/constrained_0.25cpu.json" \
  --cpus="0.25" --memory="256m" --memory-swap="256m"

run_scenario \
  "Edge scenario 2 — Loihi host-class (0.5 CPU / 512MB)" \
  "results/constrained_0.5cpu.json" \
  --cpus="0.5" --memory="512m" --memory-swap="512m"

echo ""
echo "All scenarios complete. Results saved to results/"
echo "  results/baseline.json"
echo "  results/constrained_0.25cpu.json"
echo "  results/constrained_0.5cpu.json"
