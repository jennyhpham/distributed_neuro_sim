#!/usr/bin/env bash
# Run all three benchmark scenarios against the neuro-sim container.
# Each scenario starts a container with different resource limits, waits
# for it to be healthy, runs the benchmark, then stops it.
#
# Usage:
#   ./scripts/run_benchmarks.sh
#   ./scripts/run_benchmarks.sh --n-runs 50
#   ./scripts/run_benchmarks.sh --image neuro-sim:v2 --port 9090 --output-dir results/run1
#   ./scripts/run_benchmarks.sh --timeout 120 --n-runs 20

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

IMAGE="neuro-sim:latest"
PORT=8000
N_RUNS=20
TIMEOUT=30
OUTPUT_DIR="results"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)       IMAGE="$2";      shift 2 ;;
    --image=*)     IMAGE="${1#*=}"; shift ;;
    --port)        PORT="$2";       shift 2 ;;
    --port=*)      PORT="${1#*=}";  shift ;;
    --n-runs)      N_RUNS="$2";     shift 2 ;;
    --n-runs=*)    N_RUNS="${1#*=}"; shift ;;
    --timeout)     TIMEOUT="$2";    shift 2 ;;
    --timeout=*)   TIMEOUT="${1#*=}"; shift ;;
    --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
    --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

URL="http://localhost:${PORT}"

mkdir -p "${OUTPUT_DIR}"

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
    --timeout "${TIMEOUT}" \
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
echo "Image      : ${IMAGE}"
echo "Port       : ${PORT}"
echo "Runs       : ${N_RUNS} per digit × 10 digits = $((N_RUNS * 10)) requests per scenario"
echo "Timeout    : ${TIMEOUT}s per request"
echo "Output dir : ${OUTPUT_DIR}/"

run_scenario \
  "Baseline — no resource limits" \
  "${OUTPUT_DIR}/baseline.json"

run_scenario \
  "Edge scenario 1 — SpiNNaker-class (0.25 CPU / 256MB)" \
  "${OUTPUT_DIR}/constrained_0.25cpu.json" \
  --cpus="0.25" --memory="256m" --memory-swap="256m"

run_scenario \
  "Edge scenario 2 — Loihi host-class (0.5 CPU / 512MB)" \
  "${OUTPUT_DIR}/constrained_0.5cpu.json" \
  --cpus="0.5" --memory="512m" --memory-swap="512m"

echo ""
echo "All scenarios complete. Results saved to ${OUTPUT_DIR}/"
echo "  ${OUTPUT_DIR}/baseline.json"
echo "  ${OUTPUT_DIR}/constrained_0.25cpu.json"
echo "  ${OUTPUT_DIR}/constrained_0.5cpu.json"
