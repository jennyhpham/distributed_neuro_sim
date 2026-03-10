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

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

wait_healthy() {
  local container_id="$1"
  local url="$2"
  echo -n "    Waiting for container to be ready"
  for i in $(seq 1 30); do
    # Fail fast if the container has already exited
    if ! docker inspect --format '{{.State.Running}}' "${container_id}" 2>/dev/null | grep -q "true"; then
      echo " EXITED"
      echo "  Container exited unexpectedly. Logs:" >&2
      docker logs "${container_id}" 2>&1 | tail -20 >&2
      return 1
    fi
    if curl -sf "${url}/health" > /dev/null 2>&1; then
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

  # Start container — no host-port binding; connect via container IP instead
  # (required when running inside a devcontainer where localhost != Docker host)
  # No --rm: we keep the container so we can capture logs even if it exits early
  CONTAINER_ID=$(docker run -d \
    "${docker_args[@]}" \
    "${IMAGE}")
  echo "  Container: ${CONTAINER_ID:0:12}"

  # Resolve container's bridge IP
  CONTAINER_IP=$(docker inspect --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "${CONTAINER_ID}")
  URL="http://${CONTAINER_IP}:${PORT}"

  # Wait for healthy or abort
  if ! wait_healthy "${CONTAINER_ID}" "${URL}"; then
    OOM=$(docker inspect --format '{{.State.OOMKilled}}' "${CONTAINER_ID}" 2>/dev/null || echo "false")
    if [[ "${OOM}" == "true" ]]; then
      echo "  SKIP: Container was OOM-killed — memory limit too low for this workload." >&2
      echo '{"error":"oom_killed","message":"Container OOM-killed before becoming healthy"}' > "${output}"
    else
      echo "  SKIP: Container did not become healthy in time." >&2
      echo '{"error":"startup_timeout","message":"Container did not become healthy within the wait window"}' > "${output}"
    fi
    docker rm -f "${CONTAINER_ID}" > /dev/null 2>&1 || true
    return 0
  fi

  # Run benchmark
  python scripts/benchmark.py \
    --url "${URL}" \
    --n-runs "${N_RUNS}" \
    --timeout "${TIMEOUT}" \
    --output-json "${output}"

  # Stop and remove container
  echo "  Stopping container..."
  docker rm -f "${CONTAINER_ID}" > /dev/null 2>&1 || true
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
echo "Network    : direct container IP (devcontainer-safe)"

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
