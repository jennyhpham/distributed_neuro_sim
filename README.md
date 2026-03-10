# NeuroSim - MNIST SNN Training and Inference

A production-grade application for training and running inference with Spiking Neural Networks (SNNs) on the MNIST dataset using BindsNET. Designed as the workload for a thesis on container-based orchestration of neuromorphic computing simulations at the virtual edge.

## Features

- **Modular Architecture**: Clean separation of training, evaluation, and data handling
- **Configuration Management**: YAML-based configuration system for easy experimentation
- **Model Persistence**: Save and load trained models and checkpoints
- **Training Pipeline**: Full training loop with progress tracking and checkpointing
- **Inference Pipeline**: Easy-to-use inference interface for predictions
- **HTTP Inference API**: FastAPI server with `/predict`, `/health`, and `/metrics` endpoints
- **Container Packaging**: Docker image with resource-constraint flags for edge simulation
- **K3d Orchestration**: Kubernetes manifests for deploying and scaling the SNN as a managed service
- **Benchmark Suite**: Latency, accuracy, and throughput measurement across deployment scenarios
- **Logging**: Comprehensive logging with file and console output

---

## Prerequisites

### Git LFS (required for model files)

Git LFS is required to download model files (`*.pt`). If you clone the repo and get an error like `invalid load key, 'v'` when loading a model, Git LFS was not set up and the file is just a pointer stub.

```bash
# Install git-lfs (once per machine)
sudo apt-get install git-lfs   # Ubuntu/Debian
# brew install git-lfs         # macOS

# Then initialize and pull the actual model files
git lfs install
git lfs pull
```

Or just run `./setup.sh`, which handles this automatically.

### For K3d orchestration (Windows)

| Tool | Purpose | Install |
| --- | --- | --- |
| Docker Desktop | Runs containers and K3d nodes | [docker.com](https://docker.com) |
| k3d | Creates a local K3s cluster using Docker | `choco install k3d` |
| kubectl | Sends commands to the cluster | `choco install kubernetes-cli` |

Make sure Docker Desktop is running before using k3d or kubectl.

---

## Installation

Install the package in development mode:

```bash
pip install -e .
```

This installs dependencies from `pyproject.toml` and registers the CLI entry points (`neuro-sim-train`, `neuro-sim-infer`).

**Note on BindsNET:** This project targets **BindsNET 0.2.x APIs**. Prefer `pip install -e .` over manual `pip install -r requirements.txt` to ensure dependency constraints are applied consistently.

**Note:** If you encounter `libGL.so.1: cannot open shared object file` errors in headless environments:

```bash
# Debian/Ubuntu (bookworm/trixie)
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
```

---

## Quick Start

### Training

Train with default configuration:

```bash
neuro-sim-train
```

Train with a custom config file:

```bash
neuro-sim-train --config configs/som_lm_snn.yaml
```

Train with CLI overrides:

```bash
neuro-sim-train --n-epochs 5 --batch-size 64 --gpu
```

Resume from a checkpoint:

```bash
neuro-sim-train --resume-from checkpoints/checkpoint_epoch_3.pt
```

### Inference (batch)

Run inference against the full MNIST test set:

```bash
neuro-sim-infer --model-path models/increasing_inhibition_network_400.pt
```

---

## Container Deployment

### Build

```bash
docker build -t neuro-sim:latest .
```

> **Note:** The image is large (~9 GB) due to the PyTorch CPU runtime. This is expected and is itself a finding for the thesis — it reflects the real overhead of packaging a framework-heavy neuromorphic workload.

The Dockerfile default CMD runs batch inference. When deployed via Kubernetes, the manifest overrides this with the FastAPI server command (see [K3d Orchestration](#k3d-orchestration)).

### Run standalone — baseline (no resource limits)

```bash
docker run --rm -p 8000:8000 \
  neuro-sim:latest \
  uvicorn neuro_sim.api.server:app --host 0.0.0.0 --port 8000
```

### Run standalone — resource-constrained (edge simulation)

Use `--cpus` and `--memory` to approximate edge hardware limits:

```bash
# Simulate a mid-tier edge node (Loihi host-class)
docker run --rm -p 8000:8000 \
  --cpus="0.5" --memory="512m" --memory-swap="512m" \
  neuro-sim:latest \
  uvicorn neuro_sim.api.server:app --host 0.0.0.0 --port 8000

# Simulate a constrained edge node (SpiNNaker-class)
docker run --rm -p 8000:8000 \
  --cpus="0.25" --memory="256m" --memory-swap="256m" \
  neuro-sim:latest \
  uvicorn neuro_sim.api.server:app --host 0.0.0.0 --port 8000
```

### API endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/predict` | POST | Upload a grayscale MNIST image, returns predicted digit and inference metadata |
| `/health` | GET | Liveness and readiness check — returns model info when ready |
| `/metrics` | GET | Prometheus-formatted metrics (request count, latency histograms) |

**Example `/predict` call:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/digit.png"
```

**Response:**

```json
{
  "digit": 7,
  "all_activity_digit": 7,
  "inference_time_ms": 243.5,
  "simulation_time_ms": 100,
  "spike_count": 842
}
```

**`/health` response:**

```json
{
  "status": "healthy",
  "model": "IncreasingInhibitionNetwork",
  "n_neurons": 400,
  "simulation_time_ms": 100,
  "device": "cpu"
}
```

---

## K3d Orchestration

This section covers deploying the SNN as a managed Kubernetes workload using K3d on a local Windows machine.

### 1. Create the cluster

```powershell
k3d cluster create neuro-sim --agents 2 --port "8080:80@loadbalancer"
```

This creates a 3-node virtual cluster: 1 server (control plane) + 2 agent (worker) nodes, all running as Docker containers.

Verify nodes are ready:

```powershell
kubectl get nodes
# NAME                     STATUS   ROLES                  AGE   VERSION
# k3d-neuro-sim-server-0   Ready    control-plane,master   ...
# k3d-neuro-sim-agent-0    Ready    <none>                 ...
# k3d-neuro-sim-agent-1    Ready    <none>                 ...
```

> **Windows note:** If `kubectl get nodes` returns a connection error, run:
> ```powershell
> kubectl config use-context k3d-neuro-sim
> kubectl config set-cluster k3d-neuro-sim --server=https://127.0.0.1:<port>
> ```
> where `<port>` is visible in `kubectl config view` under the k3d-neuro-sim cluster.

### 2. Build and import the image

K3d nodes cannot see your local Docker images by default. Import the image directly into the cluster after building:

```powershell
docker build -t neuro-sim:latest .
k3d image import neuro-sim:latest -c neuro-sim
```

Repeat the import step any time you rebuild the image.

### 3. Apply manifests

Manifests live in `k3d/`. Apply in this order:

```powershell
kubectl apply -f k3d/namespace.yaml   # creates the neuro-sim namespace
kubectl apply -f k3d/pvc.yaml         # creates persistent storage for logs
kubectl apply -f k3d/deployment.yaml  # deploys the FastAPI server
kubectl apply -f k3d/service.yaml     # exposes the server inside the cluster
```

Watch the pod come up:

```powershell
kubectl get pods -n neuro-sim -w
# Wait for READY = 1/1 (readiness probe must pass before traffic is accepted)
```

### 4. Access the API

The LoadBalancer external IP stays `<pending>` on Windows. Use port-forward instead:

```powershell
# Run in a dedicated terminal — keep it open while testing
kubectl port-forward svc/neuro-sim-svc 9090:80 -n neuro-sim
```

Then in a separate terminal:

```powershell
curl http://localhost:9090/health
```

### 5. Scale for the orchestration experiment

```powershell
kubectl scale deployment neuro-sim --replicas=3 -n neuro-sim
kubectl get pods -n neuro-sim -w   # wait for all 3 pods to show 1/1 Running
```

K8s automatically spreads pods across the two agent nodes and the Service load balances traffic across all running pods.

### Manifest reference

| File | Kind | Purpose |
| --- | --- | --- |
| `k3d/namespace.yaml` | Namespace | Isolates all neuro-sim resources |
| `k3d/pvc.yaml` | PersistentVolumeClaim | 1Gi storage for inference logs |
| `k3d/job.yaml` | Job | One-shot batch inference (Phase 1 baseline) |
| `k3d/deployment.yaml` | Deployment | Always-running API server with probes and resource limits |
| `k3d/service.yaml` | Service | LoadBalancer routing traffic to pods |

---

## Benchmarking

Two scripts are provided under `scripts/` to measure performance across deployment scenarios.

### `scripts/benchmark.py` — single scenario

Runs N inference requests against a live server and reports latency percentiles, accuracy, and throughput.

```bash
# Against a standalone Docker container
python scripts/benchmark.py \
  --url http://localhost:8000 \
  --n-runs 20 \
  --output-json results/run.json

# Against the K3d deployment (port-forward must be running)
python scripts/benchmark.py \
  --url http://host.docker.internal:9090 \
  --n-runs 20 \
  --output-json results/k8s_1pod.json
```

| Flag | Default | Description |
| --- | --- | --- |
| `--url` | `http://localhost:8000` | Base URL of the running API server |
| `--n-runs` | `10` | Number of inference requests per digit class (×10 digits = total requests) |
| `--output-json` | *(none)* | Optional path to write full results as JSON |

**Output includes:**
- Accuracy (proportion-weighted and all-activity)
- Latency: min, p50, p95, p99, max
- Neuromorphic floor latency (irreducible SNN simulation cost)
- Orchestration overhead (p50 minus neuromorphic floor)
- Throughput (requests/second)
- Per-digit spike counts

### `scripts/run_benchmarks.sh` — all three Docker scenarios automated

Builds and runs all three standalone Docker scenarios back-to-back, saves results to `results/`. Designed for the resource-constraint experiment (not K3d).

```bash
# Default: 20 runs per digit per scenario
./scripts/run_benchmarks.sh

# Custom run count
./scripts/run_benchmarks.sh --n-runs 50
```

Scenarios run in order:
1. **Baseline** — no resource limits → `results/baseline.json`
2. **0.5 CPU / 512 MB** — Loihi host-class → `results/constrained_0.5cpu.json`
3. **0.25 CPU / 256 MB** — SpiNNaker-class → `results/constrained_0.25cpu.json`

The script handles container startup, health-check polling, and teardown automatically between scenarios.

---

## Configuration

Configuration is managed via YAML files in `configs/`.

| File | Model | Use |
| --- | --- | --- |
| `configs/default.yaml` | DiehlAndCook2015 (100 neurons) | Baseline reference |
| `configs/som_lm_snn.yaml` | IncreasingInhibitionNetwork (400 neurons) | Primary thesis model |

Pass a config via `--config`:

```bash
neuro-sim-train --config configs/som_lm_snn.yaml
neuro-sim-infer --config configs/som_lm_snn.yaml
```

CLI arguments always take precedence over config file values.

### Available CLI Options

**`neuro-sim-train`**

| Flag | Description |
| --- | --- |
| `--config` | Path to YAML config file |
| `--batch-size` | Override `training.batch_size` |
| `--n-epochs` | Override `training.n_epochs` |
| `--gpu` / `--no-gpu` | Override `training.gpu` |
| `--checkpoint-dir` | Override `paths.checkpoint_dir` |
| `--resume-from` | Path to checkpoint to resume from |
| `--log-dir` | Override `paths.log_dir` |

**`neuro-sim-infer`**

| Flag | Description |
| --- | --- |
| `--config` | Path to YAML config file |
| `--model-path` | Path to trained model file |
| `--gpu` / `--no-gpu` | Override `inference.gpu` |

---

## Project Structure

```
distributed-neuro-sim/
├── src/
│   └── neuro_sim/                  # Main package
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       ├── cli.py                  # CLI entry points
│       ├── compat.py               # Compatibility patches
│       ├── api/
│       │   └── server.py           # FastAPI inference server
│       ├── data/
│       │   └── dataset.py          # Data loading utilities
│       ├── training/
│       │   ├── trainer.py          # Training logic
│       │   └── evaluator.py        # Inference/evaluation logic
│       ├── utils/
│       │   ├── logging.py          # Logging setup
│       │   └── model_persistence.py
│       └── models/                 # Model architecture definitions
├── k3d/
│   ├── namespace.yaml              # Kubernetes namespace
│   ├── pvc.yaml                    # Persistent volume claim for logs
│   ├── job.yaml                    # One-shot batch inference job
│   ├── deployment.yaml             # Always-running API deployment
│   └── service.yaml                # LoadBalancer service
├── scripts/
│   ├── benchmark.py                # Single-scenario latency/accuracy benchmark
│   └── run_benchmarks.sh           # Automated multi-scenario benchmark runner
├── results/                        # Benchmark output JSON files
├── configs/
│   ├── default.yaml                # DiehlAndCook2015 config
│   └── som_lm_snn.yaml             # IncreasingInhibitionNetwork config
├── Dockerfile                      # Container image definition
├── checkpoints/                    # Training checkpoints (gitignored)
├── models/                         # Saved models (git-lfs tracked)
├── logs/                           # Log files (gitignored)
├── data/                           # Datasets (gitignored)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Dependencies

**Core SNN simulation:**
- `torch>=2.0.0` - PyTorch
- `torchvision>=0.15.0` - Vision datasets and transforms
- `bindsnet>=0.2.7` - Spiking neural network simulation (0.2.x API)
- `numpy>=1.20.0` - Numerical computations
- `matplotlib>=3.3.0` - Plotting
- `tqdm>=4.65.0` - Progress bars
- `pyyaml>=6.0` - YAML config parsing

**API server:**
- `fastapi>=0.110.0` - HTTP API framework
- `uvicorn[standard]>=0.29.0` - ASGI server
- `python-multipart>=0.0.9` - File upload support
- `pillow>=10.0.0` - Image decoding for `/predict`
- `prometheus-fastapi-instrumentator>=7.0.0` - Prometheus metrics endpoint
