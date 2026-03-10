# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Context

**Thesis:** Evaluating Container-Based Orchestration for Neuromorphic Computing Simulations at the Virtual Edge

**Research question:** "To what extent can simulated neuromorphic-inspired workloads (SNNs) be orchestrated and managed in a virtual edge environment using modern container-based technologies?"

**Primary focus:** Investigate how SNN workloads can be orchestrated and managed using K3d and MicroK8s in a fully virtual setup (Windows 11/WSL2 — no physical hardware).

**Secondary focus:** Explore how neuromorphic hardware characteristics can be approximated in software, and assess where those abstractions conflict with cloud-native orchestration assumptions.

### Critical Distinction: Simulation vs. Emulation

NEST, Brian2, SpikingJelly, and BindsNET **only simulate SNNs** — they do not emulate neuromorphic hardware (e.g., Intel Loihi, SpiNNaker). The chosen approach for this project is:

> **Simulate SNNs → deploy on virtual edge** (not hardware emulation)

The goal is to simulate SNNs in a way that approximates real neuromorphic hardware behavior as closely as possible within software constraints, then evaluate how well container orchestration platforms handle these workloads.

### What Is Being Evaluated

**A. Simulation environments** — which SNN simulators can run in Docker, how flexible they are, how they perform under load.

**B. Virtual edge orchestration** — can SNN workloads be distributed across multiple virtual "edge" nodes? Which platform (K3d vs MicroK8s) works best? Does the workload scale or break when distributed?

**C. Benchmarking metrics** — CPU usage, memory footprint, execution latency, deployment complexity, stability under load. Compared across K3d and MicroK8s.

### Infrastructure Stack

All virtual, no hardware:
- **Container runtime:** Docker
- **Edge cluster option 1:** K3d (K3s in Docker containers)
- **Edge cluster option 2:** MicroK8s
- **SNN simulator:** BindsNET (PyTorch-based), backed by Brian2
- **Monitoring:** Prometheus + Kubernetes metrics (planned)

## Commands

```bash
# Install (required before any commands work)
pip install -e .

# API server (inference endpoint)
uvicorn neuro_sim.api.server:app --host 0.0.0.0 --port 8000 --workers 1

# Build production container (run from host, not devcontainer)
docker build -t neuro-sim:latest .

# Run container — no limits (baseline)
docker run --rm -p 8000:8000 neuro-sim:latest

# Run container — resource-constrained (edge simulation)
docker run --rm -p 8000:8000 --cpus="0.5" --memory="512m" --memory-swap="512m" neuro-sim:latest
docker run --rm -p 8000:8000 --cpus="0.25" --memory="256m" --memory-swap="256m" neuro-sim:latest

# Benchmark — single scenario
python scripts/benchmark.py --url http://localhost:8000 --n-runs 20 --output-json results/run.json

# Benchmark — all three scenarios automated (requires docker)
./scripts/run_benchmarks.sh
./scripts/run_benchmarks.sh --n-runs 50

# Training
neuro-sim-train                                          # default config
neuro-sim-train --config configs/som_lm_snn.yaml        # SOM-LM-SNN model
neuro-sim-train --n-epochs 5 --batch-size 64 --gpu
neuro-sim-train --resume-from checkpoints/checkpoint_epoch_3.pt

# Inference
neuro-sim-infer
neuro-sim-infer --model-path models/mnist_snn_model.pt

# Alternative entry point
python -m neuro_sim.cli train --config configs/default.yaml
python -m neuro_sim.cli infer --model-path models/mnist_snn_model.pt

# Git LFS (required for model files)
git lfs install && git lfs pull
# Or just run:
./setup.sh
```

There are no automated tests in this project currently.

## Architecture

The package lives in `src/neuro_sim/` and is installed as `distributed-neuro-sim`.

### Key Layers

**Config** (`config.py`): YAML-based config with deep merging. CLI args always override config file values. Access via `config.model`, `config.training`, `config.inference`, `config.data`, `config.paths`. Set programmatically with `config.set("training.n_epochs", 10)`.

**Models** (`models/`): Registry-based factory pattern. `ModelFactory` maps string names to `BaseModelWrapper` subclasses. Two models are registered:
- `"DiehlAndCook2015"` → `DiehlCookModel` — standard STDP-based SNN
- `"IncreasingInhibitionNetwork"` → `IncreasingInhibitionModel` — SOM-LM-SNN variant with growing lateral inhibition; **requires `batch_size=1`**

**Training Strategies** (`training/strategies/`): `TrainingStrategy` ABC with hooks (`pre_sample_hook`, `post_sample_hook`, `periodic_update_hook`, `on_epoch_start`, `on_epoch_end`). Each model has a corresponding strategy that encapsulates model-specific behavior, keeping `Trainer` model-agnostic.

**Trainer / Evaluator** (`training/trainer.py`, `training/evaluator.py`): `Trainer` drives the training loop and delegates model-specific behavior to a strategy. `Evaluator` loads a saved model and runs inference, returning two accuracy metrics: `all` (winner-takes-all spike count) and `proportion` (weighted by learned neuron-class proportions).

**Compatibility** (`compat.py`): Patches BindsNET 0.2.x for device compatibility. Must be imported before using BindsNET. Fixed issue: `RuntimeError: indices should be either on cpu or on the same device`.

**Model Persistence** (`utils/model_persistence.py`): Checkpoints save weights + assignments + proportions + rates (not epoch counter or optimizer state). Loaded back to specified device.

### Data Flow

Images → Poisson spike encoding → BindsNET `DiehlAndCook2015` / `IncreasingInhibitionNetwork` simulation (`time` ms) → spike counts from excitatory layer (Ae/Y) → `assign_labels` maps neurons to MNIST classes → accuracy via all-activity or proportion-weighting.

### Config Files

- `configs/default.yaml` — DiehlAndCook2015, `n_neurons=100`, `batch_size=32`
- `configs/som_lm_snn.yaml` — IncreasingInhibitionNetwork, `n_neurons=400`, **`batch_size=1` required**, lower intensity (64 vs 128)

### Dependencies

Targets **BindsNET 0.2.x API**. Use `pip install -e .` (not `pip install -r requirements.txt`) to ensure consistent dependency resolution. Headless environments need `libgl1 libglib2.0-0` or `opencv-python-headless`.

### Devcontainer

`Dockerfile` uses `nvidia/cuda:12.6.0-runtime-ubuntu24.04`. Git LFS is initialized in the devcontainer `postCreateCommand` (`setup.sh`).
