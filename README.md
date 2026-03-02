# NeuroSim - MNIST SNN Training and Inference

A production-grade application for training and running inference with Spiking Neural Networks (SNNs) on the MNIST dataset using BindsNET.

## Features

- **Modular Architecture**: Clean separation of training, evaluation, and data handling
- **Configuration Management**: YAML-based configuration system for easy experimentation
- **Model Persistence**: Save and load trained models and checkpoints
- **Training Pipeline**: Full training loop with progress tracking and checkpointing
- **Inference Pipeline**: Easy-to-use inference interface for predictions
- **Logging**: Comprehensive logging with file and console output

## Prerequisites

**Git LFS** is required to download model files (`*.pt`). If you clone the repo and get an error like `invalid load key, 'v'` when loading a model, it means Git LFS was not set up and the file is just a pointer stub.

```bash
# Install git-lfs (once per machine)
sudo apt-get install git-lfs   # Ubuntu/Debian
# brew install git-lfs         # macOS

# Then initialize and pull the actual model files
git lfs install
git lfs pull
```

Or just run `./setup.sh`, which handles this automatically.

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

# Or use a headless OpenCV build
pip uninstall opencv-python opencv-python-headless -y && pip install opencv-python-headless
```

## Quick Start

### Training

Train with default configuration:

```bash
neuro-sim-train
```

Train with a custom config file:

```bash
neuro-sim-train --config configs/default.yaml
```

Train with CLI overrides:

```bash
neuro-sim-train --n-epochs 5 --batch-size 64 --gpu
```

Resume from a checkpoint:

```bash
neuro-sim-train --resume-from checkpoints/checkpoint_epoch_3.pt
```

### Inference

Run inference with the default trained model:

```bash
neuro-sim-infer
```

Run inference with a specific model:

```bash
neuro-sim-infer --model-path models/mnist_snn_model.pt
```

## Configuration

Configuration is managed via YAML files. Copy the default config as a starting point:

```bash
cp configs/default.yaml configs/my_experiment.yaml
```

Then pass it via `--config`:

```bash
neuro-sim-train --config configs/my_experiment.yaml
neuro-sim-infer --config configs/my_experiment.yaml
```

CLI arguments always take precedence over config file values:

```bash
neuro-sim-train --config configs/default.yaml --n-epochs 20 --batch-size 128 --gpu
```

### Config File Structure

```yaml
model:
  name: "DiehlAndCook2015"
  n_neurons: 100
  n_inpt: 784
  exc: 22.5
  inh: 120.0
  theta_plus: 0.05
  dt: 1.0
  norm: 78.4
  nu: [0.0001, 0.01]
  inpt_shape: [1, 28, 28]
  w_dtype: "float32"
  sparse: false

training:
  batch_size: 32
  n_epochs: 5
  n_train: 60000
  n_updates: 10
  time: 100
  intensity: 128.0
  progress_interval: 10
  checkpoint_interval: 1
  seed: 42
  gpu: true

inference:
  batch_size: 32
  n_test: 10000
  time: 100
  intensity: 128.0
  gpu: true

data:
  dataset: "MNIST"
  root: null
  n_workers: 4

paths:
  checkpoint_dir: "checkpoints"
  model_dir: "models"
  log_dir: "logs"
  data_dir: "data"
```

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

## Project Structure

```
distributed-neuro-sim/
├── src/
│   └── neuro_sim/                  # Main package
│       ├── __init__.py
│       ├── config.py                # Configuration management
│       ├── cli.py                   # CLI entry points
│       ├── compat.py                # Compatibility patches
│       ├── main.py                  # Main entry point
│       ├── data/
│       │   └── dataset.py           # Data loading utilities
│       ├── training/
│       │   ├── trainer.py           # Training logic
│       │   └── evaluator.py         # Inference/evaluation logic
│       ├── utils/
│       │   ├── logging.py           # Logging setup
│       │   └── model_persistence.py # Model save/load
│       └── models/
├── scripts/
│   └── batch_eth_mnist.py          # Original BindsNET reference script
├── configs/
│   └── default.yaml
├── docs/
│   └── strategy.md
├── checkpoints/                    # Training checkpoints (gitignored)
├── models/                         # Saved models (gitignored)
├── logs/                           # Log files (gitignored)
├── data/                           # Datasets (gitignored)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Dependencies

- `torch>=2.0.0` - PyTorch
- `torchvision>=0.15.0` - Vision datasets and transforms
- `bindsnet>=0.2.7` - Spiking neural network simulation (0.2.x API)
- `numpy>=1.20.0` - Numerical computations
- `matplotlib>=3.3.0` - Plotting
- `tqdm>=4.65.0` - Progress bars
- `pyyaml>=6.0` - YAML config parsing
