# NeuroSim - Production-Grade MNIST SNN Training and Inference

A production-grade machine learning application for training and running inference with Spiking Neural Networks (SNNs) on the MNIST dataset using BindsNET.

## Features

- **Modular Architecture**: Clean separation of training, evaluation, and data handling
- **Configuration Management**: YAML-based configuration system for easy experimentation
- **Model Persistence**: Save and load trained models and checkpoints
- **Training Pipeline**: Full training loop with progress tracking and checkpointing
- **Inference Pipeline**: Easy-to-use inference interface for predictions
- **Logging**: Comprehensive logging with file and console output
- **CLI Interface**: Command-line tools for training and inference
- **Error Handling**: Robust error handling and validation

## Installation

### Install Package

First, install the package in development mode (this makes the `neuro_sim` module available):

```bash
pip install -e .
```

This will install dependencies declared in `pyproject.toml` and make the CLI commands available.

### Manual Installation

Alternatively, install dependencies manually:

```bash
pip install -r requirements.txt
```

**Note on BindsNET version:** This project targets **BindsNET 0.2.x APIs** (see `pyproject.toml`), and the `requirements.txt` installs BindsNET from GitHub. If you run into version/API mismatches, prefer installing with `pip install -e .` so the dependency constraints are applied consistently.

**Note:** If you encounter `libGL.so.1: cannot open shared object file` errors (common in headless environments), OpenCV requires system libraries. You can either:

1. Install system dependencies (Debian/Ubuntu):
   ```bash
   # For newer Debian (trixie/bookworm) - use libgl1
   sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
   
   # For older Debian/Ubuntu - use libgl1-mesa-glx
   # sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
   ```

2. Or use a headless OpenCV build (if your environment doesn't need GUI):
   ```bash
   pip uninstall opencv-python opencv-python-headless -y
   pip install opencv-python-headless
   ```

### Verify Installation

After installation, verify the CLI works:

```bash
python -m neuro_sim.cli --help
```

## Quick Start

### Training a Model

Train a model with default configuration:

```bash
python -m neuro_sim.cli train
```

Train with a custom config file:

```bash
python -m neuro_sim.cli train --config configs/default.yaml
```

Train with command-line overrides:

```bash
python -m neuro_sim.cli train --n-epochs 5 --batch-size 64 --gpu
```

Resume training from a checkpoint:

```bash
python -m neuro_sim.cli train --resume-from checkpoints/checkpoint_epoch_3.pt
```

### Running Inference

Run inference with the trained model:

```bash
python -m neuro_sim.cli infer
```

Run inference with a specific model:

```bash
python -m neuro_sim.cli infer --model-path models/mnist_snn_model.pt
```

### Using the CLI Entry Points

After installation, you can also use the entry points:

```bash
neuro-sim-train --n-epochs 5 --batch-size 64
neuro-sim-infer --model-path models/mnist_snn_model.pt
neuro-sim --help
```

## Configuration

Configuration is managed via YAML files, allowing you to easily experiment with different hyperparameters and settings without modifying code.

### Using YAML Config Files

#### Basic Usage

Train with a custom config file:

```bash
python -m neuro_sim.cli train --config configs/default.yaml
```

Or using the entry point:

```bash
neuro-sim-train --config configs/default.yaml
```

#### Creating Custom Config Files

1. **Copy the default config** as a starting point:
   ```bash
   cp configs/default.yaml configs/my_experiment.yaml
   ```

2. **Edit the config file** to customize parameters. See `configs/default.yaml` for the complete structure:

```yaml
# Example: configs/my_experiment.yaml
model:
  name: "DiehlAndCook2015"
  n_neurons: 200          # Increase neurons for better capacity
  n_inpt: 784
  exc: 22.5
  inh: 120.0
  theta_plus: 0.05
  dt: 1.0
  norm: 78.4
  nu: [0.0001, 0.01]      # Learning rates [min, max]
  inpt_shape: [1, 28, 28]
  w_dtype: "float32"
  sparse: false

training:
  batch_size: 64          # Larger batch size
  n_epochs: 10             # More epochs
  n_train: 60000
  n_updates: 10
  time: 100
  intensity: 128.0
  progress_interval: 10
  checkpoint_interval: 2   # Save checkpoint every 2 epochs
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
  root: null               # Uses bindsnet default
  n_workers: 8

paths:
  checkpoint_dir: "checkpoints"
  model_dir: "models"
  log_dir: "logs"
  data_dir: "data"
```

3. **Train with your custom config**:
   ```bash
   python -m neuro_sim.cli train --config configs/my_experiment.yaml
   ```

#### Overriding Config Values with CLI Arguments

CLI arguments take precedence over config file values. This allows you to quickly experiment without editing files:

```bash
# Use config file but override specific values
python -m neuro_sim.cli train \
  --config configs/default.yaml \
  --n-epochs 20 \
  --batch-size 128 \
  --gpu
```

Available CLI overrides for training:
- `--config`: Path to config file
- `--batch-size`: Override `training.batch_size`
- `--n-epochs`: Override `training.n_epochs`
- `--gpu` / `--no-gpu`: Override `training.gpu`
- `--checkpoint-dir`: Override `paths.checkpoint_dir`
- `--resume-from`: Path to checkpoint to resume from
- `--log-dir`: Override `paths.log_dir`

#### Configuration Sections

**Model Parameters** (`model:`):
- `n_neurons`: Number of excitatory neurons (default: 100)
- `exc`: Excitatory weight (default: 22.5)
- `inh`: Inhibitory weight (default: 120.0)
- `theta_plus`: Threshold increment for STDP (default: 0.05)
- `nu`: Learning rates `[min, max]` (default: [0.0001, 0.01])
- `dt`: Time step in ms (default: 1.0)
- `norm`: Input normalization factor (default: 78.4)

**Training Parameters** (`training:`):
- `batch_size`: Batch size (default: 32)
- `n_epochs`: Number of training epochs (default: 5)
- `n_train`: Number of training samples (default: 60000)
- `n_updates`: Number of assignment updates per epoch (default: 10)
- `time`: Simulation time in ms (default: 100)
- `intensity`: Input intensity scaling (default: 128.0)
- `checkpoint_interval`: Save checkpoint every N epochs (default: 1)
- `gpu`: Use GPU if available (default: true)

**Inference Parameters** (`inference:`):
- `batch_size`: Batch size for evaluation (default: 32)
- `n_test`: Number of test samples (default: 10000)
- `time`: Simulation time in ms (default: 100)
- `intensity`: Input intensity scaling (default: 128.0)

**Paths** (`paths:`):
- `checkpoint_dir`: Directory for training checkpoints
- `model_dir`: Directory for saved models
- `log_dir`: Directory for log files
- `data_dir`: Directory for datasets

## Project Structure

```
distributed-neuro-sim/
├── src/
│   └── neuro_sim/                  # Main package
│       ├── __init__.py
│       ├── config.py                # Configuration management
│       ├── cli.py                   # Command-line interface
│       ├── compat.py                # Compatibility patches
│       ├── main.py                  # Main entry point
│       ├── data/                    # Data loading utilities
│       │   ├── __init__.py
│       │   └── dataset.py
│       ├── training/                # Training and evaluation
│       │   ├── __init__.py
│       │   ├── trainer.py           # Training logic
│       │   └── evaluator.py         # Inference/evaluation logic
│       ├── utils/                   # Utilities
│       │   ├── __init__.py
│       │   ├── logging.py           # Logging setup
│       │   └── model_persistence.py # Model save/load
│       └── models/                  # Model definitions
│           └── __init__.py
├── scripts/                        # Utility scripts
│   └── batch_eth_mnist.py          # Original BindsNET reference script
├── configs/                        # Configuration files
│   └── default.yaml
├── docs/                           # Documentation
│   └── strategy.md
├── checkpoints/                    # Training checkpoints (created, gitignored)
├── models/                         # Saved models (created, gitignored)
├── logs/                           # Log files (created, gitignored)
├── data/                           # Data directory (created, gitignored)
├── requirements.txt
├── pyproject.toml
├── README.md
└── TROUBLESHOOTING.md
```

## Python API

### Training

```python
from neuro_sim.config import Config
from neuro_sim.training import Trainer
from neuro_sim.utils.logging import setup_logger

# Setup
logger = setup_logger("train")
config = Config(config_path="configs/default.yaml")
trainer = Trainer(config)

# Train
metrics = trainer.train()
print(f"Training metrics: {metrics}")
```

### Inference

```python
from neuro_sim.config import Config
from neuro_sim.training import Evaluator
from neuro_sim.utils.logging import setup_logger

# Setup
logger = setup_logger("infer")
config = Config(config_path="configs/default.yaml")
evaluator = Evaluator(config, model_path="models/mnist_snn_model.pt")

# Evaluate
metrics = evaluator.evaluate()
print(f"Inference metrics: {metrics}")

# Single batch prediction
predictions_all, predictions_prop = evaluator.predict(batch)
```

## Model Persistence

Models are automatically saved after training to `models/mnist_snn_model.pt`. Checkpoints are saved during training to `checkpoints/checkpoint_epoch_N.pt`.

To manually save/load:

```python
from neuro_sim.utils.model_persistence import ModelPersistence

# Save
ModelPersistence.save_model(
    model_path="models/my_model.pt",
    network=network,
    assignments=assignments,
    proportions=proportions,
    rates=rates,
    config=config.to_dict(),
)

# Load
model_data = ModelPersistence.load_model("models/my_model.pt", device="cuda")
```

## Logging

Logs are written to both console and files (in `logs/` directory by default). Log levels can be configured:

```python
from neuro_sim.utils.logging import setup_logger
import logging

logger = setup_logger(
    name="my_logger",
    log_dir="logs",
    log_level=logging.INFO,
    console=True,
)
```

## Dependencies

- `torch>=2.0.0` - PyTorch for deep learning
- `torchvision>=0.15.0` - Vision datasets and transforms
- `bindsnet>=0.2.7` - Spiking neural network simulation (0.2.x API)
- `numpy>=1.20.0` - Numerical computations
- `matplotlib>=3.3.0` - Plotting (optional)
- `tqdm>=4.65.0` - Progress bars
- `pyyaml>=6.0` - Configuration file parsing

## Original Script

The original script from BindsNET (`batch_eth_mnist.py`) is preserved in `scripts/batch_eth_mnist.py` for reference.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
