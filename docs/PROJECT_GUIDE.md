# NeuroSim Project Guide

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Installation and Setup](#installation-and-setup)
4. [Configuration System](#configuration-system)
5. [Training Workflow](#training-workflow)
6. [Inference and Evaluation](#inference-and-evaluation)
7. [Checkpointing and Resuming](#checkpointing-and-resuming)
8. [Model Architecture](#model-architecture)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)
11. [Examples](#examples)

---

## Project Overview

**NeuroSim** is a production-grade machine learning application for training and running inference with Spiking Neural Networks (SNNs) on the MNIST dataset using BindsNET. The project provides a clean, modular architecture that separates concerns between training, evaluation, data handling, and configuration management.

### Key Features

- **Modular Architecture**: Clean separation of training, evaluation, and data handling
- **Configuration Management**: YAML-based configuration system for easy experimentation
- **Model Persistence**: Save and load trained models and checkpoints
- **Training Pipeline**: Full training loop with progress tracking and checkpointing
- **Inference Pipeline**: Easy-to-use inference interface for predictions
- **Logging**: Comprehensive logging with file and console output
- **CLI Interface**: Command-line tools for training and inference
- **Error Handling**: Robust error handling and validation
- **Device Management**: Automatic GPU/CPU device handling with proper tensor placement

### Technology Stack

- **PyTorch**: Deep learning framework
- **BindsNET 0.2.x**: Spiking neural network simulation library
- **Brian2**: Underlying spiking neuron simulation engine
- **NumPy**: Numerical computations
- **PyYAML**: Configuration file parsing

---

## Architecture

### Project Structure

```
distributed-neuro-sim/
├── src/
│   └── neuro_sim/                  # Main package
│       ├── __init__.py
│       ├── config.py                # Configuration management
│       ├── cli.py                   # Command-line interface
│       ├── compat.py                # Compatibility patches for BindsNET
│       ├── main.py                  # Main entry point
│       ├── data/                    # Data loading utilities
│       │   ├── __init__.py
│       │   └── dataset.py           # Dataset and DataLoader wrappers
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
│   └── PROJECT_GUIDE.md            # This file
├── checkpoints/                    # Training checkpoints (created, gitignored)
├── models/                         # Saved models (created, gitignored)
├── logs/                           # Log files (created, gitignored)
├── data/                           # Data directory (created, gitignored)
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Component Overview

#### 1. **Config Module** (`config.py`)
- Manages configuration loading from YAML files
- Provides default configurations
- Validates configuration values
- Handles path resolution and directory creation

#### 2. **CLI Module** (`cli.py`)
- Command-line interface for training and inference
- Parses arguments and delegates to Trainer/Evaluator
- Handles logging setup

#### 3. **Training Module** (`training/trainer.py`)
- Implements the training loop
- Manages network state, assignments, and accuracy tracking
- Handles checkpoint saving
- Supports resuming from checkpoints

#### 4. **Evaluation Module** (`training/evaluator.py`)
- Loads trained models
- Runs inference on test data
- Computes accuracy metrics
- Provides prediction interface

#### 5. **Data Module** (`data/dataset.py`)
- Wraps BindsNET dataset loading
- Provides DataLoader utilities
- Handles encoding and preprocessing

#### 6. **Model Persistence** (`utils/model_persistence.py`)
- Saves and loads model checkpoints
- Handles weight extraction from BindsNET networks
- Manages device placement for tensors

#### 7. **Compatibility Module** (`compat.py`)
- Patches BindsNET functions for device compatibility
- Fixes known issues with BindsNET 0.2.x API

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)
- Linux/macOS/Windows with WSL2

### Installation Steps

#### 1. Install Package

```bash
pip install -e .
```

This installs the package in development mode and makes CLI commands available.

#### 2. Verify Installation

```bash
python -m neuro_sim.cli --help
```

Or test the entry points:

```bash
neuro-sim --help
neuro-sim-train --help
neuro-sim-infer --help
```

### System Dependencies

If you encounter `libGL.so.1` errors (common in headless environments):

**Debian/Ubuntu (newer versions):**
```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
```

**Debian/Ubuntu (older versions):**
```bash
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

**Or use headless OpenCV:**
```bash
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python-headless
```

### Dependencies

The project requires:
- `torch>=2.0.0` - PyTorch for deep learning
- `torchvision>=0.15.0` - Vision datasets and transforms
- `bindsnet>=0.2.7` - Spiking neural network simulation (0.2.x API)
- `brian2>=2.5.0` - Underlying spiking neuron simulator
- `numpy>=1.20.0` - Numerical computations
- `matplotlib>=3.3.0` - Plotting (optional)
- `tqdm>=4.65.0` - Progress bars
- `pyyaml>=6.0` - Configuration file parsing

**Note:** BindsNET is installed from GitHub in `requirements.txt`, but `pyproject.toml` specifies `bindsnet>=0.2.7`. Use `pip install -e .` for consistent dependency resolution.

---

## Configuration System

### Overview

The configuration system uses YAML files to manage all hyperparameters and settings. Configurations are hierarchical and support deep merging, allowing you to override specific values without rewriting entire files.

### Configuration Structure

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
  seed: 0
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
  n_workers: 8

paths:
  checkpoint_dir: "checkpoints"
  model_dir: "models"
  log_dir: "logs"
  data_dir: "data"
```

### Configuration Parameters

#### Model Parameters (`model:`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | "DiehlAndCook2015" | Model architecture name |
| `n_neurons` | int | 100 | Number of excitatory neurons |
| `n_inpt` | int | 784 | Number of input neurons (28×28 for MNIST) |
| `exc` | float | 22.5 | Excitatory connection strength |
| `inh` | float | 120.0 | Inhibitory connection strength |
| `theta_plus` | float | 0.05 | STDP threshold increment |
| `dt` | float | 1.0 | Simulation time step (ms) |
| `norm` | float | 78.4 | Input normalization factor |
| `nu` | list[float] | [0.0001, 0.01] | Learning rates [min, max] |
| `inpt_shape` | list[int] | [1, 28, 28] | Input shape (channels, height, width) |
| `w_dtype` | string | "float32" | Weight data type |
| `sparse` | bool | false | Use sparse tensors |

#### Training Parameters (`training:`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 32 | Batch size for training |
| `n_epochs` | int | 5 | Number of training epochs |
| `n_train` | int | 60000 | Number of training samples |
| `n_updates` | int | 10 | Number of assignment updates per epoch |
| `time` | int | 100 | Simulation time window (ms) |
| `intensity` | float | 128.0 | Input intensity scaling |
| `progress_interval` | int | 10 | Log progress every N epochs |
| `checkpoint_interval` | int | 1 | Save checkpoint every N epochs |
| `seed` | int | 0 | Random seed for reproducibility |
| `gpu` | bool | true | Use GPU if available |

#### Inference Parameters (`inference:`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 32 | Batch size for evaluation |
| `n_test` | int | 10000 | Number of test samples |
| `time` | int | 100 | Simulation time window (ms) |
| `intensity` | float | 128.0 | Input intensity scaling |
| `gpu` | bool | true | Use GPU if available |

#### Data Parameters (`data:`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | string | "MNIST" | Dataset name |
| `root` | string/null | null | Dataset root directory (null = BindsNET default) |
| `n_workers` | int | 8 | Number of data loading workers |

#### Path Parameters (`paths:`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_dir` | string | "checkpoints" | Directory for training checkpoints |
| `model_dir` | string | "models" | Directory for saved models |
| `log_dir` | string | "logs" | Directory for log files |
| `data_dir` | string | "data" | Directory for datasets |

### Using Configuration Files

#### 1. Create Custom Config

```bash
cp configs/default.yaml configs/my_experiment.yaml
```

Edit `configs/my_experiment.yaml` to customize parameters.

#### 2. Use Config File

```bash
python -m neuro_sim.cli train --config configs/my_experiment.yaml
```

#### 3. Override with CLI Arguments

CLI arguments take precedence over config file values:

```bash
python -m neuro_sim.cli train \
  --config configs/default.yaml \
  --n-epochs 20 \
  --batch-size 128 \
  --gpu
```

### Programmatic Configuration

```python
from neuro_sim.config import Config

# Load from file
config = Config(config_path="configs/default.yaml")

# Override values
config.set("training.n_epochs", 10)
config.set("model.n_neurons", 200)

# Access values
n_neurons = config.model["n_neurons"]
batch_size = config.training["batch_size"]

# Get full config dict
config_dict = config.to_dict()
```

---

## Training Workflow

### Overview

The training workflow consists of:
1. **Initialization**: Build network, setup monitors, initialize assignments
2. **Training Loop**: Process batches, update assignments, track accuracy
3. **Checkpointing**: Save model state periodically
4. **Final Save**: Save final trained model

### Training Process

#### 1. Network Initialization

- Builds `DiehlAndCook2015` network from config
- Moves network to GPU/CPU based on config
- Sets up voltage and spike monitors
- Initializes spike recording buffers

#### 2. Training Loop

For each epoch:
- Loads dataset with Poisson encoding
- Processes batches:
  - Encodes images to spike trains
  - Runs network simulation
  - Records spikes
  - Updates neuron assignments periodically
- Saves checkpoint (if `checkpoint_interval` reached)
- Logs progress and accuracy

#### 3. Assignment Updates

During training, neuron assignments are updated periodically:
- Every `n_train / (batch_size * n_updates)` batches
- Uses `assign_labels` to map neurons to classes
- Tracks two accuracy metrics:
  - **All Activity**: Highest spike count wins
  - **Proportion Weighting**: Weighted by learned proportions

#### 4. Checkpoint Saving

Checkpoints contain:
- Network weights (connection weights only)
- Neuron assignments
- Proportions and rates
- Epoch number
- Accuracy metrics
- Configuration

### Running Training

#### Basic Training

```bash
python -m neuro_sim.cli train
```

#### With Config File

```bash
python -m neuro_sim.cli train --config configs/default.yaml
```

#### With Overrides

```bash
python -m neuro_sim.cli train \
  --config configs/default.yaml \
  --n-epochs 10 \
  --batch-size 64 \
  --gpu
```

#### Resume from Checkpoint

```bash
python -m neuro_sim.cli train \
  --resume-from checkpoints/checkpoint_epoch_3.pt \
  --config configs/default.yaml
```

### Training Output

Training produces:
- **Console logs**: Progress bars, epoch info, accuracy metrics
- **Log files**: Detailed logs in `logs/` directory
- **Checkpoints**: Saved in `checkpoints/checkpoint_epoch_N.pt`
- **Final model**: Saved in `models/mnist_snn_model.pt`

### Monitoring Training

Check logs for:
- Device being used (CPU/GPU)
- Epoch progress
- Assignment update accuracy
- Checkpoint saves
- Training time

---

## Inference and Evaluation

### Overview

The evaluation workflow:
1. Loads trained model
2. Runs inference on test set
3. Computes accuracy metrics
4. Returns results

### Running Inference

#### Basic Inference

```bash
python -m neuro_sim.cli infer
```

This uses the default model path: `models/mnist_snn_model.pt`

#### With Specific Model

```bash
python -m neuro_sim.cli infer --model-path models/mnist_snn_model.pt
```

#### With Config File

```bash
python -m neuro_sim.cli infer \
  --config configs/default.yaml \
  --model-path models/mnist_snn_model.pt
```

### Evaluation Metrics

The evaluator computes two accuracy metrics:

1. **All Activity Accuracy**: 
   - Uses neuron with highest spike count
   - Simple winner-takes-all approach

2. **Proportion Weighting Accuracy**:
   - Uses learned neuron-to-class proportions
   - More sophisticated prediction method

### Programmatic Evaluation

```python
from neuro_sim.config import Config
from neuro_sim.training import Evaluator

config = Config(config_path="configs/default.yaml")
evaluator = Evaluator(config, model_path="models/mnist_snn_model.pt")

# Evaluate on test set
metrics = evaluator.evaluate()
print(f"All activity accuracy: {metrics['all']:.2f}%")
print(f"Proportion accuracy: {metrics['proportion']:.2f}%")

# Predict on a batch
predictions_all, predictions_prop = evaluator.predict(batch)
```

---

## Checkpointing and Resuming

### Checkpoint Contents

Each checkpoint (`checkpoint_epoch_N.pt`) contains:

- **`epoch`**: Epoch number when checkpoint was saved
- **`network_state`**: Connection weights (not neuron states)
- **`assignments`**: Neuron-to-class assignments
- **`proportions`**: Learned spike proportions
- **`rates`**: Neuron firing rates
- **`accuracy`**: Accuracy metrics at checkpoint time
- **`config`**: Training configuration used
- **`metadata`**: Optional additional metadata

### Saving Checkpoints

Checkpoints are automatically saved:
- Every `checkpoint_interval` epochs
- At the end of training (final model)

Checkpoint naming: `checkpoints/checkpoint_epoch_{epoch}.pt`

### Loading Checkpoints

#### Via CLI

```bash
python -m neuro_sim.cli train \
  --resume-from checkpoints/checkpoint_epoch_3.pt \
  --config configs/default.yaml
```

#### Programmatically

```python
trainer = Trainer(config)
trainer.train(resume_from="checkpoints/checkpoint_epoch_3.pt")
```

### Verifying Checkpoint Loading

When resuming, you should see log messages:

```
INFO - Checkpoint loaded: checkpoints/checkpoint_epoch_3.pt
INFO -   Resuming from epoch: 3
INFO -   Previous accuracy - All activity: 30.89%, Proportion: 29.56%
```

### Important Notes

1. **Epoch Counter**: The training loop still shows "Epoch 0" after resuming because it doesn't resume the epoch counter. The important part is that **model state** (weights, assignments, proportions) is restored.

2. **Configuration**: When resuming, make sure the config matches the original training config, especially model architecture parameters (`n_neurons`, `n_inpt`, etc.).

3. **Device**: Checkpoints are saved on CPU but loaded to the device specified in config.

4. **What Gets Restored**:
   - ✅ Network connection weights
   - ✅ Neuron assignments
   - ✅ Proportions and rates
   - ❌ Epoch counter (starts from 0)
   - ❌ Optimizer state (not used in this SNN)

### Manual Checkpoint Management

```python
from neuro_sim.utils.model_persistence import ModelPersistence

# Load checkpoint manually
checkpoint = ModelPersistence.load_checkpoint(
    "checkpoints/checkpoint_epoch_3.pt",
    device="cuda"
)

print(f"Epoch: {checkpoint['epoch']}")
print(f"Accuracy: {checkpoint['accuracy']}")
print(f"Config: {checkpoint['config']}")
```

---

## Model Architecture

### DiehlAndCook2015 Network

The project uses the **DiehlAndCook2015** architecture from BindsNET, which implements:

- **Input Layer**: 784 neurons (28×28 MNIST pixels)
- **Excitatory Layer (Ae)**: `n_neurons` excitatory neurons
- **Inhibitory Layer (Ai)**: Same number of inhibitory neurons

### Architecture Details

#### Connections

1. **X → Ae**: Input to excitatory neurons
   - Weights initialized randomly
   - STDP learning enabled

2. **Ae → Ai**: Excitatory to inhibitory neurons
   - Fixed excitatory weights (`exc` parameter)

3. **Ai → Ae**: Inhibitory to excitatory neurons
   - Fixed inhibitory weights (`inh` parameter)

#### Learning Rule

- **STDP (Spike-Timing-Dependent Plasticity)**: 
  - Learning rates: `nu[0]` (min) to `nu[1]` (max)
  - Threshold increment: `theta_plus`
  - Only applied to X → Ae connections

#### Neuron Model

- Uses Brian2's leaky integrate-and-fire (LIF) neurons
- Time step: `dt` ms
- Simulation window: `time` ms per sample

### Training Process

1. **Encoding**: Images converted to Poisson spike trains
2. **Simulation**: Network runs for `time` ms
3. **Spike Recording**: Spikes from Ae layer recorded
4. **Assignment**: Neurons assigned to classes based on spike patterns
5. **Prediction**: Uses spike counts or proportions for classification

### Key Parameters

- **`n_neurons`**: More neurons = more capacity, slower training
- **`exc`/`inh`**: Balance excitation/inhibition
- **`nu`**: Learning rate range for STDP
- **`theta_plus`**: STDP threshold increment
- **`time`**: Longer simulation = more spikes, slower training

---

## API Reference

### Config Class

```python
class Config:
    def __init__(self, config_path: Optional[str] = None, **kwargs)
    def load_from_file(self, config_path: str)
    def get(self, key: str, default: Any = None) -> Any
    def set(self, key: str, value: Any)
    def to_dict(self) -> Dict[str, Any]
    def save(self, path: str)
    
    @property
    def model(self) -> Dict[str, Any]
    @property
    def training(self) -> Dict[str, Any]
    @property
    def inference(self) -> Dict[str, Any]
    @property
    def data(self) -> Dict[str, Any]
    @property
    def paths(self) -> Dict[str, str]
```

### Trainer Class

```python
class Trainer:
    def __init__(self, config: Config, checkpoint_dir: Optional[str] = None)
    def train(self, resume_from: Optional[str] = None) -> Dict[str, float]
    def save_checkpoint(self, epoch: int)
    def load_checkpoint(self, checkpoint_path: str)
    def save_model(self) -> str
```

### Evaluator Class

```python
class Evaluator:
    def __init__(self, config: Config, model_path: Optional[str] = None)
    def evaluate(self) -> Dict[str, float]
    def predict(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
```

### ModelPersistence Class

```python
class ModelPersistence:
    @staticmethod
    def save_checkpoint(
        checkpoint_path: str,
        network: Any,
        assignments: torch.Tensor,
        proportions: torch.Tensor,
        rates: torch.Tensor,
        epoch: int,
        accuracy: Dict[str, float],
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    )
    
    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        device: str = "cpu"
    ) -> Dict[str, Any]
    
    @staticmethod
    def save_model(
        model_path: str,
        network: Any,
        assignments: torch.Tensor,
        proportions: torch.Tensor,
        rates: torch.Tensor,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    )
    
    @staticmethod
    def load_model(
        model_path: str,
        device: str = "cpu"
    ) -> Dict[str, Any]
```

### Data Utilities

```python
def get_dataset(
    dataset_name: str = "MNIST",
    encoder: Optional[PoissonEncoder] = None,
    root: Optional[str] = None,
    train: bool = True,
    download: bool = True,
    intensity: float = 128.0,
    time: int = 100,
    dt: float = 1.0
) -> MNIST

def get_dataloader(
    dataset: MNIST,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = -1,
    pin_memory: bool = False
) -> DataLoader
```

---

## Troubleshooting

### Common Issues

#### 1. Device Mismatch Errors

**Error**: `RuntimeError: indices should be either on cpu or on the same device`

**Solution**: This has been fixed in the codebase. Ensure you're using the latest version. The evaluator now properly moves all tensors to the same device.

#### 2. Checkpoint Not Found

**Error**: `FileNotFoundError: Checkpoint not found`

**Solution**: 
- Check the checkpoint path is correct
- Verify checkpoint exists: `ls checkpoints/`
- Use absolute path if relative path fails

#### 3. GPU Not Available

**Error**: Training/inference runs on CPU despite `gpu: true`

**Solution**:
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA-enabled PyTorch if needed
- Set `gpu: false` in config to use CPU

#### 4. Low Accuracy

**Symptoms**: Accuracy around 30% (close to random)

**Solutions**:
- Train for more epochs (`n_epochs: 10` or more)
- Increase number of neurons (`n_neurons: 200`)
- Adjust learning rates (`nu: [0.0001, 0.01]`)
- Increase simulation time (`time: 200`)
- Resume from checkpoint to continue training

#### 5. Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `batch_size` (e.g., 32 → 16)
- Reduce `n_neurons` (e.g., 200 → 100)
- Use CPU: set `gpu: false`
- Reduce `n_train` for testing

#### 6. BindsNET Version Issues

**Error**: API mismatches or import errors

**Solution**:
- Ensure BindsNET 0.2.x is installed
- Use `pip install -e .` for consistent dependencies
- Check `compat.py` for compatibility patches

#### 7. Data Loading Errors

**Error**: Dataset not found or download fails

**Solution**:
- Set `data.root` to a valid directory
- Set `data.root: null` to use BindsNET default
- Check internet connection for download
- Manually download MNIST if needed

### Debugging Tips

1. **Enable Verbose Logging**: Check `logs/` directory for detailed logs
2. **Check Device**: Verify GPU/CPU usage in logs
3. **Verify Config**: Print config with `config.to_dict()`
4. **Test Components**: Test data loading, network building separately
5. **Monitor Memory**: Use `nvidia-smi` for GPU memory usage

---

## Examples

### Example 1: Basic Training

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
print(f"Final metrics: {metrics}")
```

### Example 2: Custom Configuration

```python
from neuro_sim.config import Config

# Create custom config
config = Config()
config.set("model.n_neurons", 200)
config.set("training.n_epochs", 10)
config.set("training.batch_size", 64)
config.set("training.gpu", True)

# Save config
config.save("configs/custom.yaml")

# Train with custom config
trainer = Trainer(config)
metrics = trainer.train()
```

### Example 3: Resume Training

```python
from neuro_sim.config import Config
from neuro_sim.training import Trainer

config = Config(config_path="configs/default.yaml")
trainer = Trainer(config)

# Resume from checkpoint
metrics = trainer.train(resume_from="checkpoints/checkpoint_epoch_3.pt")
```

### Example 4: Evaluation

```python
from neuro_sim.config import Config
from neuro_sim.training import Evaluator

config = Config(config_path="configs/default.yaml")
evaluator = Evaluator(config, model_path="models/mnist_snn_model.pt")

# Evaluate
metrics = evaluator.evaluate()
print(f"All activity: {metrics['all']:.2f}%")
print(f"Proportion: {metrics['proportion']:.2f}%")
```

### Example 5: Manual Checkpoint Inspection

```python
from neuro_sim.utils.model_persistence import ModelPersistence
import torch

# Load checkpoint
checkpoint = ModelPersistence.load_checkpoint(
    "checkpoints/checkpoint_epoch_5.pt",
    device="cpu"
)

# Inspect contents
print(f"Epoch: {checkpoint['epoch']}")
print(f"Accuracy: {checkpoint['accuracy']}")
print(f"Assignments shape: {checkpoint['assignments'].shape}")
print(f"Config: {checkpoint['config']}")
```

### Example 6: Training with Progress Tracking

```python
from neuro_sim.config import Config
from neuro_sim.training import Trainer
import logging

# Setup detailed logging
logging.basicConfig(level=logging.INFO)

config = Config(config_path="configs/default.yaml")
trainer = Trainer(config)

# Train with progress tracking
metrics = trainer.train()

# Access accuracy history
print(f"All activity history: {trainer.accuracy_history['all']}")
print(f"Proportion history: {trainer.accuracy_history['proportion']}")
```

### Example 7: Batch Prediction

```python
from neuro_sim.config import Config
from neuro_sim.training import Evaluator
from neuro_sim.data.dataset import get_dataset, get_dataloader

config = Config(config_path="configs/default.yaml")
evaluator = Evaluator(config, model_path="models/mnist_snn_model.pt")

# Get a batch
dataset = get_dataset(train=False, intensity=128.0, time=100, dt=1.0)
dataloader = get_dataloader(dataset, batch_size=32, shuffle=False)
batch = next(iter(dataloader))

# Predict
predictions_all, predictions_prop = evaluator.predict(batch)
print(f"All activity predictions: {predictions_all}")
print(f"Proportion predictions: {predictions_prop}")
```

---

## Additional Resources

- **BindsNET Documentation**: https://github.com/BindsNET/bindsnet
- **Brian2 Documentation**: https://brian2.readthedocs.io/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Original Paper**: Diehl & Cook (2015) - "Unsupervised learning of digit recognition using spike-timing-dependent plasticity"

---

## Contributing

When contributing to this project:

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure compatibility with BindsNET 0.2.x
5. Test on both CPU and GPU
6. Update this guide if adding new features

---

## License

[Add your license information here]

---

**Last Updated**: 2024
**Version**: 0.1.0








