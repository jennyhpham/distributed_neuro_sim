"""Configuration management for neuro_sim."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for training and inference."""

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file (optional)
            **kwargs: Override config values with keyword arguments
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config validation fails
        """
        self._config: Dict[str, Any] = {}
        
        # Load default config
        default_config = self._get_default_config()
        self._config.update(default_config)
        
        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)
        
        # Override with kwargs
        if kwargs:
            self._config.update(kwargs)
        
        # Validate config
        self._validate()
        
        # Set up paths
        self._setup_paths()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model": {
                "name": "DiehlAndCook2015",
                "n_neurons": 100,
                "n_inpt": 784,
                "exc": 22.5,
                "inh": 120.0,
                "theta_plus": 0.05,
                "dt": 1.0,
                "norm": 78.4,
                "nu": [0.0001, 0.01],  # Learning rates [min, max]
                "inpt_shape": [1, 28, 28],
                "w_dtype": "float32",
                "sparse": False,
            },
            "training": {
                "batch_size": 32,
                "n_epochs": 1,
                "n_train": 60000,
                "n_updates": 10,
                "time": 100,
                "intensity": 128.0,
                "progress_interval": 10,
                "seed": 0,
                "gpu": True,
            },
            "inference": {
                "batch_size": 32,
                "n_test": 10000,
                "time": 100,
                "intensity": 128.0,
            },
            "data": {
                "dataset": "MNIST",
                "root": None,  # Will default to bindsnet ROOT_DIR
                "n_workers": -1,
            },
            "paths": {
                "checkpoint_dir": "checkpoints",
                "model_dir": "models",
                "log_dir": "logs",
                "data_dir": "data",
            },
        }
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            file_config = yaml.safe_load(f)
        
        # Deep merge with existing config
        self._deep_update(self._config, file_config or {})
    
    def _deep_update(self, base: Dict, update: Dict):
        """Recursively update nested dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def _validate(self):
        """Validate configuration values."""
        # Validate model config
        model_config = self._config.get("model", {})
        if model_config.get("n_neurons", 0) <= 0:
            raise ValueError("model.n_neurons must be positive")
        if model_config.get("n_inpt", 0) <= 0:
            raise ValueError("model.n_inpt must be positive")
        
        # Validate training config
        training_config = self._config.get("training", {})
        if training_config.get("batch_size", 0) <= 0:
            raise ValueError("training.batch_size must be positive")
        if training_config.get("n_epochs", 0) < 0:
            raise ValueError("training.n_epochs must be non-negative")
        
        # Validate inference config
        inference_config = self._config.get("inference", {})
        if inference_config.get("batch_size", 0) <= 0:
            raise ValueError("inference.batch_size must be positive")
    
    def _setup_paths(self):
        """Set up and create necessary directories."""
        paths = self._config["paths"]
        
        # Find project root by looking for pyproject.toml or setup.py
        project_root = self._find_project_root()
        
        # Create directories if they don't exist
        for key, path in paths.items():
            full_path = Path(path)
            if not full_path.is_absolute():
                # Relative to project root
                full_path = project_root / path
            
            full_path.mkdir(parents=True, exist_ok=True)
            paths[key] = str(full_path.resolve())
    
    def _find_project_root(self) -> Path:
        """Find project root by looking for pyproject.toml or setup.py."""
        current = Path(__file__).resolve()
        
        # Start from config.py and go up until we find project root markers
        for parent in [current.parent] + list(current.parents):
            if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
                return parent
        
        # Fallback: assume src/neuro_sim/config.py structure
        # Go up 3 levels: config.py -> neuro_sim -> src -> project_root
        return current.parent.parent.parent
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-separated key (e.g., 'model.n_neurons')."""
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set config value by dot-separated key."""
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self._config.copy()
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config["model"]
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._config["training"]
    
    @property
    def inference(self) -> Dict[str, Any]:
        """Get inference configuration."""
        return self._config["inference"]
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config["data"]
    
    @property
    def paths(self) -> Dict[str, str]:
        """Get paths configuration."""
        return self._config["paths"]

