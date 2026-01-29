"""Model factory for creating SNN model wrappers.

Provides a registry-based factory pattern for instantiating different
SNN architectures based on configuration.
"""

from typing import Any, Dict, List, Type

import torch

from neuro_sim.models.base import BaseModelWrapper
from neuro_sim.models.diehl_cook import DiehlCookModel
from neuro_sim.models.increasing_inhibition import IncreasingInhibitionModel


class ModelFactory:
    """Factory for creating model wrappers based on configuration.

    Uses a registry pattern to map model names to their wrapper classes.
    New models can be registered dynamically.

    Example:
        >>> config = {"name": "DiehlAndCook2015", "n_neurons": 400, ...}
        >>> device = torch.device("cuda")
        >>> model = ModelFactory.create(config, device)
        >>> network = model.build(batch_size=32)
    """

    _registry: Dict[str, Type[BaseModelWrapper]] = {
        "DiehlAndCook2015": DiehlCookModel,
        "IncreasingInhibitionNetwork": IncreasingInhibitionModel,
    }

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModelWrapper]) -> None:
        """Register a new model type.

        Args:
            name: Identifier for the model type.
            model_class: The model wrapper class to register.
        """
        cls._registry[name] = model_class

    @classmethod
    def create(cls, config: Dict[str, Any], device: torch.device) -> BaseModelWrapper:
        """Create a model wrapper based on configuration.

        Args:
            config: Model configuration dictionary. Must contain "name" key
                   or defaults to "DiehlAndCook2015".
            device: Torch device for computation.

        Returns:
            Instantiated model wrapper.

        Raises:
            ValueError: If model name is not in registry.
        """
        model_name = config.get("name", "DiehlAndCook2015")

        if model_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown model: '{model_name}'. Available models: {available}"
            )

        model_class = cls._registry[model_name]
        return model_class(config, device)

    @classmethod
    def get_default_config(cls, model_name: str) -> Dict[str, Any]:
        """Get default configuration for a model type.

        Args:
            model_name: Name of the model type.

        Returns:
            Dictionary of default configuration values.

        Raises:
            ValueError: If model name is not in registry.
        """
        if model_name not in cls._registry:
            raise ValueError(f"Unknown model: '{model_name}'")
        return cls._registry[model_name].get_default_config()

    @classmethod
    def list_models(cls) -> List[str]:
        """Return list of available model names.

        Returns:
            List of registered model names.
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        """Check if a model is registered.

        Args:
            model_name: Name to check.

        Returns:
            True if model is registered, False otherwise.
        """
        return model_name in cls._registry
