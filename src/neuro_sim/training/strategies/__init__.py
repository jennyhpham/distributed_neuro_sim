"""Training strategies for different SNN architectures.

This module provides strategy classes that encapsulate model-specific
training behaviors, allowing the Trainer to remain model-agnostic.

Available strategies:
    - StandardTrainingStrategy: Standard batched STDP for DiehlAndCook2015
    - IncreasingInhibitionStrategy: Sample-by-sample with increasing inhibition
"""

from typing import Any, Dict, Type

import torch

from neuro_sim.models.base import BaseModelWrapper
from neuro_sim.training.strategies.base import TrainingStrategy
from neuro_sim.training.strategies.increasing_inhib import IncreasingInhibitionStrategy
from neuro_sim.training.strategies.standard import StandardTrainingStrategy

# Registry mapping model names to their training strategies
_STRATEGY_REGISTRY: Dict[str, Type[TrainingStrategy]] = {
    "DiehlAndCook2015": StandardTrainingStrategy,
    "IncreasingInhibitionNetwork": IncreasingInhibitionStrategy,
}


def get_training_strategy(
    model_name: str,
    model: BaseModelWrapper,
    config: Dict[str, Any],
    device: torch.device,
) -> TrainingStrategy:
    """Get the appropriate training strategy for a model.

    Args:
        model_name: Name of the model type.
        model: The model wrapper instance.
        config: Training configuration dictionary.
        device: Torch device for computation.

    Returns:
        Instantiated training strategy for the model.
    """
    strategy_class = _STRATEGY_REGISTRY.get(model_name, StandardTrainingStrategy)
    return strategy_class(model, config, device)


def register_strategy(model_name: str, strategy_class: Type[TrainingStrategy]) -> None:
    """Register a training strategy for a model type.

    Args:
        model_name: Name of the model type.
        strategy_class: The strategy class to register.
    """
    _STRATEGY_REGISTRY[model_name] = strategy_class


__all__ = [
    "TrainingStrategy",
    "StandardTrainingStrategy",
    "IncreasingInhibitionStrategy",
    "get_training_strategy",
    "register_strategy",
]
