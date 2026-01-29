"""Model abstraction layer for neuro_sim.

This module provides a unified interface for different SNN architectures,
enabling model-agnostic training and evaluation.

Available models:
    - DiehlAndCook2015: Three-layer network with separate inhibitory layer
    - IncreasingInhibitionNetwork: Two-layer network with distance-weighted inhibition

Example:
    >>> from neuro_sim.models import ModelFactory
    >>> config = {"name": "DiehlAndCook2015", "n_neurons": 400}
    >>> model = ModelFactory.create(config, torch.device("cuda"))
    >>> network = model.build(batch_size=32)
"""

from neuro_sim.models.base import BaseModelWrapper
from neuro_sim.models.diehl_cook import DiehlCookModel
from neuro_sim.models.factory import ModelFactory
from neuro_sim.models.increasing_inhibition import IncreasingInhibitionModel

__all__ = [
    "BaseModelWrapper",
    "DiehlCookModel",
    "IncreasingInhibitionModel",
    "ModelFactory",
]
