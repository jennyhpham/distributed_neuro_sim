"""Standard STDP training strategy.

Implements standard spike-timing-dependent plasticity training
for the DiehlAndCook2015 network architecture.
"""

from typing import Any, Dict

import torch

from neuro_sim.models.base import BaseModelWrapper
from neuro_sim.training.strategies.base import TrainingStrategy


class StandardTrainingStrategy(TrainingStrategy):
    """Standard STDP training strategy for DiehlAndCook2015.

    This strategy implements standard batched STDP training without
    any special mechanisms like retry or weight modification during training.
    """

    def __init__(
        self,
        model: BaseModelWrapper,
        config: Dict[str, Any],
        device: torch.device,
    ):
        """Initialize the standard training strategy.

        Args:
            model: The model wrapper being trained.
            config: Training configuration dictionary.
            device: Torch device for computation.
        """
        super().__init__(model, config, device)

    def pre_sample_hook(
        self, step: int, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """No preprocessing needed for standard training.

        Args:
            step: Current training step.
            batch: Input batch dictionary.

        Returns:
            Unchanged batch dictionary.
        """
        return batch

    def post_sample_hook(self, step: int, spikes: torch.Tensor) -> bool:
        """No retry mechanism in standard training.

        Args:
            step: Current training step.
            spikes: Spike tensor from the output layer.

        Returns:
            Always False (no retry).
        """
        return False

    def periodic_update_hook(self, step: int) -> None:
        """Standard training has no periodic weight updates.

        Weight normalization is handled separately by the Trainer.

        Args:
            step: Current training step.
        """
        pass
