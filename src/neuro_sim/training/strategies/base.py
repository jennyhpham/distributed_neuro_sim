"""Abstract base class for training strategies.

Training strategies encapsulate model-specific training behaviors,
allowing the main Trainer class to remain model-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from neuro_sim.models.base import BaseModelWrapper


class TrainingStrategy(ABC):
    """Abstract base class for training strategies.

    Defines hooks that are called at various points during training
    to implement model-specific behaviors.

    Attributes:
        model: The model wrapper being trained.
        config: Training configuration dictionary.
        device: Torch device for computation.
    """

    def __init__(
        self,
        model: BaseModelWrapper,
        config: Dict[str, Any],
        device: torch.device,
    ):
        """Initialize the training strategy.

        Args:
            model: The model wrapper being trained.
            config: Training configuration dictionary.
            device: Torch device for computation.
        """
        self.model = model
        self.config = config
        self.device = device

    @abstractmethod
    def pre_sample_hook(
        self, step: int, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Hook called before processing each sample/batch.

        Can be used to preprocess inputs or reset state.

        Args:
            step: Current training step.
            batch: Input batch dictionary.

        Returns:
            Potentially modified batch dictionary.
        """
        pass

    @abstractmethod
    def post_sample_hook(self, step: int, spikes: torch.Tensor) -> bool:
        """Hook called after processing each sample/batch.

        Can be used to check results and determine if retry is needed.

        Args:
            step: Current training step.
            spikes: Spike tensor from the output layer.

        Returns:
            True if sample should be retried, False otherwise.
        """
        pass

    @abstractmethod
    def periodic_update_hook(self, step: int) -> None:
        """Hook called at each step for periodic updates.

        Can be used to update network weights or other parameters
        at regular intervals.

        Args:
            step: Current training step.
        """
        pass

    def get_batch_size(self) -> Optional[int]:
        """Return the required batch size for this strategy.

        Returns:
            Required batch size, or None to use config default.
        """
        return None

    def get_adjusted_input(
        self,
        batch: Dict[str, torch.Tensor],
        time: int,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        """Get adjusted input for retry mechanism.

        Default implementation returns batch unchanged.

        Args:
            batch: Input batch dictionary.
            time: Simulation time in ms.
            dt: Time step in ms.

        Returns:
            Adjusted batch dictionary.
        """
        return batch

    def on_epoch_start(self, epoch: int) -> None:
        """Hook called at the start of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).
        """
        pass

    def on_epoch_end(self, epoch: int) -> None:
        """Hook called at the end of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).
        """
        pass
