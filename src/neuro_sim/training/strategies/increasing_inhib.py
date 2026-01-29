"""Increasing Inhibition training strategy for SOM-LM-SNN.

Implements the training strategy from Hazan et al. 2018 with:
- Sample-by-sample training (batch_size=1)
- Gradually increasing lateral inhibition
- Retry mechanism for low-spiking samples
"""

import logging
from typing import Any, Dict, Optional

import torch
from bindsnet.encoding import poisson

from neuro_sim.models.base import BaseModelWrapper
from neuro_sim.models.increasing_inhibition import IncreasingInhibitionModel
from neuro_sim.training.strategies.base import TrainingStrategy

logger = logging.getLogger(__name__)


class IncreasingInhibitionStrategy(TrainingStrategy):
    """Training strategy for IncreasingInhibitionNetwork (SOM-LM-SNN).

    Implements special training behaviors:
    - Requires batch_size=1 (sample-by-sample STDP)
    - Increases lateral inhibition weights at regular intervals
    - Retries samples that produce too few spikes

    Attributes:
        update_inhib_interval: Steps between inhibition weight updates.
        inhib_decrease_small: Amount to decrease inhibition (small updates).
        inhib_decrease_large: Amount to decrease inhibition (large updates).
        large_decrease_multiplier: Interval multiplier for large decreases.
        min_spikes: Minimum spikes required (retry if fewer).
        max_retries: Maximum retry attempts per sample.
        intensity_factor: Factor to increase input intensity on retry.
    """

    def __init__(
        self,
        model: BaseModelWrapper,
        config: Dict[str, Any],
        device: torch.device,
    ):
        """Initialize the increasing inhibition strategy.

        Args:
            model: The model wrapper (should be IncreasingInhibitionModel).
            config: Training configuration dictionary.
            device: Torch device for computation.
        """
        super().__init__(model, config, device)

        # Strategy-specific configuration
        self.update_inhib_interval = config.get("update_inhibition_interval", 500)
        self.inhib_decrease_small = config.get("inhib_decrease_small", 0.5)
        self.inhib_decrease_large = config.get("inhib_decrease_large", 50.0)
        self.large_decrease_multiplier = config.get("large_decrease_multiplier", 10)
        self.min_spikes = config.get("min_spikes", 2)
        self.max_retries = config.get("max_retries", 5)
        self.intensity_factor = config.get("intensity_factor", 1.2)

        # Create weight mask (zeros on diagonal, ones elsewhere)
        n_neurons = config.get("n_neurons", 100)
        self.weights_mask = (1 - torch.diag(torch.ones(n_neurons))).to(device)

        # Track retry state for current sample
        self._current_retry = 0
        self._current_intensity_factor = 1.0
        self._current_batch: Optional[Dict[str, torch.Tensor]] = None

    def get_batch_size(self) -> int:
        """SOM-LM-SNN requires batch_size=1 for proper STDP learning.

        Returns:
            Always returns 1.
        """
        return 1

    def pre_sample_hook(
        self, step: int, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Reset retry state before each new sample.

        Args:
            step: Current training step.
            batch: Input batch dictionary.

        Returns:
            Unchanged batch dictionary.
        """
        self._current_retry = 0
        self._current_intensity_factor = 1.0
        self._current_batch = batch
        return batch

    def post_sample_hook(self, step: int, spikes: torch.Tensor) -> bool:
        """Check if sample needs retry due to low spiking.

        Args:
            step: Current training step.
            spikes: Spike tensor from the output layer.

        Returns:
            True if should retry, False otherwise.
        """
        spike_count = spikes.sum().item()

        if spike_count < self.min_spikes and self._current_retry < self.max_retries:
            self._current_retry += 1
            self._current_intensity_factor *= self.intensity_factor

            if self._current_retry == 1:
                logger.debug(
                    f"Step {step}: Low spike count ({spike_count}), retrying..."
                )

            return True

        if self._current_retry > 0:
            logger.debug(
                f"Step {step}: Completed after {self._current_retry} retries, "
                f"final spike count: {spike_count}"
            )

        return False

    def get_adjusted_input(
        self,
        batch: Dict[str, torch.Tensor],
        time: int,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        """Get input with adjusted intensity for retry.

        Re-encodes the image with higher intensity to increase firing.

        Args:
            batch: Input batch dictionary.
            time: Simulation time in ms.
            dt: Time step in ms.

        Returns:
            Batch with re-encoded input at higher intensity.
        """
        if self._current_intensity_factor <= 1.0:
            return batch

        # Get original image and scale up
        if "image" not in batch:
            return batch

        adjusted_image = batch["image"].clamp(min=0) * self._current_intensity_factor

        # Re-encode with Poisson encoding
        time_steps = int(time / dt)
        encoded = poisson(
            datum=adjusted_image,
            dt=dt,
            time=time_steps,
        ).to(self.device)

        # Update encoded image in batch
        # Shape: (time_steps, batch=1, channels=1, height=28, width=28)
        batch["encoded_image"] = encoded.view(time_steps, 1, 1, 28, 28)

        return batch

    def periodic_update_hook(self, step: int) -> None:
        """Update inhibitory weights at specified intervals.

        Gradually increases lateral inhibition to encourage competition
        and specialization of neurons.

        Args:
            step: Current training step.
        """
        if step == 0:
            return

        if step % self.update_inhib_interval != 0:
            return

        # Get the recurrent inhibitory connection
        if not isinstance(self.model, IncreasingInhibitionModel):
            return

        inhib_conn = self.model.get_recurrent_connection()
        if inhib_conn is None:
            return

        # Determine decrease amount
        is_large_update = (
            step % (self.update_inhib_interval * self.large_decrease_multiplier) == 0
        )

        if is_large_update:
            # Large decrease every N*10 samples
            decrease_amount = self.inhib_decrease_large
            logger.debug(f"Step {step}: Large inhibition decrease ({decrease_amount})")
        else:
            # Small decrease every N samples
            decrease_amount = self.inhib_decrease_small
            logger.debug(f"Step {step}: Small inhibition decrease ({decrease_amount})")

        # Update weights (subtract to increase inhibition magnitude)
        # Note: Inhibition weights are negative, so subtracting makes them more negative
        inhib_conn.w -= self.weights_mask * decrease_amount
