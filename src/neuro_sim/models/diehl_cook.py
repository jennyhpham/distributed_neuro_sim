"""DiehlAndCook2015 model wrapper.

Wrapper for the Diehl & Cook 2015 spiking neural network architecture
for unsupervised learning of MNIST digit recognition.
"""

from typing import Any, Dict, List, Tuple

import torch
from bindsnet.models import DiehlAndCook2015

from neuro_sim.models.base import BaseModelWrapper


class DiehlCookModel(BaseModelWrapper):
    """Wrapper for DiehlAndCook2015 network.

    Implements the three-layer architecture from Diehl & Cook 2015:
    - Input layer (X): Poisson-encoded MNIST images
    - Excitatory layer (Ae): DiehlAndCookNodes with adaptive thresholds
    - Inhibitory layer (Ai): LIF neurons providing lateral inhibition

    Attributes:
        MODEL_NAME: "DiehlAndCook2015"
        OUTPUT_LAYER: "Ae" (excitatory layer for spike recording)
        INHIBITORY_LAYER: "Ai" (inhibitory feedback layer)
    """

    MODEL_NAME = "DiehlAndCook2015"
    OUTPUT_LAYER = "Ae"
    INHIBITORY_LAYER = "Ai"

    def build(
        self,
        batch_size: int,
        learning_enabled: bool = True,
        sparse: bool = False,
    ) -> DiehlAndCook2015:
        """Build the DiehlAndCook2015 network.

        Args:
            batch_size: Batch size for training/inference.
            learning_enabled: Whether STDP learning is enabled.
            sparse: Whether to use sparse tensors for weights.

        Returns:
            Configured DiehlAndCook2015 network instance.
        """
        model_config = self.config

        # Parse weight dtype
        w_dtype_str = model_config.get("w_dtype", "float32")
        w_dtype = getattr(torch, w_dtype_str, torch.float32)

        # Get learning rates (disable if learning not enabled)
        nu = tuple(model_config.get("nu", [0.0001, 0.01]))
        if not learning_enabled:
            nu = (0.0, 0.0)

        self.network = DiehlAndCook2015(
            device=self.device,
            batch_size=batch_size,
            sparse=sparse,
            n_inpt=model_config["n_inpt"],
            n_neurons=model_config["n_neurons"],
            exc=model_config.get("exc", 22.5),
            inh=model_config.get("inh", 120.0),
            dt=model_config.get("dt", 1.0),
            norm=model_config.get("norm", 78.4),
            nu=nu,
            theta_plus=model_config.get("theta_plus", 0.05),
            inpt_shape=tuple(model_config.get("inpt_shape", [1, 28, 28])),
            w_dtype=w_dtype,
            inh_thresh=model_config.get("inh_thresh", -40.0),
            exc_thresh=model_config.get("exc_thresh", -52.0),
        )

        self.move_to_device()
        return self.network

    def get_connection_keys(self) -> List[str]:
        """Return connection keys for weight saving/loading.

        Returns:
            List of connection key strings for this architecture.
        """
        return ["X_to_Ae.w", "Ae_to_Ai.w", "Ai_to_Ae.w"]

    def get_learnable_connection(self) -> Tuple[str, str]:
        """Return the learnable connection for weight normalization.

        Returns:
            Tuple of (source, target) layer names.
        """
        return ("X", "Ae")

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Return default configuration for DiehlAndCook2015.

        Returns:
            Dictionary of default model parameters.
        """
        return {
            "name": "DiehlAndCook2015",
            "n_neurons": 100,
            "n_inpt": 784,
            "exc": 22.5,
            "inh": 120.0,
            "theta_plus": 0.05,
            "dt": 1.0,
            "norm": 78.4,
            "nu": [0.0001, 0.01],
            "inpt_shape": [1, 28, 28],
            "w_dtype": "float32",
            "sparse": False,
            "inh_thresh": -40.0,
            "exc_thresh": -52.0,
        }
