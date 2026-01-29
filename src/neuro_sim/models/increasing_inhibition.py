"""IncreasingInhibitionNetwork (SOM-LM-SNN) model wrapper.

Wrapper for the Increasing Inhibition Network architecture from Hazan et al. 2018,
which uses distance-weighted lateral inhibition for self-organizing map properties.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from bindsnet.models import IncreasingInhibitionNetwork
from bindsnet.network.topology import Connection

from neuro_sim.models.base import BaseModelWrapper


class IncreasingInhibitionModel(BaseModelWrapper):
    """Wrapper for IncreasingInhibitionNetwork (SOM-LM-SNN).

    Implements a two-layer architecture with distance-weighted lateral inhibition:
    - Input layer (X): Poisson-encoded MNIST images
    - Output layer (Y): DiehlAndCookNodes with adaptive thresholds and
      recurrent inhibitory connections weighted by spatial distance

    Key features:
    - Self-organizing map properties through distance-weighted inhibition
    - Requires sample-by-sample training (batch_size=1)
    - Training includes gradually increasing inhibition weights

    Attributes:
        MODEL_NAME: "IncreasingInhibitionNetwork"
        OUTPUT_LAYER: "Y" (output layer for spike recording)
    """

    MODEL_NAME = "IncreasingInhibitionNetwork"
    OUTPUT_LAYER = "Y"

    def build(
        self,
        batch_size: int,
        learning_enabled: bool = True,
        sparse: bool = False,
    ) -> IncreasingInhibitionNetwork:
        """Build the IncreasingInhibitionNetwork.

        Args:
            batch_size: Batch size (should be 1 for this model).
            learning_enabled: Whether STDP learning is enabled.
            sparse: Whether to use sparse tensors (not typically used).

        Returns:
            Configured IncreasingInhibitionNetwork instance.
        """
        model_config = self.config

        # Get learning rates (disable if learning not enabled)
        nu = tuple(model_config.get("nu", [0.0001, 0.01]))
        if not learning_enabled:
            nu = (0.0, 0.0)

        self.network = IncreasingInhibitionNetwork(
            n_input=model_config["n_inpt"],
            n_neurons=model_config["n_neurons"],
            start_inhib=model_config.get("start_inhib", 10.0),
            max_inhib=model_config.get("max_inhib", -40.0),
            dt=model_config.get("dt", 1.0),
            nu=nu,
            wmin=model_config.get("wmin", 0.0),
            wmax=model_config.get("wmax", 1.0),
            norm=model_config.get("norm", 78.4),
            theta_plus=model_config.get("theta_plus", 0.05),
            tc_theta_decay=model_config.get("tc_theta_decay", 1e7),
            inpt_shape=tuple(model_config.get("inpt_shape", [1, 28, 28])),
            exc_thresh=model_config.get("exc_thresh", -52.0),
        )

        self.move_to_device()
        return self.network

    def get_connection_keys(self) -> List[str]:
        """Return connection keys for weight saving/loading.

        Returns:
            List of connection key strings for this architecture.
        """
        return ["X_to_Y.w", "Y_to_Y.w"]

    def get_learnable_connection(self) -> Tuple[str, str]:
        """Return the learnable connection for weight normalization.

        Returns:
            Tuple of (source, target) layer names.
        """
        return ("X", "Y")

    def get_recurrent_connection(self) -> Optional[Connection]:
        """Return the recurrent inhibitory connection for weight updates.

        This connection is modified during training to increase inhibition.

        Returns:
            The Y->Y Connection object, or None if network not built.
        """
        if self.network is None:
            return None

        conn_key = ("Y", "Y")
        if conn_key in self.network.connections:
            return self.network.connections[conn_key]
        return None

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Return default configuration for IncreasingInhibitionNetwork.

        Returns:
            Dictionary of default model parameters.
        """
        return {
            "name": "IncreasingInhibitionNetwork",
            "n_neurons": 100,
            "n_inpt": 784,
            "start_inhib": 10.0,
            "max_inhib": -40.0,
            "theta_plus": 0.05,
            "tc_theta_decay": 1e7,
            "dt": 1.0,
            "norm": 78.4,
            "nu": [0.0001, 0.01],
            "inpt_shape": [1, 28, 28],
            "wmin": 0.0,
            "wmax": 1.0,
            "exc_thresh": -52.0,
        }
