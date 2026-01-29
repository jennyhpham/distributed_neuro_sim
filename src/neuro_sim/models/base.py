"""Abstract base class for SNN model wrappers.

This module provides the foundation for model-agnostic training and evaluation
by defining a common interface for different SNN architectures.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
from bindsnet.network import Network


class BaseModelWrapper(ABC):
    """Abstract base class for SNN model wrappers.

    Provides a unified interface for creating, configuring, and managing
    different SNN architectures (e.g., DiehlAndCook2015, IncreasingInhibitionNetwork).

    Attributes:
        MODEL_NAME: Identifier for this model type.
        OUTPUT_LAYER: Name of the output layer for spike recording.
        INPUT_LAYER: Name of the input layer.
        config: Model configuration dictionary.
        device: Torch device for computation.
        network: The underlying BindsNET network instance.
    """

    MODEL_NAME: str = ""
    OUTPUT_LAYER: str = ""
    INPUT_LAYER: str = "X"

    def __init__(self, config: Dict[str, Any], device: torch.device):
        """Initialize the model wrapper.

        Args:
            config: Model configuration dictionary containing architecture parameters.
            device: Torch device (CPU or CUDA) for computation.
        """
        self.config = config
        self.device = device
        self.network: Optional[Network] = None

    @abstractmethod
    def build(
        self,
        batch_size: int,
        learning_enabled: bool = True,
        sparse: bool = False,
    ) -> Network:
        """Build and return the network.

        Args:
            batch_size: Batch size for training/inference.
            learning_enabled: Whether STDP learning is enabled.
            sparse: Whether to use sparse tensors for weights.

        Returns:
            The constructed BindsNET Network instance.
        """
        pass

    @abstractmethod
    def get_connection_keys(self) -> List[str]:
        """Return list of connection keys for weight saving/loading.

        Returns:
            List of connection key strings (e.g., ["X_to_Ae.w", "Ae_to_Ai.w"]).
        """
        pass

    @abstractmethod
    def get_learnable_connection(self) -> Tuple[str, str]:
        """Return the connection tuple that has learnable weights.

        Used for weight normalization during training.

        Returns:
            Tuple of (source_layer, target_layer) for the learnable connection.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_default_config() -> Dict[str, Any]:
        """Return default configuration for this model type.

        Returns:
            Dictionary of default configuration values.
        """
        pass

    @property
    def model_name(self) -> str:
        """Return the name/identifier of this model type."""
        return self.MODEL_NAME

    @property
    def output_layer(self) -> str:
        """Return the name of the output layer for spike recording."""
        return self.OUTPUT_LAYER

    @property
    def input_layer(self) -> str:
        """Return the name of the input layer."""
        return self.INPUT_LAYER

    def get_theta_key(self) -> Optional[str]:
        """Return the key for theta (adaptive threshold) values.

        Returns:
            Key string for theta values, or None if not applicable.
        """
        return f"{self.OUTPUT_LAYER}.theta"

    def move_to_device(self, network: Optional[Network] = None) -> None:
        """Move network and all components to the configured device.

        Args:
            network: Optional network to move. If None, uses self.network.
        """
        target_network = network or self.network
        if target_network is None:
            return

        # Update self.network if a new network was provided
        if network is not None:
            self.network = network

        if self.device.type == "cuda":
            target_network.to("cuda")
            self._move_connections_to_device()

    def _move_connections_to_device(self) -> None:
        """Move connection weights and pipeline features to device.

        Handles the BindsNET-specific requirement of explicitly moving
        connection weights and learning rule attributes to GPU.
        """
        if self.network is None:
            return

        for conn_name, connection in self.network.connections.items():
            # Move main weight tensor
            if hasattr(connection, "w") and isinstance(connection.w, torch.Tensor):
                connection.w = connection.w.to(self.device)

            # Handle MulticompartmentConnection pipeline features
            if hasattr(connection, "pipeline"):
                for feature in connection.pipeline:
                    # Move feature value
                    if hasattr(feature, "value") and isinstance(
                        feature.value, torch.Tensor
                    ):
                        feature.value = feature.value.to(self.device)

                    # Move learning rule attributes
                    if hasattr(feature, "learning_rule") and feature.learning_rule:
                        rule = feature.learning_rule
                        for attr in ["reduction", "weight_decay"]:
                            if hasattr(rule, attr):
                                val = getattr(rule, attr)
                                if isinstance(val, torch.Tensor):
                                    setattr(rule, attr, val.to(self.device))

    def get_weight_from_connection(
        self, connection_key: Tuple[str, str]
    ) -> Optional[torch.Tensor]:
        """Extract weight tensor from a connection.

        Args:
            connection_key: Tuple of (source_layer, target_layer).

        Returns:
            Weight tensor if found, None otherwise.
        """
        if self.network is None:
            return None

        if connection_key not in self.network.connections:
            return None

        connection = self.network.connections[connection_key]

        # Check direct weight attribute
        if hasattr(connection, "w") and connection.w is not None:
            return connection.w

        # Check pipeline features (MulticompartmentConnection)
        if hasattr(connection, "pipeline"):
            for feature in connection.pipeline:
                if hasattr(feature, "value") and feature.value is not None:
                    return feature.value

        return None
