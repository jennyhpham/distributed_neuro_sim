"""Model persistence utilities for saving and loading trained models."""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class ModelPersistence:
    """Handles saving and loading of trained models and their state."""

    @staticmethod
    def save_checkpoint(
        checkpoint_path: str,
        network: Any,
        assignments: torch.Tensor,
        proportions: torch.Tensor,
        rates: torch.Tensor,
        epoch: int,
        accuracy: Dict[str, float],
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save model checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
            network: The network model
            assignments: Neuron label assignments
            proportions: Neuron spike proportions
            rates: Neuron firing rates
            epoch: Current epoch number
            accuracy: Dictionary of accuracy metrics
            config: Training configuration
            metadata: Optional additional metadata
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract only weights directly from connections, not batch-specific neuron states
        weights_state_dict = ModelPersistence._get_weights_state_dict(network)
        
        checkpoint = {
            "epoch": epoch,
            "network_state": weights_state_dict,
            "assignments": assignments.cpu() if isinstance(assignments, torch.Tensor) else assignments,
            "proportions": proportions.cpu() if isinstance(proportions, torch.Tensor) else proportions,
            "rates": rates.cpu() if isinstance(rates, torch.Tensor) else rates,
            "accuracy": accuracy,
            "config": config,
            "metadata": metadata or {},
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load tensors to

        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # PyTorch 2.6+ defaults to weights_only=True, but we need to load numpy scalars
        # Since these are our own checkpoints, it's safe to use weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Move tensors to device
        if isinstance(checkpoint.get("assignments"), torch.Tensor):
            checkpoint["assignments"] = checkpoint["assignments"].to(device)
        if isinstance(checkpoint.get("proportions"), torch.Tensor):
            checkpoint["proportions"] = checkpoint["proportions"].to(device)
        if isinstance(checkpoint.get("rates"), torch.Tensor):
            checkpoint["rates"] = checkpoint["rates"].to(device)
        
        return checkpoint
    
    @staticmethod
    def _get_weights_state_dict(network: Any) -> Dict[str, torch.Tensor]:
        """
        Extract only connection weights directly from network connections.
        In bindsnet, weights are stored in pipeline features (pipeline[0].value).
        
        Args:
            network: The network model
            
        Returns:
            Dictionary containing only weight parameters
        """
        weights_state_dict = {}
        if hasattr(network, "connections"):
            for conn_name, connection in network.connections.items():
                # Check if connection has direct w attribute
                if hasattr(connection, "w") and connection.w is not None:
                    # Use connection name as key (e.g., ('X', 'Ae') -> 'X_to_Ae.w')
                    if isinstance(conn_name, tuple):
                        key = f"{conn_name[0]}_to_{conn_name[1]}.w"
                    else:
                        key = f"{conn_name}.w"
                    w = connection.w
                    weights_state_dict[key] = w.cpu() if isinstance(w, torch.Tensor) else w
                # Check pipeline for weights (bindsnet stores weights in pipeline features)
                elif hasattr(connection, "pipeline") and connection.pipeline:
                    # Get weight from first pipeline feature (usually Weight object)
                    for i, feature in enumerate(connection.pipeline):
                        if hasattr(feature, "value") and feature.value is not None:
                            # Use connection name as key
                            if isinstance(conn_name, tuple):
                                key = f"{conn_name[0]}_to_{conn_name[1]}.w"
                            else:
                                key = f"{conn_name}.w"
                            value = feature.value
                            weights_state_dict[key] = value.cpu() if isinstance(value, torch.Tensor) else value
                            break  # Only take first weight feature
        return weights_state_dict
    
    @staticmethod
    def _extract_weights_only(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract only connection weights from state_dict, excluding neuron states.
        
        Args:
            state_dict: Full network state dictionary
            
        Returns:
            Dictionary containing only weight parameters
        """
        weights_only = {}
        for key, value in state_dict.items():
            # Only save connection weights (w parameters), not neuron states (s, x, v, refrac_count, etc.)
            # Weight keys are like: X_to_Ae.w, Ae_to_Ai.w, Ai_to_Ae.w
            if '.w' in key or key.endswith('.w'):
                weights_only[key] = value.cpu() if isinstance(value, torch.Tensor) else value
        return weights_only
    
    @staticmethod
    def save_model(
        model_path: str,
        network: Any,
        assignments: torch.Tensor,
        proportions: torch.Tensor,
        rates: torch.Tensor,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save final trained model.

        Args:
            model_path: Path to save model
            network: The network model
            assignments: Neuron label assignments
            proportions: Neuron spike proportions
            rates: Neuron firing rates
            config: Model configuration
            metadata: Optional additional metadata
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract only weights directly from connections, not batch-specific neuron states
        weights_state_dict = ModelPersistence._get_weights_state_dict(network)
        
        model_data = {
            "network_state": weights_state_dict,
            "assignments": assignments.cpu() if isinstance(assignments, torch.Tensor) else assignments,
            "proportions": proportions.cpu() if isinstance(proportions, torch.Tensor) else proportions,
            "rates": rates.cpu() if isinstance(rates, torch.Tensor) else rates,
            "config": config,
            "metadata": metadata or {},
        }
        
        torch.save(model_data, model_path)
    
    @staticmethod
    def load_model(
        model_path: str,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load trained model.

        Args:
            model_path: Path to model file
            device: Device to load tensors to

        Returns:
            Dictionary containing model data
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # PyTorch 2.6+ defaults to weights_only=True, but we need to load numpy scalars
        # Since these are our own models, it's safe to use weights_only=False
        model_data = torch.load(model_path, map_location=device, weights_only=False)
        
        # Move tensors to device
        if isinstance(model_data.get("assignments"), torch.Tensor):
            model_data["assignments"] = model_data["assignments"].to(device)
        if isinstance(model_data.get("proportions"), torch.Tensor):
            model_data["proportions"] = model_data["proportions"].to(device)
        if isinstance(model_data.get("rates"), torch.Tensor):
            model_data["rates"] = model_data["rates"].to(device)
        
        return model_data

