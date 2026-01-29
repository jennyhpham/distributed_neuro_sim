"""Evaluation/inference module for MNIST SNN model."""

# Apply compatibility patches BEFORE importing bindsnet
import neuro_sim.compat  # noqa: F401

import logging
from pathlib import Path
from time import time as t
from typing import Any, Dict, Optional, Tuple

import torch
from bindsnet.evaluation import all_activity
import bindsnet.evaluation as bindsnet_eval
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from tqdm import tqdm

# Patch proportion_weighting AFTER import to fix device handling
neuro_sim.compat.patch_proportion_weighting()

# Use the patched version from the module
proportion_weighting = bindsnet_eval.proportion_weighting

from neuro_sim.config import Config
from neuro_sim.data.dataset import get_dataset, get_dataloader
from neuro_sim.models import ModelFactory
from neuro_sim.utils.model_persistence import ModelPersistence

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for SNN models on MNIST.

    Supports multiple model architectures through the ModelFactory pattern:
    - DiehlAndCook2015: Standard STDP model with separate Ae/Ai layers
    - IncreasingInhibitionNetwork: SOM-LM-SNN with distance-weighted inhibition
    """

    def __init__(
        self,
        config: Config,
        model_path: Optional[str] = None,
    ):
        """
        Initialize evaluator.

        Args:
            config: Configuration object
            model_path: Path to trained model (if None, uses default from config)
        """
        self.config = config
        self.model_path = model_path
        self.model_wrapper = None
        self.output_layer = None

        # Setup device
        self.device = self._setup_device()

        # Load model
        self.network, self.assignments, self.proportions = self._load_model()

        # Setup monitors
        self._setup_monitors()
    
    def _setup_device(self) -> torch.device:
        """Set up compute device."""
        gpu = self.config.inference.get("gpu", self.config.training.get("gpu", True))
        
        if gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        logger.info(f"Using device: {device}")
        return device
    
    def _load_model(self) -> Tuple[Network, torch.Tensor, torch.Tensor]:
        """Load trained model using the ModelFactory pattern."""
        if self.model_path is None:
            model_dir = Path(self.config.paths["model_dir"])
            model_path = model_dir / "mnist_snn_model.pt"
        else:
            model_path = Path(self.model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Please train the model first."
            )

        logger.info(f"Loading model from: {model_path}")
        model_data = ModelPersistence.load_model(str(model_path), device=str(self.device))

        # Build network using factory - config is nested, extract model config
        model_config = model_data["config"].get("model", model_data["config"])
        network = self._build_network(model_config)

        # Load network weights if available (support both old and new format)
        weights = model_data.get("network_weights") or model_data.get("network_state")
        if weights:
            self._load_weights(network, weights)

        assignments = model_data["assignments"]
        proportions = model_data["proportions"]

        # Debug: Check assignments validity
        if isinstance(assignments, torch.Tensor):
            assignments = assignments.to(self.device)
            unique_assignments = torch.unique(assignments)
            valid_assignments = (assignments >= 0).sum().item()
            logger.info(
                f"Loaded assignments: {assignments.shape[0]} neurons, "
                f"{valid_assignments} assigned, "
                f"classes: {sorted(unique_assignments[unique_assignments >= 0].tolist())}"
            )
            if valid_assignments == 0:
                logger.error(
                    "WARNING: No neurons are assigned to any class! "
                    "This will cause poor accuracy."
                )
            if len(unique_assignments[unique_assignments >= 0]) < 10:
                logger.warning(
                    f"Only {len(unique_assignments[unique_assignments >= 0])} "
                    f"classes have assigned neurons (expected 10)"
                )

        if isinstance(proportions, torch.Tensor):
            proportions = proportions.to(self.device)

        logger.info("Model loaded successfully")
        return network, assignments, proportions

    def _load_weights(self, network: Network, weights: Dict[str, Any]) -> None:
        """Load weights into the network.

        Args:
            network: The network to load weights into
            weights: Dictionary of weight tensors from checkpoint
        """
        # Load connection weights directly into connections
        for conn_name, connection in network.connections.items():
            # Generate key for this connection
            if isinstance(conn_name, tuple):
                key = f"{conn_name[0]}_to_{conn_name[1]}.w"
            else:
                key = f"{conn_name}.w"

            if key in weights:
                weight_value = weights[key].to(self.device)
                # Try to set weight in pipeline feature (bindsnet 0.2.7)
                if hasattr(connection, "pipeline") and connection.pipeline:
                    for feature in connection.pipeline:
                        if hasattr(feature, "value"):
                            # Match tensor type (sparse/dense) of the existing value
                            if feature.value.is_sparse and not weight_value.is_sparse:
                                weight_value = weight_value.to_sparse()
                            elif not feature.value.is_sparse and weight_value.is_sparse:
                                weight_value = weight_value.to_dense()
                            feature.value.data = weight_value
                            break
                # Fallback: try direct w attribute
                elif hasattr(connection, "w") and connection.w is not None:
                    # Match tensor type (sparse/dense) of the existing w
                    if connection.w.is_sparse and not weight_value.is_sparse:
                        weight_value = weight_value.to_sparse()
                    elif not connection.w.is_sparse and weight_value.is_sparse:
                        weight_value = weight_value.to_dense()
                    # Use .data to update Parameter values
                    connection.w.data = weight_value

        # Load theta (adaptive threshold) values if available
        # Try model-specific output layer first, then fallback to "Ae" for backwards compat
        theta_key = f"{self.output_layer}.theta"
        fallback_theta_key = "Ae.theta"

        theta_loaded = False
        for key in [theta_key, fallback_theta_key]:
            if key in weights:
                theta_value = weights[key].to(self.device)
                layer_name = key.split(".")[0]
                if layer_name in network.layers and hasattr(network.layers[layer_name], "theta"):
                    network.layers[layer_name].theta.data = theta_value
                    logger.info(
                        f"Loaded theta values for {layer_name}: "
                        f"min={theta_value.min():.2f}mV, "
                        f"max={theta_value.max():.2f}mV, "
                        f"mean={theta_value.mean():.2f}mV"
                    )
                    theta_loaded = True
                    break

        if not theta_loaded:
            logger.debug("No theta values found in checkpoint")
    
    def _build_network(self, model_config: Dict) -> Network:
        """Build the network model using the factory pattern.

        Args:
            model_config: Model configuration dictionary

        Returns:
            Built network ready for inference
        """
        inference_config = self.config.inference

        # Create model wrapper using factory
        self.model_wrapper = ModelFactory.create(model_config, self.device)
        self.output_layer = self.model_wrapper.output_layer

        logger.info(
            f"Building {self.model_wrapper.model_name} network with "
            f"learning disabled for inference (output layer: {self.output_layer})"
        )

        # Build network with learning disabled (learning_enabled=False sets nu=(0,0))
        network = self.model_wrapper.build(
            batch_size=inference_config["batch_size"],
            learning_enabled=False,
            sparse=bool(model_config.get("sparse", False)),
        )

        # Move network to device
        self.model_wrapper.move_to_device(network)

        return network
    
    def _setup_monitors(self):
        """Set up network monitors."""
        time = self.config.inference["time"]
        dt = self.config.model["dt"]
        time_steps = int(time / dt)
        sparse = bool(self.config.model.get("sparse", False))
        
        # Spike monitors
        inference_config = self.config.inference
        self.spikes = {}
        for layer in set(self.network.layers):
            self.spikes[layer] = Monitor(
                self.network.layers[layer],
                state_vars=["s"],
                time=time_steps,
                batch_size=inference_config["batch_size"],
                device=self.device,
                sparse=sparse,
            )
            self.network.add_monitor(self.spikes[layer], name=f"{layer}_spikes")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Returns:
            Dictionary of accuracy metrics
        """
        inference_config = self.config.inference
        logger.info("Starting evaluation...")
        logger.info(f"Inference config: time={inference_config['time']}ms, intensity={inference_config['intensity']}, "
                   f"batch_size={inference_config['batch_size']}")
        
        # Check if inference config matches training config
        training_config = self.config.training
        if training_config.get("time") != inference_config.get("time"):
            logger.warning(f"Time mismatch: training={training_config.get('time')}ms, inference={inference_config.get('time')}ms")
        if training_config.get("intensity") != inference_config.get("intensity"):
            logger.warning(f"Intensity mismatch: training={training_config.get('intensity')}, inference={inference_config.get('intensity')}")
        
        start_time = t()
        
        # Load test dataset
        dataset = get_dataset(
            root=self.config.data.get("root"),
            train=False,
            intensity=inference_config["intensity"],
            time=inference_config["time"],
            dt=self.config.model["dt"],
        )
        
        test_dataloader = get_dataloader(
            dataset,
            batch_size=inference_config["batch_size"],
            shuffle=False,
            num_workers=self.config.data["n_workers"],
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.data["n_workers"] > 0,  # Keep workers alive for faster loading
        )
        
        n_test = inference_config["n_test"]
        accuracy = {"all": 0.0, "proportion": 0.0}
        n_classes = 10
        total_samples_processed = 0
        
        pbar = tqdm(total=n_test, desc="Evaluating")
        
        for step, batch in enumerate(test_dataloader):
            # Get actual batch size (may be smaller for last batch)
            # PoissonEncoder output shape is (time, batch, ...), so batch dim is index 1
            actual_batch_size = batch["encoded_image"].shape[1]
            expected_batch_size = inference_config["batch_size"]
            
            # Skip incomplete batches to avoid batch size mismatch
            # This is safer than padding since BindsNET's internal state is fixed-size
            if actual_batch_size < expected_batch_size:
                logger.debug(f"Skipping incomplete batch with {actual_batch_size} samples (expected {expected_batch_size})")
                break
            
            # Check if we've processed enough samples
            if total_samples_processed >= n_test:
                break
            
            # Prepare inputs
            inputs = {"X": batch["encoded_image"]}
            if self.device.type == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Reset network state before each batch to ensure clean state
            self.network.reset_state_variables()
            
            # Run network
            self.network.run(inputs=inputs, time=inference_config["time"])
            
            # Get spike record from the output layer
            spike_record = self.spikes[self.output_layer].get("s").permute((1, 0, 2))
            # Ensure everything used by BindsNET evaluators is on the same device.
            # Some Monitor outputs can remain on CPU even if the network runs on CUDA.
            spike_record = spike_record.to(self.device)
            
            # Convert sparse to dense for evaluation functions (they don't support sparse indexing)
            if spike_record.is_sparse:
                spike_record_dense = spike_record.to_dense()
            else:
                spike_record_dense = spike_record
            
            # Debug: Check spike record on first batch
            if step == 0:
                logger.info(f"Spike record shape: {spike_record.shape} (expected: (batch_size={actual_batch_size}, time_steps, n_neurons))")
                logger.info(f"Spike record stats: sum={spike_record_dense.sum().item()}, mean={spike_record_dense.float().mean().item():.6f}, "
                           f"max={spike_record_dense.float().max().item()}, non_zero={spike_record_dense.nonzero().shape[0]}")
            
            assignments = self.assignments.to(self.device)
            proportions = self.proportions.to(self.device)
            
            # Get predictions
            label_tensor = batch["label"].to(self.device) if isinstance(batch["label"], torch.Tensor) else torch.tensor(batch["label"], device=self.device)
            
            # Debug: Check assignments before prediction
            if step == 0:
                valid_assignments = (assignments >= 0).sum().item()
                unique_classes = sorted(torch.unique(assignments[assignments >= 0]).tolist())
                logger.info(f"Assignments: {valid_assignments}/{len(assignments)} valid, classes: {unique_classes}")
            
            # Use dense spike record for evaluation functions
            all_activity_pred = all_activity(
                spikes=spike_record_dense,
                assignments=assignments,
                n_labels=n_classes,
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record_dense,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )
            
            # Debug: Check predictions on first batch
            if step == 0:
                logger.info(f"First batch - Labels: {label_tensor[:10].tolist()}")
                logger.info(f"First batch - All activity pred: {all_activity_pred[:10].tolist()}")
                logger.info(f"First batch - Proportion pred: {proportion_pred[:10].tolist()}")
                matches_all = (label_tensor[:10].long() == all_activity_pred[:10]).sum().item()
                matches_prop = (label_tensor[:10].long() == proportion_pred[:10]).sum().item()
                logger.info(f"First 10 matches - All activity: {matches_all}/10, Proportion: {matches_prop}/10")
            
            # Compute accuracy
            accuracy["all"] += float(
                torch.sum(label_tensor.long() == all_activity_pred.to(self.device)).item()
            )
            accuracy["proportion"] += float(
                torch.sum(label_tensor.long() == proportion_pred.to(self.device)).item()
            )
            
            total_samples_processed += actual_batch_size
            pbar.update(actual_batch_size)
        
        pbar.close()

        # Normalize by number of samples actually processed
        # (may be slightly less than n_test if last batch was incomplete and skipped)
        actual_n_test = total_samples_processed if total_samples_processed > 0 else n_test
        accuracy["all"] = accuracy["all"] / actual_n_test
        accuracy["proportion"] = accuracy["proportion"] / actual_n_test

        if total_samples_processed < n_test:
            logger.warning(
                f"Processed {total_samples_processed} samples instead of {n_test} "
                f"(skipped incomplete last batch to avoid batch size mismatch)"
            )

        total_time = t() - start_time
        logger.info(f"Evaluation completed in {total_time:.2f}s")
        logger.info(f"All activity accuracy: {accuracy['all']*100:.2f}%")
        logger.info(f"Proportion weighting accuracy: {accuracy['proportion']*100:.2f}%")
        
        return accuracy
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on a batch of data.

        Args:
            batch: Batch dictionary with 'encoded_image' key

        Returns:
            Tuple of (all_activity_predictions, proportion_predictions)
        """
        inference_config = self.config.inference
        n_classes = 10
        
        # Get actual batch size
        actual_batch_size = batch["encoded_image"].shape[0]
        expected_batch_size = inference_config["batch_size"]
        
        # Prepare inputs
        inputs = {"X": batch["encoded_image"]}
        if self.device.type == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Handle variable batch sizes: pad if necessary
        if actual_batch_size < expected_batch_size:
            pad_size = expected_batch_size - actual_batch_size
            if inputs["X"].dim() == 3:  # (batch, time, features)
                padding = torch.zeros(
                    (pad_size, inputs["X"].shape[1], inputs["X"].shape[2]),
                    device=inputs["X"].device,
                    dtype=inputs["X"].dtype
                )
            else:
                padding = torch.zeros(
                    (pad_size, *inputs["X"].shape[1:]),
                    device=inputs["X"].device,
                    dtype=inputs["X"].dtype
                )
            inputs["X"] = torch.cat([inputs["X"], padding], dim=0)
        
        # Run network
        self.network.run(inputs=inputs, time=inference_config["time"])

        # Get spike record from the output layer
        spike_record = self.spikes[self.output_layer].get("s").permute((1, 0, 2))
        spike_record = spike_record.to(self.device)
        
        # Only use the actual batch size (slice out padding)
        spike_record = spike_record[:actual_batch_size]
        
        assignments = self.assignments.to(self.device)
        proportions = self.proportions.to(self.device)
        
        # Get predictions
        all_activity_pred = all_activity(
            spikes=spike_record,
            assignments=assignments,
            n_labels=n_classes,
        )
        proportion_pred = proportion_weighting(
            spikes=spike_record,
            assignments=assignments,
            proportions=proportions,
            n_labels=n_classes,
        )
        
        # Reset state
        self.network.reset_state_variables()
        
        return all_activity_pred, proportion_pred

