"""Training module for MNIST SNN model."""

# Apply compatibility patches BEFORE importing bindsnet
import neuro_sim.compat  # noqa: F401

import logging
from pathlib import Path
from time import time as t
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from bindsnet.evaluation import all_activity, assign_labels
import bindsnet.evaluation as bindsnet_eval

# Patch proportion_weighting AFTER import to fix device handling
neuro_sim.compat.patch_proportion_weighting()

# Use the patched version from the module
proportion_weighting = bindsnet_eval.proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor

from neuro_sim.config import Config
from neuro_sim.data.dataset import get_dataset, get_dataloader
from neuro_sim.utils.model_persistence import ModelPersistence

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for DiehlAndCook2015 SNN model on MNIST."""

    def __init__(
        self,
        config: Config,
        checkpoint_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Configuration object
            checkpoint_dir: Directory for checkpoints (overrides config if provided)
            model_name: Name for the saved model file (without .pt extension). 
                       If None, uses default "mnist_snn_model"
        """
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir or config.paths["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name or "mnist_snn_model"
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize network
        self.network = self._build_network()
        
        # Initialize neuron assignments
        n_neurons = config.model["n_neurons"]
        n_classes = 10
        self.assignments = -torch.ones(n_neurons, device=self.device)
        self.proportions = torch.zeros((n_neurons, n_classes), device=self.device)
        self.rates = torch.zeros((n_neurons, n_classes), device=self.device)
        
        # Training state
        self.accuracy_history = {"all": [], "proportion": []}
        
        # Get optimization flags (default to False for speed)
        self.enable_monitoring = self.config.training.get("enable_monitoring", False)
        self.enable_grad_tracking = self.config.training.get("enable_grad_tracking", False)
        self.enable_weight_norm = self.config.training.get("enable_weight_norm", False)
        
        # Disable gradient tracking globally if not enabled (STDP doesn't need it)
        if not self.enable_grad_tracking:
            torch.set_grad_enabled(False)
            logger.info("Gradient tracking disabled for faster training (STDP doesn't require gradients)")
        
        # Setup monitors (conditionally)
        self._setup_monitors()
        
        # Setup spike recording
        self._setup_spike_recording()
    
    def _setup_device(self) -> torch.device:
        """Set up compute device."""
        gpu = self.config.training["gpu"]
        
        if gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.config.training["seed"])
        else:
            device = torch.device("cpu")
            torch.manual_seed(self.config.training["seed"])
        
        logger.info(f"Using device: {device}")
        return device
    
    def _build_network(self) -> DiehlAndCook2015:
        """Build the network model."""
        model_config = self.config.model
        training_config = self.config.training

        w_dtype_str = model_config.get("w_dtype", "float32")
        try:
            w_dtype = getattr(torch, w_dtype_str)
        except AttributeError:
            logger.warning(f"Unsupported w_dtype '{w_dtype_str}', defaulting to float32")
            w_dtype = torch.float32
        
        network = DiehlAndCook2015(
            device=self.device,
            batch_size=training_config["batch_size"],
            sparse=bool(model_config.get("sparse", False)),
            n_inpt=model_config["n_inpt"],
            n_neurons=model_config["n_neurons"],
            exc=model_config["exc"],
            inh=model_config["inh"],
            dt=model_config["dt"],
            norm=model_config["norm"],
            nu=tuple(model_config["nu"]),
            theta_plus=model_config["theta_plus"],
            inpt_shape=tuple(model_config["inpt_shape"]),
            w_dtype=w_dtype,
            inh_thresh=model_config.get("inh_thresh", -40.0),
            exc_thresh=model_config.get("exc_thresh", -52.0),
        )
        
        # Move network to device after creation
        if self.device.type == "cuda":
            network.to("cuda")
            # Explicitly move connection weights and pipeline features to GPU
            # bindsnet's network.to() doesn't always move pipeline feature .value attributes
            for conn_name, connection in network.connections.items():
                # Move connection weights
                if hasattr(connection, 'w') and isinstance(connection.w, torch.Tensor):
                    connection.w = connection.w.to(self.device)
                # Move pipeline features (like Weight objects with .value)
                if hasattr(connection, 'pipeline'):
                    for feature in connection.pipeline:
                        # Move feature value
                        if hasattr(feature, 'value') and isinstance(feature.value, torch.Tensor):
                            feature.value = feature.value.to(self.device)
                        # Move learning rule tensor attributes (like feature_value, nu)
                        if hasattr(feature, 'learning_rule'):
                            learning_rule = feature.learning_rule
                            for attr_name in dir(learning_rule):
                                if not attr_name.startswith('_'):
                                    try:
                                        attr = getattr(learning_rule, attr_name)
                                        if isinstance(attr, torch.Tensor) and attr.device.type != self.device.type:
                                            setattr(learning_rule, attr_name, attr.to(self.device))
                                    except (AttributeError, RuntimeError):
                                        pass
                        # Also move any other tensor attributes in features
                        for attr_name in dir(feature):
                            if not attr_name.startswith('_') and attr_name not in ['value', 'learning_rule']:
                                try:
                                    attr = getattr(feature, attr_name)
                                    if isinstance(attr, torch.Tensor) and attr.device.type != self.device.type:
                                        setattr(feature, attr_name, attr.to(self.device))
                                except (AttributeError, RuntimeError):
                                    pass
        
        logger.info("Network built successfully")
        return network
    
    def _setup_monitors(self):
        """Set up network monitors (conditionally based on enable_monitoring flag)."""
        if not self.enable_monitoring:
            logger.info("Monitors disabled for faster training (use --enable-monitoring to enable)")
            # Create minimal spike monitor only for Ae layer (needed for spike recording)
            time = self.config.training["time"]
            dt = self.config.model["dt"]
            time_steps = int(time / dt)
            sparse = bool(self.config.model.get("sparse", False))
            training_config = self.config.training
            
            # Only create spike monitor for Ae layer (required for training)
            self.spikes = {
                "Ae": Monitor(
                    self.network.layers["Ae"],
                    state_vars=["s"],
                    time=time_steps,
                    batch_size=training_config["batch_size"],
                    device=self.device,
                    sparse=sparse,
                )
            }
            self.network.add_monitor(self.spikes["Ae"], name="Ae_spikes")
            
            # Set voltage monitors to None
            self.exc_voltage_monitor = None
            self.inh_voltage_monitor = None
            return
        
        # Full monitoring enabled
        time = self.config.training["time"]
        dt = self.config.model["dt"]
        time_steps = int(time / dt)
        sparse = bool(self.config.model.get("sparse", False))
        
        # Voltage monitors
        training_config = self.config.training
        self.exc_voltage_monitor = Monitor(
            self.network.layers["Ae"],
            ["v"],
            time=time_steps,
            batch_size=training_config["batch_size"],
            device=self.device,
        )
        self.inh_voltage_monitor = Monitor(
            self.network.layers["Ai"],
            ["v"],
            time=time_steps,
            batch_size=training_config["batch_size"],
            device=self.device,
        )
        self.network.add_monitor(self.exc_voltage_monitor, name="exc_voltage")
        self.network.add_monitor(self.inh_voltage_monitor, name="inh_voltage")
        
        # Spike monitors
        self.spikes = {}
        for layer in set(self.network.layers):
            self.spikes[layer] = Monitor(
                self.network.layers[layer],
                state_vars=["s"],
                time=time_steps,
                batch_size=training_config["batch_size"],
                device=self.device,
                sparse=sparse,
            )
            self.network.add_monitor(self.spikes[layer], name=f"{layer}_spikes")
    
    def _setup_spike_recording(self):
        """Set up spike recording buffers."""
        training_config = self.config.training
        n_train = training_config["n_train"]
        batch_size = training_config["batch_size"]
        n_updates = training_config["n_updates"]
        time = training_config["time"]
        dt = self.config.model["dt"]
        
        # Use update_interval from config if provided, otherwise calculate it
        if "update_interval" in training_config:
            update_interval = training_config["update_interval"]
            update_steps = update_interval // batch_size
        else:
            update_steps = int(n_train / batch_size / n_updates)
            update_interval = update_steps * batch_size
        
        self.update_steps = update_steps
        self.update_interval = update_interval
        
        # Initialize spike record buffers on the correct device
        self.spike_record = []
        for _ in range(update_interval // batch_size):
            spike_buffer = torch.zeros(
                (batch_size, int(time / dt), self.config.model["n_neurons"]),
                device=self.device,
            )
            if self.config.model.get("sparse", False):
                spike_buffer = spike_buffer.to_sparse()
            self.spike_record.append(spike_buffer)
        self.spike_record_idx = 0
    
    def train(self, resume_from: Optional[str] = None) -> Dict[str, float]:
        """
        Train the model.

        Args:
            resume_from: Optional path to checkpoint to resume from

        Returns:
            Final accuracy metrics
        """
        if resume_from:
            self.load_checkpoint(resume_from)
        
        training_config = self.config.training
        logger.info("Starting training...")
        start_time = t()
        
        # Load dataset
        dataset = get_dataset(
            root=self.config.data.get("root"),
            train=True,
            intensity=training_config["intensity"],
            time=training_config["time"],
            dt=self.config.model["dt"],
        )
        
        for epoch in range(training_config["n_epochs"]):
            if epoch % training_config["progress_interval"] == 0:
                logger.info(f"Epoch {epoch}/{training_config['n_epochs']}")
            
            epoch_start = t()
            labels = []
            
            train_dataloader = get_dataloader(
                dataset,
                batch_size=training_config["batch_size"],
                shuffle=True,
                num_workers=self.config.data["n_workers"],
                pin_memory=self.device.type == "cuda",
            )
            
            pbar = tqdm(total=training_config["n_train"], desc=f"Epoch {epoch}")
            
            for step, batch in enumerate(train_dataloader):
                if step * training_config["batch_size"] > training_config["n_train"]:
                    break
                
                # Update assignments periodically
                if step % self.update_steps == 0 and step > 0:
                    self._update_assignments(labels)
                    labels = []
                    
                    # Normalize weights at update interval if weight normalization is disabled during training
                    if not self.enable_weight_norm:
                        self._normalize_weights()
                
                # Normalize weights every sample if enabled (matches Brian2 behavior)
                # Brian2 calls normalize_weights() BEFORE each sample during training
                if self.enable_weight_norm:
                    self._normalize_weights()
                
                # Prepare inputs
                inputs = {"X": batch["encoded_image"]}
                if self.device.type == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                labels.extend(batch["label"].tolist())
                
                # Run network
                self.network.run(inputs=inputs, time=training_config["time"])
                
                # Record spikes - ensure they're on the correct device
                s = self.spikes["Ae"].get("s").permute((1, 0, 2))
                # Move to device if needed (handles both sparse and dense tensors)
                if s.is_sparse:
                    s = s.to(self.device)
                else:
                    s = s.to(self.device)
                self.spike_record[self.spike_record_idx] = s
                self.spike_record_idx += 1
                if self.spike_record_idx == len(self.spike_record):
                    self.spike_record_idx = 0
                
                # Reset state
                self.network.reset_state_variables()
                pbar.update(training_config["batch_size"])
            
            pbar.close()
            
            # Final assignment update at end of epoch if there are remaining labels
            # Only update if we have enough labels to match the spike records
            # (spike_record contains update_interval samples, so we need at least that many labels)
            if labels and len(labels) >= self.update_interval:
                self._update_assignments(labels)
                # Normalize weights after final assignment update if weight normalization is disabled
                if not self.enable_weight_norm:
                    self._normalize_weights()
            elif labels:
                # If we have some labels but not enough, log it but don't update
                logger.debug(
                    f"End of epoch {epoch + 1}: Skipping assignment update - "
                    f"only {len(labels)} labels remaining (need {self.update_interval} for full update)"
                )
            
            # Final weight normalization at end of epoch if disabled during training
            if not self.enable_weight_norm:
                self._normalize_weights()
            
            # Validate assignments at end of epoch
            assigned_classes = torch.unique(self.assignments).tolist()
            missing_classes = set(range(10)) - set(assigned_classes)
            if missing_classes:
                logger.warning(
                    f"End of epoch {epoch + 1}: Missing class assignments: {missing_classes}. "
                    f"Attempting to fix..."
                )
                self.assignments = self._ensure_all_classes_assigned(
                    self.assignments, self.proportions, self.rates, missing_classes
                )
                final_assigned = torch.unique(self.assignments).tolist()
                logger.info(f"After fix: Assigned classes: {sorted(final_assigned)}")
            
            # Save checkpoint
            if (epoch + 1) % training_config.get("checkpoint_interval", 1) == 0:
                self.save_checkpoint(epoch + 1)
            
            epoch_time = t() - epoch_start
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        
        total_time = t() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # Final validation: ensure all classes are assigned
        final_assigned_classes = torch.unique(self.assignments).tolist()
        final_missing = set(range(10)) - set(final_assigned_classes)
        if final_missing:
            logger.warning(
                f"Final validation: Still missing classes {final_missing}. "
                f"Attempting final fix..."
            )
            self.assignments = self._ensure_all_classes_assigned(
                self.assignments, self.proportions, self.rates, final_missing
            )
            final_assigned = torch.unique(self.assignments).tolist()
            logger.info(f"Final assigned classes: {sorted(final_assigned)}")
        
        # Save final model
        self.save_model()
        
        # Return final metrics
        if self.accuracy_history["all"]:
            return {
                "all_activity_accuracy": np.mean(self.accuracy_history["all"]),
                "proportion_accuracy": np.mean(self.accuracy_history["proportion"]),
            }
        return {}
    
    def _update_assignments(self, labels: list):
        """Update neuron label assignments."""
        if not labels:
            return
        
        label_tensor = torch.tensor(labels, device=self.device)
        n_classes = 10
        
        # Ensure spike_record_tensor is on the correct device
        # For sparse tensors, we need to convert to dense first, then back to sparse if needed
        spike_records = []
        for record in self.spike_record:
            if record.is_sparse:
                record = record.to_dense().to(self.device)
            else:
                record = record.to(self.device)
            spike_records.append(record)
        spike_record_tensor = torch.cat(spike_records, dim=0)
        
        # Ensure labels match spike records - spike_record contains update_interval samples
        # If we have fewer labels, we need to match them properly
        n_spike_samples = spike_record_tensor.shape[0]
        n_label_samples = len(label_tensor)
        
        if n_label_samples != n_spike_samples:
            # If we have fewer labels than spikes, only use the last n_label_samples spikes
            # This can happen at the end of an epoch
            if n_label_samples < n_spike_samples:
                logger.debug(
                    f"Label count ({n_label_samples}) < spike count ({n_spike_samples}). "
                    f"Using last {n_label_samples} spikes to match labels."
                )
                spike_record_tensor = spike_record_tensor[-n_label_samples:]
            else:
                # If we have more labels than spikes (shouldn't happen), truncate labels
                logger.warning(
                    f"Label count ({n_label_samples}) > spike count ({n_spike_samples}). "
                    f"Truncating labels to match spikes."
                )
                label_tensor = label_tensor[:n_spike_samples]
        
        # Ensure assignments and proportions are on the correct device
        assignments = self.assignments.to(self.device)
        proportions = self.proportions.to(self.device)
        
        # Log class distribution in this batch
        unique_labels, counts = torch.unique(label_tensor, return_counts=True)
        label_distribution = {int(l): int(c) for l, c in zip(unique_labels, counts)}
        missing_classes = set(range(n_classes)) - set(label_distribution.keys())
        if missing_classes:
            logger.debug(f"Missing classes in batch: {missing_classes}")
        
        # Get predictions
        all_activity_pred = all_activity(
            spikes=spike_record_tensor,
            assignments=assignments,
            n_labels=n_classes,
        )
        proportion_pred = proportion_weighting(
            spikes=spike_record_tensor,
            assignments=assignments,
            proportions=proportions,
            n_labels=n_classes,
        )
        
        # Compute accuracy
        all_acc = (
            100
            * torch.sum(label_tensor.long() == all_activity_pred.to(self.device)).item()
            / len(label_tensor)
        )
        prop_acc = (
            100
            * torch.sum(label_tensor.long() == proportion_pred.to(self.device)).item()
            / len(label_tensor)
        )
        
        self.accuracy_history["all"].append(all_acc)
        self.accuracy_history["proportion"].append(prop_acc)
        
        # Update assignments
        new_assignments, new_proportions, new_rates = assign_labels(
            spikes=spike_record_tensor,
            labels=label_tensor,
            n_labels=n_classes,
            rates=self.rates,
        )
        
        # Check for missing classes in assignments
        assigned_classes = torch.unique(new_assignments).tolist()
        missing_assigned_classes = set(range(n_classes)) - set(assigned_classes)
        
        if missing_assigned_classes:
            logger.warning(
                f"Classes without assigned neurons: {missing_assigned_classes}. "
                f"Attempting to assign neurons to missing classes..."
            )
            # Try to assign neurons to missing classes
            new_assignments = self._ensure_all_classes_assigned(
                new_assignments, new_proportions, new_rates, missing_assigned_classes
            )
        
        self.assignments = new_assignments
        self.proportions = new_proportions
        self.rates = new_rates
        
        # Log assignment distribution
        final_assigned_classes = torch.unique(self.assignments).tolist()
        assignment_counts = {
            int(c): int((self.assignments == c).sum().item())
            for c in final_assigned_classes
        }
        logger.info(
            f"All activity accuracy: {all_acc:.2f}%, "
            f"Proportion accuracy: {prop_acc:.2f}% | "
            f"Assigned classes: {sorted(final_assigned_classes)} | "
            f"Assignment counts: {assignment_counts}"
        )
    
    def _ensure_all_classes_assigned(
        self,
        assignments: torch.Tensor,
        proportions: torch.Tensor,
        rates: torch.Tensor,
        missing_classes: set,
    ) -> torch.Tensor:
        """
        Ensure all classes have at least one neuron assigned.
        
        For missing classes, assign neurons that have the highest firing rate
        for that class, even if it's not their primary class.
        
        Args:
            assignments: Current neuron assignments
            proportions: Current proportions tensor
            rates: Current rates tensor
            missing_classes: Set of class indices that need assignments
            
        Returns:
            Updated assignments tensor
        """
        n_neurons = assignments.shape[0]
        updated_assignments = assignments.clone()
        
        for missing_class in missing_classes:
            # Find neurons that fire most for this missing class
            # (even if they're currently assigned to another class)
            class_rates = rates[:, missing_class]
            
            if class_rates.sum() > 0:
                # Find neurons with highest rates for this class
                # Prefer neurons that are currently assigned to classes with many neurons
                # to avoid taking neurons from underrepresented classes
                assignment_counts = {
                    int(c): int((assignments == c).sum().item())
                    for c in torch.unique(assignments).tolist()
                }
                
                # Sort neurons by their rate for the missing class
                sorted_indices = torch.argsort(class_rates, descending=True)
                
                # Find a neuron to reassign
                # Prefer reassigning from classes with many neurons
                for neuron_idx in sorted_indices:
                    current_class = int(assignments[neuron_idx].item())
                    # Only reassign if the current class has multiple neurons
                    if assignment_counts.get(current_class, 0) > 1:
                        updated_assignments[neuron_idx] = missing_class
                        logger.debug(
                            f"Reassigned neuron {neuron_idx.item()} from class {current_class} "
                            f"to missing class {missing_class}"
                        )
                        break
                else:
                    # If we couldn't find a good candidate, just assign the highest-rate neuron
                    if len(sorted_indices) > 0:
                        neuron_idx = sorted_indices[0]
                        old_class = int(assignments[neuron_idx].item())
                        updated_assignments[neuron_idx] = missing_class
                        logger.debug(
                            f"Reassigned neuron {neuron_idx.item()} from class {old_class} "
                            f"to missing class {missing_class} (forced)"
                        )
            else:
                # If no neurons fire for this class, assign a random neuron
                # Prefer reassigning from classes with many neurons
                assignment_counts = {
                    int(c): int((assignments == c).sum().item())
                    for c in torch.unique(assignments).tolist()
                }
                
                # Find class with most neurons
                if assignment_counts:
                    max_class = max(assignment_counts.items(), key=lambda x: x[1])[0]
                    # Find a neuron assigned to that class
                    candidates = torch.nonzero(assignments == max_class).view(-1)
                    if len(candidates) > 0:
                        neuron_idx = candidates[0]
                        updated_assignments[neuron_idx] = missing_class
                        logger.debug(
                            f"Randomly reassigned neuron {neuron_idx.item()} from class {max_class} "
                            f"to missing class {missing_class} (no firing data)"
                        )
        
        return updated_assignments
    
    def _normalize_weights(self):
        """Normalize connection weights (called periodically when weight normalization is disabled during training)."""
        if not hasattr(self.network, "connections"):
            return
        
        norm_value = self.config.model.get("norm", 78.4)
        
        for conn_name, connection in self.network.connections.items():
            # Only normalize X -> Ae connections (input to excitatory)
            if isinstance(conn_name, tuple) and conn_name[0] == "X" and conn_name[1] == "Ae":
                w = None
                w_is_sparse = False
                
                # Try to get weight from pipeline feature first (bindsnet standard)
                if hasattr(connection, "pipeline") and connection.pipeline:
                    for feature in connection.pipeline:
                        if hasattr(feature, "value") and feature.value is not None:
                            w = feature.value
                            w_is_sparse = w.is_sparse if isinstance(w, torch.Tensor) else False
                            break
                
                # Fallback to direct w attribute
                if w is None and hasattr(connection, "w") and connection.w is not None:
                    w = connection.w
                    w_is_sparse = w.is_sparse if isinstance(w, torch.Tensor) else False
                
                if w is not None and isinstance(w, torch.Tensor):
                    # Optimized normalization: minimize sparse<->dense conversions
                    # For sparse tensors, we can compute sum without full conversion
                    if w_is_sparse:
                        # Compute sum efficiently: sparse sum is O(nnz), much faster than full conversion
                        # Sum over input dimension (dim=0) for each neuron
                        w_sum = w.sum(dim=0).to_dense()  # Result is small (n_neurons), so to_dense() is cheap
                        w_sum = torch.clamp(w_sum, min=1e-8)
                        
                        # For normalization, we need to divide each weight by its column sum
                        # This requires accessing all weights, so we convert to dense
                        # But we do it in-place where possible to minimize memory overhead
                        w_dense = w.to_dense()
                        w_normalized = w_dense / w_sum * norm_value
                        w_normalized = w_normalized.to_sparse()
                    else:
                        # Dense tensor: straightforward normalization
                        w_sum = w.sum(dim=0)
                        w_sum = torch.clamp(w_sum, min=1e-8)
                        w_normalized = w / w_sum * norm_value
                    
                    # Update the weight tensor
                    if hasattr(connection, "pipeline") and connection.pipeline:
                        for feature in connection.pipeline:
                            if hasattr(feature, "value") and feature.value is not None:
                                feature.value.data = w_normalized
                                break
                    elif hasattr(connection, "w"):
                        connection.w.data = w_normalized
                    
                    # Logging: get sum for verification (minimal overhead)
                    final_sum = w_normalized.sum(dim=0)
                    if final_sum.is_sparse:
                        final_sum = final_sum.to_dense()
                    logger.debug(f"Normalized {conn_name} weights (sum per neuron: {final_sum.min().item():.2f}-{final_sum.max().item():.2f})")
    
    def save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        accuracy = {
            "all": self.accuracy_history["all"][-1] if self.accuracy_history["all"] else 0.0,
            "proportion": self.accuracy_history["proportion"][-1]
            if self.accuracy_history["proportion"]
            else 0.0,
        }
        
        ModelPersistence.save_checkpoint(
            checkpoint_path=str(checkpoint_path),
            network=self.network,
            assignments=self.assignments,
            proportions=self.proportions,
            rates=self.rates,
            epoch=epoch,
            accuracy=accuracy,
            config=self.config.to_dict(),
        )
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = ModelPersistence.load_checkpoint(checkpoint_path, device=str(self.device))
        
        # Load network weights (support both old and new format)
        weights = checkpoint.get("network_weights") or checkpoint.get("network_state")
        if weights and hasattr(self.network, "load_state_dict"):
            # Only load weights that exist in the current network (ignore neuron states)
            network_state = self.network.state_dict()
            filtered_weights = {k: v for k, v in weights.items() if k in network_state and '.w' in k}
            if filtered_weights:
                network_state.update(filtered_weights)
                self.network.load_state_dict(network_state, strict=False)
        
        self.assignments = checkpoint["assignments"]
        self.proportions = checkpoint["proportions"]
        self.rates = checkpoint["rates"]
        
        # Log checkpoint details for verification
        epoch = checkpoint.get("epoch", "unknown")
        accuracy = checkpoint.get("accuracy", {})
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"  Resuming from epoch: {epoch}")
        if accuracy:
            logger.info(f"  Previous accuracy - All activity: {accuracy.get('all', 'N/A'):.2f}%, "
                        f"Proportion: {accuracy.get('proportion', 'N/A'):.2f}%")
    
    def save_model(self):
        """Save final trained model."""
        model_dir = Path(self.config.paths["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        # Ensure .pt extension
        model_filename = self.model_name if self.model_name.endswith('.pt') else f"{self.model_name}.pt"
        model_path = model_dir / model_filename
        
        accuracy = {
            "all": np.mean(self.accuracy_history["all"]) if self.accuracy_history["all"] else 0.0,
            "proportion": np.mean(self.accuracy_history["proportion"])
            if self.accuracy_history["proportion"]
            else 0.0,
        }
        
        ModelPersistence.save_model(
            model_path=str(model_path),
            network=self.network,
            assignments=self.assignments,
            proportions=self.proportions,
            rates=self.rates,
            config=self.config.to_dict(),
            metadata={"accuracy": accuracy},
        )
        
        logger.info(f"Model saved: {model_path}")
        return str(model_path)

