"""Training module for SNN models.

Supports multiple model architectures through the model factory pattern
and training strategies for model-specific behaviors.
"""

# Apply compatibility patches BEFORE importing bindsnet
import neuro_sim.compat  # noqa: F401

import logging
from pathlib import Path
from time import time as t
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

import bindsnet.evaluation as bindsnet_eval
from bindsnet.evaluation import all_activity, assign_labels
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor

# Patch proportion_weighting AFTER import to fix device handling
neuro_sim.compat.patch_proportion_weighting()

# Use the patched version from the module
proportion_weighting = bindsnet_eval.proportion_weighting

from neuro_sim.config import Config
from neuro_sim.data.dataset import get_dataloader, get_dataset
from neuro_sim.models import ModelFactory
from neuro_sim.models.base import BaseModelWrapper
from neuro_sim.training.strategies import get_training_strategy
from neuro_sim.utils.model_persistence import ModelPersistence

logger = logging.getLogger(__name__)


class Trainer:
    """Model-agnostic trainer for SNN models on MNIST.

    Supports multiple architectures (DiehlAndCook2015, IncreasingInhibitionNetwork)
    through the model factory and training strategy patterns.

    Attributes:
        config: Configuration object.
        model_wrapper: Model wrapper providing architecture-specific methods.
        network: The underlying BindsNET network.
        strategy: Training strategy for model-specific behaviors.
    """

    def __init__(
        self,
        config: Config,
        checkpoint_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """Initialize trainer.

        Args:
            config: Configuration object.
            checkpoint_dir: Directory for checkpoints (overrides config if provided).
            model_name: Name for the saved model file (without .pt extension).
                       If None, uses default based on model type.
        """
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir or config.paths["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Default model name based on model type
        model_type = config.model.get("name", "DiehlAndCook2015")
        default_name = f"{model_type.lower()}_mnist"
        self.model_name = model_name or default_name

        # Setup device
        self.device = self._setup_device()

        # Create model wrapper using factory
        self.model_wrapper: BaseModelWrapper = ModelFactory.create(
            config.model, self.device
        )
        self.output_layer = self.model_wrapper.output_layer

        # Build network
        training_config = config.training
        self.network: Network = self.model_wrapper.build(
            batch_size=training_config["batch_size"],
            learning_enabled=True,
            sparse=bool(config.model.get("sparse", False)),
        )

        # Create training strategy
        self.strategy = get_training_strategy(
            model_type,
            self.model_wrapper,
            {**training_config, "n_neurons": config.model["n_neurons"]},
            self.device,
        )

        # Get effective batch size from strategy (may override config)
        strategy_batch_size = self.strategy.get_batch_size()
        if strategy_batch_size is not None:
            self.effective_batch_size = strategy_batch_size
            if strategy_batch_size != training_config["batch_size"]:
                logger.info(
                    f"Strategy overrides batch_size: {training_config['batch_size']} -> {strategy_batch_size}"
                )
        else:
            self.effective_batch_size = training_config["batch_size"]

        # Initialize neuron assignments
        n_neurons = config.model["n_neurons"]
        n_classes = 10
        self.assignments = -torch.ones(n_neurons, device=self.device)
        self.proportions = torch.zeros((n_neurons, n_classes), device=self.device)
        self.rates = torch.zeros((n_neurons, n_classes), device=self.device)

        # Training state
        self.accuracy_history: Dict[str, list] = {"all": [], "proportion": []}

        # Get optimization flags (default to False for speed)
        self.enable_monitoring = self.config.training.get("enable_monitoring", False)
        self.enable_grad_tracking = self.config.training.get(
            "enable_grad_tracking", False
        )
        self.enable_weight_norm = self.config.training.get("enable_weight_norm", False)

        # Disable gradient tracking globally if not enabled (STDP doesn't need it)
        if not self.enable_grad_tracking:
            torch.set_grad_enabled(False)
            logger.info(
                "Gradient tracking disabled for faster training "
                "(STDP doesn't require gradients)"
            )

        # Setup monitors (conditionally)
        self._setup_monitors()

        # Setup spike recording
        self._setup_spike_recording()

        logger.info(f"Trainer initialized for model: {model_type}")

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

    def _setup_monitors(self) -> None:
        """Set up network monitors (conditionally based on enable_monitoring flag)."""
        time = self.config.training["time"]
        dt = self.config.model["dt"]
        time_steps = int(time / dt)
        sparse = bool(self.config.model.get("sparse", False))

        if not self.enable_monitoring:
            logger.info(
                "Monitors disabled for faster training "
                "(use --enable-monitoring to enable)"
            )
            # Create minimal spike monitor only for output layer (needed for training)
            self.spikes = {
                self.output_layer: Monitor(
                    self.network.layers[self.output_layer],
                    state_vars=["s"],
                    time=time_steps,
                    batch_size=self.effective_batch_size,
                    device=self.device,
                    sparse=sparse,
                )
            }
            self.network.add_monitor(
                self.spikes[self.output_layer], name=f"{self.output_layer}_spikes"
            )

            # Set voltage monitors to None
            self.exc_voltage_monitor = None
            self.inh_voltage_monitor = None
            return

        # Full monitoring enabled - monitor all layers
        self.spikes = {}
        for layer in set(self.network.layers):
            self.spikes[layer] = Monitor(
                self.network.layers[layer],
                state_vars=["s"],
                time=time_steps,
                batch_size=self.effective_batch_size,
                device=self.device,
                sparse=sparse,
            )
            self.network.add_monitor(self.spikes[layer], name=f"{layer}_spikes")

        # Voltage monitor for output layer
        self.exc_voltage_monitor = Monitor(
            self.network.layers[self.output_layer],
            ["v"],
            time=time_steps,
            batch_size=self.effective_batch_size,
            device=self.device,
        )
        self.network.add_monitor(self.exc_voltage_monitor, name="exc_voltage")

        # Inhibitory voltage monitor only for DiehlAndCook2015
        if "Ai" in self.network.layers:
            self.inh_voltage_monitor = Monitor(
                self.network.layers["Ai"],
                ["v"],
                time=time_steps,
                batch_size=self.effective_batch_size,
                device=self.device,
            )
            self.network.add_monitor(self.inh_voltage_monitor, name="inh_voltage")
        else:
            self.inh_voltage_monitor = None

    def _setup_spike_recording(self) -> None:
        """Set up spike recording buffers.

        Uses a single pre-allocated tensor instead of a list of tensors
        to avoid memory allocations and concatenation overhead during training.
        """
        training_config = self.config.training
        n_train = training_config["n_train"]
        n_updates = training_config["n_updates"]
        time = training_config["time"]
        dt = self.config.model["dt"]

        # Use update_interval from config if provided, otherwise calculate it
        if "update_interval" in training_config:
            update_interval = training_config["update_interval"]
            update_steps = max(1, update_interval // self.effective_batch_size)
        else:
            update_steps = max(1, int(n_train / self.effective_batch_size / n_updates))
            update_interval = max(self.effective_batch_size, update_steps * self.effective_batch_size)

        self.update_steps = update_steps
        self.update_interval = update_interval

        # Pre-allocate a single contiguous tensor for all spike records
        n_samples = max(update_interval // self.effective_batch_size, 1)
        time_steps = int(time / dt)
        n_neurons = self.config.model["n_neurons"]

        # Keep spike records DENSE during training for faster operations
        self.spike_record = torch.zeros(
            (n_samples, self.effective_batch_size, time_steps, n_neurons),
            device=self.device,
            dtype=torch.float32,
        )
        self.spike_record_idx = 0

        logger.debug(
            f"Allocated spike recording buffer: "
            f"{self.spike_record.shape} = {self.spike_record.numel() * 4 / 1e9:.2f} GB"
        )

    def train(self, resume_from: Optional[str] = None) -> Dict[str, float]:
        """Train the model.

        Args:
            resume_from: Optional path to checkpoint to resume from.

        Returns:
            Final accuracy metrics.
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
            labels: list = []

            # Strategy epoch start hook
            self.strategy.on_epoch_start(epoch)

            train_dataloader = get_dataloader(
                dataset,
                batch_size=self.effective_batch_size,
                shuffle=True,
                num_workers=self.config.data["n_workers"],
                pin_memory=self.device.type == "cuda",
                persistent_workers=self.config.data["n_workers"] > 0,
            )

            pbar = tqdm(total=training_config["n_train"], desc=f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                if step * self.effective_batch_size > training_config["n_train"]:
                    break

                # Strategy pre-sample hook
                batch = self.strategy.pre_sample_hook(step, batch)

                # Strategy periodic update hook (e.g., increasing inhibition)
                self.strategy.periodic_update_hook(step)

                # Update assignments periodically
                if step % self.update_steps == 0 and step > 0:
                    self._update_assignments(labels)
                    labels = []

                    # Normalize weights at update interval
                    if not self.enable_weight_norm:
                        self._normalize_weights()

                # Normalize weights every sample if enabled (matches Brian2 behavior)
                if self.enable_weight_norm:
                    self._normalize_weights()

                # Run sample (with retry support from strategy)
                retry = True
                while retry:
                    # Prepare inputs
                    inputs = {"X": batch["encoded_image"]}
                    if self.device.type == "cuda":
                        inputs = {k: v.cuda() for k, v in inputs.items()}

                    # Run network
                    self.network.run(inputs=inputs, time=training_config["time"])

                    # Record spikes from output layer
                    s = self.spikes[self.output_layer].get("s").permute((1, 0, 2))

                    # Convert to dense and ensure on correct device
                    if s.is_sparse:
                        s = s.to_dense()
                    s = s.to(self.device)

                    # Strategy post-sample hook (retry check)
                    retry = self.strategy.post_sample_hook(step, s)

                    if retry:
                        # Get adjusted input for retry
                        batch = self.strategy.get_adjusted_input(
                            batch,
                            training_config["time"],
                            self.config.model["dt"],
                        )
                        self.network.reset_state_variables()

                # Store labels
                if isinstance(batch["label"], torch.Tensor):
                    labels.extend(batch["label"].tolist())
                else:
                    labels.append(batch["label"])

                # Record spikes into pre-allocated buffer
                self.spike_record[self.spike_record_idx] = s
                self.spike_record_idx += 1
                if self.spike_record_idx == self.spike_record.shape[0]:
                    self.spike_record_idx = 0

                # Reset state
                self.network.reset_state_variables()
                pbar.update(self.effective_batch_size)

            pbar.close()

            # Strategy epoch end hook
            self.strategy.on_epoch_end(epoch)

            # Final assignment update at end of epoch if there are remaining labels
            if labels and len(labels) >= self.update_interval:
                self._update_assignments(labels)
                if not self.enable_weight_norm:
                    self._normalize_weights()
            elif labels:
                logger.debug(
                    f"End of epoch {epoch + 1}: Skipping assignment update - "
                    f"only {len(labels)} labels remaining "
                    f"(need {self.update_interval} for full update)"
                )

            # Final weight normalization at end of epoch
            if not self.enable_weight_norm:
                self._normalize_weights()

            # Validate assignments at end of epoch
            self._validate_assignments(epoch)

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

    def _validate_assignments(self, epoch: int) -> None:
        """Validate and fix missing class assignments."""
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

    def _update_assignments(self, labels: list) -> None:
        """Update neuron label assignments.

        Uses pre-allocated spike_record tensor (no concatenation needed).
        """
        if not labels:
            return

        label_tensor = torch.tensor(labels, device=self.device)
        n_classes = 10

        # Reshape pre-allocated tensor
        spike_record_tensor = self.spike_record.reshape(
            -1, self.spike_record.shape[2], self.spike_record.shape[3]
        )

        # Ensure labels match spike records
        n_spike_samples = spike_record_tensor.shape[0]
        n_label_samples = len(label_tensor)

        if n_label_samples != n_spike_samples:
            if n_label_samples < n_spike_samples:
                logger.debug(
                    f"Label count ({n_label_samples}) < spike count ({n_spike_samples}). "
                    f"Using last {n_label_samples} spikes to match labels."
                )
                spike_record_tensor = spike_record_tensor[-n_label_samples:]
            else:
                logger.warning(
                    f"Label count ({n_label_samples}) > spike count ({n_spike_samples}). "
                    f"Truncating labels to match spikes."
                )
                label_tensor = label_tensor[:n_spike_samples]

        # Ensure tensors on correct device
        assignments = self.assignments.to(self.device)
        proportions = self.proportions.to(self.device)

        # Log class distribution
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
        """Ensure all classes have at least one neuron assigned.

        For missing classes, assign neurons that have the highest firing rate
        for that class, even if it's not their primary class.
        """
        updated_assignments = assignments.clone()

        for missing_class in missing_classes:
            class_rates = rates[:, missing_class]

            if class_rates.sum() > 0:
                assignment_counts = {
                    int(c): int((assignments == c).sum().item())
                    for c in torch.unique(assignments).tolist()
                }

                sorted_indices = torch.argsort(class_rates, descending=True)

                for neuron_idx in sorted_indices:
                    current_class = int(assignments[neuron_idx].item())
                    if assignment_counts.get(current_class, 0) > 1:
                        updated_assignments[neuron_idx] = missing_class
                        logger.debug(
                            f"Reassigned neuron {neuron_idx.item()} from class "
                            f"{current_class} to missing class {missing_class}"
                        )
                        break
                else:
                    if len(sorted_indices) > 0:
                        neuron_idx = sorted_indices[0]
                        old_class = int(assignments[neuron_idx].item())
                        updated_assignments[neuron_idx] = missing_class
                        logger.debug(
                            f"Reassigned neuron {neuron_idx.item()} from class "
                            f"{old_class} to missing class {missing_class} (forced)"
                        )
            else:
                assignment_counts = {
                    int(c): int((assignments == c).sum().item())
                    for c in torch.unique(assignments).tolist()
                }

                if assignment_counts:
                    max_class = max(assignment_counts.items(), key=lambda x: x[1])[0]
                    candidates = torch.nonzero(assignments == max_class).view(-1)
                    if len(candidates) > 0:
                        neuron_idx = candidates[0]
                        updated_assignments[neuron_idx] = missing_class
                        logger.debug(
                            f"Randomly reassigned neuron {neuron_idx.item()} from "
                            f"class {max_class} to missing class {missing_class}"
                        )

        return updated_assignments

    def _normalize_weights(self) -> None:
        """Normalize connection weights.

        Uses the model wrapper to identify the learnable connection.
        """
        if not hasattr(self.network, "connections"):
            return

        norm_value = self.config.model.get("norm", 78.4)
        learnable_conn = self.model_wrapper.get_learnable_connection()

        for conn_name, connection in self.network.connections.items():
            # Only normalize the learnable connection
            if conn_name != learnable_conn:
                continue

            w = None
            feature_ref = None

            # Try to get weight from pipeline feature first
            if hasattr(connection, "pipeline") and connection.pipeline:
                for feature in connection.pipeline:
                    if hasattr(feature, "value") and feature.value is not None:
                        w = feature.value
                        feature_ref = feature
                        break

            # Fallback to direct w attribute
            if w is None and hasattr(connection, "w") and connection.w is not None:
                w = connection.w

            if w is not None and isinstance(w, torch.Tensor):
                # Keep weights dense during training
                if w.is_sparse:
                    w = w.to_dense()
                    if feature_ref is not None:
                        feature_ref.value = w
                    elif hasattr(connection, "w"):
                        connection.w = w

                # Fast in-place normalization
                w_sum = w.sum(dim=0, keepdim=True)
                w_sum.clamp_(min=1e-8)
                w.mul_(norm_value / w_sum)

    def save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        accuracy = {
            "all": self.accuracy_history["all"][-1]
            if self.accuracy_history["all"]
            else 0.0,
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

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint = ModelPersistence.load_checkpoint(
            checkpoint_path, device=str(self.device)
        )

        # Load network weights
        weights = checkpoint.get("network_weights") or checkpoint.get("network_state")
        if weights and hasattr(self.network, "load_state_dict"):
            network_state = self.network.state_dict()
            filtered_weights = {
                k: v for k, v in weights.items() if k in network_state and ".w" in k
            }
            if filtered_weights:
                network_state.update(filtered_weights)
                self.network.load_state_dict(network_state, strict=False)

        self.assignments = checkpoint["assignments"]
        self.proportions = checkpoint["proportions"]
        self.rates = checkpoint["rates"]

        # Log checkpoint details
        epoch = checkpoint.get("epoch", "unknown")
        accuracy = checkpoint.get("accuracy", {})
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"  Resuming from epoch: {epoch}")
        if accuracy:
            logger.info(
                f"  Previous accuracy - All activity: {accuracy.get('all', 'N/A'):.2f}%, "
                f"Proportion: {accuracy.get('proportion', 'N/A'):.2f}%"
            )

    def save_model(self) -> str:
        """Save final trained model."""
        model_dir = Path(self.config.paths["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)

        model_filename = (
            self.model_name
            if self.model_name.endswith(".pt")
            else f"{self.model_name}.pt"
        )
        model_path = model_dir / model_filename

        accuracy = {
            "all": np.mean(self.accuracy_history["all"])
            if self.accuracy_history["all"]
            else 0.0,
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
