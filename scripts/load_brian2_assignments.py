#!/usr/bin/env python3
"""
Load neuron assignments from Brian2 activity files and update checkpoint.

Brian2 saves activity data (resultPopVecs and inputNumbers) which can be used
to compute neuron assignments using the same algorithm as Brian2.
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from neuro_sim.utils.model_persistence import ModelPersistence


def get_new_assignments(result_monitor, input_numbers, n_e=400):
    """
    Compute neuron assignments from activity data (matches Brian2's algorithm).
    
    Args:
        result_monitor: Array of shape (n_samples, n_e) with spike counts per sample
        input_numbers: Array of shape (n_samples,) with true labels
        n_e: Number of excitatory neurons
        
    Returns:
        Array of shape (n_e,) with class assignments for each neuron
    """
    assignments = np.ones(n_e) * -1  # Initialize as not assigned
    input_nums = np.asarray(input_numbers)
    maximum_rate = np.zeros(n_e)
    
    for j in range(10):  # 10 digit classes
        num_inputs = len(np.where(input_nums == j)[0])
        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis=0) / num_inputs
            for i in range(n_e):
                if rate[i] > maximum_rate[i]:
                    maximum_rate[i] = rate[i]
                    assignments[i] = j
    
    return assignments


def load_brian2_assignments(
    activity_dir: str,
    training_ending: str = "10000",
    n_e: int = 400,
) -> tuple:
    """
    Load assignments from Brian2 activity files.
    
    Args:
        activity_dir: Directory containing activity files
        training_ending: Suffix for training files (e.g., "10000")
        n_e: Number of excitatory neurons
        
    Returns:
        Tuple of (assignments, proportions, rates) as numpy arrays
    """
    activity_dir = Path(activity_dir)
    
    # Load training activity data
    result_monitor_path = activity_dir / f"resultPopVecs{training_ending}.npy"
    input_numbers_path = activity_dir / f"inputNumbers{training_ending}.npy"
    
    if not result_monitor_path.exists():
        raise FileNotFoundError(f"Activity file not found: {result_monitor_path}")
    if not input_numbers_path.exists():
        raise FileNotFoundError(f"Input numbers file not found: {input_numbers_path}")
    
    print(f"Loading {result_monitor_path}...")
    result_monitor = np.load(result_monitor_path)
    print(f"  Shape: {result_monitor.shape}")
    
    print(f"Loading {input_numbers_path}...")
    input_numbers = np.load(input_numbers_path)
    print(f"  Shape: {input_numbers.shape}")
    
    # Compute assignments
    print("\nComputing assignments from activity data...")
    assignments = get_new_assignments(result_monitor, input_numbers, n_e=n_e)
    
    # Count assignments per class
    assignment_counts = {i: np.sum(assignments == i) for i in range(10)}
    print(f"  Assignment counts: {assignment_counts}")
    print(f"  Unassigned neurons: {np.sum(assignments == -1)}")
    
    # Compute proportions and rates (simplified - would need full spike data for accurate rates)
    proportions = np.zeros((n_e, 10))
    rates = np.zeros((n_e, 10))
    
    for j in range(10):
        num_inputs = len(np.where(input_numbers == j)[0])
        if num_inputs > 0:
            class_rates = np.sum(result_monitor[input_numbers == j], axis=0) / num_inputs
            rates[:, j] = class_rates
            # Normalize to get proportions
            total_rate = class_rates.sum()
            if total_rate > 0:
                proportions[:, j] = class_rates / total_rate
    
    return assignments, proportions, rates


def update_checkpoint_with_assignments(
    checkpoint_path: str,
    assignments: np.ndarray,
    proportions: np.ndarray,
    rates: np.ndarray,
    device: str = "cpu",
):
    """
    Update checkpoint with assignments from Brian2.
    
    Args:
        checkpoint_path: Path to checkpoint file
        assignments: Array of assignments (n_e,)
        proportions: Array of proportions (n_e, 10)
        rates: Array of rates (n_e, 10)
        device: Device to create tensors on
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = ModelPersistence.load_checkpoint(checkpoint_path, device=device)
    
    # Convert to tensors
    assignments_tensor = torch.from_numpy(assignments).long().to(device)
    proportions_tensor = torch.from_numpy(proportions).float().to(device)
    rates_tensor = torch.from_numpy(rates).float().to(device)
    
    # Update checkpoint
    checkpoint["assignments"] = assignments_tensor
    checkpoint["proportions"] = proportions_tensor
    checkpoint["rates"] = rates_tensor
    
    # Save updated checkpoint
    print(f"Saving updated checkpoint...")
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint updated with assignments")


def main():
    parser = argparse.ArgumentParser(
        description="Load neuron assignments from Brian2 activity files"
    )
    parser.add_argument(
        "--activity-dir",
        type=str,
        default="brian2/activity",
        help="Directory containing Brian2 activity files (default: brian2/activity)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/brian2_pretrained.pt",
        help="Path to checkpoint file to update (default: models/brian2_pretrained.pt)",
    )
    parser.add_argument(
        "--training-ending",
        type=str,
        default="10000",
        help="Suffix for training files (default: 10000)",
    )
    parser.add_argument(
        "--n-e",
        type=int,
        default=400,
        help="Number of excitatory neurons (default: 400)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to create tensors on (default: cpu)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Brian2 Assignment Loader")
    print("=" * 60)
    
    # Load assignments
    assignments, proportions, rates = load_brian2_assignments(
        args.activity_dir,
        training_ending=args.training_ending,
        n_e=args.n_e,
    )
    
    # Update checkpoint
    update_checkpoint_with_assignments(
        args.checkpoint,
        assignments,
        proportions,
        rates,
        device=args.device,
    )
    
    print("\n" + "=" * 60)
    print("Assignment loading complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

