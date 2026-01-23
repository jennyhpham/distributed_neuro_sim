#!/usr/bin/env python3
"""
Convert Brian2 pretrained weights to BindsNET format.

This script loads weights from Brian2's numpy format and converts them
to a BindsNET-compatible checkpoint file that can be loaded directly.
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional

from neuro_sim.utils.model_persistence import ModelPersistence
from neuro_sim.config import Config
from bindsnet.models import DiehlAndCook2015


def load_brian2_weights(
    weights_dir: str,
    n_input: int = 784,
    n_e: int = 400,
    n_i: int = 400,
) -> Dict[str, np.ndarray]:
    """
    Load weights from Brian2 format.
    
    Brian2 saves weights as sparse format: (i, j, w) where:
    - i: source neuron index
    - j: target neuron index  
    - w: weight value
    
    Args:
        weights_dir: Directory containing Brian2 weight files
        n_input: Number of input neurons (784 for MNIST)
        n_e: Number of excitatory neurons (400)
        n_i: Number of inhibitory neurons (400)
        
    Returns:
        Dictionary mapping connection names to weight matrices
    """
    weights_dir = Path(weights_dir)
    weights = {}
    
    # Load input to excitatory weights (XeAe)
    xeae_path = weights_dir / "XeAe.npy"
    if xeae_path.exists():
        print(f"Loading {xeae_path}...")
        sparse_weights = np.load(xeae_path)
        
        # Convert sparse format to dense matrix
        weight_matrix = np.zeros((n_input, n_e))
        if len(sparse_weights) > 0:
            i_indices = np.int32(sparse_weights[:, 0])
            j_indices = np.int32(sparse_weights[:, 1])
            w_values = sparse_weights[:, 2]
            weight_matrix[i_indices, j_indices] = w_values
        
        weights["XeAe"] = weight_matrix
        print(f"  Loaded XeAe: shape {weight_matrix.shape}, "
              f"non-zero: {np.count_nonzero(weight_matrix)}, "
              f"min: {weight_matrix.min():.6f}, max: {weight_matrix.max():.6f}")
    else:
        print(f"Warning: {xeae_path} not found")
    
    # Load theta (adaptive threshold) values
    theta_path = weights_dir / "theta_A.npy"
    if theta_path.exists():
        print(f"Loading {theta_path}...")
        theta = np.load(theta_path)
        weights["theta_A"] = theta
        print(f"  Loaded theta_A: shape {theta.shape}, "
              f"min: {theta.min():.6f}, max: {theta.max():.6f}")
    else:
        print(f"Warning: {theta_path} not found")
    
    # Load recurrent connections from random directory (if available)
    random_dir = weights_dir.parent / "random"
    if random_dir.exists():
        for conn_name in ["AeAi", "AiAe"]:
            conn_path = random_dir / f"{conn_name}.npy"
            if conn_path.exists():
                print(f"Loading {conn_path}...")
                sparse_weights = np.load(conn_path)
                
                # Determine dimensions
                if conn_name == "AeAi":
                    n_src, n_tgt = n_e, n_i
                else:  # AiAe
                    n_src, n_tgt = n_i, n_e
                
                weight_matrix = np.zeros((n_src, n_tgt))
                if len(sparse_weights) > 0:
                    i_indices = np.int32(sparse_weights[:, 0])
                    j_indices = np.int32(sparse_weights[:, 1])
                    w_values = sparse_weights[:, 2]
                    weight_matrix[i_indices, j_indices] = w_values
                
                weights[conn_name] = weight_matrix
                print(f"  Loaded {conn_name}: shape {weight_matrix.shape}, "
                      f"non-zero: {np.count_nonzero(weight_matrix)}")
    
    return weights


def convert_to_bindsnet_checkpoint(
    brian2_weights: Dict[str, np.ndarray],
    config_path: Optional[str] = None,
    output_path: str = "models/brian2_pretrained.pt",
    device: str = "cpu",
) -> None:
    """
    Convert Brian2 weights to BindsNET checkpoint format.
    
    Args:
        brian2_weights: Dictionary of Brian2 weights
        config_path: Path to config file (for model architecture)
        output_path: Path to save the checkpoint
        device: Device to create tensors on
    """
    # Load config if provided
    if config_path:
        config = Config(config_path)
        model_config = config.model
    else:
        # Use defaults matching Brian2
        model_config = {
            "n_neurons": 400,
            "n_inpt": 784,
            "exc": 22.5,
            "inh": 17.5,
            "theta_plus": 0.05,
            "dt": 1.0,
            "norm": 78.0,
            "nu": [0.0001, 0.01],
            "inpt_shape": [1, 28, 28],
            "w_dtype": "float32",
            "sparse": True,
        }
    
    # Create network (we'll extract weights from it)
    print("\nCreating BindsNET network...")
    w_dtype = getattr(torch, model_config.get("w_dtype", "float32"))
    
    network = DiehlAndCook2015(
        device=device,
        sparse=model_config.get("sparse", True),
        batch_size=1,  # Doesn't matter for weight loading
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
    
    # Convert and set weights
    print("\nConverting weights to BindsNET format...")
    network_weights = {}

    # Convert XeAe weights (input to excitatory)
    if "XeAe" in brian2_weights:
        xeae_weights = torch.from_numpy(brian2_weights["XeAe"]).to(w_dtype).to(device)
        network_weights["X_to_Ae.w"] = xeae_weights
        print(f"  Converted XeAe: {xeae_weights.shape}, dtype: {xeae_weights.dtype}")

        # Set weights in network connection
        if ("X", "Ae") in network.connections:
            conn = network.connections[("X", "Ae")]
            if hasattr(conn, "pipeline") and conn.pipeline:
                for feature in conn.pipeline:
                    if hasattr(feature, "value"):
                        # Get the existing value to match its type and device
                        existing_value = feature.value
                        # Convert to match existing tensor type (sparse/dense) and device
                        if existing_value.is_sparse:
                            xeae_weights_sparse = xeae_weights.to_sparse()
                            feature.value.data = xeae_weights_sparse
                        else:
                            feature.value.data = xeae_weights
                        print(f"  Set weights in connection pipeline")
                        break
            elif hasattr(conn, "w"):
                if conn.w.is_sparse:
                    conn.w.data = xeae_weights.to_sparse()
                else:
                    conn.w.data = xeae_weights
                print(f"  Set weights in connection.w")

    # Convert theta (adaptive threshold) values
    if "theta_A" in brian2_weights:
        theta_values = torch.from_numpy(brian2_weights["theta_A"]).to(w_dtype).to(device)
        # Brian2 saves theta in volts (0.024-0.050 V), but BindsNET uses millivolts
        # Convert from volts to millivolts by multiplying by 1000
        theta_values_mV = theta_values * 1000.0
        network_weights["Ae.theta"] = theta_values_mV
        print(f"  Converted theta_A: {theta_values.shape}, dtype: {theta_values.dtype}")
        print(f"  Converted from volts to mV: {theta_values.min():.4f}V -> {theta_values_mV.min():.2f}mV")

        # Set theta in network layer
        if "Ae" in network.layers:
            ae_layer = network.layers["Ae"]
            if hasattr(ae_layer, "theta"):
                ae_layer.theta.data = theta_values_mV
                print(f"  Set theta in Ae layer: min={theta_values_mV.min():.2f}mV, max={theta_values_mV.max():.2f}mV")
    
    # Note: Recurrent connections (AeAi, AiAe) are typically initialized randomly
    # in BindsNET and may not match Brian2's initialization exactly.
    # We can load them if available, but they may need retraining.
    if "AeAi" in brian2_weights:
        aeai_weights = torch.from_numpy(brian2_weights["AeAi"]).to(w_dtype).to(device)
        network_weights["Ae_to_Ai.w"] = aeai_weights
        print(f"  Converted AeAi: {aeai_weights.shape}")
    
    if "AiAe" in brian2_weights:
        aiae_weights = torch.from_numpy(brian2_weights["AiAe"]).to(w_dtype).to(device)
        network_weights["Ai_to_Ae.w"] = aiae_weights
        print(f"  Converted AiAe: {aiae_weights.shape}")
    
    # Create dummy assignments and proportions (will need to be trained/loaded separately)
    # Brian2 doesn't save assignments in the weight files, so we create empty ones
    n_neurons = model_config["n_neurons"]
    assignments = torch.full((n_neurons,), -1, dtype=torch.long, device=device)
    proportions = torch.zeros((n_neurons, 10), dtype=torch.float32, device=device)
    rates = torch.zeros((n_neurons, 10), dtype=torch.float32, device=device)
    
    print("\nCreating checkpoint...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract weights from network using ModelPersistence helper
    # This gets connection weights but not layer attributes like theta
    extracted_weights = ModelPersistence._get_weights_state_dict(network)

    # Fix recurrent connection shapes - remove extra batch dimension if present
    for key in list(extracted_weights.keys()):
        if key in ["Ae_to_Ai.w", "Ai_to_Ae.w"]:
            weight = extracted_weights[key]
            # If sparse, convert to dense first
            if weight.is_sparse:
                weight = weight.to_dense()
            # Remove batch dimension if present
            if weight.dim() == 3 and weight.shape[0] == 1:
                weight = weight.squeeze(0)
                print(f"  Removed batch dimension from {key}: {extracted_weights[key].shape} -> {weight.shape}")
                extracted_weights[key] = weight

    # Manually add theta to the weights dict (not extracted by default)
    if "theta_A" in brian2_weights:
        theta_values = torch.from_numpy(brian2_weights["theta_A"]).to(w_dtype).to(device)
        # Convert from volts to millivolts
        theta_values_mV = theta_values * 1000.0
        extracted_weights["Ae.theta"] = theta_values_mV
        print(f"  Added theta to checkpoint: {theta_values.shape} (converted to mV: {theta_values_mV.min():.2f}-{theta_values_mV.max():.2f}mV)")

    # Create checkpoint manually (instead of using save_checkpoint which extracts weights again)
    checkpoint = {
        "epoch": 0,
        "network_state": extracted_weights,  # Use our manually constructed weights dict
        "assignments": assignments.cpu() if isinstance(assignments, torch.Tensor) else assignments,
        "proportions": proportions.cpu() if isinstance(proportions, torch.Tensor) else proportions,
        "rates": rates.cpu() if isinstance(rates, torch.Tensor) else rates,
        "accuracy": {"all": 0.0, "proportion": 0.0},
        "config": {"model": model_config},
        "metadata": {
            "source": "brian2",
            "weights_loaded": list(brian2_weights.keys()),
            "note": "Assignments need to be trained/loaded from activity files",
        },
    }

    # Save checkpoint
    torch.save(checkpoint, output_path)
    
    print(f"\n✓ Checkpoint saved to: {output_path}")
    print(f"\nNote: This checkpoint contains weights but not neuron assignments.")
    print(f"      You may need to train assignments or load them from activity files.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Brian2 pretrained weights to BindsNET format"
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="brian2/weights",
        help="Directory containing Brian2 weight files (default: brian2/weights)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (optional, uses defaults if not provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/brian2_pretrained.pt",
        help="Output path for checkpoint (default: models/brian2_pretrained.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to create tensors on (default: cpu)",
    )
    parser.add_argument(
        "--n-input",
        type=int,
        default=784,
        help="Number of input neurons (default: 784)",
    )
    parser.add_argument(
        "--n-e",
        type=int,
        default=400,
        help="Number of excitatory neurons (default: 400)",
    )
    parser.add_argument(
        "--n-i",
        type=int,
        default=400,
        help="Number of inhibitory neurons (default: 400)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Brian2 to BindsNET Weight Converter")
    print("=" * 60)
    
    # Load Brian2 weights
    print(f"\nLoading weights from: {args.weights_dir}")
    brian2_weights = load_brian2_weights(
        args.weights_dir,
        n_input=args.n_input,
        n_e=args.n_e,
        n_i=args.n_i,
    )
    
    if not brian2_weights:
        print("\nError: No weights loaded!")
        return
    
    # Convert to BindsNET format
    convert_to_bindsnet_checkpoint(
        brian2_weights,
        config_path=args.config,
        output_path=args.output,
        device=args.device,
    )
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

