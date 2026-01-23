"""Compatibility patches for bindsnet with PyTorch 2.0+."""

# IMPORTANT: This must be imported BEFORE any bindsnet imports
# Patch torch._six for PyTorch 2.0+ compatibility
# torch._six was removed in PyTorch 2.0, but bindsnet 0.2.7 still uses it

import sys
import collections.abc as container_abcs
from functools import wraps

# Patch torch._six BEFORE importing torch (if possible) or immediately after
try:
    import torch
    if not hasattr(torch, '_six'):
        # Create a compatibility module
        class _TorchSix:
            """Compatibility shim for torch._six removed in PyTorch 2.0+."""
            container_abcs = container_abcs
            string_classes = str
            int_classes = int
        
        torch._six = _TorchSix()
        # Also patch it in sys.modules for early imports
        import types
        _six_module = types.ModuleType('torch._six')
        _six_module.container_abcs = container_abcs
        _six_module.string_classes = str
        _six_module.int_classes = int
        sys.modules['torch._six'] = _six_module
except ImportError:
    pass


def patch_proportion_weighting():
    """Patch proportion_weighting to handle device correctly."""
    try:
        # Import bindsnet evaluation module
        import bindsnet.evaluation as eval_module
        original_proportion_weighting = eval_module.proportion_weighting
        
        def proportion_weighting_patched(
            spikes: torch.Tensor,
            assignments: torch.Tensor,
            proportions: torch.Tensor,
            n_labels: int,
        ) -> torch.Tensor:
            """Wrapper that ensures device consistency."""
            # Infer device from input tensors
            device = spikes.device if isinstance(spikes, torch.Tensor) else assignments.device
            
            # Get n_samples from spikes tensor
            n_samples = spikes.shape[0]
            
            # Create rates tensor on the correct device
            rates = torch.zeros(n_samples, n_labels, device=device)
            
            for i in range(n_labels):
                # Count the number of neurons with this label assignment.
                n_assigns = torch.sum(assignments == i).float()

                if n_assigns > 0:
                    # Get indices of samples with this label.
                    indices = torch.nonzero(assignments == i).view(-1)
                    
                    # Get proportion of spikes for this label.
                    prop = proportions[indices, i].unsqueeze(0)
                    
                    # Get spike counts for these neurons.
                    spike_counts = spikes[:, :, indices].sum(dim=1)
                    
                    # Weight by proportion and sum.
                    rates[:, i] += (prop * spike_counts).sum(dim=1)
            
            # Return label with highest rate.
            return rates.argmax(dim=1)
        
        # Copy docstring and metadata
        proportion_weighting_patched.__doc__ = original_proportion_weighting.__doc__
        proportion_weighting_patched.__name__ = original_proportion_weighting.__name__
        
        # Replace the function in the module
        eval_module.proportion_weighting = proportion_weighting_patched
        
    except Exception as e:
        # If patching fails, use original
        import warnings
        warnings.warn(f"Failed to patch proportion_weighting: {e}")
        pass

