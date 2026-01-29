"""Dataset loading utilities."""

# Apply compatibility patches BEFORE importing bindsnet
import neuro_sim.compat  # noqa: F401

import os
from pathlib import Path
from typing import Optional

import torch
from torchvision import transforms

from bindsnet import ROOT_DIR
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder


def get_dataset(
    dataset_name: str = "MNIST",
    encoder: Optional[PoissonEncoder] = None,
    root: Optional[str] = None,
    train: bool = True,
    download: bool = True,
    intensity: float = 128.0,
    time: int = 100,
    dt: float = 1.0,
) -> MNIST:
    """
    Get dataset for training or testing.

    Args:
        dataset_name: Name of dataset (currently only "MNIST")
        encoder: Optional encoder (if None, creates PoissonEncoder)
        root: Root directory for data (if None, uses bindsnet ROOT_DIR)
        train: Whether to use training set
        download: Whether to download if not present
        intensity: Input intensity scaling
        time: Encoding time window
        dt: Time step

    Returns:
        Dataset instance
    """
    if encoder is None:
        encoder = PoissonEncoder(time=time, dt=dt)
    
    if root is None:
        root = os.path.join(ROOT_DIR, "data", "MNIST")
    
    if dataset_name == "MNIST":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        )
        
        # MNIST in bindsnet 0.2.7 takes image_encoder and label_encoder as first two positional args
        dataset = MNIST(
            encoder,  # image_encoder
            None,     # label_encoder
            root,
            download=download,
            train=train,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def get_dataloader(
    dataset: MNIST,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = -1,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """
    Get DataLoader for dataset.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes (-1 for auto)
        pin_memory: Whether to pin memory (faster GPU transfers)
        persistent_workers: Whether to keep workers alive between epochs (not supported by bindsnet DataLoader)

    Returns:
        DataLoader instance
    """
    if num_workers == -1:
        num_workers = 0

    # Note: bindsnet.datasets.DataLoader doesn't support persistent_workers parameter,
    # so we don't pass it

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataloader

