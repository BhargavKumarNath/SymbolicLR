try:
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import random
from typing import Tuple, Any
import numpy as np
from sklearn.model_selection import train_test_split

def set_seed(seed: int = 42) -> None:
    """Enforces reproducibility across PyTorch, Numpy and Python's random module
    
    Args:
        seed (int): The deterministic seed to apply globally
    """
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def get_tier1_dataloaders(
        data_dir: str = "./data_cache",
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """Tier 1 Data Loader: MNIST for ultra-rapid fitness screening.
    
    Args:
        data_dir (str): Directory to cache the dataset.
        batch_size (int): Training batch size. 256 optimal for RTX 4070.
        num_workers (int): Subprocess workers. 4 balances CPU/GPU queues on Ryzen 9.
        pin_memory (bool): Pins host memory to speed up PCIe transfers to GPU.
        seed (int): Random seed for the validation split.
        val_split (float): Fraction of the training set used for validation.
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation dataloaders.
    """
    set_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)

    targets = dataset.targets.numpy()
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_split,
        stratify=targets,
        random_state=seed
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader

def get_tier2_dataloaders(
    data_dir: str = "./data_cache",
    batch_size: int = 256,
    subset_fraction: float = 0.1,
    val_split: float = 0.2,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Tier 2 Data Loader: 10% stratified subset of CIFAR-10 for schedule refinement.
    
    Args:
        data_dir (str): Directory to cache the dataset.
        batch_size (int): Training batch size.
        subset_fraction (float): Fraction of the full dataset to utilize.
        val_split (float): Fraction of the extracted subset to use for validation.
        num_workers (int): Thread allowance for data fetching.
        pin_memory (bool): Enable non-paged memory for direct VRAM mapping.
        seed (int): Deterministic seed for stratified subsetting.
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation dataloaders.
    """
    set_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    
    targets = np.array(dataset.targets)
    
    # 1. Extract the 10% subset
    subset_idx, _ = train_test_split(
        np.arange(len(targets)),
        train_size=subset_fraction,
        stratify=targets,
        random_state=seed
    )
    
    # 2. Split the subset into Train/Val
    subset_targets = targets[subset_idx]
    train_idx, val_idx = train_test_split(
        subset_idx,
        test_size=val_split,
        stratify=subset_targets,
        random_state=seed
    )
    
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader
