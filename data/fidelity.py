import torch
import numpy as np
from typing import Tuple, Iterator
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

class VRAMDataLoader:
    """
    A custom fast DataLoader that iterates over tensors already pinned in GPU VRAM. Btpasses the CPU to GPU PCIe transfer bottleneck entirely
    """
    def __init__(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor, batch_size: int, shuffle: bool = True, drop_last: bool = True):
        self.x = x_tensor
        self.y = y_tensor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_samples = len(self.x)
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.shuffle:
            indices = torch.randperm(self.n_samples, device=self.x.device)
        else:
            indices = torch.arange(self.n_samples, device=self.x.device)
        
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = start_idx + self.batch_size
            if end_idx > self.n_samples:
                if self.drop_last:
                    break
                end_idx = self.n_samples
            
            batch_idx = indices[start_idx:end_idx]
            yield self.x[batch_idx], self.y[batch_idx]
        
    def __len__(self) -> int:
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size

class FidelityManager:
    """
    Manages multi-fidelity datasets (Low=MNIST, Med/High=CIFAR10).
    Applies strict stratified sampling to maintain class distributions, 
    and directly caches the result into the RTX 4070 VRAM.
    """
    def __init__(self, data_dir: str = "./data_cache", seed: int = 42):
        self.data_dir = data_dir
        self.seed = seed

    def _prepare_vram_split(
        self,
        dataset: torch.utils.data.Dataset,
        device: torch.device,
        fraction: float,
        val_split: float,
        batch_size: int
    ) -> Tuple[VRAMDataLoader, VRAMDataLoader]:
        """Core logic to stratify, tensorize, and push data to VRAM."""
        # 1. Extract raw data and targets efficiently
        if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
            # Works for MNIST and CIFAR10
            raw_x = dataset.data
            raw_y = np.array(dataset.targets)
        else:
            raise ValueError("Dataset must have 'data' and 'targets' attributes.")

        # 2. Extract Fidelity Subset (Stratified)
        if fraction < 1.0:
            subset_idx, _ = train_test_split(
                np.arange(len(raw_y)),
                train_size=fraction,
                stratify=raw_y,
                random_state=self.seed
            )
            raw_x = raw_x[subset_idx]
            raw_y = raw_y[subset_idx]

        # 3. Train/Validation Split (Stratified)
        train_idx, val_idx = train_test_split(
            np.arange(len(raw_y)),
            test_size=val_split,
            stratify=raw_y,
            random_state=self.seed
        )

        # 4. Apply transforms and push directly to VRAM
        def to_vram_tensor(indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
            # Use the stratified subset tensors, not the original dataset backing store,
            # or images and labels drift out of sync once fraction < 1.0.
            x_tensors = [dataset.transform(raw_x[i]) for i in indices]
            # Stack creates a contiguous block of memory
            x_tensor = torch.stack(x_tensors).to(device, non_blocking=True)
            y_tensor = torch.tensor(raw_y[indices], dtype=torch.long, device=device)
            return x_tensor, y_tensor

        x_train, y_train = to_vram_tensor(train_idx)
        x_val, y_val = to_vram_tensor(val_idx)

        train_loader = VRAMDataLoader(x_train, y_train, batch_size, shuffle=True, drop_last=True)
        val_loader = VRAMDataLoader(x_val, y_val, batch_size, shuffle=False, drop_last=False)

        return train_loader, val_loader

    def get_low_fidelity(self, device: torch.device, batch_size: int = 256) -> Tuple[VRAMDataLoader, VRAMDataLoader]:
        """Low Fidelity: 5% of MNIST for rapid Sanity Checks."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=transform)
        # Using numpy array mapping directly for consistency
        if isinstance(dataset.data, torch.Tensor):
            dataset.data = dataset.data.numpy()
            
        return self._prepare_vram_split(dataset, device, fraction=0.05, val_split=0.2, batch_size=batch_size)

    def get_medium_fidelity(self, device: torch.device, batch_size: int = 256) -> Tuple[VRAMDataLoader, VRAMDataLoader]:
        """Medium Fidelity: 20% of CIFAR-10 for Refinement."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=transform)
        return self._prepare_vram_split(dataset, device, fraction=0.20, val_split=0.2, batch_size=batch_size)

    def get_high_fidelity(self, device: torch.device, batch_size: int = 256) -> Tuple[VRAMDataLoader, VRAMDataLoader]:
        """High Fidelity: 100% of CIFAR-10 for final Elite Evaluation."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=transform)
        return self._prepare_vram_split(dataset, device, fraction=1.0, val_split=0.2, batch_size=batch_size)
