import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from data.fidelity import VRAMDataLoader, FidelityManager

@pytest.fixture
def dummy_tensors():
    """Generates synthetic contiguous tensors mapped to CPU to represent VRAM"""
    x = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 10, (100,))
    return x, y

def test_vram_dataloader_iteration_and_drop_last(dummy_tensors):
    """Verifies that the custom VRAM loader correctly chunks batches and handles drop_last"""
    x, y = dummy_tensors

    # 100 samples, batch_size=30 -> 3 full batches, 10 leftovers
    loader_drop = VRAMDataLoader(x, y, batch_size=30, shuffle=False, drop_last=True)
    loader_keep = VRAMDataLoader(x, y, batch_size=30, shuffle=False, drop_last=False)

    assert len(loader_drop) == 3
    assert len(loader_keep) == 4

    batches_drop = list(loader_drop)
    assert len(batches_drop) == 3
    assert batches_drop[0][0].shape == (30, 3, 32, 32)

    batches_keep = list(loader_keep)
    assert len(batches_keep) == 4
    assert batches_keep[-1][0].shape == (10, 3, 32, 32) # The leftover batch

def test_vram_dataloader_shuffling(dummy_tensors):
    """Ensures GPU-side shuffle successfully reorders indices"""
    x, y = dummy_tensors
    loader = VRAMDataLoader(x, y, batch_size=100, shuffle=True, drop_last=False)

    # Extract first batch (which is the entire dataset)
    x_batch, _ = next(iter(loader))

    assert not torch.equal(x, x_batch), "VRAM DataLoader failed to shuffle tensors"

@pytest.fixture
def mock_dataset():
    """Mocks a standard Torchvision dataset interface"""
    dataset = MagicMock()
    dataset.data = np.zeros((10000, 28, 28, 1))
    dataset.targets = np.tile(np.arange(10), 1000).tolist()

    dataset.transform = lambda img: torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    return dataset

@patch("data.fidelity.datasets.MNIST")
def test_fidelity_manager_low_tier(mock_mnist_cls, mock_dataset):
    """Validates multi-tier data pipeline extracts exactly 5% for the Low Fidelity tier."""
    mock_mnist_cls.return_value = mock_dataset
    
    manager = FidelityManager(seed=42)
    device = torch.device("cpu") # Force CPU for CI testing
    
    train_loader, val_loader = manager.get_low_fidelity(device, batch_size=16)
    
    # Total Data: 10,000. 
    # 5% Subset: 500 samples.
    # 80/20 Train/Val Split on subset -> 400 Train, 100 Val.
    
    # Drop_last is True for train (400 // 16 = 25)
    # Drop_last is False for val (100 / 16 = 6.25 -> 7 batches)
    assert len(train_loader) == 25
    assert len(val_loader) == 7
    
    # Verify device pinning logic
    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.device.type == "cpu"
    assert y_batch.device.type == "cpu"
    