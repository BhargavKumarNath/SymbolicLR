import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader

# Import the module under test
from data.loader import get_tier1_dataloaders, get_tier2_dataloaders, set_seed

@pytest.fixture
def mock_mnist():
    """Mocks the MNIST dataset to avoid network calls during testing."""
    with patch("data.loader.datasets.MNIST") as mock:
        instance = MagicMock()
        instance.__len__.return_value = 60000
        # Stratified targets logic requirement
        instance.targets = MagicMock()
        instance.targets.numpy.return_value = np.random.randint(0, 10, 60000)
        mock.return_value = instance
        yield mock

@pytest.fixture
def mock_cifar10():
    """Mocks the CIFAR-10 dataset to test stratification limits securely."""
    with patch("data.loader.datasets.CIFAR10") as mock:
        instance = MagicMock()
        instance.__len__.return_value = 50000
        # Provide balanced dummy targets for stratification
        instance.targets = np.tile(np.arange(10), 5000).tolist()
        mock.return_value = instance
        yield mock

def test_set_seed_reproducibility():
    """Validates pseudo-random deterministic synchronization."""
    set_seed(42)
    val1 = np.random.rand()
    set_seed(42)
    val2 = np.random.rand()
    assert val1 == val2, "Seed stabilization failed across numpy contexts."

def test_tier1_dataloader_shapes_and_pinning(mock_mnist):
    """Ensures MNIST dataset correctly separates and applies GPU pinning."""
    train_loader, val_loader = get_tier1_dataloaders(batch_size=128, val_split=0.2, num_workers=0)
    
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    
    # 60,000 total -> 48,000 train, 12,000 val
    assert len(train_loader.dataset) == 48000
    assert len(val_loader.dataset) == 12000
    
    # Hardware verification
    assert train_loader.pin_memory is True, "DataLoader must use pinned memory for fast PCIe transfers."
    assert train_loader.drop_last is True, "Drop last must be True to prevent mixed-shape batch slowdowns in AMP."

def test_tier2_cifar10_stratified_subset(mock_cifar10):
    """Ensures CIFAR-10 extracts precisely a 10% stratified subset, then properly splits to 80/20 train/val."""
    train_loader, val_loader = get_tier2_dataloaders(batch_size=64, subset_fraction=0.1, val_split=0.2, num_workers=0)
    
    # Total CIFAR is 50,000. 10% subset = 5,000.
    # 80/20 split of 5,000 -> 4,000 train, 1,000 val.
    assert len(train_loader.dataset) == 4000
    assert len(val_loader.dataset) == 1000

    assert train_loader.pin_memory is True

def test_seeding_preserves_stratified_indices(mock_cifar10):
    """Ensures two consecutive executions with the same seed result in identical subsets."""
    tl_1, _ = get_tier2_dataloaders(seed=123, num_workers=0)
    tl_2, _ = get_tier2_dataloaders(seed=123, num_workers=0)
    
    # Extract underlying indices from the Subset wrapper
    indices_1 = tl_1.dataset.indices
    indices_2 = tl_2.dataset.indices
    
    np.testing.assert_array_equal(
        indices_1, 
        indices_2, 
        err_msg="Dataloader subsetting is non-deterministic between identical seed calls."
    )