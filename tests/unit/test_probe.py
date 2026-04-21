import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.probe import FastConvNet, ProbeTrainer

@pytest.fixture
def dummy_loaders():
    """
    Provides small synthetic batches to isolate model logic from data loading.
    """
    x_train = torch.randn(64, 1, 28, 28)
    y_train = torch.randint(0, 10, (64,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16)

    x_val = torch.randn(32, 1, 28, 28)
    y_val = torch.randint(0, 10, (32,))
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=16)

    return train_loader, val_loader

def test_fast_convnet_forward():
    """Validates the architectural forward pass and tensor shapes."""
    model = FastConvNet(in_channels=1, num_classes=10)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    
    assert out.shape == (2, 10)
    assert not torch.isnan(out).any(), "Forward pass emitted NaN values."


def test_probe_trainer_normal_execution(dummy_loaders):
    """Ensures a valid schedule trains properly and returns a legitimate loss."""
    train_loader, val_loader = dummy_loaders
    device = torch.device('cpu')  # Force CPU to bypass local GPU requirements in CI
    trainer = ProbeTrainer(device=device, amp_enabled=False, patience=5)
    model = FastConvNet(in_channels=1, num_classes=10)
    
    # Schedule covers 2 epochs * 4 batches = 8 steps
    lr_schedule = np.linspace(0.1, 0.01, 8)
    
    best_loss = trainer.evaluate_schedule(model, train_loader, val_loader, lr_schedule, epochs=2)
    
    assert isinstance(best_loss, float)
    assert best_loss < float('inf')


def test_probe_trainer_exploding_loss(dummy_loaders):
    """Simulates an inherently unstable formula triggering the early-exit explosion threshold."""
    train_loader, val_loader = dummy_loaders
    device = torch.device('cpu')
    
    # Trigger immediate explosion by capping the max allowed loss drastically low
    trainer = ProbeTrainer(device=device, explode_threshold=-1.0, amp_enabled=False)
    model = FastConvNet(in_channels=1, num_classes=10)
    
    lr_schedule = np.ones(10) * 0.1
    best_loss = trainer.evaluate_schedule(model, train_loader, val_loader, lr_schedule, epochs=1)
    
    assert best_loss == float('inf'), "Model failed to early-exit on exploding loss condition."


def test_probe_trainer_stagnation_early_exit(dummy_loaders):
    """Ensures that stagnant loss correctly triggers patience-based early stopping."""
    train_loader, val_loader = dummy_loaders
    device = torch.device('cpu')
    
    # Force an impossible delta requirement to guarantee stagnation triggers
    trainer = ProbeTrainer(device=device, patience=1, min_delta=10.0, amp_enabled=False)
    model = FastConvNet(in_channels=1, num_classes=10)
    
    lr_schedule = np.ones(50) * 0.01
    
    # Requesting 10 epochs. It should exit after Epoch 1 due to patience=1
    best_loss = trainer.evaluate_schedule(model, train_loader, val_loader, lr_schedule, epochs=10)
    
    assert best_loss < float('inf'), "Model diverged incorrectly during stagnation testing."


def test_probe_trainer_nan_lr(dummy_loaders):
    """Validates the GP defense mechanism against mathematically flawed schedules."""
    train_loader, val_loader = dummy_loaders
    device = torch.device('cpu')
    trainer = ProbeTrainer(device=device, amp_enabled=False)
    model = FastConvNet(in_channels=1, num_classes=10)
    
    # Pass a mathematically impossible learning rate (NaN)
    lr_schedule = np.array([np.nan, 0.1])
    best_loss = trainer.evaluate_schedule(model, train_loader, val_loader, lr_schedule, epochs=1)
    
    assert best_loss == float('inf'), "Failed to penalize NaN learning rate."
