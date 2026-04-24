import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastConvNet(nn.Module):
    """Lightweight CNN optimized for RTX 4070 throughput."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_compiled_model(device: torch.device, in_channels: int = 1) -> nn.Module:
    """
    Factory function to instantiate and fuse the model using PyTorch 2.0's compiler.
    Uses 'reduce-overhead' to minimize CPU overhead during small batched VRAM loading.
    """
    model = FastConvNet(in_channels=in_channels).to(device)
    
    # Attempt to compile the model to Triton kernels
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead", disable=False)
        except Exception:
            pass
            
    return model


class ProbeTrainer:
    """Handles aggressive early-stopping and AMP for rapid GP schedule evaluations."""
    def __init__(
        self,
        device: torch.device,
        patience: int = 3,
        min_delta: float = 1e-3,
        explode_threshold: float = 10.0,
        amp_enabled: bool = True
    ):
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
        self.explode_threshold = explode_threshold
        self.amp_enabled = amp_enabled and device.type == 'cuda'

    def evaluate_schedule(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        lr_schedule: np.ndarray,
        epochs: int
    ) -> float:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        
        best_val_loss = float('inf')
        stagnant_epochs = 0
        global_step = 0
        schedule_length = len(lr_schedule)

        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                current_lr = lr_schedule[global_step] if global_step < schedule_length else lr_schedule[-1]
                
                # Defend against NaN/Inf LR from unconstrained GP mutations
                if not math.isfinite(current_lr) or current_lr <= 0.0 or current_lr > 10.0:
                    return float('inf')
                    
                for param_group in optimizer.param_groups:
                    param_group['lr'] = float(current_lr)
                
                # If tensors are from VRAMDataLoader, this is a zero-cost operation.
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                if not math.isfinite(loss.item()) or loss.item() > self.explode_threshold:
                    return float('inf')
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                global_step += 1
                
            val_loss = self._validate(model, val_loader, criterion)
            
            if not math.isfinite(val_loss):
                return float('inf')
            
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                stagnant_epochs = 0
            else:
                stagnant_epochs += 1
                
            if stagnant_epochs >= self.patience:
                break

        return float(best_val_loss)

    @torch.no_grad()
    def _validate(self, model: nn.Module, val_loader, criterion: nn.Module) -> float:
        model.eval()
        total_loss, batches = 0.0, 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                loss = criterion(model(inputs), targets)
            total_loss += loss.item()
            batches += 1
        return total_loss / max(1, batches)