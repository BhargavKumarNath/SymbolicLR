try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module: pass

import math
import numpy as np
from typing import Any

class FastConvNet(nn.Module):
    """Lightweight CNN optimized for RTX 4070 throughput."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: Any) -> Any:
        if not TORCH_AVAILABLE:
            return None
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_compiled_model(device: Any, in_channels: int = 1, init_seed: int = None) -> Any:
    """
    Factory function to instantiate and fuse the model using PyTorch 2.0's compiler.
    Uses 'reduce-overhead' to minimize CPU overhead during small batched VRAM loading.
    """
    if not TORCH_AVAILABLE:
        return "MockModel"

    import sys
    
    if init_seed is not None:
        torch.manual_seed(init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(init_seed)
        
    model = FastConvNet(in_channels=in_channels).to(device)
    
    # Attempt to compile the model to Triton kernels
    if hasattr(torch, "compile") and sys.platform != "win32":
        try:
            model = torch.compile(model, mode="reduce-overhead", disable=False)
        except Exception:
            pass
            
    return model


class ProbeTrainer:
    """Handles aggressive early-stopping and AMP for rapid GP schedule evaluations."""
    def __init__(
        self,
        device: Any,
        patience: int = 3,
        min_delta: float = 1e-3,
        explode_threshold: float = 10.0,
        amp_enabled: bool = True
    ):
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
        self.explode_threshold = explode_threshold
        if TORCH_AVAILABLE:
            self.amp_enabled = amp_enabled and device.type == 'cuda'
        else:
            self.amp_enabled = False

    def evaluate_schedule(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Any,
        lr_schedule: np.ndarray,
        epochs: int
    ) -> float:
        if not TORCH_AVAILABLE:
            # Mock evaluation logic
            if not np.all(np.isfinite(lr_schedule)) or np.any(lr_schedule < 1e-7) or np.any(lr_schedule > 10):
                return float('inf')
            base = 2.5
            std = np.std(lr_schedule)
            trend = np.abs(lr_schedule[-1] - lr_schedule[0])
            is_decay = 1.0 if lr_schedule[0] > lr_schedule[-1] else 0.0
            complexity = len(str(model))
            gene_variety = (len(str(lr_schedule)) % 100) / 1000.0
            mock_loss = base - (std * 15.0) - (trend * 10.0) - (is_decay * 0.2) - gene_variety
            return float(np.clip(mock_loss, 0.1, 3.5))

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler(enabled=self.amp_enabled)
        
        best_val_loss = float('inf')
        stagnant_epochs = 0
        global_step = 0
        schedule_length = len(lr_schedule)
        
        # Map the schedule to actual training steps via interpolation
        total_steps = len(train_loader) * epochs
        if total_steps > 0 and total_steps != schedule_length:
            step_indices = np.linspace(0, schedule_length - 1, total_steps)
            lr_schedule = np.interp(step_indices, np.arange(schedule_length), lr_schedule)
            schedule_length = total_steps

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

    def _validate(self, model: Any, val_loader: Any, criterion: Any) -> float:
        if not TORCH_AVAILABLE: return 0.0
        model.eval()
        total_loss, batches = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    loss = criterion(model(inputs), targets)
                total_loss += loss.item()
                batches += 1
        return total_loss / max(1, batches)


class CUDABatchEvaluator:
    """
    Evaluates genetic programming prefix formulas against a surrogate dataset
    entirely on the GPU using PyTorch tensor operations.
    """
    def __init__(self, data_labels: np.ndarray, device: str = 'cuda:0'):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for CUDABatchEvaluator")
            
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load the surrogate dataset directly into VRAM
        self.labels = torch.tensor(data_labels, dtype=torch.float32, device=self.device)
        self.time_steps = len(self.labels)
        
        # Pre-allocate time tensor t in VRAM
        self.t = torch.linspace(0.0, 1.0, self.time_steps, dtype=torch.float32, device=self.device)
        
        # Constants cache on GPU
        self._const_cache = {}

        if self.device.type == 'cuda':
            print(f"[System] RTX CUDA Device Initialized. VRAM Allocated for {self.time_steps} records.")

    def _get_const(self, val: float) -> torch.Tensor:
        if val not in self._const_cache:
            self._const_cache[val] = torch.full_like(self.t, val)
        return self._const_cache[val]

    def _parse_and_evaluate(self, prefix_str: str) -> torch.Tensor:
        """
        Parses a prefix string and maps it directly to PyTorch tensor operations.
        Prefix parsing works cleanly by scanning the tokens right-to-left.
        """
        tokens = prefix_str.strip().split()
        stack = []
        
        for token in reversed(tokens):
            if token == 'x':
                stack.append(self.t)
            elif token == '+':
                left = stack.pop()
                right = stack.pop()
                stack.append(left + right)
            elif token == '-':
                left = stack.pop()
                right = stack.pop()
                stack.append(left - right)
            elif token == '*':
                left = stack.pop()
                right = stack.pop()
                stack.append(left * right)
            elif token == '/':
                left = stack.pop()
                right = stack.pop()
                # Protected division: if right is 0, return 1 (standard GP protection)
                # But to vectorize, we add a small epsilon
                epsilon = 1e-6
                safe_right = torch.where(torch.abs(right) < epsilon, torch.sign(right) * epsilon + epsilon * (right == 0), right)
                stack.append(left / safe_right)
            elif token == 'sin':
                stack.append(torch.sin(stack.pop()))
            elif token == 'cos':
                stack.append(torch.cos(stack.pop()))
            elif token == 'exp':
                # Protected exp to prevent Inf
                val = torch.clamp(stack.pop(), max=20.0)
                stack.append(torch.exp(val))
            elif token == 'log':
                # Protected log
                val = stack.pop()
                safe_val = torch.where(torch.abs(val) < 1e-6, torch.full_like(val, 1e-6), torch.abs(val))
                stack.append(torch.log(safe_val))
            else:
                # Constant
                try:
                    val = float(token)
                    stack.append(self._get_const(val))
                except ValueError:
                    # Fallback for unrecognized tokens
                    stack.append(self._get_const(0.0))
                    
        return stack[0]

    def evaluate_batch(self, formulas: list[str]) -> list[float]:
        """
        Invoked via PyO3 from Rust.
        Accepts a batch of prefix strings and returns their MSE losses.
        """
        if not formulas:
            return []
            
        losses = []
        
        with torch.no_grad():
            for formula in formulas:
                try:
                    y_pred = self._parse_and_evaluate(formula)
                    
                    # Compute MSE
                    mse = torch.mean((y_pred - self.labels) ** 2)
                    
                    # Add parsimony pressure based on string length (proxy for AST size)
                    # Node count is approximately len(formulas.split())
                    size = len(formula.split())
                    parsimony = 0.01 * size
                    
                    loss = mse.item() + parsimony
                    
                    if not math.isfinite(loss):
                        loss = float('inf')
                        
                except Exception as e:
                    loss = float('inf')
                    
                losses.append(loss)
                
        return losses
