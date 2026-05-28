import math
import torch
import numpy as np
from src.symbolr.engine.evaluator import BaseEvaluator

class CUDABatchEvaluator(BaseEvaluator):
    """
    Evaluates genetic programming prefix formulas against a surrogate dataset
    entirely on the GPU using PyTorch tensor operations.
    """
    def __init__(self, data_labels: np.ndarray, device: str = 'cuda:0'):
        if not torch.cuda.is_available() and device.startswith('cuda'):
            device = 'cpu'
            
        self.device = torch.device(device)
        
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
            if token == 'x' or token == 't': # support both x and t as time variables
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
                epsilon = 1e-6
                safe_right = torch.where(torch.abs(right) < epsilon, torch.sign(right) * epsilon + epsilon * (right == 0), right)
                stack.append(left / safe_right)
            elif token == 'sin':
                stack.append(torch.sin(stack.pop()))
            elif token == 'cos':
                stack.append(torch.cos(stack.pop()))
            elif token == 'exp':
                val = torch.clamp(stack.pop(), max=20.0)
                stack.append(torch.exp(val))
            elif token == 'log':
                val = stack.pop()
                safe_val = torch.where(torch.abs(val) < 1e-6, torch.full_like(val, 1e-6), torch.abs(val))
                stack.append(torch.log(safe_val))
            else:
                try:
                    val = float(token)
                    stack.append(self._get_const(val))
                except ValueError:
                    stack.append(self._get_const(0.0))
                    
        return stack[0]

    def evaluate(self, formulas: list[str]) -> list[float]:
        """
        Evaluates a batch of formulas via PyTorch GPU ops and returns MSE losses.
        Implements VRAM-aware batch chunking and successive halving triage.
        """
        if not formulas:
            return []
            
        losses = [float('inf')] * len(formulas)
        
        # Determine safe batch size based on VRAM (heuristic)
        batch_size = 64
        if self.device.type == 'cuda':
            total_vram = torch.cuda.get_device_properties(self.device).total_memory
            batch_size = max(16, int(total_vram / max(1, self.time_steps * 400)))
            batch_size = min(batch_size, 256)
            
        # Successive halving setup
        half_idx = self.time_steps // 2
        t_half = self.t[:half_idx]
        labels_half = self.labels[:half_idx]

        with torch.no_grad():
            for i in range(0, len(formulas), batch_size):
                batch_indices = list(range(i, min(i + batch_size, len(formulas))))
                batch_formulas = [formulas[idx] for idx in batch_indices]
                
                # Phase 1: Evaluate on first half of time steps
                half_losses = []
                for formula in batch_formulas:
                    try:
                        # Temporarily override self.t for parsing
                        original_t = self.t
                        self.t = t_half
                        y_pred = self._parse_and_evaluate(formula)
                        mse = torch.mean((y_pred - labels_half) ** 2).item()
                        self.t = original_t
                        if not math.isfinite(mse): mse = float('inf')
                    except Exception:
                        self.t = original_t
                        mse = float('inf')
                    half_losses.append(mse)
                
                # Successive halving: find the median loss of valid ones
                valid_losses = [l for l in half_losses if l != float('inf')]
                if not valid_losses:
                    continue
                cutoff = np.median(valid_losses)
                
                # Phase 2: Evaluate survivors on full time steps
                for local_idx, global_idx in enumerate(batch_indices):
                    if half_losses[local_idx] <= cutoff:
                        try:
                            y_pred = self._parse_and_evaluate(formulas[global_idx])
                            mse = torch.mean((y_pred - self.labels) ** 2).item()
                            size = len(formulas[global_idx].split())
                            parsimony = 0.01 * size
                            final_loss = mse + parsimony
                            if not math.isfinite(final_loss):
                                final_loss = float('inf')
                        except Exception:
                            final_loss = float('inf')
                        losses[global_idx] = final_loss
                    else:
                        # Pruned early
                        losses[global_idx] = float('inf')
                        
        return losses
