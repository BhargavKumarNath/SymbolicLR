import concurrent.futures
import numpy as np
from typing import List, Callable, Optional
from gp.tree import Node

try:
    import symbolr_rust
except ImportError:
    class MockSymbolrRust:
        @staticmethod
        def evaluate_fast(prefix, t_array):
            # Fallback for mock mode
            return 0.1 * (1.0 - t_array)
    symbolr_rust = MockSymbolrRust()

class ParallelEvaluator:
    """
    Orchestrates high-speed concurrent evaluations.
    Uses ThreadPools to overlap Rust CPU evaluations and PyTorch GPU training.
    """
    def __init__(self, trainer, train_loader, val_loader, epochs: int, time_steps: int):
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        # Pre-compute the normalized time array for Rust evaluation
        self.t_array = np.linspace(0.0, 1.0, time_steps, dtype=np.float64)

    def evaluate_individual(self, tree: Node, model_factory: Callable) -> float:
        """Evaluates a single individual, hitting the cache if it already exists."""
        # 1. Zero-Cost Cache Hit
        if hasattr(tree, "fitness") and getattr(tree, "fitness") is not None:
            return getattr(tree, "fitness")

        # 2. Rust CPU Execution
        prefix = tree.to_prefix()
        try:
            lr_schedule = symbolr_rust.evaluate_fast(prefix, self.t_array)
        except Exception:
            return float('inf') # Fail gracefully on mathematically impossible syntax

        # 3. Dedicated GPU Execution
        model = model_factory()
        loss = self.trainer.evaluate_schedule(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            lr_schedule=lr_schedule,
            epochs=self.epochs
        )

        # Cache result
        tree.fitness = loss
        return loss

    def evaluate_population(
        self,
        population: List[Node],
        model_factory: Callable,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[float]:
        """
        Evaluates the entire generation concurrently. 
        max_workers=4 is heavily optimized for an 8GB RTX 4070 avoiding OOM issues.
        """
        total = len(population)
        if total == 0:
            return []

        fitnesses = [float('inf')] * total
        safe_workers = max(1, min(max_workers, total))
        completed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=safe_workers) as executor:
            # Map futures to their original index to maintain population order
            future_to_idx = {
                executor.submit(self.evaluate_individual, ind, model_factory): i
                for i, ind in enumerate(population)
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    fitnesses[idx] = future.result()
                except Exception:
                    fitnesses[idx] = float('inf')
                finally:
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(completed, total)
                    
        return fitnesses
