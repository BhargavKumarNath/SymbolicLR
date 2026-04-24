import concurrent.futures
import numpy as np
from typing import List, Callable
from gp.tree import Node
import symbolr_rust

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

    def evaluate_population(self, population: List[Node], model_factory: Callable, max_workers: int = 4) -> List[float]:
        """
        Evaluates the entire generation concurrently. 
        max_workers=4 is heavily optimized for an 8GB RTX 4070 avoiding OOM issues.
        """
        fitnesses =[float('inf')] * len(population)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                    
        return fitnesses
