"""
gp/evaluator.py - Parallel population evaluation.

Orchestrates concurrent fitness evaluation using either real PyTorch
training or the synthetic fitness function, depending on runtime mode.
Uses the unified rust_bridge for schedule computation.
"""

import concurrent.futures
import numpy as np
from typing import List, Callable, Optional, Any
from gp.tree import Node
from gp.fitness import evaluate_fitness
from config.settings import get_config


class ParallelEvaluator:
    """
    Orchestrates high-speed concurrent evaluations.
    Uses ThreadPools to overlap Rust CPU evaluations and PyTorch GPU training.
    """

    def __init__(
        self,
        trainer: Any = None,
        train_loader: Any = None,
        val_loader: Any = None,
        epochs: int = 1,
        time_steps: int = 100,
    ):
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.t_array = np.linspace(0.0, 1.0, time_steps, dtype=np.float64)

    def evaluate_individual(self, tree: Node, model_factory: Optional[Callable] = None) -> float:
        """Evaluates a single individual using the unified fitness function."""
        return evaluate_fitness(
            tree=tree,
            t_array=self.t_array,
            trainer=self.trainer,
            model_factory=model_factory,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=self.epochs,
        )

    def evaluate_population(
        self,
        population: List[Node],
        model_factory: Optional[Callable] = None,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[float]:
        """
        Evaluates the entire generation concurrently.
        """
        total = len(population)
        if total == 0:
            return []

        fitnesses = [float("inf")] * total
        safe_workers = max(1, min(max_workers, total))
        completed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=safe_workers) as executor:
            future_to_idx = {
                executor.submit(self.evaluate_individual, ind, model_factory): i
                for i, ind in enumerate(population)
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    fitnesses[idx] = future.result()
                except Exception:
                    fitnesses[idx] = float("inf")
                finally:
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(completed, total)

        return fitnesses
