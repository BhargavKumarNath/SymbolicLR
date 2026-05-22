"""
gp/evaluator.py - Parallel population evaluation with optional surrogate triage.

Phase 3 upgrades over Phase 2:
- Optional surrogate integration: LightweightSurrogate pre-ranks candidates
  and allows the real evaluator to focus on the most promising ones.
- always_evaluate mask: structurally new + high-novelty candidates always
  go to real evaluation regardless of surrogate predictions.
- Surrogate is updated after every real evaluation (online learning).

All surrogate logic is independently disable-able:
- Pass surrogate=None (default) → identical to Phase 2 behavior.
- Surrogate disabled via config → evaluator falls back to full evaluation.
"""

import concurrent.futures
import warnings
from typing import Callable, Dict, List, Optional, Any

import numpy as np

from gp.tree import Node
from gp.fitness import evaluate_fitness
from config.settings import get_config


class ParallelEvaluator:
    """
    Orchestrates high-speed concurrent fitness evaluation with optional surrogate triage.

    When a surrogate is provided and ready, candidates are split into:
    - evaluate: sent to real GPU/CPU fitness evaluation
    - skip: assigned predicted fitness from surrogate (not entered into archive
            unless they happen to be re-evaluated in a later generation)

    Note: 'skip' candidates with unseen hashes or high novelty are always promoted
    to the evaluate group, regardless of surrogate predictions.
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

        # Structural hash -> fitness cache (avoids re-evaluating duplicates)
        self._fitness_cache: Dict[str, float] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def evaluate_individual(
        self,
        tree: Node,
        model_factory: Optional[Callable] = None,
        surrogate: Optional[Any] = None,
    ) -> float:
        """
        Evaluate a single individual using the unified fitness function.
        Updates surrogate after real evaluation if provided.

        Uses structural hash caching to avoid redundant GPU evaluations.
        """
        tree_hash = tree.get_hash()
        if tree_hash in self._fitness_cache:
            self._cache_hits += 1
            cached = self._fitness_cache[tree_hash]
            tree.fitness = cached
            return cached

        self._cache_misses += 1
        result = evaluate_fitness(
            tree=tree,
            t_array=self.t_array,
            trainer=self.trainer,
            model_factory=model_factory,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=self.epochs,
        )
        self._fitness_cache[tree_hash] = result

        # Update surrogate with real evaluation result
        if surrogate is not None:
            try:
                surrogate.update(tree, result, self.t_array)
            except Exception:
                pass  # surrogate update failure must never crash evaluation

        return result

    def evaluate_population(
        self,
        population: List[Node],
        model_factory: Optional[Callable] = None,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        surrogate: Optional[Any] = None,
        always_evaluate: Optional[List[bool]] = None,
    ) -> List[float]:
        """
        Evaluate the entire generation concurrently, with optional surrogate triage.

        When surrogate is provided and ready:
        - Candidates marked always_evaluate=True are always fully evaluated.
        - Remaining candidates are ranked by surrogate; top eval_fraction go to
          real evaluation; the rest receive surrogate predictions.
        - Cache hits bypass real evaluation regardless of always_evaluate.

        Args:
            population:        List of candidate trees to evaluate.
            model_factory:     Factory for creating probe models.
            max_workers:       Thread pool size.
            progress_callback: Optional (completed, total) callback.
            surrogate:         Optional LightweightSurrogate instance.
            always_evaluate:   Boolean mask for mandatory real evaluation.

        Returns:
            List[float]: Fitness scores in the same order as population.
        """
        total = len(population)
        if total == 0:
            return []

        fitnesses = [float("inf")] * total
        safe_workers = max(1, min(max_workers, total))

        # Determine which candidates need real evaluation vs surrogate prediction
        if surrogate is not None and surrogate.is_ready():
            _always = always_evaluate if always_evaluate is not None else [False] * total
            # Cache hits don't need real evaluation regardless
            for i, tree in enumerate(population):
                if tree.get_hash() in self._fitness_cache:
                    _always[i] = False  # will be served from cache, not triage

            eval_indices, skip_indices = surrogate.rank_candidates(
                population, self.t_array, always_evaluate=_always
            )

            # Assign surrogate predictions to skipped candidates
            for i in skip_indices:
                pred = surrogate.predict(population[i], self.t_array)
                fitnesses[i] = pred if np.isfinite(pred) else float("inf")
        else:
            eval_indices = list(range(total))
            skip_indices = []

        # Real evaluation for selected candidates (threaded)
        completed = 0
        eval_population = [population[i] for i in eval_indices]

        with concurrent.futures.ThreadPoolExecutor(max_workers=safe_workers) as executor:
            future_to_local_idx = {
                executor.submit(
                    self.evaluate_individual, ind, model_factory, surrogate
                ): local_i
                for local_i, ind in enumerate(eval_population)
            }

            for future in concurrent.futures.as_completed(future_to_local_idx):
                local_i = future_to_local_idx[future]
                global_i = eval_indices[local_i]
                try:
                    fitnesses[global_i] = future.result()
                except Exception as e:
                    warnings.warn(
                        f"Fitness evaluation failed for individual {global_i}: {e}",
                        RuntimeWarning,
                    )
                    fitnesses[global_i] = float("inf")
                finally:
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(completed, total)

        # Advance progress bar for surrogate-predicted candidates too
        if progress_callback is not None:
            for _ in skip_indices:
                completed += 1
                progress_callback(completed, total)

        return fitnesses

    def get_cache_stats(self) -> dict:
        total = self._cache_hits + self._cache_misses
        return {
            "cache_size": len(self._fitness_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": round(self._cache_hits / max(1, total), 3),
        }
