"""
gp/map_elites.py - MAP-Elites++ Quality-Diversity Archive.

Phase 3 upgrades over Phase 2:
- Elite aging: archive entries track age (generations since last improvement).
  A soft age_penalty slightly favors fresh contenders in competitive replacement.
- Pareto front: secondary tracking of (loss, tree_size) Pareto-optimal solutions.
- try_add() accepts optional effective_loss for novelty-augmented comparison
  while storing raw_loss for interpretable Hall of Fame display.
- increment_ages(): call once per generation to age all archive entries.

Behavioral descriptors (3D grid):
    size_bins       — AST node count (proxy for formula complexity)
    com_bins        — Center of mass (temporal learning concentration)
    smoothness_bins — Total variation (schedule smoothness profile)

Uses the unified rust_bridge for schedule evaluation.
"""

import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from gp.tree import Node
from gp.rust_bridge import evaluate_schedule


class MAPElitesArchive:
    """
    3D quality-diversity archive with elite aging and Pareto-front tracking.

    Archive entries: Dict[(size_idx, com_idx, smoothness_idx)] -> (raw_loss, Node, age)
        raw_loss: actual validation loss (stored for interpretability)
        age: generations since this niche was last improved (incremented per gen)

    Competitive replacement uses effective_loss (novelty-augmented if provided),
    but raw_loss is what gets stored and reported in Hall of Fame.
    """

    def __init__(
        self,
        size_bins: int = 30,
        com_bins: int = 20,
        smoothness_bins: int = 10,
        time_steps: int = 100,
        age_penalty_coeff: float = 0.001,
    ):
        self.size_bins = size_bins
        self.com_bins = com_bins
        self.smoothness_bins = smoothness_bins
        self.age_penalty_coeff = age_penalty_coeff

        # Primary archive: niche -> (raw_loss, Node, age)
        self.archive: Dict[Tuple[int, int, int], Tuple[float, Node, int]] = {}

        # Structural dedup set (cross-niche)
        self._expression_hashes: set = set()

        # Niche curiosity tracking
        self._niche_update_counts: Dict[Tuple, int] = {}

        # Global counters
        self._total_attempts: int = 0
        self._total_additions: int = 0
        self._current_generation: int = 0

        # Pre-compute time array for descriptor extraction
        self.t_array = np.linspace(0.0, 1.0, time_steps, dtype=np.float64)

    def _compute_descriptors(
        self, tree: Node
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Compute (complexity, center_of_mass, smoothness) behavioral descriptors.
        Returns (None, None, None) if the formula evaluates to invalid math.
        """
        size = tree.size()
        try:
            lr_schedule = evaluate_schedule(tree, self.t_array)

            total_lr = np.sum(lr_schedule)
            if total_lr == 0 or not np.isfinite(total_lr):
                return None, None, None

            # Center of Mass = sum(t * LR(t)) / sum(LR(t))
            com = np.sum(self.t_array * lr_schedule) / total_lr
            com = float(max(0.0, min(1.0, com)))

            # Smoothness = normalized total variation
            diffs = np.abs(np.diff(lr_schedule))
            total_variation = np.sum(diffs)
            max_tv = (
                np.max(lr_schedule) - np.min(lr_schedule)
                if np.max(lr_schedule) > np.min(lr_schedule)
                else 1.0
            )
            normalized_tv = min(1.0, total_variation / (max_tv * len(diffs) + 1e-8))

        except Exception:
            return None, None, None

        size_idx = min(size, self.size_bins - 1)
        com_idx = int(com * (self.com_bins - 1))
        smoothness_idx = int(normalized_tv * (self.smoothness_bins - 1))

        return size_idx, com_idx, smoothness_idx

    def try_add(
        self,
        tree: Node,
        raw_loss: float,
        effective_loss: Optional[float] = None,
    ) -> bool:
        """
        Attempt to place a tree into the archive.

        Competitive replacement uses effective_loss (novelty-augmented if provided)
        to decide whether to replace the incumbent. The stored value is always raw_loss,
        so the Hall of Fame displays real validation loss.

        An age penalty is applied to the incumbent's effective loss to slightly
        favor fresh contenders: incumbent_compare = incumbent_raw + age * age_penalty.

        Args:
            tree:           The candidate tree.
            raw_loss:       The real validation loss (stored in archive).
            effective_loss: Novelty-augmented loss for comparison (optional).
                            If None, raw_loss is used for comparison.

        Returns:
            True if the tree occupied a new or improved niche.
        """
        if not np.isfinite(raw_loss):
            return False

        self._total_attempts += 1
        compare_loss = effective_loss if effective_loss is not None else raw_loss

        size_idx, com_idx, smoothness_idx = self._compute_descriptors(tree)
        if size_idx is None:
            return False

        niche = (size_idx, com_idx, smoothness_idx)
        tree_hash = tree.get_hash()

        if niche in self.archive:
            incumbent_raw, incumbent_tree, incumbent_age = self.archive[niche]

            # Exact structural duplicate in same niche: reject
            if incumbent_tree.get_hash() == tree_hash:
                return False

            # Compute age-adjusted incumbent loss (favors fresh contenders)
            incumbent_adjusted = incumbent_raw + self.age_penalty_coeff * incumbent_age

            if compare_loss >= incumbent_adjusted:
                return False  # incumbent wins

        # Insert / replace
        self.archive[niche] = (raw_loss, copy.deepcopy(tree), 0)
        self._expression_hashes.add(tree_hash)
        self._niche_update_counts[niche] = self._niche_update_counts.get(niche, 0) + 1
        self._total_additions += 1
        return True

    def increment_ages(self) -> None:
        """
        Increment the age counter for all current archive entries.
        Call exactly once per generation, after all try_add() calls complete.
        """
        self._current_generation += 1
        updated = {}
        for niche, (loss, tree, age) in self.archive.items():
            updated[niche] = (loss, tree, age + 1)
        self.archive = updated

    def sample_parents(self, batch_size: int) -> List[Node]:
        """
        Hybrid parent selection: fitness-weighted + curiosity + uniform.

        60% fitness-proportionate (inverse loss weighting)
        20% curiosity (least-recently-updated niches — structural exploration)
        20% uniform (unbiased diversity)
        """
        if not self.archive:
            return []

        occupied = list(self.archive.keys())
        losses = [self.archive[k][0] for k in occupied]

        # Fitness-proportionate weights
        max_loss = max(losses) + 1e-8
        weights = [(max_loss - l + 1e-8) for l in losses]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        n_fitness = int(batch_size * 0.6)
        n_curiosity = int(batch_size * 0.2)
        n_uniform = batch_size - n_fitness - n_curiosity

        parents = []

        if n_fitness > 0:
            indices = random.choices(range(len(occupied)), weights=probs, k=n_fitness)
            for idx in indices:
                parents.append(copy.deepcopy(self.archive[occupied[idx]][1]))

        if n_curiosity > 0:
            update_counts = [
                (self._niche_update_counts.get(k, 0), k) for k in occupied
            ]
            update_counts.sort()  # least-updated first
            curiosity_niches = [
                k for _, k in update_counts[: max(1, len(update_counts) // 3)]
            ]
            for _ in range(n_curiosity):
                niche = random.choice(curiosity_niches)
                parents.append(copy.deepcopy(self.archive[niche][1]))

        for _ in range(n_uniform):
            niche = random.choice(occupied)
            parents.append(copy.deepcopy(self.archive[niche][1]))

        return parents

    def get_hall_of_fame(self, top_k: int = 5) -> List[Tuple[float, Node]]:
        """
        Returns the top-k globally best formulas across all niches.

        Returns List[Tuple[raw_loss, Node]] — backward-compatible with benchmark.py.
        """
        all_elites = [(loss, tree) for loss, tree, age in self.archive.values()]
        all_elites.sort(key=lambda x: x[0])
        return all_elites[:top_k]

    def get_pareto_front(self) -> List[Tuple[float, int, Node]]:
        """
        Returns the (loss, tree_size, Node) Pareto-optimal front for
        (minimize loss, minimize size) objectives.

        Useful for the Hall of Fame when you want compact AND accurate schedules.
        """
        all_elites = [
            (loss, tree.size(), tree)
            for loss, tree, age in self.archive.values()
        ]
        all_elites.sort(key=lambda x: x[0])  # sort by loss

        pareto = []
        min_size_seen = float("inf")
        for loss, size, tree in all_elites:
            if size < min_size_seen:
                pareto.append((loss, size, tree))
                min_size_seen = size

        return pareto

    def get_stats(self) -> dict:
        """Return archive statistics for diagnostics."""
        max_niches = self.size_bins * self.com_bins * self.smoothness_bins
        occupied = len(self.archive)
        unique_hashes = len(self._expression_hashes)
        losses = [v[0] for v in self.archive.values()] if self.archive else []
        ages = [v[2] for v in self.archive.values()] if self.archive else []

        return {
            "occupied_niches": occupied,
            "max_niches": max_niches,
            "occupancy_pct": round(100 * occupied / max(1, max_niches), 1),
            "unique_expressions": unique_hashes,
            "total_attempts": self._total_attempts,
            "total_additions": self._total_additions,
            "acceptance_rate": round(
                self._total_additions / max(1, self._total_attempts), 3
            ),
            "best_loss": min(losses) if losses else float("inf"),
            "median_loss": float(np.median(losses)) if losses else float("inf"),
            "loss_std": float(np.std(losses)) if losses else 0.0,
            "mean_elite_age": round(float(np.mean(ages)), 1) if ages else 0.0,
        }