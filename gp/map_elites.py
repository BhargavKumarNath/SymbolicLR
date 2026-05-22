"""
gp/map_elites.py - MAP-Elites Quality-Diversity Archive.

Maintains a 3D grid of elite individuals based on behavioral descriptors.
Optimizes for Quality-Diversity: finding the best loss in every possible niche.

Uses the unified rust_bridge for schedule evaluation.
"""

import random
import numpy as np
from typing import Dict, Tuple, List, Optional
import copy
from gp.tree import Node
from gp.rust_bridge import evaluate_schedule


class MAPElitesArchive:
    """
    Maintains a 3D grid of elite individuals based on behavioral descriptors.
    Optimizes for Quality-Diversity: Finding the best loss in every possible niche.
    """

    def __init__(self, size_bins: int = 30, com_bins: int = 20, smoothness_bins: int = 10, time_steps: int = 100):
        self.size_bins = size_bins
        self.com_bins = com_bins
        self.smoothness_bins = smoothness_bins

        # Grid dimensions: [Size, Center_of_Mass, Smoothness]
        # Stores tuples of (loss: float, tree: Node)
        self.archive: Dict[Tuple[int, int, int], Tuple[float, Node]] = {}
        self._expression_hashes: set = set()  # structural dedup
        self._niche_update_counts: Dict[Tuple, int] = {}  # curiosity tracking
        self._total_attempts: int = 0
        self._total_additions: int = 0

        # Pre-compute time array for behavior extraction
        self.t_array = np.linspace(0.0, 1.0, time_steps, dtype=np.float64)

    def _compute_descriptors(self, tree: Node) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Computes (Complexity, Center of Mass, Smoothness) descriptors.
        Returns (None, None, None) if the formula evaluates to invalid math.
        """
        size = tree.size()
        try:
            lr_schedule = evaluate_schedule(tree, self.t_array)

            # Prevent sum(0) division errors
            total_lr = np.sum(lr_schedule)
            if total_lr == 0 or not np.isfinite(total_lr):
                return None, None, None

            # Center of Mass = sum(t * LR(t)) / sum(LR(t))
            com = np.sum(self.t_array * lr_schedule) / total_lr

            # Clamp COM strictly between 0.0 and 1.0
            com = max(0.0, min(1.0, float(com)))
            
            # Smoothness = total variation of schedule (lower = smoother)
            diffs = np.abs(np.diff(lr_schedule))
            total_variation = np.sum(diffs)
            # Normalize TV to [0, 1] range
            max_tv = np.max(lr_schedule) - np.min(lr_schedule) if np.max(lr_schedule) > np.min(lr_schedule) else 1.0
            normalized_tv = min(1.0, total_variation / (max_tv * len(diffs) + 1e-8))

        except Exception:
            return None, None, None

        # Discretize into bins
        size_idx = min(size, self.size_bins - 1)
        com_idx = int(com * (self.com_bins - 1))
        smoothness_idx = int(normalized_tv * (self.smoothness_bins - 1))

        return size_idx, com_idx, smoothness_idx

    def try_add(self, tree: Node, loss: float) -> bool:
        """
        Attempts to place a tree into the archive.
        Returns True if it occupies an empty niche or beats the current niche champion.
        """
        if not np.isfinite(loss):
            return False

        self._total_attempts += 1
        size_idx, com_idx, smoothness_idx = self._compute_descriptors(tree)
        if size_idx is None or com_idx is None or smoothness_idx is None:
            return False

        niche = (size_idx, com_idx, smoothness_idx)
        tree_hash = tree.get_hash()

        # Allow structural duplicates in different niches, but not in same niche
        if niche in self.archive:
            existing_hash = self.archive[niche][1].get_hash()
            if existing_hash == tree_hash:
                return False  # exact duplicate in same niche
                
        if niche not in self.archive or loss < self.archive[niche][0]:
            self.archive[niche] = (loss, copy.deepcopy(tree))
            self._expression_hashes.add(tree_hash)
            self._niche_update_counts[niche] = self._niche_update_counts.get(niche, 0) + 1
            self._total_additions += 1
            return True

        return False

    def sample_parents(self, batch_size: int) -> List[Node]:
        """
        Hybrid selection: fitness-proportionate + curiosity + uniform.
        """
        if not self.archive:
            return []
            
        parents = []
        occupied = list(self.archive.keys())
        losses = [self.archive[k][0] for k in occupied]
        
        # Compute fitness-proportionate weights (inverse loss)
        max_loss = max(losses) + 1e-8
        weights = [(max_loss - l + 1e-8) for l in losses]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        
        n_fitness = int(batch_size * 0.6)   # 60% fitness-weighted
        n_curiosity = int(batch_size * 0.2)  # 20% least-recently-updated
        n_uniform = batch_size - n_fitness - n_curiosity  # 20% uniform
        
        # Fitness-weighted
        if n_fitness > 0:
            indices = random.choices(range(len(occupied)), weights=probs, k=n_fitness)
            for idx in indices:
                parents.append(copy.deepcopy(self.archive[occupied[idx]][1]))
        
        # Curiosity (least recently updated niches)
        if n_curiosity > 0:
            update_counts = [(self._niche_update_counts.get(k, 0), k) for k in occupied]
            update_counts.sort()  # least updated first
            curiosity_niches = [k for _, k in update_counts[:max(1, len(update_counts)//3)]]
            for _ in range(n_curiosity):
                niche = random.choice(curiosity_niches)
                parents.append(copy.deepcopy(self.archive[niche][1]))
        
        # Uniform
        for _ in range(n_uniform):
            niche = random.choice(occupied)
            parents.append(copy.deepcopy(self.archive[niche][1]))
            
        return parents

    def get_hall_of_fame(self, top_k: int = 5) -> List[Tuple[float, Node]]:
        """Returns the absolute best globally performing formulas across all niches."""
        all_elites = list(self.archive.values())
        all_elites.sort(key=lambda x: x[0])  # Sort by loss ascending
        return all_elites[:top_k]
        
    def get_stats(self) -> dict:
        max_niches = self.size_bins * self.com_bins * self.smoothness_bins
        occupied = len(self.archive)
        unique_hashes = len(self._expression_hashes)
        losses = [v[0] for v in self.archive.values()] if self.archive else []
        return {
            'occupied_niches': occupied,
            'max_niches': max_niches,
            'occupancy_pct': round(100 * occupied / max(1, max_niches), 1),
            'unique_expressions': unique_hashes,
            'total_attempts': self._total_attempts,
            'total_additions': self._total_additions,
            'acceptance_rate': round(self._total_additions / max(1, self._total_attempts), 3),
            'best_loss': min(losses) if losses else float('inf'),
            'median_loss': float(np.median(losses)) if losses else float('inf'),
            'loss_std': float(np.std(losses)) if losses else 0.0,
        }