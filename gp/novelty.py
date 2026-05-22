"""
gp/novelty.py - Lightweight novelty search module.

Computes behavioral novelty scores using k-nearest-neighbour distance
in a compact 6-dimensional schedule fingerprint space.

Augments fitness to reward genuinely different schedule behaviors.
Designed to be independently disable-able via config.novelty_enabled.

All randomness uses Python's random module (respects global seed).
No GPU cost. Hot path safe (fingerprint computation is pure numpy).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


def compute_fingerprint(lr_schedule: np.ndarray) -> np.ndarray:
    """
    Compute a compact 6-element behavioral fingerprint for a learning rate schedule.

    Args:
        lr_schedule: 1D numpy array of learning rate values over normalized time.

    Returns:
        np.ndarray: [mean, std, min, max, center_of_mass, total_variation]
        All values are finite floats. Returns zeros for empty/invalid input.
    """
    n = len(lr_schedule)
    if n == 0 or not np.all(np.isfinite(lr_schedule)):
        return np.zeros(6, dtype=np.float64)

    t = np.linspace(0.0, 1.0, n)
    abs_lr = np.abs(lr_schedule)
    total_lr = float(np.sum(abs_lr))

    mean_lr = float(np.mean(lr_schedule))
    std_lr = float(np.std(lr_schedule))
    min_lr = float(np.min(lr_schedule))
    max_lr = float(np.max(lr_schedule))
    com = float(np.sum(t * abs_lr) / total_lr) if total_lr > 1e-10 else 0.5
    tv = float(np.sum(np.abs(np.diff(lr_schedule))))

    return np.array([mean_lr, std_lr, min_lr, max_lr, com, tv], dtype=np.float64)


class NoveltyArchive:
    """
    Rolling archive of behavioral fingerprints for novelty scoring.

    Uses k-nearest-neighbour mean distance as the novelty metric.
    Maintains a rolling deque to prevent unbounded memory growth.

    Key properties:
    - Novelty score = mean Euclidean distance to k nearest neighbours
    - Returns 0.0 if archive has fewer entries than k (safe default)
    - All operations are O(|archive| * d) where d=6 — extremely cheap
    - No GPU dependency. No sklearn dependency.

    Args:
        max_size: Maximum number of fingerprints to keep (oldest evicted first).
        k_neighbours: Number of nearest neighbours for scoring.
    """

    def __init__(self, max_size: int = 500, k_neighbours: int = 5):
        self.max_size = max_size
        self.k_neighbours = k_neighbours
        self._archive: deque = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self._archive)

    def add(self, fingerprint: np.ndarray) -> None:
        """Add a fingerprint to the rolling archive. Evicts oldest if full."""
        self._archive.append(fingerprint.copy())

    def novelty_score(self, fingerprint: np.ndarray) -> float:
        """
        Compute novelty as mean Euclidean distance to k nearest neighbours.

        Returns 0.0 if the archive is empty or has fewer entries than k.
        This is a safe default: zero novelty = no augmentation of fitness.

        Args:
            fingerprint: 6-element behavioral fingerprint (from compute_fingerprint).

        Returns:
            float: Non-negative novelty score. Higher = more behaviorally novel.
        """
        n = len(self._archive)
        if n == 0:
            return 0.0

        k = min(self.k_neighbours, n)
        archive_array = np.array(self._archive)  # shape (n, 6)

        # Vectorized Euclidean distances
        diffs = archive_array - fingerprint
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))

        # k-NN mean distance (partitioned sort — O(n) not O(n log n))
        nearest_k = np.partition(distances, k - 1)[:k]
        return float(np.mean(nearest_k))

    def is_novel(self, fingerprint: np.ndarray, threshold: float = 0.02) -> bool:
        """
        True if the fingerprint is NOT a near-duplicate of any archived entry.

        Uses minimum distance (not mean) to detect near-duplicates robustly.
        Returns True (novel) if the archive is empty — safe default.

        Args:
            fingerprint: 6-element behavioral fingerprint.
            threshold: Minimum distance below which a fingerprint is considered
                       a duplicate. Default 0.02 is conservative.
        """
        if len(self._archive) == 0:
            return True
        archive_array = np.array(self._archive)
        diffs = archive_array - fingerprint
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        return float(np.min(distances)) > threshold

    def get_stats(self) -> dict:
        """Return summary stats for diagnostics."""
        return {
            "size": len(self._archive),
            "max_size": self.max_size,
            "k_neighbours": self.k_neighbours,
        }
