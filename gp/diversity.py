"""
gp/diversity.py - Diversity tracking and semantic duplicate filtering.

Independently disable-able. Computes two complementary diversity metrics:
    - Structural diversity: ratio of unique tree hashes / archive size
    - Behavioral diversity: sampled mean pairwise fingerprint distance

Semantic duplicate filtering: optionally rejects schedules that are
structurally different but behaviorally near-identical, preventing archive
pollution with meaningless variants.

Key design constraints:
- Zero overhead on the hot evaluation path (call once per generation, after archive update)
- Behavioral diversity uses random sampling to avoid O(n^2) cost
- All randomness goes through Python's random module (respects global seed)
- Independently disable-able: set diversity_tracking_enabled=False in config
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, List

import numpy as np

from gp.novelty import NoveltyArchive, compute_fingerprint

if TYPE_CHECKING:
    from gp.map_elites import MAPElitesArchive


class DiversityTracker:
    """
    Tracks structural and behavioral diversity of the MAP-Elites archive.

    All metrics are computed from the archive state after generation completion.
    Designed to be called once per generation — not on the hot evaluation path.

    Args:
        sample_size:        Max number of individuals to sample for behavioral diversity.
                            Keeps computation O(sample_size^2) not O(archive_size^2).
        collapse_threshold: Structural diversity below this triggers DIVERSIFY phase.
    """

    def __init__(self, sample_size: int = 50, collapse_threshold: float = 0.30):
        self.sample_size = sample_size
        self.collapse_threshold = collapse_threshold

        self._structural: float = 1.0
        self._behavioral: float = 1.0

    def update(self, archive: "MAPElitesArchive") -> None:
        """
        Recompute diversity metrics from the current archive state.

        Call once per generation after archive.try_add() calls complete.
        Modifies internal state; read metrics via properties afterward.

        Args:
            archive: The MAP-Elites archive to compute metrics from.
        """
        entries = list(archive.archive.values())
        if not entries:
            self._structural = 1.0
            self._behavioral = 1.0
            return

        # --- Structural diversity: unique hashes / archive size ---
        hashes = set()
        for entry in entries:
            # entries are (loss, Node, age) tuples
            hashes.add(entry[1].get_hash())
        self._structural = len(hashes) / len(entries)

        # --- Behavioral diversity: sampled mean pairwise fingerprint distance ---
        if len(entries) < 2:
            self._behavioral = 1.0
            return

        t_array = archive.t_array
        trees = [entry[1] for entry in entries]

        # Sample to avoid O(n^2) cost
        sample_count = min(self.sample_size, len(trees))
        sampled = random.sample(trees, sample_count)

        try:
            from gp.rust_bridge import evaluate_schedule

            fingerprints = []
            for tree in sampled:
                try:
                    schedule = evaluate_schedule(tree, t_array)
                    if np.all(np.isfinite(schedule)):
                        fingerprints.append(compute_fingerprint(schedule))
                except Exception:
                    continue

            n = len(fingerprints)
            if n < 2:
                self._behavioral = 1.0
                return

            fp_array = np.array(fingerprints)

            # Normalize each dimension to [0,1] range for fair comparison
            ranges = np.ptp(fp_array, axis=0)
            ranges[ranges < 1e-10] = 1.0  # avoid division by zero
            fp_norm = fp_array / ranges

            # Mean pairwise distance (all unique pairs)
            total_dist = 0.0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    d = float(np.sqrt(np.sum((fp_norm[i] - fp_norm[j]) ** 2)))
                    total_dist += d
                    count += 1

            self._behavioral = min(1.0, total_dist / count) if count > 0 else 1.0

        except Exception:
            self._behavioral = 1.0

    @property
    def structural_diversity(self) -> float:
        """Ratio of unique tree hashes to archive size. Range [0, 1]."""
        return self._structural

    @property
    def behavioral_diversity(self) -> float:
        """Sampled mean pairwise behavioral fingerprint distance. Range [0, 1]."""
        return self._behavioral

    def is_collapsing(self) -> bool:
        """True if structural diversity has dropped below the collapse threshold."""
        return self._structural < self.collapse_threshold

    def get_stats(self) -> dict:
        """Return diversity metrics as a dict for diagnostics logging."""
        return {
            "structural_diversity": round(self._structural, 4),
            "behavioral_diversity": round(self._behavioral, 4),
            "collapsing": self.is_collapsing(),
        }


def is_semantic_duplicate(
    fingerprint: np.ndarray,
    novelty_archive: NoveltyArchive,
    threshold: float = 0.02,
) -> bool:
    """
    Return True if the behavioral fingerprint is a near-duplicate of an
    existing archived fingerprint.

    Used as a lightweight pre-rejection filter before archive insertion to
    prevent structurally different but behaviorally identical schedules from
    displacing genuinely diverse occupants.

    Safety: Returns False (not a duplicate) if the novelty archive is empty,
    so this never blocks insertion into a fresh archive.

    Args:
        fingerprint:    6-element behavioral fingerprint (from compute_fingerprint).
        novelty_archive: NoveltyArchive instance containing behavioral history.
        threshold:      Minimum distance below which fingerprint is a duplicate.

    Returns:
        bool: True if fingerprint is a near-duplicate, False otherwise.
    """
    return not novelty_archive.is_novel(fingerprint, threshold=threshold)
