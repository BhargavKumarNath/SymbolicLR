"""
gp/meta_controller.py - Rule-based meta-evolutionary control system.

Implements a 3-phase state machine (EXPLOIT / EXPLORE / DIVERSIFY) that
replaces the ad-hoc stagnation counter previously inlined in benchmark.py.

Key design constraints:
- All transitions are explicit threshold rules — fully interpretable
- No reinforcement learning, no neural controllers, no learned policies
- Independently disable-able: falls back to Phase 2 stagnation logic
- All state is deterministic given the evolution metrics stream
- Outputs are control signals (multipliers), not hard overrides

Phases:
    EXPLOIT:    Normal operation. Archive growing, loss improving.
    EXPLORE:    Stagnation detected. Boost mutation + inject immigrants.
    DIVERSIFY:  Stagnation persistent AND diversity collapsing.
                Maximum mutation + maximum immigrants + max novelty.
"""

from __future__ import annotations

import enum
from typing import Optional


class Phase(enum.Enum):
    EXPLOIT = "exploit"
    EXPLORE = "explore"
    DIVERSIFY = "diversify"


class MetaController:
    """
    3-phase rule-based evolutionary meta-controller.

    Reads evolution metrics once per generation and outputs control signals
    that tune mutation intensity, novelty pressure, and immigrant injection.

    State machine transitions:
        EXPLOIT → EXPLORE:    stagnation_epochs >= stagnation_threshold
        EXPLORE → DIVERSIFY:  stagnation persists (2x threshold) AND diversity collapsing
        DIVERSIFY → EXPLOIT:  loss improves OR diversity recovers
        EXPLORE → EXPLOIT:    loss improves by > improvement_delta

    Control outputs (read as properties after calling update()):
        mutation_boost           float   Multiplier for mutation_rate
        novelty_weight           float   Novelty pressure [base, max_novelty]
        immigrant_count          int     Random immigrants to inject
        crossover_rate_delta     float   Signed adjustment to crossover rate

    Args:
        stagnation_threshold:   Generations without improvement before EXPLORE.
        collapse_threshold:     Structural diversity ratio below which DIVERSIFY triggers.
        improvement_delta:      Minimum loss delta to count as improvement.
        base_novelty_weight:    Default novelty weight in EXPLOIT phase.
        max_novelty_weight:     Maximum novelty weight (cap, per user constraint).
        base_mutation_boost:    Extra mutation rate multiplier in EXPLORE phase.
        base_immigrant_fraction: Fraction of pop_size to inject as immigrants.
        pop_size:               Population size (for immigrant count calculation).
    """

    def __init__(
        self,
        stagnation_threshold: int = 5,
        collapse_threshold: float = 0.30,
        improvement_delta: float = 1e-4,
        base_novelty_weight: float = 0.10,
        max_novelty_weight: float = 0.25,
        base_mutation_boost: float = 0.50,
        base_immigrant_fraction: float = 0.12,
        pop_size: int = 50,
    ):
        self.stagnation_threshold = stagnation_threshold
        self.collapse_threshold = collapse_threshold
        self.improvement_delta = improvement_delta
        self.base_novelty_weight = base_novelty_weight
        self.max_novelty_weight = max_novelty_weight
        self.base_mutation_boost = base_mutation_boost
        self.base_immigrant_fraction = base_immigrant_fraction
        self.pop_size = pop_size

        # Internal state
        self._phase: Phase = Phase.EXPLOIT
        self._stagnation_epochs: int = 0
        self._previous_best: float = float("inf")

    def update(
        self,
        best_loss: float,
        archive_size: int,
        structural_diversity: float,
        generation: int,
    ) -> None:
        """
        Update controller state from current generation metrics.

        Call exactly once per generation, after archive update and before
        offspring generation begins.

        Args:
            best_loss:            Current best validation loss in archive.
            archive_size:         Number of occupied niches (unused in transitions,
                                  available for future extensions).
            structural_diversity: Ratio of unique hashes to archive size [0, 1].
            generation:           Current generation index (1-indexed).
        """
        loss_improved = best_loss < self._previous_best - self.improvement_delta
        diversity_collapsing = structural_diversity < self.collapse_threshold

        if loss_improved:
            self._stagnation_epochs = 0
            self._previous_best = best_loss
        else:
            self._stagnation_epochs += 1

        # State machine transitions (explicit, ordered)
        if self._phase == Phase.EXPLOIT:
            if self._stagnation_epochs >= self.stagnation_threshold:
                self._phase = Phase.EXPLORE

        elif self._phase == Phase.EXPLORE:
            if loss_improved:
                self._phase = Phase.EXPLOIT
            elif (
                self._stagnation_epochs >= self.stagnation_threshold * 2
                and diversity_collapsing
            ):
                self._phase = Phase.DIVERSIFY

        elif self._phase == Phase.DIVERSIFY:
            if loss_improved or not diversity_collapsing:
                self._phase = Phase.EXPLOIT
                self._stagnation_epochs = 0

    # -----------------------------------------------------------------------
    # Control signal properties
    # -----------------------------------------------------------------------

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def stagnation_epochs(self) -> int:
        return self._stagnation_epochs

    @property
    def mutation_boost(self) -> float:
        """
        Multiplier applied to base mutation_rate.
        1.0 in EXPLOIT, escalates in EXPLORE and DIVERSIFY.
        """
        if self._phase == Phase.EXPLOIT:
            return 1.0
        elif self._phase == Phase.EXPLORE:
            return 1.0 + self.base_mutation_boost
        else:  # DIVERSIFY
            return 1.0 + self.base_mutation_boost * 2.0

    @property
    def novelty_weight(self) -> float:
        """
        Novelty augmentation weight within [base_novelty_weight, max_novelty_weight].
        Increases gradually with stagnation duration in EXPLORE phase.
        """
        if self._phase == Phase.EXPLOIT:
            return self.base_novelty_weight
        elif self._phase == Phase.EXPLORE:
            # Scale up proportionally to stagnation duration (capped at max)
            scale = min(
                1.5,
                1.0 + self._stagnation_epochs / (self.stagnation_threshold * 2),
            )
            return min(self.max_novelty_weight, self.base_novelty_weight * scale)
        else:  # DIVERSIFY
            return self.max_novelty_weight

    @property
    def immigrant_count(self) -> int:
        """Number of random immigrants to inject into parent pool this generation."""
        if self._phase == Phase.EXPLOIT:
            return 0
        elif self._phase == Phase.EXPLORE:
            return int(self.pop_size * self.base_immigrant_fraction)
        else:  # DIVERSIFY
            return int(self.pop_size * self.base_immigrant_fraction * 2)

    @property
    def crossover_rate_delta(self) -> float:
        """
        Signed adjustment to crossover rate.
        Reduces crossover slightly in non-EXPLOIT phases to make room for mutations.
        """
        if self._phase == Phase.EXPLOIT:
            return 0.0
        return -0.10  # slight reduction

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def get_status(self) -> dict:
        """Human-readable status summary for terminal display and diagnostics."""
        return {
            "phase": self._phase.value,
            "stagnation_epochs": self._stagnation_epochs,
            "mutation_boost": round(self.mutation_boost, 2),
            "novelty_weight": round(self.novelty_weight, 3),
            "immigrant_count": self.immigrant_count,
        }
