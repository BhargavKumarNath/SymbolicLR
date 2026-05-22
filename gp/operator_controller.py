"""
gp/operator_controller.py - Adaptive operator selection via bandit-style learning.

Tracks per-operator success rates using exponential moving average (EMA).
Replaces static operator probabilities with evidence-based adaptive selection.

'Success' is defined as an offspring entering the MAP-Elites archive.

Key design constraints:
- Minimum 5% floor probability per operator (no starvation)
- EMA update once per generation (not per offspring — avoids noise)
- Fully deterministic under fixed seed (uses Python's random module)
- Independently disable-able: falls back to static config rates when disabled.

Args:
    min_prob:   Minimum selection probability per operator (default 5%).
    ema_alpha:  EMA smoothing factor. Higher = faster adaptation (default 0.3).
"""

from __future__ import annotations

import random
from typing import Dict


class OperatorController:
    """
    Bandit-style adaptive operator controller.

    Tracks per-operator success rates using EMA and selects operators
    proportional to their evidence-based success rates, with a minimum
    floor to prevent complete starvation of any operator.

    Usage:
        controller = OperatorController()
        op = controller.select_operator()          # during offspring creation
        controller.record_outcome(op, success)     # after archive.try_add()
        controller.end_generation()                # once per generation end
    """

    OPERATORS = [
        "crossover",
        "subtree_mutation",
        "hoist_mutation",
        "point_mutation",
        "constant_perturbation",
    ]

    def __init__(self, min_prob: float = 0.05, ema_alpha: float = 0.30):
        self.min_prob = min_prob
        self.ema_alpha = ema_alpha

        # Initialize equal success rates (0.5 = neutral starting assumption)
        self._success_rates: Dict[str, float] = {op: 0.5 for op in self.OPERATORS}

        # Lifetime counters (for diagnostics)
        self._total_uses: Dict[str, int] = {op: 0 for op in self.OPERATORS}
        self._total_wins: Dict[str, int] = {op: 0 for op in self.OPERATORS}

        # Per-generation counters (reset each generation)
        self._gen_uses: Dict[str, int] = {op: 0 for op in self.OPERATORS}
        self._gen_wins: Dict[str, int] = {op: 0 for op in self.OPERATORS}

    def select_operator(self) -> str:
        """
        Select an operator using weighted random choice based on EMA success rates.

        Respects minimum floor probability for all operators.
        Uses Python's random.choices() — deterministic under global seed.

        Returns:
            str: Operator name (one of OPERATORS).
        """
        probs = self.get_probabilities()
        ops = list(probs.keys())
        weights = list(probs.values())
        return random.choices(ops, weights=weights, k=1)[0]

    def record_outcome(self, op_name: str, success: bool) -> None:
        """
        Record the result of applying a genetic operator.

        Args:
            op_name: The operator that was applied.
            success: True if the offspring entered the MAP-Elites archive.
        """
        if op_name not in self._success_rates:
            return
        self._total_uses[op_name] += 1
        self._gen_uses[op_name] += 1
        if success:
            self._total_wins[op_name] += 1
            self._gen_wins[op_name] += 1

    def end_generation(self) -> None:
        """
        Update EMA success rates from this generation's outcomes.

        Must be called exactly once per generation, after all offspring
        have been evaluated and archive insertions recorded.
        """
        for op in self.OPERATORS:
            gen_uses = self._gen_uses[op]
            if gen_uses > 0:
                gen_rate = self._gen_wins[op] / gen_uses
                # EMA update: blend new generation rate into running average
                self._success_rates[op] = (
                    self.ema_alpha * gen_rate
                    + (1 - self.ema_alpha) * self._success_rates[op]
                )
            # Reset per-generation counters
            self._gen_uses[op] = 0
            self._gen_wins[op] = 0

    def get_probabilities(self) -> Dict[str, float]:
        """
        Compute current normalized selection probabilities.

        Applies minimum floor before normalizing so no operator is ever
        completely excluded from selection.

        Returns:
            Dict mapping operator name -> selection probability (sums to 1.0).
        """
        floored = {
            op: max(self.min_prob, self._success_rates[op])
            for op in self.OPERATORS
        }
        total = sum(floored.values())
        return {op: v / total for op, v in floored.items()}

    def get_stats(self) -> dict:
        """Returns per-operator statistics for diagnostics and terminal display."""
        probs = self.get_probabilities()
        return {
            op: {
                "probability": round(probs[op], 3),
                "success_rate": round(self._success_rates[op], 3),
                "total_uses": self._total_uses[op],
                "total_wins": self._total_wins[op],
            }
            for op in self.OPERATORS
        }

    def get_dominant_operator(self) -> str:
        """Returns the operator with the highest current selection probability."""
        probs = self.get_probabilities()
        return max(probs, key=lambda k: probs[k])

    def format_probabilities(self) -> str:
        """Format operator probabilities as a compact terminal string."""
        probs = self.get_probabilities()
        abbrev = {
            "crossover": "xo",
            "subtree_mutation": "sub",
            "hoist_mutation": "hoist",
            "point_mutation": "pt",
            "constant_perturbation": "perturb",
        }
        parts = [f"{abbrev[op]}={probs[op]:.0%}" for op in self.OPERATORS]
        return " ".join(parts)
