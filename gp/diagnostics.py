"""
gp/diagnostics.py - Research-grade per-generation diagnostics.

Collects concise, research-relevant metrics per generation.
Exports to JSON and CSV for offline analysis.

Key design constraints:
- Zero overhead on the hot evaluation path: all collection happens AFTER evaluation
- Compact storage: Python primitives only (no large arrays, no tensors)
- Export happens once at run end (no per-generation disk I/O)
- Independently disable-able via config.diagnostics_enabled
- Family detection via lightweight regex motif matching on LaTeX strings
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class GenerationMetrics:
    """Per-generation metrics snapshot. All fields are simple Python primitives."""

    generation: int = 0
    best_loss: float = float("inf")
    median_loss: float = float("inf")
    archive_size: int = 0
    occupancy_pct: float = 0.0
    structural_diversity: float = 1.0
    behavioral_diversity: float = 1.0
    novelty_mean: float = 0.0
    novelty_max: float = 0.0
    new_niches: int = 0
    controller_phase: str = "exploit"
    stagnation_epochs: int = 0
    mutation_boost: float = 1.0
    operator_dominant: str = "crossover"
    operator_probs: Dict[str, float] = field(default_factory=dict)
    surrogate_rmse: float = -1.0   # -1 = surrogate not active/ready
    surrogate_buffer: int = 0
    gen_time_s: float = 0.0


def classify_family(latex_formula: str) -> str:
    """
    Classify a LaTeX formula string into a human-readable schedule family.

    Uses simple regex-based motif detection. No ML, no parsing overhead.

    Returns one of: 'cyclical', 'exponential', 'inverse', 'linear', 'constant', 'polynomial'
    """
    f = latex_formula.lower()
    if re.search(r"\\cos|\\sin", f):
        return "cyclical"
    if re.search(r"e\^|\\exp|\\mathrm\{e\}", f):
        return "exponential"
    if re.search(r"\\frac\{[^}]+\}\{t\}|t\^\{-", f):
        return "inverse"
    # linear: contains t but no t^2 or higher
    if re.search(r"\bt\b", f) and not re.search(r"t\^\{[2-9]", f):
        return "linear"
    if not re.search(r"\bt\b", f):
        return "constant"
    return "polynomial"


class DiagnosticsLog:
    """
    Collects per-generation research metrics and exports to JSON/CSV.

    Designed to be lightweight:
    - All data stored as compact Python primitives
    - Export happens once at run end, not every generation
    - No disk writes in the hot evaluation path
    - Novelty scores buffered internally, flushed once per generation

    Usage:
        log = DiagnosticsLog()
        # per generation:
        log.record_novelty(score)      # called per offspring
        mean_n, max_n = log.flush_novelty_stats()
        log.record(GenerationMetrics(...))
        # at end:
        log.export_json("results/run.json")
        log.export_csv("results/run.csv")
        print(log.summary())
    """

    def __init__(self):
        self._records: List[GenerationMetrics] = []
        self._novelty_buffer: List[float] = []
        self._run_start: float = time.time()

    def record(self, metrics: GenerationMetrics) -> None:
        """Add a generation's metrics to the log."""
        self._records.append(metrics)

    def record_novelty(self, score: float) -> None:
        """
        Buffer a novelty score for mean/max tracking.
        Call once per offspring during the offspring generation phase.
        """
        if score > 0.0:
            self._novelty_buffer.append(score)

    def flush_novelty_stats(self) -> tuple:
        """
        Return (mean, max) of buffered novelty scores and clear buffer.
        Call once per generation after all offspring novelty scores are buffered.
        """
        if not self._novelty_buffer:
            return 0.0, 0.0
        mean_n = float(sum(self._novelty_buffer) / len(self._novelty_buffer))
        max_n = float(max(self._novelty_buffer))
        self._novelty_buffer.clear()
        return mean_n, max_n

    def summary(self) -> dict:
        """Concise end-of-run summary for terminal display."""
        if not self._records:
            return {}

        initial_loss = self._records[0].best_loss
        final = self._records[-1]

        # Find dominant phase across all generations
        phase_counts: Dict[str, int] = {}
        for r in self._records:
            phase_counts[r.controller_phase] = phase_counts.get(r.controller_phase, 0) + 1
        dominant_phase = max(phase_counts, key=lambda p: phase_counts[p]) if phase_counts else "exploit"

        return {
            "total_generations": len(self._records),
            "initial_best_loss": round(initial_loss, 4),
            "final_best_loss": round(final.best_loss, 4),
            "improvement": round(initial_loss - final.best_loss, 4),
            "final_archive_size": final.archive_size,
            "final_structural_diversity": round(final.structural_diversity, 3),
            "final_behavioral_diversity": round(final.behavioral_diversity, 3),
            "dominant_phase": dominant_phase,
            "dominant_operator": final.operator_dominant,
            "surrogate_active": final.surrogate_buffer >= 50,
            "total_run_time_s": round(time.time() - self._run_start, 1),
        }

    def export_json(self, path: str) -> None:
        """Export full diagnostics log to a JSON file."""
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        data = {
            "summary": self.summary(),
            "generations": [asdict(r) for r in self._records],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, path: str) -> None:
        """Export per-generation metrics to a CSV file (operator_probs excluded)."""
        if not self._records:
            return

        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        fieldnames = [
            "generation", "best_loss", "median_loss", "archive_size",
            "occupancy_pct", "structural_diversity", "behavioral_diversity",
            "novelty_mean", "novelty_max", "new_niches", "controller_phase",
            "stagnation_epochs", "mutation_boost", "operator_dominant",
            "surrogate_rmse", "surrogate_buffer", "gen_time_s",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._records:
                row = asdict(r)
                row.pop("operator_probs", None)
                writer.writerow({k: row.get(k, "") for k in fieldnames})
