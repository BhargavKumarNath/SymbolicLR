"""
optimiser/compare.py — Run and cache baseline comparisons.

Provides cached baseline benchmark results that are computed once
per session and reused across dashboard rerenders.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from baselines.schedules import (
    BASELINE_SCHEDULES,
    evaluate_all_baselines,
    benchmark_all_baselines,
)

# Module-level cache
_cached_results: Optional[Dict[str, float]] = None
_cached_curves: Optional[Dict[str, np.ndarray]] = None


def get_baseline_results(time_steps: int = 100, force: bool = False) -> Dict[str, float]:
    """
    Get benchmark losses for all baseline schedules.
    Results are cached after first computation.
    """
    global _cached_results
    if _cached_results is not None and not force:
        return _cached_results

    t_array = np.linspace(0.0, 1.0, time_steps, dtype=np.float64)
    _cached_results = benchmark_all_baselines(t_array)
    return _cached_results


def get_baseline_curves(time_steps: int = 100) -> Dict[str, np.ndarray]:
    """Get LR schedule curves for all baselines."""
    global _cached_curves
    if _cached_curves is not None:
        return _cached_curves

    t_array = np.linspace(0.0, 1.0, time_steps, dtype=np.float64)
    _cached_curves = evaluate_all_baselines(t_array)
    return _cached_curves


def get_comparison_data(
    symbolr_loss: Optional[float] = None,
    time_steps: int = 100,
) -> List[Dict]:
    """
    Build comparison data for the dashboard chart.
    Includes all baselines plus the SymboLR elite if available.

    Returns:
        List of dicts with keys: Schedule, Val Loss, Type
    """
    results = get_baseline_results(time_steps)

    data = [
        {"Schedule": name, "Val Loss": loss, "Type": "Hand-crafted"}
        for name, loss in sorted(results.items(), key=lambda x: x[1])
    ]

    if symbolr_loss is not None:
        data.append({
            "Schedule": "SymboLR Elite",
            "Val Loss": round(symbolr_loss, 4),
            "Type": "Discovered",
        })

    return data
