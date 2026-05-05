"""
baselines/schedules.py — Standard LR schedule implementations.

Each function takes a normalized time array t ∈ [0, 1] and returns
an array of learning rates. These are used for:
  1. Generating baseline comparison data (real, not hardcoded)
  2. Visualization in the dashboard
  3. Benchmarking against discovered formulas
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Callable


def cosine_annealing(
    t: np.ndarray,
    lr_max: float = 0.01,
    lr_min: float = 1e-5,
) -> np.ndarray:
    """Standard cosine annealing: η(t) = η_min + 0.5(η_max - η_min)(1 + cos(πt))"""
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * t))


def step_decay(
    t: np.ndarray,
    lr_init: float = 0.01,
    gamma: float = 0.5,
    n_steps: int = 3,
) -> np.ndarray:
    """Step decay: drop LR by gamma at evenly spaced intervals."""
    step_positions = np.linspace(0, 1, n_steps + 1)[1:-1]  # drop boundaries
    lr = np.full_like(t, lr_init)
    for pos in step_positions:
        lr = np.where(t >= pos, lr * gamma, lr)
    return lr


def warm_restarts(
    t: np.ndarray,
    lr_max: float = 0.01,
    lr_min: float = 1e-5,
    n_restarts: int = 3,
) -> np.ndarray:
    """Cosine annealing with warm restarts (SGDR)."""
    cycle_t = (t * n_restarts) % 1.0
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * cycle_t))


def linear_decay(
    t: np.ndarray,
    lr_init: float = 0.01,
    lr_final: float = 1e-5,
) -> np.ndarray:
    """Simple linear decay from lr_init to lr_final."""
    return lr_init + (lr_final - lr_init) * t


def constant_lr(
    t: np.ndarray,
    lr: float = 0.01,
) -> np.ndarray:
    """Constant learning rate (baseline)."""
    return np.full_like(t, lr)


def one_cycle(
    t: np.ndarray,
    lr_max: float = 0.01,
    lr_min: float = 1e-5,
    warmup_frac: float = 0.3,
) -> np.ndarray:
    """
    1-Cycle policy: linear warmup then cosine decay.
    Warmup: lr_min → lr_max over [0, warmup_frac]
    Decay:  lr_max → lr_min over [warmup_frac, 1]
    """
    lr = np.where(
        t < warmup_frac,
        lr_min + (lr_max - lr_min) * (t / warmup_frac),
        lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(
            np.pi * (t - warmup_frac) / (1.0 - warmup_frac)
        )),
    )
    return lr


def exponential_decay(
    t: np.ndarray,
    lr_init: float = 0.01,
    decay_rate: float = 5.0,
) -> np.ndarray:
    """Exponential decay: η(t) = η_init * exp(-decay_rate * t)"""
    return lr_init * np.exp(-decay_rate * t)


# Registry for easy iteration

BASELINE_SCHEDULES: Dict[str, Callable] = {
    "Cosine Annealing": cosine_annealing,
    "Step Decay": step_decay,
    "Warm Restarts": warm_restarts,
    "Linear Decay": linear_decay,
    "Constant LR": constant_lr,
    "1-Cycle": one_cycle,
    "Exponential Decay": exponential_decay,
}


def evaluate_all_baselines(t_array: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Evaluate all baseline schedules on the given time array.

    Returns:
        Dict mapping schedule name → LR array.
    """
    return {name: fn(t_array) for name, fn in BASELINE_SCHEDULES.items()}


def benchmark_all_baselines(t_array: np.ndarray) -> Dict[str, float]:
    """
    Run all baseline schedules through the synthetic fitness function
    and return their simulated validation losses.

    """
    from gp.fitness import evaluate_synthetic

    results = {}
    for name, fn in BASELINE_SCHEDULES.items():
        lr_schedule = fn(t_array)
        loss = evaluate_synthetic(lr_schedule)
        results[name] = round(loss, 4)
    return results
