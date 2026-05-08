"""
gp/fitness.py - Unified fitness evaluation for SymboLR.

Provides both real (PyTorch-backed) and synthetic (numpy-only) fitness
evaluation. The synthetic mode simulates training dynamics on a quadratic
loss landscape, producing realistic loss curves that respond meaningfully
to schedule quality
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gp.tree import Node

from config.settings import get_config


def evaluate_synthetic(lr_schedule: np.ndarray) -> float:
    """
    Simulate training on a noisy quadratic loss landscape using the given
    LR schedule. Returns a final 'validation loss' that meaningfully
    responds to schedule quality.

    The simulation models gradient descent on:
        L(w) = 0.5 * curvature * (w - w*)^2 + noise

    where each step applies:
        w_{t+1} = w_t - lr_t * grad_L(w_t) + noise

    This produces:
    - Realistic convergence for good schedules (~0.05-0.3 final loss)
    - Divergence for too-aggressive schedules (>2.0)
    - Stagnation for too-conservative schedules (~1.5-2.0)
    - Meaningful differentiation between schedule shapes

    Args:
        lr_schedule: Array of learning rates, one per time step.

    Returns:
        Simulated validation loss (lower is better).
    """
    cfg = get_config()

    # Validate the schedule
    if not np.all(np.isfinite(lr_schedule)):
        return float("inf")
    if np.all(lr_schedule < 1e-7):
        return float("inf")
    if np.any(lr_schedule > 10.0):
        return float("inf")

    n_steps = len(lr_schedule)
    if n_steps == 0:
        return float("inf")

    # Normalize the schedule to a practical LR range
    # GP trees produce raw math values (e.g. 0.5*(1-t) ranges 0 to 0.5).
    # Real training requires LRs in ~[1e-4, 0.05]. We normalize the
    # schedule to preserve its *shape* while mapping to a practical range.
    # This mimics what a user would do: lr(t) = base_lr * formula(t).
    lr_max = np.max(lr_schedule)
    lr_min = np.min(lr_schedule)
    lr_range = lr_max - lr_min

    if lr_range > 1e-8:
        # Normalize to [target_min, target_max] preserving shape
        target_min, target_max = 1e-4, 0.03
        normalized = target_min + (lr_schedule - lr_min) / lr_range * (target_max - target_min)
    else:
        # Constant schedule - map to a fixed moderate LR
        mean_lr = np.mean(lr_schedule)
        # Constant schedules should get a reasonable but not great fitness
        normalized = np.full(n_steps, 0.01)

    # Simulate 5 independent "parameter dimensions" for robustness
    n_dims = 5
    rng = np.random.RandomState(cfg.seed)

    # Initial weights - scattered around the optimum
    w = rng.randn(n_dims) * 2.0

    # Optimal weights
    w_star = np.zeros(n_dims)

    # Curvature per dimension (heterogeneous - mimics real loss landscapes)
    curvatures = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

    # Track loss trajectory
    losses = []
    best_loss = float("inf")

    for step in range(n_steps):
        lr = float(normalized[step])

        # Compute loss: sum of per-dimension quadratics
        diff = w - w_star
        loss = 0.5 * np.sum(curvatures * diff ** 2)

        # Add observation noise (simulates mini-batch variance)
        noisy_loss = loss + rng.randn() * cfg.synth_noise_scale * (1.0 + loss)

        losses.append(max(0.0, noisy_loss))
        best_loss = min(best_loss, max(0.0, noisy_loss))

        # Check for explosion
        if loss > 1000.0 or not np.isfinite(loss):
            return float("inf")

        # Gradient: dL/dw = curvature * (w - w*)
        grad = curvatures * diff

        # Add gradient noise (simulates stochastic gradients)
        grad_noise = rng.randn(n_dims) * 0.1 * np.sqrt(1.0 + np.abs(loss))
        noisy_grad = grad + grad_noise

        # SGD step
        w = w - lr * noisy_grad

        # Clip weights to prevent numerical overflow
        w = np.clip(w, -50.0, 50.0)

    # Final validation: evaluate without noise
    final_diff = w - w_star
    val_loss = 0.5 * np.sum(curvatures * final_diff ** 2)

    # Combine: weighted average of final loss and best-seen loss
    # (rewards both convergence and stability)
    combined = 0.6 * val_loss + 0.4 * best_loss

    # Penalty for schedules that are essentially constant (no t-dependence)
    lr_std = np.std(lr_schedule)
    if lr_std < 1e-6:
        combined += 0.5  # Constant schedules are boring

    # Penalty for schedules that increase over time (generally bad)
    if lr_schedule[-1] > lr_schedule[0] * 1.5:
        combined += 0.3

    return float(np.clip(combined, 0.0, 50.0))


def evaluate_fitness(
    tree: "Node",
    t_array: np.ndarray,
    trainer=None,
    model_factory=None,
    train_loader=None,
    val_loader=None,
    epochs: int = 1,
) -> float:
    """
    Unified fitness evaluation. Routes to real PyTorch training when
    available, otherwise uses synthetic simulation.

    Args:
        tree: GP tree to evaluate.
        t_array: Normalized time array.
        trainer: ProbeTrainer instance (None in cloud mode).
        model_factory: Callable returning a fresh model (None in cloud mode).
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Training epochs per evaluation.

    Returns:
        Validation loss (lower is better).
    """
    from gp.rust_bridge import evaluate_schedule

    # Check fitness cache first
    if hasattr(tree, "fitness") and tree.fitness is not None:
        return tree.fitness

    # Compute LR schedule
    lr_schedule = evaluate_schedule(tree, t_array)

    cfg = get_config()

    if cfg.torch_available and trainer is not None and model_factory is not None:
        # Real training evaluation
        model = model_factory()
        loss = trainer.evaluate_schedule(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr_schedule=lr_schedule,
            epochs=epochs,
        )
    else:
        # Synthetic fitness evaluation
        loss = evaluate_synthetic(lr_schedule)

    # Cache the result
    tree.fitness = loss
    return loss
