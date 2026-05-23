"""
gp/fitness.py - Unified fitness evaluation for SymboLR.

Provides both real (PyTorch-backed) and synthetic (numpy-only) fitness
evaluation. The synthetic mode simulates training dynamics on a quadratic
loss landscape, producing realistic loss curves that respond meaningfully
to schedule quality
"""

from __future__ import annotations

import numpy as np
import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gp.tree import Node

from config.settings import get_config


def evaluate_synthetic(lr_schedule: np.ndarray, tree_size: int = 1) -> float:
    """
    Simulate training on a noisy quadratic loss landscape using the given
    LR schedule. Returns a final 'validation loss' that meaningfully
    responds to schedule quality.
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

    # Soft-clamp to practical LR range while preserving scale relationships
    raw_mean = np.mean(np.abs(lr_schedule))
    if raw_mean > 1e-7:
        # Scale so mean absolute value maps to ~0.01 (a reasonable base LR)
        scale_factor = 0.01 / raw_mean
        normalized = lr_schedule * scale_factor
    else:
        normalized = np.full(n_steps, 0.01)
        
    # Hard-clamp to prevent extreme values
    normalized = np.clip(normalized, 1e-5, 0.1)

    # Derive seed from schedule content for reproducibility per expression
    schedule_hash = int(hashlib.md5(lr_schedule.tobytes()).hexdigest()[:8], 16)
    
    n_evals = getattr(cfg, 'synth_n_evaluations', 3)
    ensemble_losses = []
    
    n_dims = getattr(cfg, 'synth_n_dims', 5)
    noise_scale = getattr(cfg, 'synth_noise_scale', 0.02)
    curvatures = np.array([0.5, 1.0, 2.0, 4.0, 8.0])[:n_dims]
    if len(curvatures) < n_dims:
        curvatures = np.pad(curvatures, (0, n_dims - len(curvatures)), constant_values=1.0)
    
    for eval_idx in range(n_evals):
        rng = np.random.RandomState((schedule_hash + eval_idx) % (2**32 - 1))
        
        # Pre-allocate arrays for vectorized simulation
        w = rng.randn(n_dims) * 2.0
        w_star = np.zeros(n_dims)
        losses = np.zeros(n_steps)
        
        best_loss = float("inf")
        failed = False
        
        for step in range(n_steps):
            lr = float(normalized[step])
            diff = w - w_star
            loss = 0.5 * np.sum(curvatures * diff ** 2)
            
            # Add observation noise
            noisy_loss = loss + rng.randn() * noise_scale * (1.0 + loss)
            current_loss = max(0.0, noisy_loss)
            losses[step] = current_loss
            best_loss = min(best_loss, current_loss)
            
            if loss > 1000.0 or not np.isfinite(loss):
                failed = True
                break
                
            grad = curvatures * diff
            grad_noise = rng.randn(n_dims) * 0.1 * np.sqrt(1.0 + np.abs(loss))
            w = w - lr * (grad + grad_noise)
            w = np.clip(w, -50.0, 50.0)
            
        if failed:
            ensemble_losses.append(float('inf'))
            continue
            
        # Final validation
        final_diff = w - w_star
        val_loss = 0.5 * np.sum(curvatures * final_diff ** 2)
        combined = 0.6 * val_loss + 0.4 * best_loss
        ensemble_losses.append(combined)

    if all(l == float('inf') for l in ensemble_losses):
        return float('inf')
        
    avg_loss = float(np.mean([l for l in ensemble_losses if np.isfinite(l)]))
    
    # Penalty for schedules that are essentially constant
    lr_std = np.std(lr_schedule)
    if lr_std < 1e-6:
        avg_loss += 0.5
        
    # Penalty for schedules that increase over time
    if lr_schedule[-1] > lr_schedule[0] * 1.5:
        avg_loss += 0.3
        
    # Parsimony pressure (Heavy Bloat Penalty)
    parsimony = getattr(cfg, 'parsimony_coefficient', 0.05)
    avg_loss += parsimony * tree_size
        
    return float(np.clip(avg_loss, 0.0, 50.0))


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
    """
    from gp.rust_bridge import evaluate_schedule

    # Check fitness cache first
    if hasattr(tree, "fitness") and tree.fitness is not None:
        return tree.fitness

    # Compute LR schedule
    lr_schedule = evaluate_schedule(tree, t_array)

    cfg = get_config()

    if cfg.torch_available and trainer is not None and model_factory is not None:
        # Generate a deterministic seed based on the schedule shape
        schedule_hash = int(hashlib.md5(lr_schedule.tobytes()).hexdigest()[:8], 16)
        base_seed = getattr(cfg, 'model_init_seeds', (42,))[0]
        model_seed = (schedule_hash + base_seed) % (2**32 - 1)
        
        # Real training evaluation
        model = model_factory(init_seed=model_seed)
        loss = trainer.evaluate_schedule(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr_schedule=lr_schedule,
            epochs=epochs,
        )
        
        # Heavily penalize schedules that are essentially flat constants
        # This forces the GP to discover truly dynamic curves!
        lr_std = np.std(lr_schedule)
        if lr_std < 1e-4:
            loss += 0.5  # Constant penalty
            
        # Penalty for schedules that aggressively increase over time (unstable)
        if lr_schedule[-1] > lr_schedule[0] * 1.5:
            loss += 0.3
            
        # Parsimony pressure for real evaluation too
        # Scale parsimony to actively penalize bloat (lambda * complexity_score)
        loss += 0.005 * tree.size()
    else:
        # Synthetic fitness evaluation
        loss = evaluate_synthetic(lr_schedule, tree_size=tree.size())

    # Cache the result
    tree.fitness = loss
    return loss
