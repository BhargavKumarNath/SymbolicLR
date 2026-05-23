"""
gp/rust_bridge.py - Single source of truth for Rust/mock schedule evaluation.

When the Rust extension is available, delegates to symbolr_rust.evaluate_fast.
Otherwise, falls back to the Python Node.evaluate() method
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from gp.tree import Node

# Try to load Rust extension once
try:
    import symbolr_rust as _rust_backend
    RUST_AVAILABLE = True
except ImportError:
    _rust_backend = None
    RUST_AVAILABLE = False


def evaluate_schedule(tree: "Node", t_array: np.ndarray) -> np.ndarray:
    """
    Evaluate a GP tree over a time array and return the LR schedule.

    Args:
        tree: A SymboLR AST Node representing a formula.
        t_array: 1D numpy array of normalized time steps in [0, 1].

    Returns:
        np.ndarray of learning rates, clamped to [1e-7, 10.0].
    """
    try:
        if RUST_AVAILABLE and _rust_backend is not None:
            result = _rust_backend.evaluate_fast(tree.to_prefix(), t_array)
        else:
            result = tree.evaluate(t_array)

        # Sanitise output
        result = np.asarray(result, dtype=np.float64)
        result = np.nan_to_num(result, nan=1e-4, posinf=1.0, neginf=1e-7)
        result = np.clip(result, 1e-7, 10.0)
        return result
    except Exception as e:
        warnings.warn(f"Schedule evaluation failed: {e}", RuntimeWarning, stacklevel=2)
        return np.full_like(t_array, 1e-3)

def evaluate_batch_schedules(trees: list["Node"], t_array: np.ndarray) -> np.ndarray:
    """
    Evaluates a batch of GP trees in a single PyO3 FFI call.
    Leverages Rust's Rayon parallelization and DashMap caching.
    Returns a 2D numpy array of shape (len(trees), len(t_array)).
    """
    try:
        if RUST_AVAILABLE and _rust_backend is not None and hasattr(_rust_backend, "evaluate_batch"):
            prefixes = [tree.to_prefix() for tree in trees]
            result = _rust_backend.evaluate_batch(prefixes, t_array)
            result = np.asarray(result, dtype=np.float64)
            result = np.nan_to_num(result, nan=1e-4, posinf=1.0, neginf=1e-7)
            result = np.clip(result, 1e-7, 10.0)
            return result
        else:
            return np.vstack([evaluate_schedule(t, t_array) for t in trees])
    except Exception as e:
        warnings.warn(f"Batch schedule evaluation failed: {e}", RuntimeWarning, stacklevel=2)
        return np.full((len(trees), len(t_array)), 1e-3)

def evaluate_schedule_from_prefix(prefix: str, t_array: np.ndarray) -> np.ndarray:
    """
    Evaluate a prefix-notation expression string directly.
    Used by visualization layers that don't have Node references.
    
    Falls back to a gentle decay curve if anything fails.
    """
    try:
        if RUST_AVAILABLE and _rust_backend is not None:
            result = _rust_backend.evaluate_fast(prefix, t_array)
        else:
            # Parse prefix back into a tree for Python evaluation
            from gp.tree import Node
            from gp.operators import OPERATORS
            tokens = prefix.split()
            tree, _ = _parse_prefix(tokens, 0)
            result = tree.evaluate(t_array)

        result = np.asarray(result, dtype=np.float64)
        result = np.nan_to_num(result, nan=1e-4, posinf=1.0, neginf=1e-7)
        result = np.clip(result, 1e-7, 10.0)
        return result
    except Exception as e:
        warnings.warn(f"Schedule prefix evaluation failed: {e}", RuntimeWarning, stacklevel=2)
        return np.full_like(t_array, 1e-3)


def _parse_prefix(tokens: list, pos: int) -> tuple:
    """Minimal prefix parser for fallback evaluation."""
    from gp.tree import Node
    from gp.operators import OPERATORS

    if pos >= len(tokens):
        return Node(0.1), pos + 1

    token = tokens[pos]

    # Check if it's an operator
    if token in OPERATORS:
        op = OPERATORS[token]
        children = []
        current_pos = pos + 1
        for _ in range(op.arity):
            child, current_pos = _parse_prefix(tokens, current_pos)
            children.append(child)
        return Node(token, children), current_pos

    # Check if it's the variable
    if token == "t":
        return Node("t"), pos + 1

    # Otherwise it's a float constant
    try:
        return Node(float(token)), pos + 1
    except ValueError:
        return Node(0.1), pos + 1
