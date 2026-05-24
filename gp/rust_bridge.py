"""
gp/rust_bridge.py — Single source of truth for all SymboLR ↔ Rust FFI.

## Two APIs in one module

### Legacy (Phase 1) — schedule evaluation helpers
  evaluate_schedule()              Single tree over t_array via evaluate_fast.
  evaluate_batch_schedules()       Batch trees via evaluate_batch (rayon).
  evaluate_schedule_from_prefix()  Prefix-string evaluation (viz layer).

### Streaming Bridge (Phase 3) — Thick Rust Core iterator
  RustEvolutionBridge              Wraps symbolr_rust.EvolutionStream.
  GenerationResult                 Typed telemetry payload (parsed from JSON).
  run_evolution_stream()           Convenience generator for one-liner usage.

## FFI Contract (Phase 3)

  1. probe_labels numpy array is handed to Rust **once** in the constructor.
     Rust copies it to a Vec<f64> and releases the numpy reference immediately.
  2. Every __next__ call executes one full parallel generation in Rust memory
     with zero GIL acquisition on the hot path.
  3. Per-generation telemetry returns as a JSON string — one allocation per
     generation, not one per formula.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from typing import Iterator, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gp.tree import Node

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Extension availability guard
# ─────────────────────────────────────────────────────────────────────────────

try:
    import symbolr_rust as _rust_backend
    RUST_AVAILABLE = True
except ImportError:
    _rust_backend = None  # type: ignore[assignment]
    RUST_AVAILABLE = False
    logger.warning(
        "symbolr_rust extension not found. "
        "Run `maturin develop` (or `pip install -e .`) inside rust_core/. "
        "Legacy Python fallbacks will be used for schedule evaluation; "
        "EvolutionStream will be unavailable."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Legacy Phase-1 helpers — preserved for zero regression
# ─────────────────────────────────────────────────────────────────────────────


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
            from gp.tree import Node  # noqa: PLC0415
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
    from gp.tree import Node          # noqa: PLC0415
    from gp.operators import OPERATORS  # noqa: PLC0415

    if pos >= len(tokens):
        return Node(0.1), pos + 1

    token = tokens[pos]

    if token in OPERATORS:
        op = OPERATORS[token]
        children = []
        current_pos = pos + 1
        for _ in range(op.arity):
            child, current_pos = _parse_prefix(tokens, current_pos)
            children.append(child)
        return Node(token, children), current_pos

    if token == "t":
        return Node("t"), pos + 1

    try:
        return Node(float(token)), pos + 1
    except ValueError:
        return Node(0.1), pos + 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Streaming Evolution Bridge
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GenerationResult:
    """
    Typed telemetry payload for one completed Rust generation.

    Parsed from the JSON string yielded by ``EvolutionStream.__next__``.
    All fields mirror the Rust ``GenerationStats`` struct exactly so the
    Python layer never needs to understand the Rust internals.
    """

    generation_number: int
    best_mse: float
    average_mse: float
    top_formula_latex: str
    top_formula_prefix: str
    archive_size: int
    new_entries: int
    gen_time_ms: int

    @classmethod
    def from_json(cls, json_str: str) -> "GenerationResult":
        """Parse a JSON telemetry string from ``EvolutionStream.__next__``."""
        data: dict = json.loads(json_str)
        return cls(
            generation_number=int(data.get("generation_number", 0)),
            # Rust sends `null` for non-finite values; map to Python inf.
            best_mse=float(data["best_mse"]) if data.get("best_mse") is not None else float("inf"),
            average_mse=float(data["average_mse"]) if data.get("average_mse") is not None else float("inf"),
            top_formula_latex=data.get("top_formula_latex", ""),
            top_formula_prefix=data.get("top_formula_prefix", ""),
            archive_size=int(data.get("archive_size", 0)),
            new_entries=int(data.get("new_entries", 0)),
            gen_time_ms=int(data.get("gen_time_ms", 0)),
        )

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation or API responses."""
        return {
            "generation_number": self.generation_number,
            "best_mse":          self.best_mse,
            "average_mse":       self.average_mse,
            "top_formula_latex": self.top_formula_latex,
            "top_formula_prefix": self.top_formula_prefix,
            "archive_size":      self.archive_size,
            "new_entries":       self.new_entries,
            "gen_time_ms":       self.gen_time_ms,
        }


class RustEvolutionBridge:
    """
    Python wrapper for the Rust ``EvolutionStream`` PyO3 iterator.

    ## Streaming usage

    .. code-block:: python

        bridge = RustEvolutionBridge(
            probe_labels    = y_train,   # numpy float64 array — handed to Rust once
            max_generations = 100,
            pop_size        = 50,
            seed            = 42,
        )
        for result in bridge.stream():
            print(f"Gen {result.generation_number}: {result.best_mse:.4f}")

    ## Memory contract

    ``probe_labels`` is copied into Rust-owned ``Vec<f64>`` exactly once in
    the constructor.  All ``__next__`` calls on the underlying
    ``EvolutionStream`` run on Rust memory with zero FFI overhead — no numpy
    reference counting, no GIL on the hot evolutionary path.
    """

    def __init__(
        self,
        eval_callback: callable,
        max_generations: int = 50,
        pop_size: int = 50,
        seed: int = 42,
        crossover_rate: float = 0.20,
        mutation_rate: float = 0.70,
    ) -> None:
        if not RUST_AVAILABLE or _rust_backend is None:
            raise RuntimeError(
                "symbolr_rust is not installed. "
                "Run `maturin develop` inside rust_core/ then retry."
            )

        # Construct the Rust iterator. eval_callback is passed into Rust
        # as a PyObject and is invoked during the generation loop.
        self._stream = _rust_backend.EvolutionStream(
            eval_callback   = eval_callback,
            max_generations = max_generations,
            pop_size        = pop_size,
            seed            = seed,
            crossover_rate  = crossover_rate,
            mutation_rate   = mutation_rate,
        )
        self._max_generations = max_generations

        logger.info(
            "EvolutionStream ready — max_gens=%d  pop=%d  seed=%d",
            max_generations,
            pop_size,
            seed,
        )

    # Streaming iterator

    def stream(self) -> Iterator[GenerationResult]:
        """
        Yield a :class:`GenerationResult` for each completed generation.

        Each iteration executes exactly one parallel generation inside Rust
        (``rayon::par_iter`` across all CPU cores) and yields the parsed
        telemetry.  Iteration stops automatically after ``max_generations``.
        """
        for json_str in self._stream:
            result = GenerationResult.from_json(json_str)
            logger.info(
                "Gen %d | best=%.5f  mean=%.5f  archive=%d  +%d  %dms",
                result.generation_number,
                result.best_mse,
                result.average_mse,
                result.archive_size,
                result.new_entries,
                result.gen_time_ms,
            )
            yield result

    # Archive inspection

    def archive_stats(self) -> dict:
        """
        Return a snapshot of the current archive statistics as a Python dict.

        Fields: ``occupied_niches``, ``max_niches``, ``occupancy_pct``,
        ``total_attempts``, ``total_additions``, ``best_loss``,
        ``median_loss``, ``mean_elite_age``.
        """
        return json.loads(self._stream.archive_stats())

    def hall_of_fame(self, top_k: int = 10) -> list[dict]:
        """
        Return the top-`top_k` elites from the archive.

        Each dict: ``{ "latex", "prefix", "loss", "complexity", "age" }``.
        """
        return json.loads(self._stream.hall_of_fame(top_k))

    # Properties

    @property
    def generation(self) -> int:
        """Number of completed generations (0 before first iteration)."""
        return self._stream.generation

    @property
    def archive_size(self) -> int:
        """Current number of occupied niches in the MAP-Elites archive."""
        return self._stream.archive_size

    @property
    def max_generations(self) -> int:
        """Maximum generations this stream will run."""
        return self._stream.max_generations


# Convenience generator
def run_evolution_stream(
    eval_callback: callable,
    max_generations: int = 50,
    **kwargs,
) -> Iterator[GenerationResult]:
    """
    One-liner streaming evolution.

    .. code-block:: python

        for result in run_evolution_stream(eval_callback, max_generations=100):
            api_push(result.to_dict())

    All keyword arguments are forwarded to :class:`RustEvolutionBridge`.
    """
    bridge = RustEvolutionBridge(
        eval_callback=eval_callback,
        max_generations=max_generations,
        **kwargs,
    )
    yield from bridge.stream()
