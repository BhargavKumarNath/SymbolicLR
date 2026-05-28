"""
gp/rust_bridge.py — Single source of truth for all SymboLR ↔ Rust FFI.

## Two APIs in one module

### Streaming Bridge (Phase 3) — Thick Rust Core iterator
  RustEvolutionBridge              Wraps symbolr_rust.EvolutionEngine.
  GenerationResult                 Typed telemetry payload (parsed from JSON).
  run_evolution_stream()           Convenience generator for one-liner usage.

## FFI Contract (Phase 3 Ask-and-Tell)

  1. probe_labels numpy array is evaluated natively in Python.
  2. Every generation Python calls `engine.ask()` to get AST formulas.
  3. Python evaluates the formulas and calls `engine.tell(fitnesses)`.
  4. Per-generation telemetry returns as a JSON string from `tell()`.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

# Extension availability guard
try:
    import symbolr_rust as _rust_backend
    RUST_AVAILABLE = True
except ImportError:
    raise RuntimeError(
        "symbolr_rust extension not found. "
        "Run `maturin develop` inside rust_core/ to compile the Rust core."
    )

# Phase 3 — Streaming Evolution Bridge
@dataclass
class GenerationResult:
    """
    Typed telemetry payload for one completed Rust generation.

    Parsed from the JSON string returned by ``EvolutionEngine.tell()``.
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
        """Parse a JSON telemetry string from ``EvolutionEngine.tell()``."""
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
    Python wrapper for the Rust ``EvolutionEngine`` PyO3 struct.

    ## Streaming usage

    .. code-block:: python

        bridge = RustEvolutionBridge(
            eval_callback   = evaluator.evaluate_batch,
            max_generations = 100,
            pop_size        = 50,
            seed            = 42,
        )
        for result in bridge.stream():
            print(f"Gen {result.generation_number}: {result.best_mse:.4f}")

    ## Memory contract

    Python uses `engine.ask()` to retrieve generated formulas without GIL contention.
    Formulas are evaluated in Python (e.g., PyTorch on GPU).
    Fitnesses are sent back via `engine.tell(fitnesses)`.
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

        # Construct the Rust engine. eval_callback is kept in Python
        # and is invoked during the stream() generation loop.
        self.eval_callback = eval_callback
        self._engine = _rust_backend.EvolutionEngine(
            max_generations = max_generations,
            pop_size        = pop_size,
            seed            = seed,
            crossover_rate  = crossover_rate,
            mutation_rate   = mutation_rate,
        )
        self._max_generations = max_generations

        logger.info(
            "EvolutionEngine ready — max_gens=%d  pop=%d  seed=%d",
            max_generations,
            pop_size,
            seed,
        )

    # Streaming iterator

    def stream(self) -> Iterator[GenerationResult]:
        """
        Yield a :class:`GenerationResult` for each completed generation.

        Each iteration asks Rust for generated formulas, evaluates them in Python,
        and tells Rust the fitnesses to update the MAP-Elites Archive.
        Iteration stops automatically after ``max_generations``.
        """
        for _ in range(self._max_generations):
            import time
            start_time = time.time()
            
            formulas = self._engine.ask()
            if not formulas:
                break
                
            fitnesses = self.eval_callback(formulas)
            json_str = self._engine.tell(fitnesses)
            
            result = GenerationResult.from_json(json_str)
            # Python override for gen_time_ms to capture PyTorch evaluation accurately
            result.gen_time_ms = int((time.time() - start_time) * 1000)
            
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
        return json.loads(self._engine.archive_stats())

    def hall_of_fame(self, top_k: int = 10) -> list[dict]:
        """
        Return the top-`top_k` elites from the archive.

        Each dict: ``{ "latex", "prefix", "loss", "complexity", "age" }``.
        """
        return json.loads(self._engine.hall_of_fame(top_k))

    def save_checkpoint(self, path: str) -> None:
        """Saves the current evolutionary state to disk."""
        self._engine.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        """Loads evolutionary state from disk."""
        self._engine = _rust_backend.EvolutionEngine.load_checkpoint(path)

    # Properties

    @property
    def generation(self) -> int:
        """Number of completed generations (0 before first iteration)."""
        return self._engine.generation

    @property
    def archive_size(self) -> int:
        """Current number of occupied niches in the MAP-Elites archive."""
        return self._engine.archive_size

    @property
    def max_generations(self) -> int:
        """Maximum generations this stream will run."""
        return self._engine.max_generations


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
