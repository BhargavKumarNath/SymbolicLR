"""
src/symbolr/core/bridge.py — Single source of truth for SymboLR ↔ Rust FFI.

Wraps symbolr_rust.EvolutionEngine via the Ask-and-Tell pattern:
  1. Python calls engine.ask() to receive a batch of prefix formula strings.
  2. Python evaluates each formula (any BaseEvaluator subclass).
  3. Python calls engine.tell(fitnesses) to update the MAP-Elites archive.
  4. Rust returns per-generation telemetry as a JSON string.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

try:
    import symbolr_rust as _rust_backend
    RUST_AVAILABLE = True
except ImportError:
    raise RuntimeError(
        "symbolr_rust extension not found. "
        "Run `maturin develop` inside rust_core/ to compile the Rust core."
    )


@dataclass
class GenerationResult:
    """
    Typed telemetry payload for one completed Rust generation.

    All fields mirror the Rust GenerationStats struct so the Python layer
    never needs to understand Rust internals.
    """

    generation_number: int
    best_mse:          float
    average_mse:       float
    top_formula_latex: str
    top_formula_prefix: str
    archive_size:      int
    new_entries:       int
    gen_time_ms:       int

    @classmethod
    def from_json(cls, json_str: str) -> "GenerationResult":
        data: dict = json.loads(json_str)
        return cls(
            generation_number=int(data.get("generation_number", 0)),
            best_mse=float(data["best_mse"]) if data.get("best_mse") is not None else float("inf"),
            average_mse=float(data["average_mse"]) if data.get("average_mse") is not None else float("inf"),
            top_formula_latex=data.get("top_formula_latex", ""),
            top_formula_prefix=data.get("top_formula_prefix", ""),
            archive_size=int(data.get("archive_size", 0)),
            new_entries=int(data.get("new_entries", 0)),
            gen_time_ms=int(data.get("gen_time_ms", 0)),
        )

    def to_dict(self) -> dict:
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
    Python wrapper for the Rust EvolutionEngine PyO3 struct.

    Usage:
        bridge = RustEvolutionBridge(
            eval_callback   = evaluator.evaluate,
            max_generations = 100,
            pop_size        = 50,
            seed            = 42,
        )
        for result in bridge.stream():
            print(f"Gen {result.generation_number}: {result.best_mse:.4f}")
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
        self.eval_callback = eval_callback
        self._engine = _rust_backend.EvolutionEngine(
            max_generations=max_generations,
            pop_size=pop_size,
            seed=seed,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
        )
        self._max_generations = max_generations

        logger.info(
            "EvolutionEngine ready — max_gens=%d  pop=%d  seed=%d",
            max_generations, pop_size, seed,
        )

    def stream(self) -> Iterator[GenerationResult]:
        """
        Yield a GenerationResult for each completed generation.

        Each iteration asks Rust for generated formulas, evaluates them in Python,
        and tells Rust the fitnesses to update the MAP-Elites archive.
        """
        for _ in range(self._max_generations):
            t0 = time.time()

            formulas = self._engine.ask()
            if not formulas:
                break

            fitnesses = self.eval_callback(formulas)
            json_str = self._engine.tell(fitnesses)

            result = GenerationResult.from_json(json_str)
            result.gen_time_ms = int((time.time() - t0) * 1000)

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

    def archive_stats(self) -> dict:
        """Snapshot of current archive statistics."""
        return json.loads(self._engine.archive_stats())

    def hall_of_fame(self, top_k: int = 10) -> list[dict]:
        """Top-k elites from the archive."""
        return json.loads(self._engine.hall_of_fame(top_k))

    def save_checkpoint(self, path: str) -> None:
        self._engine.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        self._engine = _rust_backend.EvolutionEngine.load_checkpoint(path)

    @property
    def generation(self) -> int:
        return self._engine.generation

    @property
    def archive_size(self) -> int:
        return self._engine.archive_size

    @property
    def max_generations(self) -> int:
        return self._engine.max_generations


def run_evolution_stream(
    eval_callback: callable,
    max_generations: int = 50,
    **kwargs,
) -> Iterator[GenerationResult]:
    """Convenience one-liner for streaming evolution."""
    bridge = RustEvolutionBridge(
        eval_callback=eval_callback,
        max_generations=max_generations,
        **kwargs,
    )
    yield from bridge.stream()
