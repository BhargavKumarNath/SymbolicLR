"""
src/symbolr/baselines/benchmark.py — Phase 4.

Canonical comparison harness: evaluates any discovered formula against all
baseline LR schedules under identical conditions.

The shared fitness metric is the synthetic quadratic landscape from
SyntheticEvaluator, but with an explicit landscape seed so that each
(formula, baseline) pair is evaluated on the SAME problem instance per trial.
This enables paired statistical tests with correct Type-I error control.

Statistical output per formula-vs-baseline comparison:
  • delta_mean      — formula_mean - baseline_mean (negative = formula wins)
  • win_rate        — fraction of seeds where formula beats baseline
  • bootstrap_ci    — 95% CI on delta_mean via percentile bootstrap (n=1000)
  • wilcoxon_p      — Wilcoxon signed-rank p-value (requires scipy; None if absent)
  • is_significant  — p < 0.05 (interpretive flag, not a claim about the formula)

Scientific integrity note:
  With n_seeds=5 the minimum achievable Wilcoxon p-value is ~0.0625 (below the
  0.05 threshold). Use n_seeds >= 7 to achieve significance at α=0.05. Results
  are reported regardless — the win_rate and CI are the primary outputs.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.symbolr.baselines.schedules import BASELINE_SCHEDULES
from src.symbolr.config import get_config

logger = logging.getLogger(__name__)

# Optional scipy
try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Optional Rust backend
try:
    import symbolr_rust as _rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


# Simulation kernel
def _simulate_seeded(lr_schedule: np.ndarray, landscape_seed: int) -> float:
    """
    Evaluate an LR schedule on a seeded synthetic quadratic landscape.

    Mirrors SyntheticEvaluator._simulate() exactly, but the landscape seed is
    explicit rather than derived from the formula hash. Using the same seed
    for a formula and a baseline ensures a paired comparison: both face the
    same curvature matrix, starting point, and noise realisation.

    Args:
        lr_schedule:    LR values at each step in [0, 1] time range.
        landscape_seed: Integer seed controlling the quadratic landscape.

    Returns:
        Fitness (lower = better). Returns inf for invalid schedules.
    """
    cfg = get_config()

    if not np.all(np.isfinite(lr_schedule)) or np.all(lr_schedule < 1e-7) or np.any(lr_schedule > 10.0):
        return float("inf")

    n_steps = len(lr_schedule)
    if n_steps == 0:
        return float("inf")

    raw_mean = np.mean(np.abs(lr_schedule))
    if raw_mean > 1e-7:
        normalized = np.clip(lr_schedule * (0.01 / raw_mean), 1e-5, 0.1)
    else:
        normalized = np.full(n_steps, 0.01)

    n_evals     = cfg.synth_n_evaluations
    n_dims      = cfg.synth_n_dims
    noise_scale = cfg.synth_noise_scale

    curvatures = np.pad(
        np.array([0.5, 1.0, 2.0, 4.0, 8.0])[:n_dims],
        (0, max(0, n_dims - 5)),
        constant_values=1.0,
    )

    ensemble_losses = []
    for eval_idx in range(n_evals):
        rng = np.random.RandomState((landscape_seed + eval_idx) % (2 ** 32 - 1))
        w = rng.randn(n_dims) * 2.0
        w_star = np.zeros(n_dims)
        best_loss = float("inf")
        failed = False

        for step in range(n_steps):
            lr   = float(normalized[step])
            diff = w - w_star
            loss = 0.5 * np.sum(curvatures * diff ** 2)

            if loss > 1000.0 or not np.isfinite(loss):
                failed = True
                break

            best_loss = min(best_loss, max(0.0, loss + rng.randn() * noise_scale * (1.0 + loss)))
            grad = curvatures * diff + rng.randn(n_dims) * 0.1 * np.sqrt(1.0 + abs(loss))
            w    = np.clip(w - lr * grad, -50.0, 50.0)

        if failed:
            ensemble_losses.append(float("inf"))
            continue

        diff     = w - w_star
        val_loss = 0.5 * np.sum(curvatures * diff ** 2)
        ensemble_losses.append(0.6 * val_loss + 0.4 * best_loss)

    finite = [l for l in ensemble_losses if np.isfinite(l)]
    if not finite:
        return float("inf")

    avg = float(np.mean(finite))
    if np.std(lr_schedule) < 1e-6:
        avg += 0.5
    if lr_schedule[-1] > lr_schedule[0] * 1.5:
        avg += 0.3

    return float(np.clip(avg, 0.0, 50.0))


def _bootstrap_ci(
    diffs: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """
    Percentile bootstrap 95% CI on the mean of diffs.

    Args:
        diffs:        Paired differences (formula - baseline) per trial seed.
        n_bootstrap:  Number of bootstrap resamples.
        alpha:        Significance level (default 0.05 → 95% CI).
        rng_seed:     Bootstrap RNG seed for reproducibility.

    Returns:
        (lower, upper) confidence interval on the mean difference.
    """
    rng   = np.random.RandomState(rng_seed)
    means = [
        float(np.mean(rng.choice(diffs, size=len(diffs), replace=True)))
        for _ in range(n_bootstrap)
    ]
    lower = float(np.percentile(means, 100.0 * alpha / 2.0))
    upper = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def _wilcoxon_p(diffs: np.ndarray) -> Optional[float]:
    """
    Wilcoxon signed-rank p-value for H0: median(diffs) = 0.

    Returns None if scipy is unavailable or if all differences are zero
    (degenerate case where the test cannot be applied).
    """
    if not SCIPY_AVAILABLE:
        return None
    finite_diffs = diffs[np.isfinite(diffs)]
    if len(finite_diffs) < 2 or np.all(finite_diffs == 0):
        return None
    try:
        result = _scipy_wilcoxon(finite_diffs, alternative="two-sided", zero_method="wilcox")
        return float(result.pvalue)
    except Exception:
        return None


# Result types
@dataclass
class TrialResult:
    """
    Fitness measurements for one candidate (formula or baseline) across K seeds.
    """
    name:      str
    fitnesses: list[float]

    @property
    def mean(self) -> float:
        finite = [f for f in self.fitnesses if math.isfinite(f)]
        return float(np.mean(finite)) if finite else float("inf")

    @property
    def std(self) -> float:
        finite = [f for f in self.fitnesses if math.isfinite(f)]
        return float(np.std(finite)) if len(finite) > 1 else 0.0

    @property
    def n_valid(self) -> int:
        return sum(1 for f in self.fitnesses if math.isfinite(f))

    def to_dict(self) -> dict:
        return {
            "name":      self.name,
            "fitnesses": self.fitnesses,
            "mean":      self.mean,
            "std":       self.std,
            "n_valid":   self.n_valid,
        }


@dataclass
class ComparisonResult:
    """
    Statistical comparison of a formula against one baseline.

    delta_mean < 0  →  formula is better (lower loss).
    win_rate   > 0.5 →  formula won on the majority of seeds.
    """
    baseline_name:  str
    formula_mean:   float
    baseline_mean:  float
    delta_mean:     float         # formula_mean - baseline_mean (negative = formula wins)
    delta_std:      float         # std of per-seed differences
    win_rate:       float         # fraction of seeds formula beats baseline (in [0, 1])
    ci_lower:       float         # 95% bootstrap CI lower bound on delta_mean
    ci_upper:       float         # 95% bootstrap CI upper bound on delta_mean
    wilcoxon_p:     Optional[float] = None  # None if scipy unavailable or n too small
    n_seeds:        int = 0

    @property
    def formula_wins(self) -> bool:
        """True if formula has lower mean fitness than this baseline."""
        return self.delta_mean < 0

    @property
    def is_significant(self) -> bool:
        """True if Wilcoxon p < 0.05. Never True without scipy or with n < 7."""
        return self.wilcoxon_p is not None and self.wilcoxon_p < 0.05

    @property
    def ci_excludes_zero(self) -> bool:
        """True if the 95% CI for delta_mean does not contain zero."""
        return self.ci_upper < 0 or self.ci_lower > 0

    def to_dict(self) -> dict:
        return {
            "baseline_name":  self.baseline_name,
            "formula_mean":   self.formula_mean,
            "baseline_mean":  self.baseline_mean,
            "delta_mean":     self.delta_mean,
            "delta_std":      self.delta_std,
            "win_rate":       self.win_rate,
            "ci_lower":       self.ci_lower,
            "ci_upper":       self.ci_upper,
            "wilcoxon_p":     self.wilcoxon_p,
            "n_seeds":        self.n_seeds,
            "formula_wins":   self.formula_wins,
            "is_significant": self.is_significant,
        }


@dataclass
class BenchmarkResult:
    """
    Full benchmark output for one formula compared against all baselines.

    The rank is the formula's position when all candidates (formula + all
    baselines) are sorted by mean fitness (1 = best). A rank of 1 means the
    formula outperforms all baselines on average; a rank of 8 (in the default
    7-baseline suite) means it is the worst performer.
    """
    formula:         str
    n_seeds:         int
    formula_trial:   TrialResult
    baseline_trials: dict[str, TrialResult]
    comparisons:     dict[str, ComparisonResult]
    rank:            int   # 1 = best among (formula + all baselines)
    n_candidates:    int   # len(baselines) + 1

    @property
    def best_baseline_name(self) -> str:
        return min(self.baseline_trials, key=lambda n: self.baseline_trials[n].mean)

    @property
    def best_baseline_fitness(self) -> float:
        return self.baseline_trials[self.best_baseline_name].mean

    @property
    def beats_best_baseline(self) -> bool:
        return self.formula_trial.mean < self.best_baseline_fitness

    @property
    def n_baselines_beaten(self) -> int:
        return sum(1 for c in self.comparisons.values() if c.formula_wins)

    def to_dict(self) -> dict:
        return {
            "formula":           self.formula,
            "n_seeds":           self.n_seeds,
            "formula_trial":     self.formula_trial.to_dict(),
            "baseline_trials":   {k: v.to_dict() for k, v in self.baseline_trials.items()},
            "comparisons":       {k: v.to_dict() for k, v in self.comparisons.items()},
            "rank":              self.rank,
            "n_candidates":      self.n_candidates,
            "best_baseline_name": self.best_baseline_name,
            "beats_best_baseline": self.beats_best_baseline,
            "n_baselines_beaten": self.n_baselines_beaten,
        }

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Benchmark results saved to %s", path)


# BenchmarkSuite
class BenchmarkSuite:
    """
    Fair comparison harness for discovered SymboLR formulas vs baseline schedules.

    Both the formula under test and each baseline are evaluated through the same
    fitness function (_simulate_seeded) with identical landscape seeds per trial.
    This paired design controls for landscape variability, improving the power
    of the statistical tests.

    Usage:
        suite   = BenchmarkSuite(n_seeds=5, time_steps=100)
        result  = suite.compare("cos * 3.14159 t")
        print(result.rank, result.n_baselines_beaten)
        result.save_json("benchmark_out.json")

    Args:
        time_steps:   Number of steps in the LR schedule (matched to evolution).
        n_seeds:      Number of repeated evaluations per candidate. Default 5
                      (min Wilcoxon p ≈ 0.0625; use ≥ 7 for α=0.05 power).
        n_bootstrap:  Bootstrap resamples for CI estimation. Default 1000.
        base_seed:    Starting landscape seed. Seeds used are base_seed, ...,
                      base_seed + n_seeds - 1.
    """

    def __init__(
        self,
        time_steps:  int = 100,
        n_seeds:     int = 5,
        n_bootstrap: int = 1000,
        base_seed:   int = 42,
    ) -> None:
        self.time_steps  = time_steps
        self.n_seeds     = n_seeds
        self.n_bootstrap = n_bootstrap
        self.base_seed   = base_seed
        self._t_array    = np.linspace(0.0, 1.0, time_steps, dtype=np.float64)

    # Public API
    def compare(
        self,
        formula: str,
        baseline_names: Optional[list[str]] = None,
    ) -> BenchmarkResult:
        """
        Evaluate a formula against baseline schedules and return statistics.

        Args:
            formula:        Prefix-notation formula string (e.g. "cos * 3.14159 t").
            baseline_names: Subset of baselines to include. All 7 if None.

        Returns:
            BenchmarkResult with per-baseline comparisons and an overall rank.
        """
        if baseline_names is None:
            baseline_names = list(BASELINE_SCHEDULES.keys())

        unknown = [n for n in baseline_names if n not in BASELINE_SCHEDULES]
        if unknown:
            raise ValueError(f"Unknown baseline(s): {unknown}. Valid: {list(BASELINE_SCHEDULES)}")

        logger.info(
            "Benchmarking formula '%s' against %d baselines over %d seeds",
            formula, len(baseline_names), self.n_seeds,
        )

        # Evaluate all candidates
        formula_fitnesses   = self._eval_formula(formula)
        baseline_fitnesses  = {name: self._eval_baseline(name) for name in baseline_names}

        formula_trial   = TrialResult(formula, formula_fitnesses)
        baseline_trials = {n: TrialResult(n, f) for n, f in baseline_fitnesses.items()}

        # Per-baseline statistical comparisons
        comparisons: dict[str, ComparisonResult] = {}
        for name, bt in baseline_trials.items():
            diffs = np.array([
                f - b
                for f, b in zip(formula_trial.fitnesses, bt.fitnesses)
                if math.isfinite(f) and math.isfinite(b)
            ])
            win_rate = (
                float(np.mean(np.array(formula_trial.fitnesses) < np.array(bt.fitnesses)))
                if formula_trial.n_valid > 0
                else 0.0
            )
            ci_lower, ci_upper = _bootstrap_ci(diffs, n_bootstrap=self.n_bootstrap) if len(diffs) > 1 else (0.0, 0.0)

            comparisons[name] = ComparisonResult(
                baseline_name  = name,
                formula_mean   = formula_trial.mean,
                baseline_mean  = bt.mean,
                delta_mean     = formula_trial.mean - bt.mean,
                delta_std      = float(np.std(diffs)) if len(diffs) > 1 else 0.0,
                win_rate       = win_rate,
                ci_lower       = ci_lower,
                ci_upper       = ci_upper,
                wilcoxon_p     = _wilcoxon_p(diffs),
                n_seeds        = len(diffs),
            )

        # Overall rank
        all_means = [formula_trial.mean] + [bt.mean for bt in baseline_trials.values()]
        rank = sorted(all_means).index(formula_trial.mean) + 1

        return BenchmarkResult(
            formula         = formula,
            n_seeds         = self.n_seeds,
            formula_trial   = formula_trial,
            baseline_trials = baseline_trials,
            comparisons     = comparisons,
            rank            = rank,
            n_candidates    = 1 + len(baseline_names),
        )

    # Internal evaluation
    def _eval_formula(self, formula: str) -> list[float]:
        """
        Evaluate a prefix formula across n_seeds landscape seeds.

        Uses symbolr_rust.evaluate_batch for formula → LR array conversion,
        then _simulate_seeded for fitness. Returns list of n_seeds floats.
        """
        if not RUST_AVAILABLE:
            logger.warning("symbolr_rust unavailable; using prefix_parser fallback for benchmarking")
            return self._eval_formula_python(formula)

        fitnesses: list[float] = []
        for k in range(self.n_seeds):
            seed = self.base_seed + k
            try:
                schedules = _rust.evaluate_batch([formula], self._t_array)
                lr_array  = np.array(schedules[0], dtype=np.float64)
                fitnesses.append(_simulate_seeded(lr_array, seed))
            except Exception as exc:
                logger.debug("Formula '%s' failed on seed %d: %s", formula, seed, exc)
                fitnesses.append(float("inf"))
        return fitnesses

    def _eval_formula_python(self, formula: str) -> list[float]:
        """Python fallback (no Rust): evaluate formula via prefix_parser."""
        from src.symbolr.artifacts.prefix_parser import evaluate_formula

        fitnesses: list[float] = []
        for k in range(self.n_seeds):
            seed = self.base_seed + k
            try:
                lr_array = np.array(
                    [evaluate_formula(formula, t=float(t)) for t in self._t_array],
                    dtype=np.float64,
                )
                fitnesses.append(_simulate_seeded(lr_array, seed))
            except Exception:
                fitnesses.append(float("inf"))
        return fitnesses

    def _eval_baseline(self, baseline_name: str) -> list[float]:
        """
        Evaluate a baseline schedule across n_seeds landscape seeds.

        The LR array is computed once (it is deterministic), then the same
        array is paired with each landscape seed. This means the landscape
        seed is the only source of variation, which is correct for paired tests.
        """
        baseline_fn = BASELINE_SCHEDULES[baseline_name]
        lr_array    = baseline_fn(self._t_array).astype(np.float64)

        fitnesses: list[float] = []
        for k in range(self.n_seeds):
            seed = self.base_seed + k
            fitnesses.append(_simulate_seeded(lr_array, seed))
        return fitnesses
