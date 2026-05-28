# Phase 0 — Bugs Found and Fixed

**Date**: 2026-05-28
**Status**: All 5 bugs fixed.

---

## BUG-001: self.t Mutation in CUDABatchEvaluator

**File**: `src/symbolr/torch_impl/evaluator.py`
**Severity**: High (silent correctness bug)

**Description**: The successive halving implementation temporarily overwrote
`self.t` with a half-length tensor during Phase 1 evaluation:

```python
# OLD (buggy)
original_t = self.t
self.t = t_half               # Mutates shared instance state
y_pred = self._parse_and_evaluate(formula)
self.t = original_t           # Restores — but breaks if exception occurs
```

If an exception was raised inside `_parse_and_evaluate`, `self.t` would be left
in its mutated state for all subsequent evaluations. This produced silently wrong
fitness values for any formula evaluated after a failed formula in Phase 1.

**Fix**: Refactored `_parse_and_evaluate(prefix_str, t=None)` to accept an
explicit `t` parameter. The const cache (`_get_const`) is bypassed when `t is not self.t`
to avoid shape mismatches. No mutation of instance state.

---

## BUG-002: Hardcoded seed=42 in API

**File**: `src/symbolr/api/main.py`
**Severity**: High (reproducibility violation)

**Description**: The `seed` parameter was ignored in the `stream_evolve` endpoint.
The `RustEvolutionBridge` constructor always received `seed=42` regardless of
what the client requested:

```python
bridge = RustEvolutionBridge(
    ...
    seed=42   # ← request parameter ignored
)
```

This meant all API runs were identical regardless of requested seed, making
it impossible to run multiple independent trials from the dashboard.

**Fix**: Added `seed: int = 42` as a query parameter to `stream_evolve` and
forwarded it to `RustEvolutionBridge`.

---

## BUG-003: Unseeded Surrogate Label Generation

**Files**: `cli/main.py`, `src/symbolr/api/main.py`
**Severity**: High (reproducibility violation)

**Description**: When `data/surrogate_labels.npy` did not exist, it was generated
using `np.random.rand()` with no prior seed call:

```python
dummy = np.random.rand(time_steps).astype(np.float64)
```

This meant the surrogate target changed every time it was regenerated (e.g., on a
fresh clone, after deleting the data directory). Different targets produce different
fitness landscapes, making it impossible to compare runs across machines.

**Fix**: Replaced with `np.random.RandomState(seed).rand(...)` using the run's
seed parameter. The surrogate target is now deterministic given the same seed.

---

## BUG-004: Broken Import in baselines/schedules.py

**File**: `baselines/schedules.py` (now `src/symbolr/baselines/schedules.py`)
**Severity**: Medium (import failure on call)

**Description**: The `benchmark_all_baselines()` function imported from the
deleted Python GP module:

```python
from gp.fitness import evaluate_synthetic   # ModuleNotFoundError
```

This function would crash immediately when called, but because it was never
called in the main execution paths, the bug was invisible at import time.

**Fix**: Deleted `benchmark_all_baselines()` entirely. Replaced with `evaluate_all_baselines()`
which returns LR arrays without evaluation (evaluation belongs in the benchmark pipeline,
Phase 4).

---

## BUG-005: Duplicate CUDABatchEvaluator Class

**File**: `src/symbolr/torch_impl/models.py`
**Severity**: Medium (dead code, incorrect API)

**Description**: `models.py` contained a second `CUDABatchEvaluator` class that
shadowed the canonical one in `evaluator.py`. Key differences:

- Used `evaluate_batch()` method (not the `evaluate()` interface from `BaseEvaluator`)
- Only supported `'x'` token (not `'t'` alias)
- No successive halving
- Never imported or called anywhere

This was dead code that would cause confusion for any developer navigating the codebase.

**Fix**: Removed lines 187–304 from `models.py`. The file now contains only
`FastConvNet`, `create_compiled_model`, and `ProbeTrainer`.

---

## Pre-existing Issues NOT Fixed in Phase 0

These are tracked for future phases:

- **Config mismatch**: Python config had `crossover_rate=0.45, mutation_rate=0.25` while
  Rust used `0.20, 0.70`. Fixed by the new `src/symbolr/config.py`. However, the Rust
  archive bin counts (`size_bins=30, com_bins=20, smoothness_bins=10`) are hardcoded in
  Rust and not yet passable from Python — to be resolved in Phase 2.

- **Dashboard pages with dead references**: `EvolutionPage.jsx` still references deleted
  `gp/*.py` files as code examples. To be fixed in Phase 5.
