# SymboLR Research Master Log

**Project**: Gradient-Health-Aware Symbolic Schedule Discovery
**Started**: 2026-05-28
**Status**: Phase 6 — Complete (main engine done)

---

## Project Identity

SymboLR is a framework for discovering symbolic learning rate schedules conditioned
on live training dynamics. Unlike all published schedules (`lr = f(t)`), SymboLR
discovers formulas of the form `lr = f(t, g, Δl)` where `g` is the gradient norm
and `Δl` is the recent loss slope.

The evolutionary engine is implemented in Rust (MAP-Elites + genetic programming)
and exposed to Python via PyO3. Evaluators are pluggable Python classes.

---

## Phase Timeline

| Phase | Name | Status | Dates |
|-------|------|--------|-------|
| 0 | Repository Cleanup & Architecture Reset | ✅ Complete | 2026-05-28 |
| 1 | Research Infrastructure & Reproducibility | Pending | — |
| 2 | Rust Core Extension (Multi-Variable AST) | ✅ Complete | 2026-05-30 |
| 3 | Gradient-Aware Interactive Evaluator | ✅ Complete | 2026-05-30 |
| 4 | Canonical Fitness Pipeline & Benchmarking | ✅ Complete | 2026-05-31 |
| 5 | Dashboard Overhaul | Pending | — |
| 6 | Research Validation & Ablation Studies | ✅ Complete | 2026-05-31 |
| 7 | Distribution, Integration & Polish | Pending | — |
| F | Showcase & Interview Preparation | Pending | — |

---

## Phase 0 Summary (2026-05-28)

**Objectives**: Remove dead code, fix critical bugs, establish clean architecture.

**Completed**:
- Deleted 6 dead directories: `optimiser/`, `graphify-out/`, `fonts/`, `results/`,
  `config/`, `baselines/`
- Deleted fabricated dashboard data: `dashboard/public/results.json`
- Restructured source: `engine/` → `core/`, `evaluators/`, `artifacts/`
- Fixed 4 critical bugs (see `phases/phase_0_cleanup/bugs_found.md`)
- Removed duplicate `CUDABatchEvaluator` class from `torch_impl/models.py`
- Config aligned with Rust core defaults (was mismatched by `crossover_rate: 0.45 vs 0.20`)
- Created `research_journal/` logging infrastructure
- Written `tests/test_smoke.py`

**New folder structure**:
```
src/symbolr/
  core/          ← RustEvolutionBridge, BaseEvaluator (was: engine/)
  evaluators/    ← SyntheticEvaluator (was: engine/synthetic.py)
  baselines/     ← 7 schedule implementations (was: root baselines/)
  artifacts/     ← prefix_parser, pytorch_export, latex_export (was: cli/artifacts.py × 5)
  torch_impl/    ← CUDABatchEvaluator, ProbeTrainer (unchanged)
  api/           ← FastAPI SSE server (bugs fixed)
  config.py      ← clean config dataclass (was: config/settings.py)
experiments/     ← mnist_example.py (was at project root)
research_journal/ ← this directory
```

---

---

## Phase 3 Summary (2026-05-30)

**Objectives**: Build GradientAwareEvaluator — a batched, interactive training
loop that evaluates formulas by actually training models, feeding real gradient
norms and loss slopes to gradient-aware formulas.

**Completed**:
- `src/symbolr/evaluators/gradient_aware.py` — full evaluator implementation:
  - `_NormStats` — warmup-fitted z-score normalization for g (log-space) and dl (tanh)
  - `_build_proxy_dataset()` — synthetic 10-class Gaussian classification, no downloads
  - `_ProxyMLP` — 2-layer MLP (64→128→10), BatchNorm-free for vmap compatibility
  - `_VmapBatchedTrainer` — N models in one GPU forward+backward via torch.func.vmap
  - `_SequentialTrainer` — Python for-loop fallback for CPU / PyTorch < 2.0
  - `GradientAwareEvaluator` — pluggable `BaseEvaluator` subclass with 3-phase protocol
- Added Phase 3 config fields to `SymboLRConfig` (5 new fields)
- Added `GradientAwareEvaluator` to `evaluators/__init__.py`
- `tests/test_phase3_gradient_aware.py` — 22 tests, all passing
- ADR-004 documenting the proxy task and vmap decisions
- Phase 3 design doc at `phases/phase_3_evaluator/design.md`

**Test results**:
- `cargo test`: 54/54
- `pytest tests/`: 50/50 (17 smoke + 11 phase2 + 22 phase3)

**Key design properties**:
- Deterministic: formula-hash seeded models + fixed batch sequence
- Gradient-reactive formulas now measurably different from time-only ones
- 3-phase protocol: warmup (10%) → phase1 (45%) → phase2 (45%)
- Successive halving: top ceil(N/2) survive from phase1 to phase2
- vmap backend: N models in one GPU pass (single kernel per step)
- CPU throughput: ~3 formulas/sec; GPU target ≥10 formulas/sec

---

---

## Phase 4 Summary (2026-05-31)

**Objectives**: Build a canonical fitness pipeline that compares discovered
formulas against 7 baseline LR schedules with paired statistical tests and
no hardcoded comparison values.

**Completed**:
- `src/symbolr/baselines/benchmark.py` — canonical harness:
  - `_simulate_seeded(lr_schedule, landscape_seed)` — shared fitness kernel
    with explicit seed (enables paired comparison between formula and baselines)
  - `_bootstrap_ci(diffs)` — 1000-resample percentile bootstrap 95% CI
  - `_wilcoxon_p(diffs)` — Wilcoxon signed-rank p-value (graceful scipy fallback)
  - `TrialResult`, `ComparisonResult`, `BenchmarkResult` — typed output hierarchy
  - `BenchmarkSuite.compare(formula)` — runs full paired comparison across n_seeds
- Updated `cli/main.py` `benchmark` command: full Rich table with rank, win rate,
  CIs, p-values, and sigma-warning for underpowered comparisons
- `src/symbolr/baselines/__init__.py` — exports `BenchmarkSuite`, `BenchmarkResult`
- `tests/test_phase4_benchmark.py` — 31 tests, all passing
- ADR-005 documenting the paired design rationale
- Phase 4 design doc at `phases/phase_4_benchmarking/design.md`

**Test results**:
- `cargo test`: 54/54
- `pytest tests/`: 83/83 (17 smoke + 11 phase2 + 22 phase3 + 31 phase4 + 2 api)

**Statistical design**:
- Paired landscape seeds per trial: formula and each baseline see the same
  quadratic landscape in each trial seed → valid paired Wilcoxon
- Bootstrap CI (primary): always available, no scipy dependency
- Wilcoxon p-value (secondary): requires scipy; documented n_seeds power caveat
- Win rate: interpretive fraction of seeds where formula beats baseline
- `save_json()`: full trial-level data exportable for downstream analysis

---

---

## Phase 6 Summary (2026-05-31)

**Objectives**: Close the gradient-signal gap in the evolution loop and build
the ablation framework to validate the core scientific claim.

**Completed**:
- **Engine wire-up** — `symbolr evolve --evaluator gradient_aware` now live.
  `GradientAwareEvaluator` is fully connected to `RustEvolutionBridge`. The
  `∇-Sens` column in the evolve table tracks archive gradient sensitivity in real time.
  `GenerationResult.gradient_sensitivity_mean` now parsed from `tell()` JSON and
  included in `to_dict()`.

- `src/symbolr/evaluators/filtered.py` — `TokenFilteredEvaluator`:
  wraps any evaluator and returns `inf` for formulas with forbidden tokens.
  Implements t-only / t+g / t+g+dl ablation from Python, no Rust changes.

- `src/symbolr/core/ablation.py` — `AblationRunner` + typed result hierarchy:
  `AblationConfig` (canonical configs), `AblationRun`, `AblationResult`.
  Tracks best formula per generation via stream (not post-stream calls, which
  return empty after the engine finalizes).

- `experiments/ablation_terminal_set.py` — standalone experiment script.
  Runs 3 configs, prints summary table, saves JSON, interprets result.
  Usage: `python experiments/ablation_terminal_set.py --evaluator gradient_aware`

- `tests/test_phase6_ablation.py` — 24 tests, all passing.
- ADR-006 documenting the Python-filter ablation design.
- Bug found and fixed: `bridge.hall_of_fame()` and `bridge.archive_stats()` return
  empty after stream exhaustion. Fixed by tracking best formula in-stream.

**Test results**:
- `cargo test`: 54/54
- `pytest tests/`: 107/107 (17 smoke + 11 phase2 + 22 phase3 + 31 phase4 + 2 api + 24 phase6)

**Signal path: now complete**
```
Rust GP engine (ask)
    → TokenFilteredEvaluator (ablation gate)
        → GradientAwareEvaluator (live g, dl)
            → formula gets real fitness advantage if gradient-adaptive
    → Rust archive (tell, gradient-sensitivity niches)
        → BenchmarkSuite (statistical comparison vs 7 baselines)
```

---

## Open Research Questions

See `OPEN_QUESTIONS.md` for the current list.
