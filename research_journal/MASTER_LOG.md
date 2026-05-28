# SymboLR Research Master Log

**Project**: Gradient-Health-Aware Symbolic Schedule Discovery
**Started**: 2026-05-28
**Status**: Phase 0 — Complete

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
| 2 | Rust Core Extension (Multi-Variable AST) | Pending | — |
| 3 | Gradient-Aware Interactive Evaluator | Pending | — |
| 4 | Canonical Fitness Pipeline & Benchmarking | Pending | — |
| 5 | Dashboard Overhaul | Pending | — |
| 6 | Research Validation & Ablation Studies | Pending | — |
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

## Open Research Questions

See `OPEN_QUESTIONS.md` for the current list.
