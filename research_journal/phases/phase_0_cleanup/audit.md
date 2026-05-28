# Phase 0 — Deletion Audit

**Date**: 2026-05-28

All deletions are recorded here with rationale.

---

## Deleted Directories

| Path | Reason |
|------|--------|
| `optimiser/` | All 4 files had broken imports from the deleted Python GP system (`gp.tree`, `gp.fitness`, `benchmark`). None were callable. |
| `graphify-out/` | Tool artifacts from a previous knowledge-graph analysis. Not project code. |
| `fonts/` (root) | Duplicate of `dashboard/public/fonts/` (same SF Pro font files). Root-level fonts serve no purpose. |
| `results/` | 5 CSV + JSON files from a deleted Python GP run (seed 42–46). Values are incompatible with the current Rust bridge. |
| `config/` | Replaced by `src/symbolr/config.py` with a clean, aligned dataclass. |
| `baselines/` (root) | Moved to `src/symbolr/baselines/` for proper package namespacing. |
| `src/symbolr/engine/` | Replaced by `src/symbolr/core/` (bridge + evaluator ABC) and `src/symbolr/evaluators/` (SyntheticEvaluator). |
| `src/symbolr/cli/` | Replaced by `src/symbolr/artifacts/` (split into 3 focused files). |

## Deleted Files

| File | Reason |
|------|--------|
| `dashboard/public/results.json` | "Golden run" from the deleted Python GP system. Was loaded by `DiagnosticsPage.jsx` as a fabricated benchmark. |

## Moved Files (not deleted)

| Original | New Location | Changes |
|----------|-------------|---------|
| `src/symbolr/engine/bridge.py` | `src/symbolr/core/bridge.py` | No logic changes |
| `src/symbolr/engine/evaluator.py` | `src/symbolr/core/evaluator.py` | Added `is_deterministic` property |
| `src/symbolr/engine/synthetic.py` | `src/symbolr/evaluators/synthetic.py` | Updated imports |
| `src/symbolr/cli/artifacts.py` | Split into `artifacts/prefix_parser.py`, `pytorch_export.py`, `latex_export.py` | Consolidated 5 duplicate parsers into 1 |
| `baselines/schedules.py` | `src/symbolr/baselines/schedules.py` | Removed `benchmark_all_baselines()` (broken import) |
| `config/settings.py` | `src/symbolr/config.py` | Simplified, removed Phase 3 features that don't exist, aligned with Rust defaults |
| `experiment_mnist.py` (root) | `experiments/mnist_probe.py` | Updated imports |
| `mnist_example.py` (root) | `experiments/mnist_example.py` | Updated imports |
