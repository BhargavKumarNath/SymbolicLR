# SymboLR System Architecture

**Last updated**: 2026-05-28 (Phase 0)

---

## Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Entry Points                          │
│  cli/main.py (Typer CLI)    src/symbolr/api/main.py (SSE)  │
└───────────────────────┬─────────────────────┬───────────────┘
                        │                     │
               ┌────────▼─────────┐  ┌────────▼─────────┐
               │  RustEvolution   │  │  BaseEvaluator   │
               │  Bridge          │  │  (ABC)           │
               │  core/bridge.py  │  │  core/evaluator  │
               └────────┬─────────┘  └────────┬─────────┘
                        │                     │
               ┌────────▼─────────────────────▼─────────┐
               │             symbolr_rust                │
               │   (PyO3 extension, compiled from        │
               │    rust_core/ via maturin develop)      │
               │                                         │
               │  EvolutionEngine.ask()  → formulas      │
               │  EvolutionEngine.tell() ← fitnesses     │
               │  MAP-Elites archive (6000 niches)       │
               └─────────────────────────────────────────┘
```

## Module Map

| Module | Responsibility |
|--------|---------------|
| `src/symbolr/core/bridge.py` | Rust FFI wrapper, Ask-and-Tell pattern, GenerationResult |
| `src/symbolr/core/evaluator.py` | BaseEvaluator ABC |
| `src/symbolr/evaluators/synthetic.py` | Fast deterministic evaluator (no GPU required) |
| `src/symbolr/torch_impl/evaluator.py` | Surrogate-based GPU evaluator (CUDABatchEvaluator) |
| `src/symbolr/baselines/schedules.py` | 7 hand-designed schedule implementations |
| `src/symbolr/artifacts/prefix_parser.py` | Canonical prefix formula parser (single source of truth) |
| `src/symbolr/artifacts/pytorch_export.py` | Prefix → PyTorch LambdaLR code |
| `src/symbolr/artifacts/latex_export.py` | Prefix → LaTeX |
| `src/symbolr/config.py` | Single config dataclass, aligned with Rust defaults |
| `src/symbolr/api/main.py` | FastAPI SSE streaming server |
| `cli/main.py` | Typer CLI (evolve, api, dashboard, benchmark) |
| `rust_core/` | Rust evolutionary engine (MAP-Elites, GP operators, AST) |
| `experiments/` | Research experiment scripts |
| `research_journal/` | This logging system |

## Data Flow (Ask-and-Tell Pattern)

```
for generation in 1..max_generations:
    formulas: list[str]  ←  engine.ask()       # Rust generates population
    fitnesses: list[float]  ←  evaluator.evaluate(formulas)  # Python evaluates
    telemetry: str  ←  engine.tell(fitnesses)  # Rust updates archive → JSON
    yield GenerationResult.from_json(telemetry)
```

## Rust Core Details

- **Language**: Rust (stable), compiled via maturin/PyO3
- **Archive**: MAP-Elites, 30×20×10 = 6000 niches
- **Behavioral axes**: formula size, output center-of-mass, output smoothness
- **Operators**: subtree crossover, subtree mutation, hoist mutation, point mutation, constant perturbation
- **PRNG**: SmallRng seeded from Python-provided seed
- **Caching**: DashMap-based formula result cache (thread-safe)

## Phase 2 Target Changes

Phase 2 will extend the Rust AST to support multi-variable formulas `f(t, g, Δl)`.
The behavioral axes will change to: formula size, gradient sensitivity, loss-slope sensitivity.
See `architecture_decisions/ADR-003_ast_variable_extension.md` (to be written in Phase 2).
