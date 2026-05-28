# ADR-001: Project Redefined as Gradient-Health-Aware Schedule Discovery

**Date**: 2026-05-28
**Status**: Accepted

---

## Context

The original SymboLR project was a genetic programming experiment for discovering
symbolic learning rate schedules as functions of time only: `lr = f(t)`.

The codebase had accumulated significant technical debt:
- Split-brain architecture (deleted Python GP + new Rust bridge coexisting)
- Fabricated dashboard metrics and hardcoded results
- Multiple broken imports from deleted modules
- 5 duplicate prefix parsers
- Config parameters mismatched between Python and Rust

## Decision

Redefine SymboLR as a **gradient-health-aware symbolic schedule discovery framework**:

- Formulas are functions of training dynamics: `lr = f(t, g, Δl)`
  where `g` = log-normalized gradient norm, `Δl` = tanh-normalized loss slope
- The evolutionary engine searches this richer formula space
- Discovered formulas are interpretable mathematical expressions
- Research framing: do training-state-aware formulas transfer better across architectures?

## Rationale

1. **Scientific novelty**: No published system discovers symbolic, interpretable schedules
   conditioned on gradient health. This is a genuine gap.
2. **Engineering depth**: Extends the Rust AST with new variable types, requires a new
   interactive evaluator, and motivates the statistical validation infrastructure.
3. **Honest claim**: "We discover diverse symbolic schedules and study their transfer
   properties" is defensible. "We beat Adam everywhere" is not.

## Alternatives Considered

- **Keep as f(t) discovery**: Less novel, adds nothing beyond existing GP literature.
- **Neural learned optimizers**: Black boxes, not interpretable, hard to publish.
- **HPO-style schedule search**: Already solved by Optuna/Ray Tune; not novel.

## Consequences

- Phase 2 requires Rust AST changes (new variable types)
- Phase 3 requires a new interactive evaluator (replaces offline surrogate)
- The "WOW FACTOR" is the 2D response surface visualization (t vs g heatmap)
