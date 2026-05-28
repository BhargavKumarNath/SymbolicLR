# ADR-002: Python Config Aligned with Rust Core Defaults

**Date**: 2026-05-28
**Status**: Accepted

---

## Context

The Python `config/settings.py` had the following defaults:

```python
crossover_rate: float = 0.45   # Python
mutation_rate:  float = 0.25   # Python
size_bins:      int   = 20     # Python
com_bins:       int   = 15     # Python
```

But the Rust core (`rust_core/src/evolution.rs`, `rust_core/src/archive.rs`) used:

```rust
crossover_rate: 0.20   // Rust
mutation_rate:  0.70   // Rust
size_bins:      30     // Rust
com_bins:       20     // Rust
smoothness_bins: 10    // Rust
```

The Python config was never forwarded to the Rust core — `RustEvolutionBridge` always
used the Rust defaults regardless of what Python set. The Python config was dead weight
that created false expectations.

## Decision

Create `src/symbolr/config.py` with defaults that **exactly match the Rust compiled defaults**.
Add inline comments citing the Rust source file for each parameter:

```python
crossover_rate: float = 0.20   # rust_core/src/evolution.rs default
mutation_rate:  float = 0.70   # rust_core/src/evolution.rs default
size_bins:      int   = 30     # rust_core/src/archive.rs default
com_bins:       int   = 20     # rust_core/src/archive.rs default
smoothness_bins: int  = 10     # rust_core/src/archive.rs default
```

Also removed 40+ Phase 3 config parameters that referenced non-existent subsystems
(novelty search, operator controller, surrogate model, meta-controller).

## Rationale

A config that doesn't match the system it configures is worse than no config —
it actively misleads developers. Alignment + comments make the mismatch impossible.

## Remaining Limitation

The archive bin counts (`size_bins`, `com_bins`, `smoothness_bins`) are compiled
into the Rust binary and currently cannot be overridden from Python at runtime.
This will be addressed in Phase 2 when the Rust API is extended for the new
multi-variable archive axes.

## Consequences

Any code that read the old `config/settings.py` must update to `src/symbolr/config.py`.
Affected files: `evaluators/synthetic.py`, `cli/main.py`, any future evaluators.
