# Open Research Questions

Questions are added when they arise and answered/closed when resolved.

---

## Active

**Q1** (added 2026-05-28): Do gradient-aware symbolic formulas (`f(t, g, Δl)`) generalize
better across architectures than time-only formulas (`f(t)`)? This is the central
hypothesis. Answered by Phase 6 cross-architecture transfer study.

**Q2** (added 2026-05-28): What fraction of archive elites use `VarG` or `VarDL` nodes
after 100 generations of search? Does the search preferentially discover gradient-reactive
formulas or time-only formulas? Answered by Phase 6 ablation study.

**Q3** (added 2026-05-28): What is the minimum-complexity formula (fewest AST nodes) that
achieves within 5% of the best discovered fitness? Is it interpretable? Answered by
Phase 6 MDL experiment.

**Q4** (added 2026-05-28): Is loss slope (`Δl`) a useful input signal beyond gradient norm
(`g`)? Or is it redundant? Addressed by the 3-condition terminal set ablation.

---

## Closed

*(None yet — Phase 0 is not an experimental phase.)*
