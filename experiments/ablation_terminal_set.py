"""
experiments/ablation_terminal_set.py — Phase 6 ablation study.

Three-way terminal-set ablation for gradient-health-aware symbolic schedules.

Compares evolution under:
  t_only  — lr = f(t)          [forbidden: g, dl]
  t_g     — lr = f(t, g)       [forbidden: dl]
  t_g_dl  — lr = f(t, g, Δl)  [no filter]

Each configuration runs for --generations generations with the same seed.
The best formula from each run is benchmarked against 7 baseline schedules
using BenchmarkSuite (paired Wilcoxon + bootstrap CI).

Scientific honesty:
  The SyntheticEvaluator is the default because it is fast and self-contained.
  For a full gradient-signal ablation, pass --evaluator gradient_aware. This
  is the scientifically correct mode — gradient-aware formulas can only gain
  a real fitness advantage when evaluated with live (g, dl) signals.

Usage:
    cd c:/Project/symbolr
    python experiments/ablation_terminal_set.py
    python experiments/ablation_terminal_set.py --generations 50 --evaluator gradient_aware
    python experiments/ablation_terminal_set.py --output research_journal/experiments/ablation_001.json
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Allow running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ablation")


def _build_evaluator(args):
    if args.evaluator == "gradient_aware":
        from src.symbolr.evaluators.gradient_aware import GradientAwareEvaluator
        ev = GradientAwareEvaluator(n_steps=args.n_steps, seed=args.seed)
        logger.info("Using GradientAwareEvaluator: n_steps=%d  device=%s", args.n_steps, ev._device)
        return ev
    else:
        from src.symbolr.evaluators.synthetic import SyntheticEvaluator
        ev = SyntheticEvaluator(time_steps=args.n_steps)
        logger.info("Using SyntheticEvaluator: time_steps=%d  (g=0, dl=0)", args.n_steps)
        return ev


def main():
    parser = argparse.ArgumentParser(description="SymboLR terminal-set ablation study")
    parser.add_argument("--generations",  type=int,   default=20,         help="Generations per config")
    parser.add_argument("--pop-size",     type=int,   default=50,         help="Population size")
    parser.add_argument("--n-steps",      type=int,   default=100,        help="Evaluator steps")
    parser.add_argument("--seed",         type=int,   default=42,         help="Random seed")
    parser.add_argument("--evaluator",    type=str,   default="synthetic",
                        choices=["synthetic", "gradient_aware"],          help="Evaluator type")
    parser.add_argument("--benchmark-seeds", type=int, default=5,         help="BenchmarkSuite seeds")
    parser.add_argument("--no-benchmark", action="store_true",            help="Skip BenchmarkSuite step")
    parser.add_argument("--configs",      type=str,   default="",
                        help="Comma-separated subset: t_only,t_g,t_g_dl  (default: all)")
    parser.add_argument("--output",       type=str,   default="",         help="Save results to JSON")
    args = parser.parse_args()

    from src.symbolr.core.ablation import AblationRunner, ABLATION_CONFIGS

    # Filter configs if requested
    if args.configs:
        requested = set(args.configs.split(","))
        selected  = [c for c in ABLATION_CONFIGS if c.name in requested]
        if not selected:
            logger.error("No valid configs in %r. Valid: %s", args.configs,
                         ", ".join(c.name for c in ABLATION_CONFIGS))
            sys.exit(1)
    else:
        selected = ABLATION_CONFIGS

    base_ev = _build_evaluator(args)

    runner = AblationRunner(
        base_evaluator  = base_ev,
        max_generations = args.generations,
        pop_size        = args.pop_size,
        seed            = args.seed,
        benchmark_seeds = args.benchmark_seeds,
        benchmark_steps = args.n_steps,
        run_benchmark   = not args.no_benchmark,
    )

    logger.info(
        "Starting ablation: %d configs × %d generations × pop=%d  seed=%d  evaluator=%s",
        len(selected), args.generations, args.pop_size, args.seed, args.evaluator,
    )

    result = runner.run_all(configs=selected)

    # Print summary table
    print("\n" + "=" * 72)
    print(f"ABLATION SUMMARY  (total: {result.total_elapsed_sec:.1f}s)")
    print("=" * 72)
    header = f"{'Config':<28}  {'Fitness':>9}  {'Archive':>7}  {'∇-Sens':>6}  {'Rank':>5}  {'Beaten':>6}"
    print(header)
    print("-" * 72)

    for row in result.summary_rows():
        rank_str   = str(row["bench_rank"])   if row["bench_rank"]   is not None else "—"
        beaten_str = str(row["bench_beaten"]) if row["bench_beaten"] is not None else "—"
        print(
            f"{row['config']:<28}  "
            f"{row['best_fitness']:>9.5f}  "
            f"{row['archive_size']:>7}  "
            f"{row['grad_sens']:>6.3f}  "
            f"{rank_str:>5}  "
            f"{beaten_str:>6}"
        )

    print("=" * 72)

    # Print per-config best formulas
    print("\nBest formulas:")
    for name, run in result.runs.items():
        print(f"  {run.config_label}")
        print(f"    prefix: {run.best_formula_prefix}")
        print(f"    latex:  {run.best_formula_latex}")
        print()

    # Interpret results
    ranks = {
        name: run.benchmark_rank
        for name, run in result.runs.items()
        if run.benchmark_rank is not None
    }
    if len(ranks) >= 2:
        best_config = min(ranks, key=lambda k: ranks[k])
        print(f"Result: '{result.runs[best_config].config_label}' produced "
              f"the highest-ranking formula (rank #{ranks[best_config]}).")
        if best_config == "t_g_dl":
            print("  ✓ Gradient conditioning improved formula quality.")
        elif best_config == "t_only":
            print("  · Time-only formulas were sufficient on this proxy task.")
        else:
            print("  · Gradient norms helped; loss-slope did not add further benefit.")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        result.save_json(args.output)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
