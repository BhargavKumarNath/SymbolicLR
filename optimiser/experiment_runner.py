"""
optimiser/experiment_runner.py - Multi-seed experiment framework.

Runs N independent evolution seeds and aggregates results across seeds.
Produces concise statistical summaries in JSON and CSV format.

This is validation/research tooling — NOT part of the standard benchmark.py workflow.
Never run automatically. Always invoked explicitly:

    python -m optimiser.experiment_runner --seeds 3 --generations 20 --pop_size 50 --epochs 5

Design constraints:
- Lightweight: no experiment management framework, no large databases
- Concise reports: mean ± std for key metrics
- Easy comparison: seed-level JSON files + aggregate summary
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from typing import List, Optional

from rich.console import Console
from rich.table import Table


def _run_single_seed(
    seed: int,
    generations: int,
    pop_size: int,
    epochs: int,
    workers: int,
    console: Console,
) -> Optional[dict]:
    """
    Run one full evolution with the given seed.
    Returns the DiagnosticsLog.summary() dict, or None on failure.
    """
    try:
        from benchmark import run_evolution
        console.print(f"\n[bold cyan]Seed {seed}[/bold cyan] — Starting evolution...")
        log = run_evolution(
            generations=generations,
            pop_size=pop_size,
            epochs=epochs,
            workers=workers,
            seed=seed,
            console=console,
            wandb_enabled=False,
        )
        if log is not None:
            summary = log.summary()
            summary["seed"] = seed
            return summary
    except Exception as e:
        console.print(f"[red]Seed {seed} failed: {e}[/red]")
    return None


def _aggregate(results: List[dict]) -> dict:
    """Compute mean ± std across seeds for numeric metrics."""
    if not results:
        return {}

    numeric_keys = [
        "initial_best_loss",
        "final_best_loss",
        "improvement",
        "final_archive_size",
        "final_structural_diversity",
        "final_behavioral_diversity",
        "total_run_time_s",
    ]

    aggregate = {"n_seeds": len(results)}

    for key in numeric_keys:
        values = [r[key] for r in results if key in r and isinstance(r[key], (int, float))]
        if values:
            mean_v = sum(values) / len(values)
            variance = sum((v - mean_v) ** 2 for v in values) / max(1, len(values))
            std_v = variance ** 0.5
            aggregate[f"{key}_mean"] = round(mean_v, 4)
            aggregate[f"{key}_std"] = round(std_v, 4)

    # Most common dominant operator
    ops = [r.get("dominant_operator", "") for r in results if r.get("dominant_operator")]
    if ops:
        from collections import Counter
        aggregate["dominant_operator_mode"] = Counter(ops).most_common(1)[0][0]

    return aggregate


def main():
    parser = argparse.ArgumentParser(
        description="SymboLR Multi-Seed Experiment Runner"
    )
    parser.add_argument("--seeds", type=int, default=3, help="Number of independent seeds to run")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed_start", type=int, default=42, help="First seed value")
    parser.add_argument(
        "--output_dir", type=str, default="./results/experiment",
        help="Directory to write aggregate JSON and CSV"
    )
    args = parser.parse_args()

    console = Console()
    console.rule("[bold cyan]SymboLR Multi-Seed Experiment Runner[/bold cyan]")
    console.print(
        f"Seeds: {args.seeds} | Generations: {args.generations} | "
        f"Pop: {args.pop_size} | Epochs: {args.epochs}"
    )

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    results = []

    total_start = time.time()
    for seed in seeds:
        result = _run_single_seed(
            seed=seed,
            generations=args.generations,
            pop_size=args.pop_size,
            epochs=args.epochs,
            workers=args.workers,
            console=console,
        )
        if result:
            results.append(result)

    total_time = round(time.time() - total_start, 1)

    # Aggregate
    aggregate = _aggregate(results)
    aggregate["total_wall_time_s"] = total_time
    aggregate["seeds_run"] = seeds
    aggregate["seeds_successful"] = len(results)

    # Display summary table
    console.print("\n")
    console.rule("[bold green]Experiment Summary[/bold green]")

    table = Table(title=f"Aggregate Results ({len(results)}/{len(seeds)} seeds succeeded)")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Std", justify="right", style="yellow")

    display_keys = [
        ("initial_best_loss", "Initial Best Loss"),
        ("final_best_loss", "Final Best Loss"),
        ("improvement", "Improvement"),
        ("final_archive_size", "Final Archive Size"),
        ("final_structural_diversity", "Structural Diversity"),
        ("final_behavioral_diversity", "Behavioral Diversity"),
        ("total_run_time_s", "Run Time (s)"),
    ]

    for key, label in display_keys:
        mean_val = aggregate.get(f"{key}_mean", "-")
        std_val = aggregate.get(f"{key}_std", "-")
        table.add_row(
            label,
            str(mean_val) if mean_val != "-" else "-",
            f"±{std_val}" if std_val != "-" else "-",
        )

    console.print(table)

    if "dominant_operator_mode" in aggregate:
        console.print(
            f"[bold]Dominant Operator (mode across seeds):[/bold] "
            f"[cyan]{aggregate['dominant_operator_mode']}[/cyan]"
        )

    # Export
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "aggregate.json")
    csv_path = os.path.join(args.output_dir, "per_seed.csv")

    with open(json_path, "w") as f:
        json.dump({"aggregate": aggregate, "per_seed": results}, f, indent=2)

    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    console.print(f"\n[dim]Results saved → {json_path}[/dim]")
    console.print(f"[dim]Per-seed CSV → {csv_path}[/dim]")


if __name__ == "__main__":
    main()
