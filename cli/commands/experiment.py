import typer
from rich.console import Console
import os
import json
import csv
import time
from typing import Optional

from config.settings import get_config
from optimiser.experiment_runner import _run_single_seed, _aggregate

app = typer.Typer(help="Run multi-seed statistical validation")
console = Console()

@app.callback(invoke_without_command=True)
def main(
    seeds: int = typer.Option(3, "--seeds", "-s", help="Number of independent seeds"),
    seed_start: int = typer.Option(42, "--seed-start", help="First seed value"),
    generations: Optional[int] = typer.Option(None, "--generations", "-g", help="Generations"),
    pop_size: Optional[int] = typer.Option(None, "--pop-size", "-p", help="Population size"),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="Epochs per evaluation"),
    output_dir: str = typer.Option("./results/experiment", "--output-dir", "-o", help="Output directory"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
):
    """Run multi-seed experiment and aggregate results."""
    cfg = get_config()

    if config_file:
        console.print(f"[dim]Loading config from: {config_file}[/dim]")
        loaded = cfg.load_yaml(config_file)
        cfg.update(**{k: getattr(loaded, k) for k in loaded.__dataclass_fields__})

    final_gens = generations or cfg.default_generations
    final_pop = pop_size or cfg.default_pop_size
    final_epochs = epochs or cfg.default_epochs
    final_workers = cfg.default_workers

    console.rule(f"[bold cyan]SymboLR Experiment: {seeds} Seeds[/bold cyan]")
    
    seed_list = list(range(seed_start, seed_start + seeds))
    results = []

    total_start = time.time()
    for s in seed_list:
        cfg.update(seed=s)
        res = _run_single_seed(
            seed=s,
            generations=final_gens,
            pop_size=final_pop,
            epochs=final_epochs,
            workers=final_workers,
            console=console,
        )
        if res:
            results.append(res)

    total_time = round(time.time() - total_start, 1)
    
    aggregate = _aggregate(results)
    aggregate["total_wall_time_s"] = total_time
    aggregate["seeds_run"] = seed_list
    aggregate["seeds_successful"] = len(results)

    # Print summary
    from rich.table import Table
    console.print("\n")
    console.rule("[bold green]Experiment Summary[/bold green]")

    table = Table(title=f"Aggregate Results ({len(results)}/{len(seed_list)} seeds succeeded)")
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
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "aggregate.json")
    csv_path = os.path.join(output_dir, "per_seed.csv")

    with open(json_path, "w") as f:
        json.dump({"aggregate": aggregate, "per_seed": results}, f, indent=2)

    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    console.print(f"\n[dim]Results saved -> {json_path}[/dim]")
    console.print(f"[dim]Per-seed CSV -> {csv_path}[/dim]")
