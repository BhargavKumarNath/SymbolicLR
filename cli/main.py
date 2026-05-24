import os
import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live

from gp.rust_bridge import run_evolution_stream

app = typer.Typer(
    name="symbolr",
    help="SymboLR: Lightweight Research CLI for Symbolic Learning Rate Discovery",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

@app.command(name="evolve", help="Run the Rust-powered Streaming Evolution")
def evolve(
    generations: int = typer.Option(50, "--generations", "-g", help="Number of generations"),
    pop_size: int = typer.Option(50, "--pop-size", "-p", help="Population size"),
    time_steps: int = typer.Option(100, "--time-steps", "-t", help="Number of evaluation time steps"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
):
    console.rule("[bold cyan]SymboLR: Rust Core Streaming CLI[/bold cyan]")
    
    # 1. Ingest Surrogate Dataset
    data_path = os.path.join("data", "surrogate_labels.npy")
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
        
    if not os.path.exists(data_path):
        console.print(f"[dim]Generating mock surrogate dataset at {data_path}...[/dim]")
        dummy = np.random.rand(time_steps).astype(np.float64)
        np.save(data_path, dummy)

    console.print(f"[dim]Memory-mapping surrogate dataset from {data_path}...[/dim]")
    probe_labels = np.load(data_path, mmap_mode='r')
    if not probe_labels.flags['C_CONTIGUOUS']:
        probe_labels = np.ascontiguousarray(probe_labels)
    
    console.print("[bold magenta]Initializing Rust Evolution Stream...[/bold magenta]")
    
    table = Table(show_lines=True, title="Evolution Telemetry")
    table.add_column("Gen", justify="right", style="cyan")
    table.add_column("Best MSE", justify="right", style="green")
    table.add_column("Avg MSE", justify="right", style="yellow")
    table.add_column("Archive", justify="right", style="blue")
    table.add_column("Time (ms)", justify="right", style="dim")
    table.add_column("Top Formula", justify="left", style="magenta")

    try:
        from models.probe import CUDABatchEvaluator
        evaluator = CUDABatchEvaluator(data_labels=probe_labels)
        
        with Live(table, console=console, refresh_per_second=10):
            for result in run_evolution_stream(
                eval_callback=evaluator.evaluate_batch,
                max_generations=generations,
                pop_size=pop_size,
                seed=seed,
            ):
                table.add_row(
                    str(result.generation_number),
                    f"{result.best_mse:.6f}",
                    f"{result.average_mse:.6f}",
                    str(result.archive_size),
                    str(result.gen_time_ms),
                    result.top_formula_latex,
                )
    except KeyboardInterrupt:
        console.print("\n[bold red]Evolution interrupted by user.[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    app()
