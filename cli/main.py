import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live

from src.symbolr.core.bridge import RustEvolutionBridge

app = typer.Typer(
    name="symbolr",
    help="[bold cyan]SymboLR[/bold cyan]: Gradient-Health-Aware Symbolic Schedule Discovery",
    no_args_is_help=False,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    """[bold cyan]SymboLR[/bold cyan] Engine Control. Use [yellow]--help[/yellow] for the manual."""
    if ctx.invoked_subcommand is None:
        from rich.panel import Panel
        from rich.align import Align
        console.print(Align.center(Panel(
            "[bold white]SymboLR[/bold white] — Gradient-Health-Aware Symbolic Schedule Discovery.\n\n"
            "Discovers symbolic LR schedules conditioned on training dynamics.\n\n"
            "Run [bold green]symbolr --help[/bold green] to see all commands.",
            title="[bold cyan]SymboLR Engine[/bold cyan]",
            border_style="magenta",
            expand=False,
            padding=(1, 2),
        )))


@app.command(name="evolve", help="[bold green]Run the Rust-powered Streaming Evolution[/bold green]")
def evolve(
    generations: int = typer.Option(50, "--generations", "-g", help="Number of generations"),
    pop_size:    int = typer.Option(50, "--pop-size",    "-p", help="Population size"),
    time_steps:  int = typer.Option(100, "--time-steps", "-t", help="Surrogate time steps"),
    seed:        int = typer.Option(42,  "--seed",       "-s", help="Random seed"),
    evaluator:   str = typer.Option("cuda_batch", "--evaluator", "-e",
                                    help="Evaluator: cuda_batch | synthetic"),
):
    console.rule("[bold cyan]SymboLR: Streaming Evolution[/bold cyan]")

    if evaluator == "cuda_batch":
        data_path = os.path.join("data", "surrogate_labels.npy")
        os.makedirs("data", exist_ok=True)

        if not os.path.exists(data_path):
            console.print(f"[dim]Generating seeded surrogate dataset at {data_path}...[/dim]")
            rng = np.random.RandomState(seed)
            probe_labels = rng.rand(time_steps).astype(np.float64)
            np.save(data_path, probe_labels)

        console.print(f"[dim]Loading surrogate dataset from {data_path}...[/dim]")
        probe_labels = np.load(data_path, mmap_mode='r')
        if not probe_labels.flags['C_CONTIGUOUS']:
            probe_labels = np.ascontiguousarray(probe_labels)

        from src.symbolr.torch_impl.evaluator import CUDABatchEvaluator
        eval_instance = CUDABatchEvaluator(data_labels=probe_labels)

    elif evaluator == "synthetic":
        from src.symbolr.evaluators.synthetic import SyntheticEvaluator
        eval_instance = SyntheticEvaluator(time_steps=time_steps)

    else:
        console.print(f"[bold red]Unknown evaluator:[/bold red] {evaluator}")
        raise typer.Exit(1)

    bridge = RustEvolutionBridge(
        eval_callback=eval_instance.evaluate,
        max_generations=generations,
        pop_size=pop_size,
        seed=seed,
    )

    table = Table(show_lines=True, title="Evolution Telemetry")
    table.add_column("Gen",       justify="right",  style="cyan")
    table.add_column("Best MSE",  justify="right",  style="green")
    table.add_column("Avg MSE",   justify="right",  style="yellow")
    table.add_column("Archive",   justify="right",  style="blue")
    table.add_column("Time (ms)", justify="right",  style="dim")
    table.add_column("Top Formula", justify="left", style="magenta")

    try:
        with Live(table, console=console, refresh_per_second=10):
            for result in bridge.stream():
                table.add_row(
                    str(result.generation_number),
                    f"{result.best_mse:.6f}",
                    f"{result.average_mse:.6f}",
                    str(result.archive_size),
                    str(result.gen_time_ms),
                    result.top_formula_latex,
                )
    except KeyboardInterrupt:
        console.print("\n[bold red]Evolution interrupted.[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")


@app.command(name="dashboard", help="[bold blue]Launch the React Telemetry Dashboard[/bold blue]")
def dashboard():
    """Start the Vite dev server for the dashboard."""
    console.print("[bold blue]Starting React Dashboard...[/bold blue]")
    os.system("npm run dev --prefix dashboard")


@app.command(name="api", help="[bold magenta]Start the FastAPI Compute Hub[/bold magenta]")
def api_server(
    port: int = typer.Option(8000, "--port", "-p", help="Port"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host"),
):
    """Boot the SSE streaming API."""
    import uvicorn
    console.print(f"[bold magenta]Booting SymboLR API on {host}:{port}...[/bold magenta]")
    uvicorn.run("src.symbolr.api.main:app", host=host, port=port, reload=True)


@app.command(name="benchmark", help="[bold yellow]Run baseline benchmark suite[/bold yellow]")
def benchmark():
    """Compare discovered formulas against all 7 baseline schedules."""
    console.print(
        "[bold yellow]Benchmark pipeline not yet configured.[/bold yellow]\n"
        "Run Phase 4 to build the fair comparison harness."
    )


if __name__ == "__main__":
    app()
