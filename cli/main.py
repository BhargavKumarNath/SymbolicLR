import os
import sys
# Add project root to path so 'src' module can be found when script is run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live

from src.symbolr.engine.bridge import RustEvolutionBridge

app = typer.Typer(
    name="symbolr",
    help="[bold cyan]SymboLR[/bold cyan]: Lightweight Research CLI for Symbolic Learning Rate Discovery",
    no_args_is_help=False,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    """
    [bold cyan]SymboLR[/bold cyan] Engine Control. Use [yellow]--help[/yellow] for the manual.
    """
    if ctx.invoked_subcommand is None:
        from rich.panel import Panel
        from rich.align import Align
        
        welcome_text = (
            "[bold white]Welcome to the SymboLR Core CLI.[/bold white]\n\n"
            "This high-performance engine uses a hybrid PyO3 / Rust architecture to dynamically\n"
            "discover mathematical learning rate schedules.\n\n"
            "Run [bold green]symbolr --help[/bold green] to see all available commands."
        )
        console.print(Align.center(Panel(
            welcome_text,
            title="[bold cyan]🚀 SymboLR Engine[/bold cyan]",
            border_style="magenta",
            expand=False,
            padding=(1, 2)
        )))

@app.command(name="evolve", help="[bold green]Run the Rust-powered Streaming Evolution[/bold green]")
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
        from src.symbolr.torch_impl.evaluator import CUDABatchEvaluator
        evaluator = CUDABatchEvaluator(data_labels=probe_labels)
        
        bridge = RustEvolutionBridge(
            eval_callback=evaluator.evaluate,
            max_generations=generations,
            pop_size=pop_size,
            seed=seed,
        )
        
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
        console.print("\n[bold red]Evolution interrupted by user.[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")

@app.command(name="dashboard", help="[bold blue]Launch the React Telemetry Dashboard[/bold blue]")
def dashboard():
    """Start the Vite development server for the FANG-grade UI."""
    console.print("[bold blue]Starting React Dashboard...[/bold blue] (Make sure you are in the dashboard directory)")
    os.system("npm run dev --prefix dashboard")

@app.command(name="api", help="[bold magenta]Start the FastAPI Compute Hub[/bold magenta]")
def api_server(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run the API on"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host address")
):
    """Boot up the API bridge to serve streaming requests from the Dashboard."""
    import uvicorn
    console.print(f"[bold magenta]Booting SymboLR Streaming Compute Hub on {host}:{port}...[/bold magenta]")
    uvicorn.run("src.symbolr.api.main:app", host=host, port=port, reload=True)

@app.command(name="benchmark", help="[bold yellow]Run full offline benchmark suite[/bold yellow]")
def benchmark():
    """Execute the full offline benchmark without UI streaming."""
    console.print("[bold yellow]Offline Benchmark Mode Not Yet Configured in CLI.[/bold yellow]")

if __name__ == "__main__":
    app()
