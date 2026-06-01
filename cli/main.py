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


@app.command(name="benchmark", help="[bold yellow]Compare a formula against 7 baseline schedules[/bold yellow]")
def benchmark(
    formula:   str = typer.Option(
        "cos * 3.14159 t",
        "--formula", "-f",
        help="Formula in prefix notation (e.g. 'cos * 3.14159 t')",
    ),
    seeds:     int = typer.Option(5,   "--seeds",      "-n", help="Evaluation seeds (≥7 for Wilcoxon power)"),
    steps:     int = typer.Option(100, "--time-steps", "-t", help="LR schedule length"),
    base_seed: int = typer.Option(42,  "--seed",       "-s", help="Base landscape seed"),
    output:    str = typer.Option("",  "--output",     "-o", help="Save results to JSON file"),
    baselines: str = typer.Option("",  "--baselines",  "-b", help="Comma-separated subset of baselines (default: all)"),
):
    """
    Compare a discovered formula against all 7 canonical baseline LR schedules.

    Both the formula and each baseline are evaluated under identical conditions:
    same synthetic quadratic landscape, same seeds, same metric. No hardcoded
    comparison values — all numbers are computed fresh.
    """
    from src.symbolr.baselines.benchmark import BenchmarkSuite

    console.rule("[bold yellow]SymboLR Benchmark Suite[/bold yellow]")
    console.print(f"Formula : [bold magenta]{formula}[/bold magenta]")
    console.print(f"Seeds   : {seeds}  |  Steps: {steps}  |  Base seed: {base_seed}\n")

    baseline_list = [b.strip() for b in baselines.split(",") if b.strip()] or None

    suite = BenchmarkSuite(time_steps=steps, n_seeds=seeds, base_seed=base_seed)

    with console.status("[dim]Running benchmark...[/dim]"):
        try:
            result = suite.compare(formula, baseline_names=baseline_list)
        except Exception as exc:
            console.print(f"[bold red]Benchmark error:[/bold red] {exc}")
            raise typer.Exit(1)

    # ── Results table ─────────────────────────────────────────────────────────
    from rich.table import Table
    from rich.text import Text

    table = Table(
        title=f"Benchmark: [magenta]{formula}[/magenta]",
        show_lines=True,
        header_style="bold",
    )
    table.add_column("Candidate",    style="cyan",   min_width=20)
    table.add_column("Mean Fitness", justify="right", style="green")
    table.add_column("Std",          justify="right", style="dim")
    table.add_column("Δ vs Formula", justify="right")
    table.add_column("Win Rate",     justify="right")
    table.add_column("95% CI",       justify="right", style="dim")
    table.add_column("p-value",      justify="right")

    ft = result.formula_trial
    table.add_row(
        Text(f"★ Formula", style="bold magenta"),
        f"{ft.mean:.5f}",
        f"±{ft.std:.5f}",
        "—",
        "—",
        "—",
        "—",
    )

    for name, cmp in sorted(result.comparisons.items(), key=lambda x: x[1].baseline_mean):
        bt  = result.baseline_trials[name]
        win = "✓" if cmp.formula_wins else "✗"
        delta_style = "green" if cmp.formula_wins else "red"
        sig_marker  = "*" if cmp.is_significant else (" ?" if cmp.wilcoxon_p is not None else "")
        p_str = f"{cmp.wilcoxon_p:.3f}{sig_marker}" if cmp.wilcoxon_p is not None else "n/a"
        table.add_row(
            name,
            f"{bt.mean:.5f}",
            f"±{bt.std:.5f}",
            Text(f"{win} {cmp.delta_mean:+.5f}", style=delta_style),
            f"{cmp.win_rate:.0%}",
            f"[{cmp.ci_lower:+.4f}, {cmp.ci_upper:+.4f}]",
            p_str,
        )

    console.print(table)

    # ── Summary ───────────────────────────────────────────────────────────────
    rank_str   = f"[bold]#{result.rank}[/bold] of {result.n_candidates}"
    beaten_str = f"{result.n_baselines_beaten}/{len(result.baseline_trials)} baselines"
    console.print(f"\nRanking: {rank_str}  |  Beats: {beaten_str}")

    if seeds < 7 and not SCIPY_AVAILABLE_IN_CLI:
        console.print(
            "[dim]Note: use --seeds ≥ 7 for Wilcoxon significance (min p ≈ 0.031).[/dim]"
        )
    elif seeds < 7:
        console.print(
            "[dim]Note: with --seeds < 7 the Wilcoxon test cannot reach α=0.05 "
            "(min p ≈ 0.063). Increase --seeds for statistical power.[/dim]"
        )

    if output:
        result.save_json(output)
        console.print(f"[dim]Results saved to {output}[/dim]")


# runtime check — used for the note above
try:
    from scipy.stats import wilcoxon as _  # noqa: F401
    SCIPY_AVAILABLE_IN_CLI = True
except ImportError:
    SCIPY_AVAILABLE_IN_CLI = False


if __name__ == "__main__":
    app()
