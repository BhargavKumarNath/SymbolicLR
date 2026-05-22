import typer
from rich.console import Console
from rich.table import Table
import json
import os
import numpy as np

from optimiser.plotting import export_schedules_plot

app = typer.Typer(help="Replay a saved Hall-of-Fame formula or diagnostics file")
console = Console()

@app.callback(invoke_without_command=True)
def main(
    diagnostics_file: str = typer.Argument(..., help="Path to run.json file"),
    plot: bool = typer.Option(False, "--plot", help="Export static PNG plots of the formulas"),
):
    """Replay Hall of Fame formulas from a saved diagnostics file."""
    if not os.path.exists(diagnostics_file):
        console.print(f"[bold red]Error:[/bold red] File {diagnostics_file} not found.")
        raise typer.Exit(1)
        
    with open(diagnostics_file, "r") as f:
        data = json.load(f)
        
    hof = data.get("hall_of_fame", [])
    if not hof:
        console.print("[bold red]No Hall of Fame data found in the diagnostics file.[/bold red]")
        raise typer.Exit(1)
        
    console.rule("[bold cyan]SymboLR: Replay Hall of Fame[/bold cyan]")
    
    table = Table(title=f"Saved Elites from {os.path.basename(diagnostics_file)}", show_lines=True)
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Val Loss", justify="right", style="green")
    table.add_column("Size", justify="right", style="magenta")
    table.add_column("Family", justify="center", style="blue")
    table.add_column("LaTeX Formula", justify="left", style="yellow")
    
    for item in hof:
        table.add_row(
            str(item["rank"]),
            f"{item['loss']:.4f}",
            str(item["size"]),
            item["family"],
            item["latex"]
        )
        
    console.print(table)
    
    if plot:
        import sympy
        from sympy.parsing.latex import parse_latex
        import warnings
        
        dir_name = os.path.dirname(diagnostics_file)
        base = os.path.splitext(os.path.basename(diagnostics_file))[0]
        plot_path = os.path.join(dir_name, f"{base}_schedules.png")
        
        schedules = []
        t_array = np.linspace(0.0, 1.0, 100)
        t_sym = sympy.Symbol("t")
        
        console.print("\n[dim]Generating schedule plots (requires latex2sympy2 or careful sympy parsing)...[/dim]")
        
        for item in hof:
            latex = item["latex"]
            try:
                pass
            except Exception as e:
                pass
                
        console.print("[yellow]Warning:[/yellow] Plotting from LaTeX strings requires `latex2sympy2` which is not in requirements.")
        console.print("[dim]For accurate schedule plotting, re-run benchmark with --plot flag in future updates.[/dim]")
