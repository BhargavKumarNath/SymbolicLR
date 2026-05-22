import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json
import os
from typing import Optional

from optimiser.plotting import export_convergence_plot, export_diversity_plot

app = typer.Typer(help="Load and summarize previous run diagnostics")
console = Console()

@app.callback(invoke_without_command=True)
def main(
    diagnostics_file: str = typer.Argument(..., help="Path to run.json file"),
    plot: bool = typer.Option(False, "--plot", help="Export static PNG plots"),
):
    """View metrics from a completed SymboLR run."""
    if not os.path.exists(diagnostics_file):
        console.print(f"[bold red]Error:[/bold red] File {diagnostics_file} not found.")
        raise typer.Exit(1)
        
    with open(diagnostics_file, "r") as f:
        data = json.load(f)
        
    summary = data.get("summary", {})
    gens = data.get("generations", [])
    
    # 1. Summary Panel
    details = (
        f"[bold]Total Generations:[/bold] {summary.get('total_generations', '?')}\n"
        f"[bold]Best Loss:[/bold] {summary.get('initial_best_loss', '?')} -> {summary.get('final_best_loss', '?')} "
        f"[green](-{summary.get('improvement', '?')})[/green]\n"
        f"[bold]Archive Niches:[/bold] {summary.get('final_archive_size', '?')}\n"
        f"[bold]Dominant Operator:[/bold] {summary.get('dominant_operator', '?')}\n"
        f"[bold]Dominant Phase:[/bold] {summary.get('dominant_phase', '?').upper()}\n"
        f"[bold]Total Runtime:[/bold] {summary.get('total_run_time_s', '?')}s"
    )
    console.print(Panel(details, title="Run Summary", border_style="cyan"))
    
    # 2. Phase transitions and operator history (sample last 10)
    if gens:
        table = Table(title="Generation History (Last 10)", show_lines=True)
        table.add_column("Gen", style="dim")
        table.add_column("Phase", style="yellow")
        table.add_column("Loss", style="green")
        table.add_column("Archive", style="cyan")
        table.add_column("Div", style="blue")
        table.add_column("Dominant Op", style="magenta")
        
        for g in gens[-10:]:
            table.add_row(
                str(g["generation"]),
                g["controller_phase"].upper(),
                f"{g['best_loss']:.4f}",
                str(g["archive_size"]),
                f"{g['structural_diversity']:.2f}",
                g["operator_dominant"]
            )
        console.print(table)
    
    if plot and gens:
        # Generate plots
        dir_name = os.path.dirname(diagnostics_file)
        base = os.path.splitext(os.path.basename(diagnostics_file))[0]
        
        gen_indices = [g["generation"] for g in gens]
        best_loss = [g["best_loss"] for g in gens]
        med_loss = [g["median_loss"] for g in gens]
        struct_div = [g["structural_diversity"] for g in gens]
        behav_div = [g["behavioral_diversity"] for g in gens]
        
        conv_path = os.path.join(dir_name, f"{base}_convergence.png")
        div_path = os.path.join(dir_name, f"{base}_diversity.png")
        
        export_convergence_plot(gen_indices, best_loss, med_loss, conv_path)
        export_diversity_plot(gen_indices, struct_div, behav_div, div_path)
        
        console.print(f"[dim]Exported convergence plot -> {conv_path}[/dim]")
        console.print(f"[dim]Exported diversity plot -> {div_path}[/dim]")
