import typer
from rich.console import Console

from cli.commands.benchmark import main as benchmark_main

app = typer.Typer(help="Quick interactive experimentation (override subsystems)")
console = Console()

@app.callback(invoke_without_command=True)
def main(
    novelty: bool = typer.Option(True, "--novelty/--no-novelty", help="Enable/disable novelty search"),
    surrogate: bool = typer.Option(True, "--surrogate/--no-surrogate", help="Enable/disable surrogate triage"),
    operator_controller: bool = typer.Option(True, "--operator-controller/--no-operator-controller", help="Enable/disable operator bandit"),
    meta_controller: bool = typer.Option(True, "--meta-controller/--no-meta-controller", help="Enable/disable meta-controller"),
    generations: int = typer.Option(10, "--generations", "-g", help="Generations"),
    pop_size: int = typer.Option(None, "--pop-size", "-p", help="Population size"),
    epochs: int = typer.Option(None, "--epochs", "-e", help="Epochs per evaluation"),
    workers: int = typer.Option(None, "--workers", "-w", help="Number of concurrent workers"),
):
    """
    Quick experimentation command. Identical to benchmark, but requires explicit 
    enable/disable flags for subsystems instead of relying on default config.
    """
    console.print("[bold yellow]Interactive Experiment Mode[/bold yellow]")
    console.print(f"[dim]Subsystems -> Novelty: {novelty} | Surrogate: {surrogate} | Ops: {operator_controller} | Meta: {meta_controller}[/dim]")
    
    from config.settings import get_config
    cfg = get_config()
    cfg.update(
        novelty_enabled=novelty,
        surrogate_enabled=surrogate,
        operator_controller_enabled=operator_controller,
        meta_controller_enabled=meta_controller,
    )
    
    benchmark_main(
        config_file=None,
        generations=generations,
        pop_size=pop_size,
        epochs=epochs,
        workers=workers,
        seed=None,
        novelty=None,
        surrogate=None,
        meta_controller=None,
    )
