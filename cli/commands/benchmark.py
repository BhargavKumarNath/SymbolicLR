import typer
from rich.console import Console
from typing import Optional

from config.settings import get_config
from benchmark import run_evolution

app = typer.Typer(help="Run a single MAP-Elites evolution benchmark")
console = Console()

@app.callback(invoke_without_command=True)
def main(
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
    generations: Optional[int] = typer.Option(None, "--generations", "-g", help="Number of generations"),
    pop_size: Optional[int] = typer.Option(None, "--pop-size", "-p", help="Population size"),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="Epochs per evaluation"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Number of concurrent workers"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    novelty: Optional[bool] = typer.Option(None, "--novelty/--no-novelty", help="Enable/disable novelty search"),
    surrogate: Optional[bool] = typer.Option(None, "--surrogate/--no-surrogate", help="Enable/disable surrogate triage"),
    meta_controller: Optional[bool] = typer.Option(None, "--meta-controller/--no-meta-controller", help="Enable/disable meta-controller"),
):
    """Run a single SymboLR evolution process."""
    cfg = get_config()

    # Load YAML if provided
    if config_file:
        console.print(f"[dim]Loading config from: {config_file}[/dim]")
        loaded = cfg.load_yaml(config_file)
        # Update the singleton directly
        cfg.update(**{k: getattr(loaded, k) for k in loaded.__dataclass_fields__})

    # CLI Overrides
    overrides = {}
    if novelty is not None: overrides["novelty_enabled"] = novelty
    if surrogate is not None: overrides["surrogate_enabled"] = surrogate
    if meta_controller is not None: overrides["meta_controller_enabled"] = meta_controller
    if seed is not None: overrides["seed"] = seed
    
    cfg.update(**overrides)
    
    final_gens = generations or cfg.default_generations
    final_pop = pop_size or cfg.default_pop_size
    final_epochs = epochs or cfg.default_epochs
    final_workers = workers or cfg.default_workers
    final_seed = seed or cfg.seed

    console.rule("[bold cyan]SymboLR: Symbolic Learning Rate Discovery System[/bold cyan]")
    
    run_evolution(
        generations=final_gens,
        pop_size=final_pop,
        epochs=final_epochs,
        workers=final_workers,
        seed=final_seed,
        console=console,
        wandb_enabled=False,
    )
