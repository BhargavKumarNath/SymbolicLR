import typer
from rich.console import Console

from cli.commands.benchmark import main as benchmark_cmd
from cli.commands.evolve import main as evolve_cmd
from cli.commands.experiment import main as experiment_cmd
from cli.commands.diagnostics import main as diagnostics_cmd
from cli.commands.replay import main as replay_cmd
from cli.commands.info import main as info_cmd

app = typer.Typer(
    name="symbolr",
    help="SymboLR: Lightweight Research CLI for Symbolic Learning Rate Discovery",
    no_args_is_help=True,
    add_completion=False,
)

app.command(name="info", help="Check system environment and GPU detection")(info_cmd)
app.command(name="benchmark", help="Run a single MAP-Elites evolution benchmark")(benchmark_cmd)
app.command(name="evolve", help="Quick interactive experimentation (override subsystems)")(evolve_cmd)
app.command(name="experiment", help="Run multi-seed statistical validation")(experiment_cmd)
app.command(name="diagnostics", help="Load and summarize previous run diagnostics")(diagnostics_cmd)
app.command(name="replay", help="Replay a saved Hall-of-Fame formula or diagnostics file")(replay_cmd)

if __name__ == "__main__":
    app()
