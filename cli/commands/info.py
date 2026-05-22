import typer
from rich.console import Console
from rich.table import Table

from config.settings import _detect_torch, _detect_cuda, _detect_rust, _detect_mode, RuntimeMode

app = typer.Typer(help="Check system environment and GPU detection")
console = Console()

@app.callback(invoke_without_command=True)
def main():
    """Print system environment, GPU detection, and hardware capabilities."""
    console.rule("[bold cyan]SymboLR System Information[/bold cyan]")

    torch_ok = _detect_torch()
    cuda_ok = _detect_cuda()
    rust_ok = _detect_rust()
    mode = _detect_mode()

    table = Table(show_header=False, box=None)
    table.add_column("Component", style="bold")
    table.add_column("Status")

    table.add_row("PyTorch", "[green]Installed[/green]" if torch_ok else "[red]Not Found[/red]")
    
    if torch_ok:
        import torch
        table.add_row("PyTorch Version", torch.__version__)
    
    table.add_row("CUDA / GPU", "[green]Detected[/green]" if cuda_ok else "[yellow]Not Detected (CPU Only)[/yellow]")
    
    if cuda_ok:
        import torch
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown GPU"
        table.add_row("GPU Model", f"[cyan]{device_name}[/cyan]")
        
    table.add_row("Rust Extension", "[green]Installed[/green]" if rust_ok else "[yellow]Not Found (Python Fallback)[/yellow]")
    
    mode_color = "cyan" if mode == RuntimeMode.LOCAL_GPU else "yellow"
    table.add_row("Runtime Mode", f"[{mode_color}]{mode.value.upper()}[/{mode_color}]")

    console.print(table)
    console.print("\n")
