import numpy as np
from rich.console import Console

from src.symbolr.engine.bridge import RustEvolutionBridge
from src.symbolr.engine.synthetic import SyntheticEvaluator
from src.symbolr.cli.artifacts import export_to_pytorch, export_to_latex

console = Console()

def run_demo():
    console.rule("[bold cyan]SymboLR End-to-End Demo[/bold cyan]")
    
    # 1. Initialize our decoupled evaluator (using Synthetic to run anywhere)
    console.print("[dim]Initializing Synthetic Evaluator...[/dim]")
    evaluator = SyntheticEvaluator(time_steps=100)
    
    # 2. Initialize the Rust Core through the new bridge
    console.print("[dim]Booting Rust EvolutionEngine (Ask-and-Tell)...[/dim]")
    bridge = RustEvolutionBridge(
        eval_callback=evaluator.evaluate,
        max_generations=10,
        pop_size=50,
        seed=42
    )
    
    # 3. Run the Streaming Loop
    console.print("\n[bold green]Starting Evolution Loop:[/bold green]")
    best_formula = ""
    for result in bridge.stream():
        # result is a GenerationResult typed object!
        console.print(f"Gen {result.generation_number:02d} | "
                      f"Best Loss: {result.best_mse:.4f} | "
                      f"Archive: {result.archive_size:03d} | "
                      f"Time: {result.gen_time_ms}ms")
        best_formula = result.top_formula_prefix
        
    console.rule("[bold yellow]Evolution Complete![/bold yellow]")
    
    # 4. Generate Artifacts from the best formula discovered
    console.print(f"\n[bold magenta]Best Prefix Formula:[/bold magenta] {best_formula}")
    
    latex_code = export_to_latex(best_formula)
    console.print(f"\n[bold magenta]Generated LaTeX:[/bold magenta]\n{latex_code}")
    
    pytorch_code = export_to_pytorch(best_formula)
    console.print(f"\n[bold magenta]Generated PyTorch Scheduler Code:[/bold magenta]\n")
    print(pytorch_code)

if __name__ == "__main__":
    run_demo()
