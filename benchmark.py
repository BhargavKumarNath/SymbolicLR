import argparse
import random
import time
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from gp.tree import Node

# Optional W&B integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from data.fidelity import FidelityManager
from models.probe import ProbeTrainer, create_compiled_model
from gp.population import ramped_half_and_half, generate_tree
from gp.evolution import subtree_crossover, subtree_mutation, hoist_mutation
from gp.evaluator import ParallelEvaluator
from gp.map_elites import MAPElitesArchive
from gp.simplify import simplify_tree, tree_to_latex
from optimiser.hybrid import hybrid_optimize_constants


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="SymboLR: MAP-Elites Learning Rate Discovery")
    parser.add_argument("--generations", type=int, default=10, help="Number of evolution cycles")
    parser.add_argument("--pop_size", type=int, default=100, help="Individuals per generation")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per evaluation (ProbeTrainer)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent Rust/GPU workers")
    parser.add_argument("--seed", type=int, default=42, help="Global deterministic seed")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    console = Console()
    console.rule("[bold cyan]SymboLR: Symbolic Learning Rate Discovery System[/bold cyan]")
    
    set_global_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    console.print(f"[bold green]Hardware Detected:[/bold green] {device.type.upper()}")
    if device.type == "cuda":
        console.print(f"[bold green]GPU Model:[/bold green] {torch.cuda.get_device_name(0)}")

    if args.wandb and WANDB_AVAILABLE:
        wandb.init(project="SymboLR", config=vars(args))
    
    # 1. Multi-Fidelity Data Setup
    console.print("\n[bold yellow]Allocating Dataset to VRAM Cache...[/bold yellow]")
    fidelity = FidelityManager(seed=args.seed)
    train_loader, val_loader = fidelity.get_low_fidelity(device, batch_size=256)
    
    # 2. GP System Initialization
    trainer = ProbeTrainer(device=device, patience=2, amp_enabled=True)
    evaluator = ParallelEvaluator(trainer, train_loader, val_loader, epochs=args.epochs, time_steps=100)
    archive = MAPElitesArchive(size_bins=30, com_bins=20, time_steps=100)
    
    model_factory = lambda: create_compiled_model(device, in_channels=1)

    # 3. Initial Population (Generation 0)
    console.print("\n[bold magenta]Initializing Generation 0 (Ramped Half-and-Half)...[/bold magenta]")
    population = ramped_half_and_half(args.pop_size, min_depth=2, max_depth=5)
    
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), console=console) as progress:
        eval_task = progress.add_task("[cyan]Evaluating Gen 0...", total=args.pop_size)
        
        fitnesses = evaluator.evaluate_population(population, model_factory, max_workers=args.workers)
        for tree, fit in zip(population, fitnesses):
            simplified_tree = simplify_tree(tree)
            archive.try_add(simplified_tree, fit)
            progress.advance(eval_task)

    # 4. MAP-Elites Evolution Loop
    console.print("\n[bold cyan]Beginning MAP-Elites Evolution...[/bold cyan]")
    for gen in range(1, args.generations + 1):
        start_time = time.time()
        
        # Select parents from occupied niches
        parents = archive.sample_parents(args.pop_size)

        if not parents:
            console.print("[yellow]Archive empty, falling back to population sampling[/yellow]")
            parents = population.copy()
        offspring =[]
        
        # Apply Genetic Operators
        for _ in range(args.pop_size // 2):
            p1, p2 = random.choice(parents), random.choice(parents)
            
            # 50% Crossover, 25% Mutation, 25% Hoist (Anti-Bloat)
            r = random.random()
            if r < 0.5:
                o1, o2 = subtree_crossover(p1, p2)
                offspring.extend([o1, o2])
            elif r < 0.75:
                offspring.extend([subtree_mutation(p1, 4), subtree_mutation(p2, 4)])
            else:
                offspring.extend([hoist_mutation(p1), hoist_mutation(p2)])

        # Simplify newly generated offspring
        offspring =[simplify_tree(ind) for ind in offspring]
        
        # Evaluate concurrently
        fitnesses = evaluator.evaluate_population(offspring, model_factory, max_workers=args.workers)
        
        # Attempt to insert into MAP-Elites Archive
        additions = 0
        for ind, fit in zip(offspring, fitnesses):
            if archive.try_add(ind, fit):
                additions += 1

        # Calculate metrics
        gen_time = time.time() - start_time
        hof = archive.get_hall_of_fame(top_k=1)
        best_loss = hof[0][0] if hof else float('inf')
        archive_size = len(archive.archive)

        console.print(f"Gen {gen:02d}/{args.generations} | "
                      f"New Niches Found: [green]{additions}[/green] | "
                      f"Archive Size: [cyan]{archive_size}[/cyan] | "
                      f"Best Loss: [bold yellow]{best_loss:.4f}[/bold yellow] | "
                      f"Time: {gen_time:.1f}s")

        if args.wandb and WANDB_AVAILABLE:
            wandb.log({
                "generation": gen,
                "best_loss": best_loss,
                "archive_size": archive_size,
                "new_niches": additions
            })

    # 5. Hybrid Optimization on Final Elites
    console.print("\n[bold yellow]Executing Memetic L-BFGS-B Optimization on Hall of Fame...[/bold yellow]")
    final_elites = archive.get_hall_of_fame(top_k=5)
    
    table = Table(title="SymboLR Final Hall of Fame (MAP-Elites Pareto Front)")
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Val Loss", justify="right", style="green")
    table.add_column("Size", justify="right", style="magenta")
    table.add_column("LaTeX Formula", justify="left", style="yellow")
    
    for idx, (loss, tree) in enumerate(final_elites):
        # Apply numerical refinement to constants
        def eval_wrapper(t: "Node") -> float:
            return evaluator.evaluate_individual(t, model_factory)
            
        optimized_tree = hybrid_optimize_constants(tree, eval_wrapper, maxiter=10)
        final_tree = simplify_tree(optimized_tree)
        
        table.add_row(
            str(idx + 1),
            f"{loss:.4f}",
            str(final_tree.size()),
            tree_to_latex(final_tree)
        )
        
    console.print(table)
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()