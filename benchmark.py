import argparse
import os
import random
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from gp.tree import Node
from gp.population import ramped_half_and_half
from gp.evolution import (
    subtree_crossover,
    subtree_mutation,
    hoist_mutation,
    point_mutation,
    constant_perturbation,
)
from gp.simplify import simplify_tree, tree_to_latex
from gp.evaluator import ParallelEvaluator
from gp.map_elites import MAPElitesArchive
from gp.novelty import NoveltyArchive, compute_fingerprint
from gp.diversity import DiversityTracker, is_semantic_duplicate
from gp.operator_controller import OperatorController
from gp.meta_controller import MetaController, Phase
from gp.diagnostics import DiagnosticsLog, GenerationMetrics, classify_family
from gp.surrogate import LightweightSurrogate
from gp.rust_bridge import evaluate_schedule

from data.fidelity import FidelityManager
from models.probe import ProbeTrainer, create_compiled_model
from config.settings import get_config
from optimiser.hybrid import hybrid_optimize_constants

# Optional W&B integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _apply_operator(
    op_name: str,
    parents: list,
    cfg,
    max_depth: int,
) -> Node:
    """Apply the named genetic operator to a randomly selected parent (or pair)."""
    p1 = random.choice(parents)
    if op_name == "crossover":
        p2 = random.choice(parents)
        return subtree_crossover(p1, p2, max_depth=max_depth)
    elif op_name == "subtree_mutation":
        return subtree_mutation(p1, max_depth=max_depth)
    elif op_name == "hoist_mutation":
        return hoist_mutation(p1)
    elif op_name == "point_mutation":
        return point_mutation(p1)
    else:  # constant_perturbation
        return constant_perturbation(p1)


def run_evolution(
    generations: int,
    pop_size: int,
    epochs: int,
    workers: int,
    seed: int,
    console: Console,
    wandb_enabled: bool = False,
) -> DiagnosticsLog:
    """
    Core evolution loop. Returns a DiagnosticsLog for offline analysis.

    Separated from main() so experiment_runner.py can call it directly
    for multi-seed experiments without subprocess overhead.
    """
    cfg = get_config()
    set_global_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Hardware Detected:[/bold green] {device.type.upper()}")
    if device.type == "cuda":
        console.print(
            f"[bold green]GPU Model:[/bold green] {torch.cuda.get_device_name(0)}"
        )

    # -----------------------------------------------------------------------
    # Data & model setup
    # -----------------------------------------------------------------------
    console.print("\n[bold yellow]Allocating Dataset to VRAM Cache...[/bold yellow]")
    fidelity = FidelityManager(seed=seed)
    train_loader, val_loader = fidelity.get_low_fidelity(device, batch_size=256)

    trainer = ProbeTrainer(device=device, patience=2, amp_enabled=True)
    evaluator = ParallelEvaluator(
        trainer, train_loader, val_loader, epochs=epochs, time_steps=100
    )
    archive = MAPElitesArchive(size_bins=30, com_bins=20, smoothness_bins=10, time_steps=100)
    model_factory = lambda init_seed=None: create_compiled_model(
        device, in_channels=1, init_seed=init_seed
    )

    # -----------------------------------------------------------------------
    # Phase 3: Subsystem initialization
    # -----------------------------------------------------------------------
    novelty_archive = NoveltyArchive(
        max_size=cfg.novelty_archive_size,
        k_neighbours=cfg.novelty_k_neighbours,
    ) if cfg.novelty_enabled else None

    diversity_tracker = DiversityTracker(
        sample_size=cfg.diversity_sample_size,
        collapse_threshold=cfg.diversity_collapse_threshold,
    ) if cfg.diversity_tracking_enabled else None

    op_controller = OperatorController(
        min_prob=cfg.operator_controller_min_prob,
        ema_alpha=cfg.operator_controller_ema_alpha,
    ) if cfg.operator_controller_enabled else None

    meta_ctrl = MetaController(
        stagnation_threshold=cfg.stagnation_threshold,
        collapse_threshold=cfg.diversity_collapse_threshold,
        base_novelty_weight=cfg.novelty_weight,
        max_novelty_weight=cfg.novelty_max_weight,
        base_mutation_boost=cfg.stagnation_mutation_boost,
        base_immigrant_fraction=cfg.immigrant_fraction,
        pop_size=pop_size,
    ) if cfg.meta_controller_enabled else None

    diagnostics = DiagnosticsLog() if cfg.diagnostics_enabled else None

    surrogate = LightweightSurrogate(
        min_samples=cfg.surrogate_min_samples,
        eval_fraction=cfg.surrogate_eval_fraction,
        buffer_size=cfg.surrogate_buffer_size,
    ) if cfg.surrogate_enabled else None

    # -----------------------------------------------------------------------
    # Generation 0: Initial population
    # -----------------------------------------------------------------------
    console.print(
        "\n[bold magenta]Initializing Generation 0 (Ramped Half-and-Half)...[/bold magenta]"
    )
    population = ramped_half_and_half(pop_size, min_depth=2, max_depth=5)

    with Progress(
        SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), console=console
    ) as progress:
        eval_task = progress.add_task("[cyan]Evaluating Gen 0...", total=pop_size)

        raw_fitnesses = evaluator.evaluate_population(
            population, model_factory,
            max_workers=cfg.resolve_workers(workers, epochs, pop_size),
            surrogate=surrogate,
        )

        for tree, raw_fit in zip(population, raw_fitnesses):
            simplified = simplify_tree(tree)

            # Novelty augmentation
            effective_fit = raw_fit
            if novelty_archive is not None and cfg.novelty_enabled:
                try:
                    schedule = evaluate_schedule(simplified, archive.t_array)
                    fp = compute_fingerprint(schedule)
                    n_score = novelty_archive.novelty_score(fp)
                    nw = cfg.novelty_weight
                    effective_fit = (1 - nw) * raw_fit - nw * n_score
                    novelty_archive.add(fp)
                    if diagnostics:
                        diagnostics.record_novelty(n_score)
                except Exception:
                    pass

            archive.try_add(simplified, raw_fit, effective_loss=effective_fit)
            progress.advance(eval_task)

    # -----------------------------------------------------------------------
    # MAP-Elites Evolution Loop
    # -----------------------------------------------------------------------
    console.print("\n[bold cyan]Beginning MAP-Elites Evolution...[/bold cyan]")

    for gen in range(1, generations + 1):
        gen_start = time.time()

        # --- Read control signals ---
        current_phase = meta_ctrl.phase if meta_ctrl else Phase.EXPLOIT
        mutation_boost = meta_ctrl.mutation_boost if meta_ctrl else 1.0
        novelty_weight = meta_ctrl.novelty_weight if meta_ctrl else cfg.novelty_weight
        immigrant_count = meta_ctrl.immigrant_count if meta_ctrl else 0
        xo_delta = meta_ctrl.crossover_rate_delta if meta_ctrl else 0.0

        effective_crossover = max(0.05, cfg.crossover_rate + xo_delta)
        effective_mutation = min(0.80, cfg.mutation_rate * mutation_boost)

        # --- Sample parents from archive + optional immigrants ---
        parents = archive.sample_parents(pop_size)
        if not parents:
            parents = population.copy()

        if immigrant_count > 0:
            immigrants = ramped_half_and_half(immigrant_count, min_depth=1, max_depth=3)
            if len(parents) > immigrant_count:
                parents = parents[:-immigrant_count] + immigrants
            else:
                parents = parents + immigrants

        # --- Generate offspring using operator controller or static rates ---
        offspring: list = []
        offspring_ops: list = []   # parallel list tracking which op created each offspring

        while len(offspring) < pop_size:
            if op_controller is not None:
                op_name = op_controller.select_operator()
            else:
                # Static rate fallback (Phase 2 behavior)
                r = random.random()
                total = 0.0
                op_name = "constant_perturbation"
                for name, rate in [
                    ("crossover", effective_crossover),
                    ("subtree_mutation", effective_mutation),
                    ("hoist_mutation", cfg.hoist_rate),
                    ("point_mutation", cfg.point_mutation_rate),
                    ("constant_perturbation", cfg.constant_perturbation_rate),
                ]:
                    total += rate
                    if r < total:
                        op_name = name
                        break

            child = _apply_operator(op_name, parents, cfg, cfg.max_tree_depth_limit)
            offspring.append(child)
            offspring_ops.append(op_name)

        # Simplify
        offspring = [simplify_tree(ind) for ind in offspring]

        # --- Evaluate ---
        # Build always_evaluate mask: structurally new = unseen hash
        seen_hashes = evaluator._fitness_cache.keys()
        always_eval = [off.get_hash() not in seen_hashes for off in offspring]

        raw_fitnesses = evaluator.evaluate_population(
            offspring,
            model_factory,
            max_workers=cfg.resolve_workers(workers, epochs, pop_size),
            surrogate=surrogate,
            always_evaluate=always_eval,
        )

        # --- Archive insertion + operator outcome recording ---
        additions = 0
        for ind, raw_fit, op_name in zip(offspring, raw_fitnesses, offspring_ops):
            if not np.isfinite(raw_fit):
                if op_controller:
                    op_controller.record_outcome(op_name, False)
                continue

            # Novelty augmentation
            effective_fit = raw_fit
            fp = None
            if novelty_archive is not None and cfg.novelty_enabled:
                try:
                    schedule = evaluate_schedule(ind, archive.t_array)
                    fp = compute_fingerprint(schedule)
                    n_score = novelty_archive.novelty_score(fp)
                    nw = novelty_weight
                    effective_fit = (1 - nw) * raw_fit - nw * n_score
                    if diagnostics:
                        diagnostics.record_novelty(n_score)
                except Exception:
                    pass

            # Semantic dedup check (optional pre-filter)
            if (
                cfg.semantic_dedup_enabled
                and novelty_archive is not None
                and fp is not None
                and is_semantic_duplicate(fp, novelty_archive, cfg.semantic_dedup_threshold)
            ):
                if op_controller:
                    op_controller.record_outcome(op_name, False)
                continue

            entered = archive.try_add(ind, raw_fit, effective_loss=effective_fit)

            if entered:
                additions += 1
                if novelty_archive is not None and fp is not None:
                    novelty_archive.add(fp)

            if op_controller:
                op_controller.record_outcome(op_name, entered)

        # --- Post-generation updates ---
        archive.increment_ages()

        # Update operator controller EMA
        if op_controller:
            op_controller.end_generation()

        # Update diversity tracker
        struct_div = 1.0
        behav_div = 1.0
        if diversity_tracker is not None:
            diversity_tracker.update(archive)
            struct_div = diversity_tracker.structural_diversity
            behav_div = diversity_tracker.behavioral_diversity

        # Update meta-controller (drives next generation's control signals)
        hof = archive.get_hall_of_fame(top_k=1)
        best_loss = hof[0][0] if hof else float("inf")
        best_tree = hof[0][1] if hof else None
        
        best_latex = ""
        if best_tree:
            best_latex = tree_to_latex(simplify_tree(best_tree))
            
        archive_size = len(archive.archive)

        if meta_ctrl:
            meta_ctrl.update(best_loss, archive_size, struct_div, gen)

        gen_time = time.time() - gen_start

        # --- Flush novelty stats for diagnostics ---
        novelty_mean, novelty_max = 0.0, 0.0
        if diagnostics:
            novelty_mean, novelty_max = diagnostics.flush_novelty_stats()

        # --- Collect diagnostics ---
        if diagnostics:
            archive_stats = archive.get_stats()
            op_stats = op_controller.get_stats() if op_controller else {}
            surrogate_stats = surrogate.get_stats() if surrogate else {"prediction_rmse": -1.0, "buffer_size": 0}
            ctrl_status = meta_ctrl.get_status() if meta_ctrl else {"phase": "exploit", "stagnation_epochs": 0, "mutation_boost": 1.0}

            losses_in_archive = [v[0] for v in archive.archive.values()]

            diagnostics.record(GenerationMetrics(
                generation=gen,
                best_loss=best_loss,
                median_loss=float(np.median(losses_in_archive)) if losses_in_archive else float("inf"),
                top_formula_latex=best_latex,
                archive_size=archive_size,
                occupancy_pct=archive_stats["occupancy_pct"],
                structural_diversity=round(struct_div, 4),
                behavioral_diversity=round(behav_div, 4),
                novelty_mean=round(novelty_mean, 4),
                novelty_max=round(novelty_max, 4),
                new_niches=additions,
                controller_phase=ctrl_status["phase"],
                stagnation_epochs=ctrl_status["stagnation_epochs"],
                mutation_boost=round(ctrl_status["mutation_boost"], 2),
                operator_dominant=op_controller.get_dominant_operator() if op_controller else "crossover",
                operator_probs=op_controller.get_probabilities() if op_controller else {},
                surrogate_rmse=surrogate_stats["prediction_rmse"],
                surrogate_buffer=surrogate_stats["buffer_size"],
                gen_time_s=round(gen_time, 1),
            ))

        # --- Terminal output ---
        phase_str = f"[yellow]{current_phase.value.upper()}[/yellow]"
        op_str = ""
        if op_controller:
            op_str = f" | Op: [dim]{op_controller.format_probabilities()}[/dim]"

        div_str = f" | Div: [blue]{struct_div:.2f}[/blue]"
        novelty_str = f" | Nov: [magenta]{novelty_mean:.3f}[/magenta]" if novelty_archive else ""

        console.print(
            f"Gen {gen:02d}/{generations} | Phase: {phase_str} | "
            f"Niches: [green]+{additions}[/green] ({archive_size}) | "
            f"Best: [bold yellow]{best_loss:.4f}[/bold yellow]"
            f"{div_str}{novelty_str}{op_str} | "
            f"[dim]{gen_time:.1f}s[/dim]"
        )

        if wandb_enabled and WANDB_AVAILABLE:
            wandb.log({
                "generation": gen,
                "best_loss": best_loss,
                "archive_size": archive_size,
                "new_niches": additions,
                "structural_diversity": struct_div,
                "novelty_mean": novelty_mean,
                "controller_phase": current_phase.value,
            })

    # -----------------------------------------------------------------------
    # Memetic Refinement + Hall of Fame
    # -----------------------------------------------------------------------
    console.print(
        "\n[bold yellow]Executing Memetic L-BFGS-B Optimization on Hall of Fame...[/bold yellow]"
    )
    final_elites = archive.get_hall_of_fame(top_k=5)

    table = Table(title="SymboLR Final Hall of Fame (MAP-Elites Pareto Front)", show_lines=True)
    table.add_column("Rank", justify="center", style="cyan", width=6)
    table.add_column("Val Loss", justify="right", style="green", width=10)
    table.add_column("Size", justify="right", style="magenta", width=6)
    table.add_column("Family", justify="center", style="blue", width=12)
    table.add_column("LaTeX Formula", justify="left", style="yellow")

    hof_results = []
    for idx, (loss, tree) in enumerate(final_elites):
        def eval_wrapper(t: Node) -> float:
            return evaluator.evaluate_individual(t, model_factory)

        optimized_tree = hybrid_optimize_constants(tree, eval_wrapper, maxiter=10)
        final_tree = simplify_tree(optimized_tree)
        latex = tree_to_latex(final_tree)
        family = classify_family(latex)

        table.add_row(
            str(idx + 1),
            f"{loss:.4f}",
            str(final_tree.size()),
            family,
            latex,
        )
        hof_results.append({
            "rank": idx + 1,
            "loss": loss,
            "size": final_tree.size(),
            "family": family,
            "latex": latex
        })

    console.print(table)

    # -----------------------------------------------------------------------
    # Diagnostics export
    # -----------------------------------------------------------------------
    if diagnostics and cfg.diagnostics_enabled:
        os.makedirs(cfg.diagnostics_export_path, exist_ok=True)
        json_path = os.path.join(cfg.diagnostics_export_path, f"run_seed{seed}.json")
        csv_path = os.path.join(cfg.diagnostics_export_path, f"run_seed{seed}.csv")
        
        # Inject HoF into the diagnostics before export
        diagnostics._records[-1].operator_probs["hall_of_fame"] = hof_results
        
        diagnostics.export_json(json_path)
        diagnostics.export_csv(csv_path)
        
        # Remove the hacky HoF injection from the exported JSON by writing a cleaner one
        # Actually, diagnostics.export_json just dumps the data. We'll append it safely.
        with open(json_path, "r") as f:
            data = __import__("json").load(f)
        data["hall_of_fame"] = hof_results
        with open(json_path, "w") as f:
            __import__("json").dump(data, f, indent=2)

        summary = diagnostics.summary()
        console.print(
            f"\n[bold green]Run Summary:[/bold green] "
            f"Loss {summary.get('initial_best_loss', '?')} -> "
            f"[bold]{summary.get('final_best_loss', '?')}[/bold] | "
            f"Archive: {summary.get('final_archive_size', '?')} niches | "
            f"Diversity: {summary.get('final_structural_diversity', '?')} | "
            f"Dominant op: [cyan]{summary.get('dominant_operator', '?')}[/cyan]"
        )
        console.print(
            f"[dim]Diagnostics saved -> {json_path}[/dim]"
        )

    return diagnostics


def main(
    config_file: str = None,
    generations: int = None,
    pop_size: int = None,
    epochs: int = None,
    workers: int = None,
    seed: int = None,
    novelty: bool = None,
    surrogate: bool = None,
    meta_controller: bool = None,
):
    """
    Programmatic entrypoint. Usually called via the Typer CLI in cli/commands/benchmark.py.
    """
    cfg = get_config()
    
    # Load YAML if provided
    if config_file:
        loaded = cfg.load_yaml(config_file)
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

    console = Console()
    console.rule("[bold cyan]SymboLR: Symbolic Learning Rate Discovery System[/bold cyan]")

    if WANDB_AVAILABLE:
        wandb.init(project="SymboLR", config=vars(cfg))

    run_evolution(
        generations=final_gens,
        pop_size=final_pop,
        epochs=final_epochs,
        workers=final_workers,
        seed=final_seed,
        console=console,
        wandb_enabled=WANDB_AVAILABLE,
    )

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()