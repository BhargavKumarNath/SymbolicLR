"""
ui/state.py - Session-state management and background evolution job runner.

Refactored to use:
  - config.settings for centralized configuration
  - gp.rust_bridge for unified schedule evaluation
  - gp.fitness for synthetic/real fitness evaluation
  - Proper device handling for CPU/GPU/Cloud modes
"""

import concurrent.futures
import copy
import random
import threading
import time
import traceback
import uuid
import numpy as np
import streamlit as st
from typing import Any, Optional

from config.settings import get_config, RuntimeMode
from gp.rust_bridge import evaluate_schedule
from gp.population import ramped_half_and_half
from gp.evolution import subtree_crossover, subtree_mutation, hoist_mutation
from gp.evaluator import ParallelEvaluator
from gp.map_elites import MAPElitesArchive
from gp.simplify import simplify_tree

# Re-export for backward compat
cfg = get_config()
TORCH_AVAILABLE = cfg.torch_available

# Default session state schema
_DEFAULTS = {
    "evolution_done": False,
    "gen_log": [],
    "hof": [],
    "archive_snapshot": [],
    "lr_curves": [],
    "total_time": 0.0,
    "device_type": "GPU" if cfg.is_gpu else ("CPU" if cfg.torch_available else "Cloud (Simulated)"),
    "effective_workers": 1,
    "run_status": "idle",
    "run_error": None,
    "current_run_id": None,
    "progress_ratio": 0.0,
    "progress_label": "",
    "phase_label": "",
    "eval_completed": 0,
    "eval_total": 0,
    "run_params": {},
    "_heartbeat": 0,
    "_done_triggered": False,
}

_RUNS_LOCK = threading.Lock()
_RUNS: dict[str, dict] = {}
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="symbolr-runner"
)

_MAX_STALE_RUNS = cfg.max_stale_runs
_PUBLISH_THROTTLE_S = cfg.publish_throttle_s


def _fresh_defaults() -> dict:
    return {
        key: (copy.deepcopy(value) if isinstance(value, (list, dict)) else value)
        for key, value in _DEFAULTS.items()
    }


def _new_run_state(run_id: str, params: dict) -> dict:
    state = _fresh_defaults()
    state.update(
        {
            "current_run_id": run_id,
            "run_status": "queued",
            "run_params": copy.deepcopy(params),
            "progress_label": "Queued",
            "phase_label": "Waiting to start",
        }
    )
    return state


def _evict_stale_runs() -> None:
    """Remove completed/failed runs beyond the retention cap."""
    with _RUNS_LOCK:
        terminal = [
            rid
            for rid, s in _RUNS.items()
            if s.get("run_status") in {"completed", "failed"}
        ]
        for rid in terminal[:-_MAX_STALE_RUNS]:
            del _RUNS[rid]


def _store_run_state(run_id: str, **updates) -> None:
    with _RUNS_LOCK:
        if run_id not in _RUNS:
            return
        _RUNS[run_id].update(copy.deepcopy(updates))
        _RUNS[run_id]["_heartbeat"] = _RUNS[run_id].get("_heartbeat", 0) + 1


def _get_run_state(run_id: Optional[str]) -> Optional[dict]:
    if not run_id:
        return None
    with _RUNS_LOCK:
        state = _RUNS.get(run_id)
        return copy.deepcopy(state) if state is not None else None


def _is_terminal(status: str) -> bool:
    return status in {"completed", "failed"}


def _sync_from_snapshot(snapshot: Optional[dict]) -> None:
    if snapshot is None:
        return
    for key in _DEFAULTS:
        if key in snapshot:
            st.session_state[key] = snapshot[key]


def init_state():
    """Idempotent state bootstrap."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = (
                copy.deepcopy(default) if isinstance(default, (list, dict)) else default
            )


def sync_state():
    """Hydrate session state from the active background run, if any."""
    snapshot = _get_run_state(st.session_state.get("current_run_id"))
    _sync_from_snapshot(snapshot)


def _clear_run_payload():
    for key in ("gen_log", "hof", "archive_snapshot", "lr_curves"):
        st.session_state[key] = []
    st.session_state["total_time"] = 0.0
    st.session_state["evolution_done"] = False
    st.session_state["run_error"] = None
    st.session_state["progress_ratio"] = 0.0
    st.session_state["progress_label"] = ""
    st.session_state["phase_label"] = ""
    st.session_state["eval_completed"] = 0
    st.session_state["eval_total"] = 0
    st.session_state["_heartbeat"] = 0
    st.session_state["_done_triggered"] = False


def _build_archive_snapshot(archive: MAPElitesArchive, max_points: int = 500) -> list[dict]:
    """
    Build a JSON-serialisable snapshot of the archive for the scatter chart.
    Capped at max_points to keep deepcopy cheap at large archive sizes.
    """
    items = list(archive.archive.items())
    if len(items) > max_points:
        items.sort(key=lambda x: x[1][0])
        keep_top = max_points // 4
        top = items[:keep_top]
        rest = random.sample(items[keep_top:], max_points - keep_top)
        items = top + rest

    return [
        {
            "Size": size_idx,
            "Center of Mass": com_idx / archive.com_bins,
            "Loss": loss,
            "Formula": str(tree),
        }
        for (size_idx, com_idx), (loss, tree) in items
    ]


def _build_lr_curves(hof: list, t_array: np.ndarray) -> list[dict]:
    """Build LR curve data for the top formulas."""
    curves = []
    for i, (loss, tree) in enumerate(hof):
        try:
            lrs = evaluate_schedule(tree, t_array)
            for t_val, lr in zip(t_array, lrs):
                curves.append({
                    "Time": float(t_val),
                    "LR": float(lr),
                    "Rank": f"#{i+1}  loss={loss:.3f}",
                })
        except Exception:
            continue
    return curves


# Track when we last did a full (expensive) snapshot publish
_last_full_publish: dict[str, float] = {}


def _publish_progress(
    run_id: str,
    archive: MAPElitesArchive,
    gen_log: list[dict],
    hof: list,
    total_time: float,
    progress_ratio: float,
    progress_label: str,
    phase_label: str,
    eval_completed: int,
    eval_total: int,
    evolution_done: bool = False,
    force: bool = False,
) -> None:
    """
    Write current run state back to _RUNS.

    Heavy fields (archive_snapshot, lr_curves) are only rebuilt when the
    throttle period has elapsed or force=True.
    """
    now = time.time()
    last = _last_full_publish.get(run_id, 0.0)
    do_full = force or evolution_done or (now - last >= _PUBLISH_THROTTLE_S)

    updates: dict = {
        "gen_log": gen_log,
        "hof": hof,
        "total_time": round(total_time, 1),
        "progress_ratio": float(progress_ratio),
        "progress_label": progress_label,
        "phase_label": phase_label,
        "eval_completed": int(eval_completed),
        "eval_total": int(eval_total),
        "evolution_done": evolution_done,
    }

    if do_full:
        updates["archive_snapshot"] = _build_archive_snapshot(archive)
        updates["lr_curves"] = _build_lr_curves(hof, archive.t_array)
        _last_full_publish[run_id] = now

    _store_run_state(run_id, **updates)


def _setup_evaluation_stack(epochs: int, workers: int):
    """
    Create trainer, loaders, evaluator, and model_factory
    appropriate for the current runtime mode.

    Real PyTorch training is only used when GPU is available.
    On CPU (local or cloud), synthetic fitness is used for speed and stability.
    """
    cfg = get_config()
    effective_workers = cfg.resolve_workers(workers, epochs, 50)

    if cfg.is_gpu:
        import torch
        device = cfg.device
        from data.fidelity import FidelityManager
        from models.probe import ProbeTrainer, create_compiled_model

        fidelity = FidelityManager(seed=cfg.seed)
        train_loader, val_loader = fidelity.get_low_fidelity(device, batch_size=cfg.batch_size)
        trainer = ProbeTrainer(device=device, patience=cfg.patience, amp_enabled=cfg.amp_enabled)
        model_factory = lambda: create_compiled_model(device, in_channels=1)
    else:
        # CPU and Cloud modes use synthetic fitness - fast and meaningful
        train_loader = None
        val_loader = None
        trainer = None
        model_factory = None

    evaluator = ParallelEvaluator(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        time_steps=cfg.time_steps,
    )

    return evaluator, model_factory, effective_workers


def _run_evolution_job(
    run_id: str, gen_count: int, pop_size: int, epochs: int, workers: int
) -> None:
    """Background evolution job. Runs in a separate thread."""
    started_at = time.time()
    try:
        cfg = get_config()
        evaluator, model_factory, effective_workers = _setup_evaluation_stack(epochs, workers)

        device_type = "GPU" if cfg.is_gpu else ("CPU" if cfg.torch_available else "Cloud (Simulated)")

        _store_run_state(
            run_id,
            run_status="running",
            device_type=device_type,
            effective_workers=effective_workers,
            progress_label="Seeding initial population",
            phase_label="Generation 0",
        )

        archive = MAPElitesArchive(
            size_bins=cfg.size_bins,
            com_bins=cfg.com_bins,
            time_steps=cfg.time_steps,
        )

        gen_log: list[dict] = []
        hof: list = []

        # Generation 0
        population = ramped_half_and_half(pop_size, min_depth=cfg.min_tree_depth, max_depth=cfg.max_tree_depth - 1)
        fitnesses = evaluator.evaluate_population(
            population,
            model_factory,
            max_workers=effective_workers,
            progress_callback=lambda completed, total: _store_run_state(
                run_id,
                eval_completed=completed,
                eval_total=total,
                phase_label=f"Generation 0: {completed}/{total} candidates evaluated",
            ),
        )

        for tree, fit in zip(population, fitnesses):
            archive.try_add(simplify_tree(tree), fit)

        hof = archive.get_hall_of_fame(top_k=5)
        _publish_progress(
            run_id,
            archive,
            gen_log,
            hof,
            total_time=time.time() - started_at,
            progress_ratio=0.0,
            progress_label="Generation 0 complete",
            phase_label="Generation 0 complete",
            eval_completed=len(population),
            eval_total=len(population),
            force=True,
        )

        # Evolution loop
        for gen in range(1, gen_count + 1):
            gen_start = time.time()
            _store_run_state(
                run_id,
                progress_ratio=(gen - 1) / gen_count,
                progress_label=f"Generation {gen} / {gen_count}",
                phase_label=f"Generation {gen}: 0/{pop_size} candidates evaluated",
                eval_completed=0,
                eval_total=pop_size,
            )

            parents = archive.sample_parents(pop_size)
            if not parents:
                parents = ramped_half_and_half(pop_size, min_depth=cfg.min_tree_depth, max_depth=cfg.max_tree_depth - 1)

            offspring = []
            for _ in range(pop_size // 2):
                p1, p2 = random.choice(parents), random.choice(parents)
                r = random.random()
                if r < cfg.crossover_rate:
                    o1, o2 = subtree_crossover(p1, p2)
                    offspring.extend([o1, o2])
                elif r < cfg.crossover_rate + cfg.mutation_rate:
                    offspring.extend(
                        [subtree_mutation(p1, 3), subtree_mutation(p2, 3)]
                    )
                else:
                    offspring.extend([hoist_mutation(p1), hoist_mutation(p2)])

            # Evaluate with dynamic worker reduction on CUDA OOM
            current_workers = effective_workers
            while True:
                try:
                    fitnesses = evaluator.evaluate_population(
                        offspring,
                        model_factory,
                        max_workers=current_workers,
                        progress_callback=lambda completed, total, gen=gen: _store_run_state(
                            run_id,
                            eval_completed=completed,
                            eval_total=total,
                            phase_label=(
                                f"Generation {gen}: {completed}/{total} candidates evaluated"
                            ),
                        ),
                    )
                    break
                except Exception as e:
                    if cfg.torch_available:
                        import torch
                        if isinstance(e, torch.cuda.OutOfMemoryError):
                            if current_workers <= 1:
                                raise
                            current_workers = max(1, current_workers - 1)
                            _store_run_state(
                                run_id,
                                phase_label=(
                                    f"Generation {gen}: CUDA OOM - retrying with "
                                    f"{current_workers} worker(s)"
                                ),
                            )
                            torch.cuda.empty_cache()
                            time.sleep(0.5)
                            continue
                    raise e

            new_niches = 0
            for ind, fit in zip(offspring, fitnesses):
                size_idx, com_idx = archive._compute_descriptors(ind)
                if size_idx is None:
                    continue
                niche = (size_idx, com_idx)
                if niche not in archive.archive or fit < archive.archive[niche][0]:
                    simplified = simplify_tree(ind)
                    if archive.try_add(simplified, fit):
                        new_niches += 1

            hof = archive.get_hall_of_fame(top_k=5)
            best_loss = hof[0][0] if hof else float("inf")
            gen_log.append(
                {
                    "gen": gen,
                    "best_loss": best_loss,
                    "niches": len(archive.archive),
                    "new_niches": new_niches,
                    "time": round(time.time() - gen_start, 1),
                }
            )

            _publish_progress(
                run_id,
                archive,
                gen_log,
                hof,
                total_time=time.time() - started_at,
                progress_ratio=gen / gen_count,
                progress_label=f"Generation {gen} / {gen_count}",
                phase_label=f"Generation {gen} complete  ·  "
                f"{new_niches} new niches  ·  best loss {best_loss:.4f}",
                eval_completed=len(offspring),
                eval_total=len(offspring),
            )

        _store_run_state(
            run_id,
            run_status="completed",
            evolution_done=True,
            progress_ratio=1.0,
            progress_label="Evolution complete",
            phase_label="Run finished successfully",
        )
        _evict_stale_runs()

    except Exception as exc:
        _store_run_state(
            run_id,
            run_status="failed",
            run_error=f"{type(exc).__name__}: {exc}",
            phase_label="Run failed",
            progress_label="Evolution failed",
            total_time=round(time.time() - started_at, 1),
        )
        _store_run_state(run_id, traceback=traceback.format_exc())
        _evict_stale_runs()
    finally:
        _last_full_publish.pop(run_id, None)


def start_evolution(
    gen_count: int, pop_size: int, epochs: int, workers: int
) -> bool:
    """
    Launch a background evolution run.
    Returns False if this session already has an active run.
    """
    sync_state()
    if st.session_state.get("run_status") in {"queued", "running"}:
        return False

    run_id = str(uuid.uuid4())
    params = {
        "generations": int(gen_count),
        "population_size": int(pop_size),
        "epochs": int(epochs),
        "workers": int(workers),
    }

    state = _new_run_state(run_id, params)
    with _RUNS_LOCK:
        _RUNS[run_id] = state

    _clear_run_payload()
    st.session_state["current_run_id"] = run_id
    _sync_from_snapshot(state)
    _EXECUTOR.submit(_run_evolution_job, run_id, gen_count, pop_size, epochs, workers)
    return True


def get_run_status() -> str:
    sync_state()
    return st.session_state.get("run_status", "idle")