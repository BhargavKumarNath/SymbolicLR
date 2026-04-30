"""
ui/state.py
Session-state schema initialisation and background evolution job management.
Keeps Streamlit reruns decoupled from the long-running MAP-Elites process.
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
import torch
from data.fidelity import FidelityManager
from models.probe import ProbeTrainer, create_compiled_model
from gp.population import ramped_half_and_half
from gp.evolution import subtree_crossover, subtree_mutation, hoist_mutation
from gp.evaluator import ParallelEvaluator
from gp.map_elites import MAPElitesArchive
from gp.simplify import simplify_tree
import symbolr_rust


_DEFAULTS = {
    "evolution_done": False,
    "gen_log": [],
    "hof": [],
    "archive_snapshot": [],
    "lr_curves": [],
    "total_time": 0.0,
    "device_type": "cpu",
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
}

_RUNS_LOCK = threading.Lock()
_RUNS: dict[str, dict] = {}
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="symbolr-runner")


def _fresh_defaults() -> dict:
    return {key: (copy.deepcopy(value) if isinstance(value, (list, dict)) else value) for key, value in _DEFAULTS.items()}


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


def _store_run_state(run_id: str, **updates) -> None:
    with _RUNS_LOCK:
        if run_id not in _RUNS:
            return
        _RUNS[run_id].update(copy.deepcopy(updates))


def _get_run_state(run_id: str | None) -> dict | None:
    if not run_id:
        return None
    with _RUNS_LOCK:
        state = _RUNS.get(run_id)
        return copy.deepcopy(state) if state is not None else None


def _is_terminal(status: str) -> bool:
    return status in {"completed", "failed"}


def _sync_from_snapshot(snapshot: dict | None) -> None:
    if snapshot is None:
        return
    for key, default in _DEFAULTS.items():
        if key in snapshot:
            value = snapshot[key]
            st.session_state[key] = copy.deepcopy(value) if isinstance(value, (list, dict)) else value


def init_state():
    """Idempotent state bootstrap."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(default) if isinstance(default, (list, dict)) else default


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


def _resolve_worker_budget(device: torch.device, requested_workers: int, epochs: int, pop_size: int) -> int:
    """Cap concurrency to keep larger runs responsive instead of overcommitting the hardware."""
    requested = max(1, int(requested_workers))

    if device.type != "cuda":
        return min(requested, max(1, min(4, pop_size)))

    if epochs >= 5 or pop_size >= 100:
        safe_cap = 2
    elif epochs >= 3 or pop_size >= 60:
        safe_cap = 3
    else:
        safe_cap = 4

    return min(requested, safe_cap, pop_size)


def _build_archive_snapshot(archive: MAPElitesArchive) -> list[dict]:
    snapshot = []
    for (size_idx, com_idx), (loss, tree) in archive.archive.items():
        snapshot.append(
            {
                "Size": size_idx,
                "Center of Mass": com_idx / archive.com_bins,
                "Loss": loss,
                "Formula": str(tree),
            }
        )
    return snapshot


def _build_lr_curves(hof: list, archive: MAPElitesArchive) -> list[dict]:
    curves = []
    for i, (loss, tree) in enumerate(hof):
        try:
            lrs = symbolr_rust.evaluate_fast(tree.to_prefix(), archive.t_array)
            for t_val, lr in zip(archive.t_array, lrs):
                curves.append({"Time": t_val, "LR": lr, "Rank": f"#{i+1}  loss={loss:.3f}"})
        except Exception:
            continue
    return curves


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
) -> None:
    _store_run_state(
        run_id,
        gen_log=copy.deepcopy(gen_log),
        hof=copy.deepcopy(hof),
        archive_snapshot=_build_archive_snapshot(archive),
        lr_curves=_build_lr_curves(hof, archive),
        total_time=round(total_time, 1),
        progress_ratio=float(progress_ratio),
        progress_label=progress_label,
        phase_label=phase_label,
        eval_completed=int(eval_completed),
        eval_total=int(eval_total),
        evolution_done=evolution_done,
    )


def _run_evolution_job(run_id: str, gen_count: int, pop_size: int, epochs: int, workers: int) -> None:
    started_at = time.time()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        effective_workers = _resolve_worker_budget(device, workers, epochs, pop_size)

        _store_run_state(
            run_id,
            run_status="running",
            device_type=device.type,
            effective_workers=effective_workers,
            progress_label="Seeding initial population",
            phase_label="Generation 0",
        )

        fidelity = FidelityManager(seed=42)
        train_loader, val_loader = fidelity.get_low_fidelity(device, batch_size=256)
        trainer = ProbeTrainer(device=device, patience=2, amp_enabled=True)
        evaluator = ParallelEvaluator(trainer, train_loader, val_loader, epochs=epochs, time_steps=100)
        archive = MAPElitesArchive(size_bins=30, com_bins=20, time_steps=100)
        model_factory = lambda: create_compiled_model(device, in_channels=1)

        gen_log: list[dict] = []
        hof: list = []

        population = ramped_half_and_half(pop_size, min_depth=2, max_depth=4)
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
        )

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
                parents = ramped_half_and_half(pop_size, min_depth=2, max_depth=4)

            offspring = []
            for _ in range(pop_size // 2):
                p1, p2 = random.choice(parents), random.choice(parents)
                r = random.random()
                if r < 0.50:
                    o1, o2 = subtree_crossover(p1, p2)
                    offspring.extend([o1, o2])
                elif r < 0.75:
                    offspring.extend([subtree_mutation(p1, 3), subtree_mutation(p2, 3)])
                else:
                    offspring.extend([hoist_mutation(p1), hoist_mutation(p2)])

            offspring = [simplify_tree(ind) for ind in offspring]
            fitnesses = evaluator.evaluate_population(
                offspring,
                model_factory,
                max_workers=effective_workers,
                progress_callback=lambda completed, total, gen=gen: _store_run_state(
                    run_id,
                    eval_completed=completed,
                    eval_total=total,
                    phase_label=f"Generation {gen}: {completed}/{total} candidates evaluated",
                ),
            )

            new_niches = 0
            for ind, fit in zip(offspring, fitnesses):
                if archive.try_add(ind, fit):
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
                phase_label=f"Generation {gen} complete",
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


def start_evolution(gen_count: int, pop_size: int, epochs: int, workers: int) -> bool:
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

