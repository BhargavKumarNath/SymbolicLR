import torch

from ui.state import _resolve_worker_budget


def test_worker_budget_caps_large_gpu_runs():
    """Large GPU workloads should be capped to avoid freezing the dashboard."""
    workers = _resolve_worker_budget(torch.device("cuda"), requested_workers=6, epochs=5, pop_size=100)
    assert workers == 2


def test_worker_budget_respects_small_cpu_runs():
    """CPU runs should still honor modest user-requested parallelism."""
    workers = _resolve_worker_budget(torch.device("cpu"), requested_workers=3, epochs=2, pop_size=20)
    assert workers == 3
