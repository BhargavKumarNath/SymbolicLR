"""
config/settings.py - Central configuration with auto-detection of runtime capabilities.

Detects whether
PyTorch, CUDA, and the Rust extension are available, then selects the
appropriate runtime mode automatically.
"""

from __future__ import annotations

import os
import enum
from dataclasses import dataclass, field
from typing import Any, FrozenSet, Optional, Tuple


class RuntimeMode(enum.Enum):
    """Execution environment."""
    CLOUD_CPU = "cloud_cpu"       # Streamlit Cloud - no torch, no Rust
    LOCAL_CPU = "local_cpu"       # Local machine - torch on CPU
    LOCAL_GPU = "local_gpu"       # Local machine - torch with CUDA


class OperatorGroup(enum.Enum):
    """Operator categories for configurable search."""
    BASIC = "basic"           # +, -, *, /
    ADVANCED = "advanced"     # abs, sqrt
    NONLINEAR = "nonlinear"   # sin, cos, exp, log


# Default active groups for this stabilization phase
DEFAULT_OPERATOR_GROUPS: FrozenSet[str] = frozenset({"basic", "advanced", "nonlinear"})


def _detect_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _detect_rust() -> bool:
    try:
        import symbolr_rust  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_mode() -> RuntimeMode:
    if not _detect_torch():
        return RuntimeMode.CLOUD_CPU
    if _detect_cuda():
        return RuntimeMode.LOCAL_GPU
    return RuntimeMode.LOCAL_CPU


# Cache device detection at module level to avoid repeated import overhead
_cached_device: Optional[Any] = None
_device_detected: bool = False


def _get_device() -> Any:
    """Return the appropriate torch.device or a string placeholder. Cached."""
    global _cached_device, _device_detected
    if _device_detected:
        return _cached_device
    mode = _detect_mode()
    if mode == RuntimeMode.CLOUD_CPU:
        _cached_device = "cpu"
    else:
        import torch
        if mode == RuntimeMode.LOCAL_GPU:
            _cached_device = torch.device("cuda")
        else:
            _cached_device = torch.device("cpu")
    _device_detected = True
    return _cached_device


@dataclass(frozen=True)
class SymboLRConfig:
    """Immutable configuration snapshot."""

    # Runtime detection
    mode: RuntimeMode = field(default_factory=_detect_mode)
    torch_available: bool = field(default_factory=_detect_torch)
    cuda_available: bool = field(default_factory=_detect_cuda)
    rust_available: bool = field(default_factory=_detect_rust)

    # Evolution defaults
    default_generations: int = 5
    default_pop_size: int = 50
    default_epochs: int = 1
    default_workers: int = 4
    seed: int = 42

    # GP parameters
    min_tree_depth: int = 2
    max_tree_depth: int = 5
    max_tree_depth_limit: int = 7
    max_simplify_size: int = 30
    crossover_rate: float = 0.45
    mutation_rate: float = 0.25
    hoist_rate: float = 0.10
    point_mutation_rate: float = 0.10
    constant_perturbation_rate: float = 0.10

    # Operator configuration
    active_operator_groups: FrozenSet[str] = DEFAULT_OPERATOR_GROUPS

    # Population diversity
    immigrant_fraction: float = 0.12     # fraction of pop_size as random immigrants
    elite_injection_count: int = 3       # top-N elites injected each generation
    terminal_erc_range: Tuple[float, float] = (0.01, 2.0)  # ephemeral random constant range
    seed_constants: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)

    # Stagnation detection
    stagnation_threshold: int = 5        # generations without improvement to trigger recovery
    stagnation_mutation_boost: float = 0.5  # temporary mutation rate increase

    # Parsimony pressure
    parsimony_coefficient: float = 0.005  # penalty per node in fitness

    # MAP-Elites
    size_bins: int = 20
    com_bins: int = 15
    smoothness_bins: int = 10
    time_steps: int = 100
    archive_snapshot_cap: int = 500

    # Probe training
    patience: int = 2
    explode_threshold: float = 10.0
    lr_guard_max: float = 10.0
    batch_size: int = 256
    model_init_seeds: Tuple[int, ...] = (42, 137, 256)  # multi-seed averaging

    # Data fidelity
    low_fidelity_fraction: float = 0.05
    val_split: float = 0.20
    data_dir: str = "./data_cache"

    # Publish throttling
    publish_throttle_s: float = 1.0
    max_stale_runs: int = 4
    fragment_interval_s: float = 1.5

    # Synthetic fitness (cloud mode)
    synth_initial_loss: float = 2.3        # ~ log(10) for 10-class CE
    synth_landscape_curvature: float = 1.0
    synth_noise_scale: float = 0.02
    synth_n_dims: int = 5                  # parameter dimensions in synthetic landscape
    synth_n_evaluations: int = 3           # ensemble evaluations for noise reduction

    @property
    def is_cloud(self) -> bool:
        return self.mode == RuntimeMode.CLOUD_CPU

    @property
    def is_gpu(self) -> bool:
        return self.mode == RuntimeMode.LOCAL_GPU

    @property
    def device(self) -> Any:
        return _get_device()

    @property
    def amp_enabled(self) -> bool:
        return self.is_gpu

    def resolve_workers(self, requested: int, epochs: int, pop_size: int) -> int:
        """Cap concurrency to keep runs stable."""
        requested = max(1, int(requested))
        if self.is_cloud:
            return 1
        if not self.is_gpu:
            return min(requested, 4, pop_size)
        # GPU caps - conservative to prevent CUDA OOM
        if epochs >= 5 or pop_size >= 100:
            safe_cap = 1
        elif epochs >= 3 or pop_size >= 60:
            safe_cap = 2
        else:
            safe_cap = 3
        return min(requested, safe_cap, pop_size)


# Singleton
_config: Optional[SymboLRConfig] = None


def get_config() -> SymboLRConfig:
    """Return the global config singleton (created on first call)."""
    global _config
    if _config is None:
        _config = SymboLRConfig()
    return _config


def reset_config() -> None:
    """Reset the config singleton. Useful for testing."""
    global _config
    _config = None
