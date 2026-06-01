"""
Phase 6 tests — terminal-set ablation and engine integration.

Tests are organized into:
  1. TokenFilteredEvaluator — filter mechanics and batch preservation
  2. GenerationResult — gradient_sensitivity_mean now parsed from JSON
  3. AblationRunner   — single-config and multi-config runs
  4. AblationResult   — summary and serialization
  5. Experiment integration — evolve CLI wires gradient_aware evaluator
"""
import importlib.util
import json
import math
import os
import tempfile

import numpy as np
import pytest

RUST_AVAILABLE = importlib.util.find_spec("symbolr_rust") is not None

# 1. TokenFilteredEvaluator
class _ConstantEvaluator:
    """Minimal BaseEvaluator stub for filter tests."""
    is_deterministic = True
    name = "Constant"

    def __init__(self, value=0.5):
        self._value = value

    def evaluate(self, formulas):
        return [self._value] * len(formulas)


def test_filter_allows_time_only_formulas():
    """Formulas with only t/constants must pass through unfiltered."""
    from src.symbolr.evaluators.filtered import TokenFilteredEvaluator

    base   = _ConstantEvaluator(value=0.3)
    filt   = TokenFilteredEvaluator(base, forbidden_tokens={"g", "dl"})
    result = filt.evaluate(["cos * 3.14159 t", "0.01", "* t 0.5"])
    assert result == [0.3, 0.3, 0.3]


def test_filter_blocks_gradient_token():
    """Formulas containing 'g' must receive fitness=inf."""
    from src.symbolr.evaluators.filtered import TokenFilteredEvaluator

    base   = _ConstantEvaluator(value=0.3)
    filt   = TokenFilteredEvaluator(base, forbidden_tokens={"g", "dl"})
    result = filt.evaluate(["* 0.01 exp * -1 g"])
    assert result == [float("inf")]


def test_filter_blocks_dl_token():
    """Formulas containing 'dl' must receive fitness=inf."""
    from src.symbolr.evaluators.filtered import TokenFilteredEvaluator

    base   = _ConstantEvaluator(value=0.3)
    filt   = TokenFilteredEvaluator(base, forbidden_tokens={"dl"})
    result = filt.evaluate(["* 0.01 + 1 dl"])
    assert result == [float("inf")]


def test_filter_mixed_batch():
    """Mixed batch: allowed formulas get base fitness; forbidden get inf."""
    from src.symbolr.evaluators.filtered import TokenFilteredEvaluator

    base   = _ConstantEvaluator(value=0.7)
    filt   = TokenFilteredEvaluator(base, forbidden_tokens={"g", "dl"})

    formulas = ["0.01", "* 0.01 g", "cos * 3.14 t", "+ t dl"]
    results  = filt.evaluate(formulas)

    assert results[0] == 0.7           # "0.01" — allowed
    assert results[1] == float("inf")  # "* 0.01 g" — forbidden
    assert results[2] == 0.7           # "cos * 3.14 t" — allowed
    assert results[3] == float("inf")  # "+ t dl" — forbidden


def test_filter_empty_forbidden_passes_all():
    """Empty forbidden set must let all formulas through to the base."""
    from src.symbolr.evaluators.filtered import TokenFilteredEvaluator

    base   = _ConstantEvaluator(value=0.4)
    filt   = TokenFilteredEvaluator(base, forbidden_tokens=set())
    result = filt.evaluate(["0.01", "* 0.01 g", "+ t dl"])
    assert result == [0.4, 0.4, 0.4]


def test_filter_empty_batch():
    """Empty formula batch returns empty list."""
    from src.symbolr.evaluators.filtered import TokenFilteredEvaluator

    filt = TokenFilteredEvaluator(_ConstantEvaluator(), forbidden_tokens={"g"})
    assert filt.evaluate([]) == []


def test_filter_name_contains_forbidden():
    """name property must expose the forbidden token set."""
    from src.symbolr.evaluators.filtered import TokenFilteredEvaluator

    filt = TokenFilteredEvaluator(_ConstantEvaluator(), forbidden_tokens={"g", "dl"})
    assert "dl" in filt.name
    assert "g"  in filt.name


def test_filter_allows_method():
    """allows() must return True iff formula has no forbidden tokens."""
    from src.symbolr.evaluators.filtered import TokenFilteredEvaluator

    filt = TokenFilteredEvaluator(_ConstantEvaluator(), forbidden_tokens={"g"})
    assert filt.allows("cos * 3.14 t")
    assert not filt.allows("* 0.01 g")


def test_filter_preserves_base_batch_call():
    """All allowed formulas must be forwarded as a single batch."""
    from src.symbolr.evaluators.filtered import TokenFilteredEvaluator

    calls = []

    class BatchRecorder:
        is_deterministic = True
        name = "Recorder"

        def evaluate(self, formulas):
            calls.append(list(formulas))
            return [0.5] * len(formulas)

    filt = TokenFilteredEvaluator(BatchRecorder(), forbidden_tokens={"g"})
    filt.evaluate(["0.01", "* g t", "cos t", "dl"])
    # Allowed: ["0.01", "cos t", "dl"] — one batch call
    assert len(calls) == 1
    assert "* g t" not in calls[0]
    assert "0.01" in calls[0]


# 2. GenerationResult — gradient_sensitivity_mean
def test_generation_result_parses_gradient_sensitivity_mean():
    """from_json must populate gradient_sensitivity_mean from the tell() JSON."""
    from src.symbolr.core.bridge import GenerationResult

    payload = json.dumps({
        "generation_number": 3,
        "best_mse": 0.12,
        "average_mse": 0.25,
        "top_formula_latex": "t",
        "top_formula_prefix": "t",
        "archive_size": 42,
        "new_entries": 5,
        "gen_time_ms": 80,
        "gradient_sensitivity_mean": 0.37,
    })
    result = GenerationResult.from_json(payload)
    assert abs(result.gradient_sensitivity_mean - 0.37) < 1e-9


def test_generation_result_defaults_sensitivity_to_zero():
    """Missing gradient_sensitivity_mean field must default to 0.0."""
    from src.symbolr.core.bridge import GenerationResult

    payload = json.dumps({
        "generation_number": 1,
        "best_mse": 0.5,
        "average_mse": 0.6,
        "top_formula_latex": "t",
        "top_formula_prefix": "t",
        "archive_size": 10,
        "new_entries": 2,
        "gen_time_ms": 50,
        # no gradient_sensitivity_mean key
    })
    result = GenerationResult.from_json(payload)
    assert result.gradient_sensitivity_mean == 0.0


def test_generation_result_to_dict_includes_sensitivity():
    """to_dict() must include gradient_sensitivity_mean."""
    from src.symbolr.core.bridge import GenerationResult

    payload = json.dumps({
        "generation_number": 1, "best_mse": 0.5, "average_mse": 0.6,
        "top_formula_latex": "t", "top_formula_prefix": "t",
        "archive_size": 10, "new_entries": 2, "gen_time_ms": 50,
        "gradient_sensitivity_mean": 0.12,
    })
    d = GenerationResult.from_json(payload).to_dict()
    assert "gradient_sensitivity_mean" in d
    assert abs(d["gradient_sensitivity_mean"] - 0.12) < 1e-9


# 3. AblationRunner
@pytest.mark.skipif(not RUST_AVAILABLE, reason="symbolr_rust not compiled")
def test_ablation_runner_single_config():
    """run_single() must return an AblationRun with correct config name."""
    from src.symbolr.core.ablation import AblationRunner, ABLATION_CONFIGS
    from src.symbolr.evaluators.synthetic import SyntheticEvaluator

    runner = AblationRunner(
        base_evaluator  = SyntheticEvaluator(time_steps=30),
        max_generations = 3,
        pop_size        = 20,
        seed            = 0,
        run_benchmark   = False,
    )
    run = runner.run_single(ABLATION_CONFIGS[0])  # t_only

    assert run.config_name == "t_only"
    assert math.isfinite(run.best_fitness)
    assert run.final_archive_size >= 0
    assert len(run.generation_log) == 3


@pytest.mark.skipif(not RUST_AVAILABLE, reason="symbolr_rust not compiled")
def test_ablation_runner_t_only_never_uses_g_or_dl():
    """
    Under t_only config, every top_formula in generation_log must not
    contain 'g' or 'dl' tokens (since they all get fitness=inf).
    """
    from src.symbolr.core.ablation import AblationRunner, ABLATION_CONFIGS
    from src.symbolr.evaluators.synthetic import SyntheticEvaluator

    runner = AblationRunner(
        base_evaluator  = SyntheticEvaluator(time_steps=30),
        max_generations = 5,
        pop_size        = 30,
        seed            = 42,
        run_benchmark   = False,
    )
    run = runner.run_single(ABLATION_CONFIGS[0])  # t_only

    for entry in run.generation_log:
        formula = entry["top_formula"]
        tokens  = set(formula.split())
        assert "g"  not in tokens, f"t_only top formula contains 'g': {formula}"
        assert "dl" not in tokens, f"t_only top formula contains 'dl': {formula}"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="symbolr_rust not compiled")
def test_ablation_runner_run_all_returns_three_runs():
    """run_all() must return one AblationRun per config."""
    from src.symbolr.core.ablation import AblationRunner, AblationResult
    from src.symbolr.evaluators.synthetic import SyntheticEvaluator

    runner = AblationRunner(
        base_evaluator  = SyntheticEvaluator(time_steps=30),
        max_generations = 2,
        pop_size        = 20,
        seed            = 0,
        run_benchmark   = False,
    )
    result = runner.run_all()

    assert isinstance(result, AblationResult)
    assert set(result.runs.keys()) == {"t_only", "t_g", "t_g_dl"}


@pytest.mark.skipif(not RUST_AVAILABLE, reason="symbolr_rust not compiled")
def test_ablation_runner_configs_use_same_seed():
    """
    t_only and t_g_dl must start from the same random population (same seed).
    Generation 1 archive size and gen_time should be comparable.
    """
    from src.symbolr.core.ablation import AblationRunner, ABLATION_CONFIGS
    from src.symbolr.evaluators.synthetic import SyntheticEvaluator

    runner = AblationRunner(
        base_evaluator  = SyntheticEvaluator(time_steps=30),
        max_generations = 2,
        pop_size        = 20,
        seed            = 7,
        run_benchmark   = False,
    )

    run_only  = runner.run_single(ABLATION_CONFIGS[0])  # t_only
    run_full  = runner.run_single(ABLATION_CONFIGS[2])  # t_g_dl

    # Both start with the same generated population (same Rust seed)
    # so their gen-1 archive sizes should be comparable (within 2x)
    size_only = run_only.generation_log[0]["archive_size"]
    size_full = run_full.generation_log[0]["archive_size"]
    assert max(size_only, size_full) <= 2 * max(min(size_only, size_full), 1)


# 4. AblationResult — serialization
@pytest.mark.skipif(not RUST_AVAILABLE, reason="symbolr_rust not compiled")
def test_ablation_result_to_dict_has_required_keys():
    """to_dict() must include runs, total_elapsed_sec, and summary."""
    from src.symbolr.core.ablation import AblationRunner
    from src.symbolr.evaluators.synthetic import SyntheticEvaluator

    runner = AblationRunner(
        SyntheticEvaluator(time_steps=30), max_generations=2, pop_size=10,
        seed=0, run_benchmark=False,
    )
    result = runner.run_all()
    d = result.to_dict()

    assert "runs"               in d
    assert "total_elapsed_sec"  in d
    assert "summary"            in d
    assert len(d["runs"])       == 3


@pytest.mark.skipif(not RUST_AVAILABLE, reason="symbolr_rust not compiled")
def test_ablation_result_save_json(tmp_path):
    """save_json() must produce valid JSON with correct top-level keys."""
    from src.symbolr.core.ablation import AblationRunner
    from src.symbolr.evaluators.synthetic import SyntheticEvaluator

    runner = AblationRunner(
        SyntheticEvaluator(time_steps=30), max_generations=2, pop_size=10,
        seed=0, run_benchmark=False,
    )
    result = runner.run_all()
    path   = str(tmp_path / "ablation.json")
    result.save_json(path)

    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert "runs" in data
    assert set(data["runs"].keys()) == {"t_only", "t_g", "t_g_dl"}


@pytest.mark.skipif(not RUST_AVAILABLE, reason="symbolr_rust not compiled")
def test_ablation_result_summary_sorted_by_fitness():
    """summary_rows() must be sorted by best_fitness ascending."""
    from src.symbolr.core.ablation import AblationRunner
    from src.symbolr.evaluators.synthetic import SyntheticEvaluator

    runner = AblationRunner(
        SyntheticEvaluator(time_steps=30), max_generations=3, pop_size=20,
        seed=42, run_benchmark=False,
    )
    result = runner.run_all()
    rows   = result.summary_rows()

    fitnesses = [r["best_fitness"] for r in rows if math.isfinite(r["best_fitness"])]
    assert fitnesses == sorted(fitnesses)


# 5. AblationConfig constants
def test_ablation_configs_are_three():
    """ABLATION_CONFIGS must have exactly three canonical configs."""
    from src.symbolr.core.ablation import ABLATION_CONFIGS
    assert len(ABLATION_CONFIGS) == 3


def test_ablation_config_names():
    """Config names must be exactly the three canonical identifiers."""
    from src.symbolr.core.ablation import ABLATION_CONFIGS
    names = {c.name for c in ABLATION_CONFIGS}
    assert names == {"t_only", "t_g", "t_g_dl"}


def test_t_only_config_forbids_g_and_dl():
    """t_only must forbid both g and dl."""
    from src.symbolr.core.ablation import ABLATION_CONFIGS
    t_only = next(c for c in ABLATION_CONFIGS if c.name == "t_only")
    assert "g"  in t_only.forbidden_tokens
    assert "dl" in t_only.forbidden_tokens


def test_t_g_config_forbids_only_dl():
    """t_g must forbid dl but not g."""
    from src.symbolr.core.ablation import ABLATION_CONFIGS
    t_g = next(c for c in ABLATION_CONFIGS if c.name == "t_g")
    assert "dl" in t_g.forbidden_tokens
    assert "g"  not in t_g.forbidden_tokens


def test_t_g_dl_config_forbids_nothing():
    """t_g_dl must have an empty forbidden set."""
    from src.symbolr.core.ablation import ABLATION_CONFIGS
    t_g_dl = next(c for c in ABLATION_CONFIGS if c.name == "t_g_dl")
    assert len(t_g_dl.forbidden_tokens) == 0
