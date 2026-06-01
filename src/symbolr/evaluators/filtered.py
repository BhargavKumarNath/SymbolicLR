"""
src/symbolr/evaluators/filtered.py — Terminal-set filter for ablation studies.

TokenFilteredEvaluator wraps any BaseEvaluator and returns float('inf') for
formulas that contain any token in `forbidden_tokens`. This controls the
*effective* terminal set from Python without modifying the Rust engine.

The Rust engine will still generate formulas with g/dl nodes and place them
in the appropriate gradient-sensitivity niches, but with fitness=inf they
will never displace time-only formulas. The result is equivalent to running
evolution with a restricted terminal set.

Usage in ablation studies:
    # Config A: time-only terminal set
    filtered_A = TokenFilteredEvaluator(base, forbidden={"g", "dl"})

    # Config B: t + gradient norm
    filtered_B = TokenFilteredEvaluator(base, forbidden={"dl"})

    # Config C: full terminal set (no filter)
    filtered_C = base  # or TokenFilteredEvaluator(base, forbidden=set())
"""
from __future__ import annotations

from src.symbolr.core.evaluator import BaseEvaluator


class TokenFilteredEvaluator(BaseEvaluator):
    """
    Wraps any BaseEvaluator and assigns fitness=inf to formulas that contain
    one or more tokens from `forbidden_tokens`.

    Batches are split: forbidden formulas get inf immediately; the remaining
    formulas are forwarded to the wrapped evaluator as a single batch so the
    base evaluator's internal batching is preserved.

    Args:
        base_evaluator:  Any BaseEvaluator subclass.
        forbidden_tokens: Token strings (prefix notation) to exclude.
                          Example: {"g", "dl"} to forbid gradient variables.
    """

    def __init__(
        self,
        base_evaluator: BaseEvaluator,
        forbidden_tokens: set[str],
    ) -> None:
        self.base     = base_evaluator
        self.forbidden = frozenset(forbidden_tokens)

    @property
    def is_deterministic(self) -> bool:
        return self.base.is_deterministic

    @property
    def name(self) -> str:
        forbidden_sorted = sorted(self.forbidden)
        return f"TokenFiltered[forbidden={forbidden_sorted}]({self.base.name})"

    def evaluate(self, formulas: list[str]) -> list[float]:
        """
        Evaluate formulas, returning inf for any that contain a forbidden token.

        Allowed formulas are forwarded as a single batch to preserve the base
        evaluator's vectorization.
        """
        fitnesses: list[float] = [float("inf")] * len(formulas)

        # Partition into allowed and forbidden
        allowed_positions: list[int] = []
        allowed_formulas:  list[str] = []

        for i, fstr in enumerate(formulas):
            tokens = set(fstr.split())
            if not (tokens & self.forbidden):
                allowed_positions.append(i)
                allowed_formulas.append(fstr)

        if allowed_formulas:
            base_results = self.base.evaluate(allowed_formulas)
            for pos, result in zip(allowed_positions, base_results):
                fitnesses[pos] = result

        return fitnesses

    @property
    def n_forbidden_tokens(self) -> int:
        return len(self.forbidden)

    def allows(self, formula: str) -> bool:
        """Return True if the formula contains no forbidden tokens."""
        return not (set(formula.split()) & self.forbidden)
