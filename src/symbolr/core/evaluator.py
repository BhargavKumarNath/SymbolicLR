from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """
    Protocol for all SymboLR evaluators.

    Evaluators take a batch of AST formula strings in prefix notation,
    evaluate their fitness, and return a list of fitness scores (lower = better).

    Subclasses must implement `evaluate`. Optionally override `is_deterministic`
    and `name` to support experiment logging and reproducibility checks.
    """

    @abstractmethod
    def evaluate(self, formulas: list[str]) -> list[float]:
        """
        Evaluate a batch of formulas and return their fitness scores.

        Args:
            formulas: Mathematical formulas in prefix notation.

        Returns:
            Fitness scores (lower = better). Non-viable formulas return float('inf').
        """
        ...

    @property
    def is_deterministic(self) -> bool:
        """True if identical inputs always produce identical outputs."""
        return False

    @property
    def name(self) -> str:
        return self.__class__.__name__
