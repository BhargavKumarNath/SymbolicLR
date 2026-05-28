from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    """
    Protocol for all SymboLR evaluators.
    
    Evaluators are responsible for taking a batch of AST formula strings
    (in prefix notation), evaluating their fitness, and returning a list of 
    fitness scores (lower is better, e.g., validation loss).
    """

    @abstractmethod
    def evaluate(self, formulas: list[str]) -> list[float]:
        """
        Evaluate a batch of formulas and return their fitness scores.

        Args:
            formulas: A list of mathematical formulas in prefix notation.

        Returns:
            A list of fitness scores corresponding to each formula.
            Non-viable or failed formulas should return float('inf').
        """
        pass
