import numpy as np
from typing import Callable, Dict, Any
from dataclasses import dataclass

@dataclass
class Operator:
    """Represents a mathematical operation within the GP syntax tree"""
    name: str
    arity: int
    func: Callable

def _protected_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Protected division: returns 1.0 if the denominator is too small."""
    safe_b = np.where(np.abs(b) < 1e-6, 1.0, b)
    return np.where(np.abs(b) < 1e-6, 1.0, a / safe_b)


def _protected_log(a: np.ndarray) -> np.ndarray:
    """Protected natural logarithm: works on absolute values, capped at lower bound."""
    return np.log(np.maximum(1e-6, np.abs(a)))


def _protected_sqrt(a: np.ndarray) -> np.ndarray:
    """Protected square root: operates on the absolute value."""
    return np.sqrt(np.abs(a))


def _protected_exp(a: np.ndarray) -> np.ndarray:
    """Protected exponential: clips the exponent to prevent inf/NaN overflow."""
    # Clip to max 10 to keep learning rates from jumping to astronomical values
    return np.exp(np.clip(a, -100.0, 10.0))

# Global Registry of Available GP Operations
OPERATORS: Dict[str, Operator] = {
    # Binary
    "+": Operator("+", 2, np.add),
    "-": Operator("-", 2, np.subtract),
    "*": Operator("*", 2, np.multiply),
    "/": Operator("/", 2, _protected_div),
    # Unary
    "sin": Operator("sin", 1, np.sin),
    "cos": Operator("cos", 1, np.cos),
    "exp": Operator("exp", 1, _protected_exp),
    "log": Operator("log", 1, _protected_log),
    "sqrt": Operator("sqrt", 1, _protected_sqrt),
    "abs": Operator("abs", 1, np.abs)

}


