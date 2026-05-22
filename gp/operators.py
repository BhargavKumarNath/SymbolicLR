import numpy as np
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class Operator:
    """Represents a mathematical operation within the GP syntax tree"""
    name: str
    arity: int
    func: Callable

def _protected_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Protected division: preserves numerator sign information if denominator is too small."""
    mask = np.abs(b) < 1e-6
    safe_b = np.where(mask, 1.0, b)
    # If denominator is near zero, return a * sign(b) (or positive if b is exactly zero)
    fallback = a * np.sign(b + 1e-30)
    return np.where(mask, fallback, a / safe_b)

def _protected_log(a: np.ndarray) -> np.ndarray:
    """Protected natural logarithm: works on absolute values, capped at lower bound."""
    return np.log(np.maximum(1e-6, np.abs(a)))

def _protected_sqrt(a: np.ndarray) -> np.ndarray:
    """Protected square root: operates on the absolute value."""
    return np.sqrt(np.abs(a))

def _protected_exp(a: np.ndarray) -> np.ndarray:
    """Protected exponential: clips the exponent to prevent inf/NaN overflow."""
    # Clip to [-10, 5] to keep learning rates from jumping to astronomical values
    return np.exp(np.clip(a, -10.0, 5.0))

def _protected_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Protected multiplication: clips result to prevent cascading overflow."""
    return np.clip(a * b, -100.0, 100.0)

# Global Registry of Available GP Operations
OPERATORS: Dict[str, Operator] = {
    # Binary
    "+": Operator("+", 2, np.add),
    "-": Operator("-", 2, np.subtract),
    "*": Operator("*", 2, _protected_mul),
    "/": Operator("/", 2, _protected_div),
    # Unary
    "sin": Operator("sin", 1, np.sin),
    "cos": Operator("cos", 1, np.cos),
    "exp": Operator("exp", 1, _protected_exp),
    "log": Operator("log", 1, _protected_log),
    "sqrt": Operator("sqrt", 1, _protected_sqrt),
    "abs": Operator("abs", 1, np.abs)
}

OPERATOR_GROUPS = {
    'basic': ['+', '-', '*', '/'],
    'advanced': ['abs', 'sqrt'],
    'nonlinear': ['sin', 'cos', 'exp', 'log'],
}

def get_active_operators(groups: Optional[List[str]] = None) -> Dict[str, Operator]:
    """Returns a filtered dictionary of operators based on the requested groups."""
    if groups is None:
        return OPERATORS
    
    active_names = []
    for group in groups:
        if group in OPERATOR_GROUPS:
            active_names.extend(OPERATOR_GROUPS[group])
            
    return {name: op for name, op in OPERATORS.items() if name in active_names}
