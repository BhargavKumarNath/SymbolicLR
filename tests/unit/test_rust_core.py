import pytest
import numpy as np
from gp.tree import Node
import symbolr_rust

def test_prefix_conversion():
    """Validates that ASTs accurately serialize into space-separated prefix strings."""
    # Tree: (t + 0.5) * exp(t)
    # Prefix should be: * + t 0.5 exp t
    tree = Node("*", [
        Node("+",[Node("t"), Node(0.5)]),
        Node("exp", [Node("t")])
    ])
    
    prefix = tree.to_prefix()
    assert prefix == "* + t 0.5 exp t"


def test_rust_vs_python_parity():
    """
    Validates that the highly-optimized Rust extension returns EXACTLY the same 
    protected numerical results as the Python NumPy implementations.
    """
    # Extremely aggressive formula testing all protected bounds
    # log(abs(t)) / sqrt(-0.5) + exp(1000.0 * t)
    tree = Node("+", [
        Node("/", [
            Node("log", [Node("abs",[Node("t")])]),
            Node("sqrt", [Node(-0.5)])
        ]),
        Node("exp", [Node("*", [Node(1000.0), Node("t")])])
    ])
    
    prefix = tree.to_prefix()
    t_array = np.linspace(-1.0, 1.0, 100, dtype=np.float64)
    
    # 1. Evaluate via Python NumPy rules
    py_result = tree.evaluate(t_array)
    py_result = np.nan_to_num(py_result, nan=1.0, posinf=1.0, neginf=1.0)
    
    # 2. Evaluate via compiled Rust Crate
    rust_result = symbolr_rust.evaluate_fast(prefix, t_array)
    
    # Require 1e-7 floating-point parity
    np.testing.assert_allclose(py_result, rust_result, atol=1e-7, err_msg="Rust core drifted from Python implementation!")


