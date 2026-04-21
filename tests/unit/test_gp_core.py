import pytest
import numpy as np
from gp.operators import OPERATORS
from gp.tree import Node

def test_protected_division():
    div = OPERATORS["/"].func
    a = np.array([10.0, 5.0, 1.0])
    b = np.array([2.0, 0.0, 1e-7]) # 0.0 and 1e-7 should trigger protection
    
    res = div(a, b)
    assert res[0] == 5.0
    assert res[1] == 1.0 # Protected from div-by-zero
    assert res[2] == 1.0 # Protected from ultra-small denominators


def test_protected_log_and_sqrt():
    log = OPERATORS["log"].func
    sqrt = OPERATORS["sqrt"].func
    
    a = np.array([-1.0, 0.0, 4.0])
    
    # Log shouldn't NaN on negative or zero
    res_log = log(a)
    assert not np.isnan(res_log).any()
    
    # Sqrt shouldn't NaN on negative
    res_sqrt = sqrt(a)
    assert res_sqrt[0] == 1.0 # sqrt(abs(-1))
    assert res_sqrt[2] == 2.0


def test_protected_exp():
    exp = OPERATORS["exp"].func
    a = np.array([1000.0]) # Massive exponent
    
    res = exp(a)
    assert np.isfinite(res).all(), "Exponential failed to clip and overflowed to inf."
    np.testing.assert_allclose(res[0], np.exp(10.0)) # Clipped to 10


def test_tree_evaluation():
    """Builds the tree 't + 0.5' and evaluates it."""
    # (t + 0.5)
    t_node = Node("t")
    const_node = Node(0.5)
    add_node = Node("+",[t_node, const_node])
    
    t_array = np.array([0.0, 0.5, 1.0])
    res = add_node.evaluate(t_array)
    
    np.testing.assert_array_almost_equal(res, np.array([0.5, 1.0, 1.5]))


def test_tree_hashing_and_string():
    """Identical trees must produce the exact same MD5 cache key."""
    # Tree 1: sin(t) * 2.0
    t1 = Node("*", [Node("sin", [Node("t")]), Node(2.0)])
    
    # Tree 2: sin(t) * 2.0
    t2 = Node("*", [Node("sin", [Node("t")]), Node(2.0)])
    
    # Verify string construction
    assert str(t1) == "(sin(t) * 2.0000)"
    
    # Verify deterministic hashing
    assert t1.get_hash() == t2.get_hash(), "Identical ASTs produced different hashes."


def test_tree_metrics():
    """Tests accurate reporting of depth and size."""
    # Tree: (t + (0.5 * t))
    tree = Node("+", [
        Node("t"),
        Node("*",[Node(0.5), Node("t")])
    ])
    
    assert tree.size() == 5
    assert tree.depth() == 3