import pytest
import numpy as np
from gp.tree import Node
from optimiser.hybrid import _get_constant_nodes, hybrid_optimize_constants

def test_get_constant_nodes():
    """
    Validates that only float terminals are extracted
    """
    tree = Node("*", [
        Node("+", [Node("t"), Node(0.5)]),
        Node(0.1)
    ])

    const_nodes = _get_constant_nodes(tree)
    
    assert len(const_nodes) == 2
    assert const_nodes[0].value == 0.5
    assert const_nodes[1].value == 0.1

def test_hybrid_optimization_no_constants():
    """If a tree has no constants, it should return instantly without crashing."""
    tree = Node("sin", [Node("t")])
    
    # Mock fitness function that just returns 1.0
    optimized = hybrid_optimize_constants(tree, lambda t: 1.0)
    
    assert optimized.value == "sin"
    assert optimized.children[0].value == "t"
    assert optimized is not tree  # Should still return a deepcopy

def test_hybrid_optimization_convergence():
    """
    Provides a synthetic parabolic fitness landscape.
    L-BFGS-B should find the global minimum easily.
    """
    # Tree: (C1 + C2) - we just use a dummy structure
    tree = Node("+", [Node(1.0), Node(2.0)])
    
    # Objective: Minimize (C1 - 0.25)^2 + (C2 - 0.75)^2
    # The optimal constants should become exactly 0.25 and 0.75
    def mock_fitness_fn(t: Node) -> float:
        c1 = t.children[0].value
        c2 = t.children[1].value
        return (c1 - 0.25)**2 + (c2 - 0.75)**2
        
    optimized = hybrid_optimize_constants(
        tree, 
        fitness_fn=mock_fitness_fn, 
        bounds=(0.0, 5.0), 
        maxiter=20
    )
    
    c1_opt = optimized.children[0].value
    c2_opt = optimized.children[1].value
    
    np.testing.assert_allclose(c1_opt, 0.25, atol=1e-3)
    np.testing.assert_allclose(c2_opt, 0.75, atol=1e-3)


def test_hybrid_optimization_bounds_clipping():
    """Ensures L-BFGS-B respects our mathematical bounds constraint."""
    tree = Node(5.0)
    
    # Objective: Minimize C1. Without bounds, it would go to negative infinity.
    def minimize_c1(t: Node) -> float:
        return t.value
        
    optimized = hybrid_optimize_constants(
        tree, 
        fitness_fn=minimize_c1, 
        bounds=(1e-6, 10.0)
    )
    
    # It should clip precisely to the lower bound
    assert optimized.value == 1e-6
