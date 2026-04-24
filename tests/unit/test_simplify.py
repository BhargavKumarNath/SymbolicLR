import pytest
from gp.tree import Node
from gp.simplify import _node_to_sympy, _sympy_to_node, simplify_tree, tree_to_latex

def test_algebraic_identity_removal():
    """Validates that addition by zero and multiplication by one are pruned."""
    # Tree: (t + 0.0) * 1.0
    bloated_tree = Node("*", [
        Node("+",[Node("t"), Node(0.0)]),
        Node(1.0)
    ])
    
    assert bloated_tree.size() == 5
    
    pruned = simplify_tree(bloated_tree)
    
    # Must reduce perfectly down to just "t"
    assert pruned.value == "t"
    assert pruned.size() == 1


def test_algebraic_cancellation():
    """Validates that self-division and self-subtraction cancel out to constants."""
    # Tree: (t / t) + (t - t)
    bloated_tree = Node("+", [
        Node("/", [Node("t"), Node("t")]),
        Node("-",[Node("t"), Node("t")])
    ])
    
    pruned = simplify_tree(bloated_tree)
    
    # t/t = 1, t-t = 0. So 1 + 0 = 1.0
    assert pruned.value == 1.0
    assert pruned.size() == 1


def test_sympy_pow_handling():
    """Ensures division is safely managed through SymPy's fractional exponent mappings."""
    # Tree: 1.0 / t
    div_tree = Node("/",[Node(1.0), Node("t")])
    
    pruned = simplify_tree(div_tree)
    
    # The structure must remain functionally a division, not crash
    assert pruned.value == "/"
    assert pruned.children[1].value == "t"


def test_simplification_fallback_on_error():
    """If the mapping fails, it MUST return the original tree untouched to protect the system."""
    # Create an illegally formatted node
    bad_tree = Node("unsupported_op", [Node("t")])
    
    pruned = simplify_tree(bad_tree)
    
    # Instead of throwing a crash that stops a 4-hour evolution loop, it returns the original
    assert pruned.value == "unsupported_op"


def test_latex_generation():
    """Tests that the UI wrapper returns a valid LaTeX string."""
    tree = Node("/", [Node("sin",[Node("t")]), Node(2.0)])
    
    latex_str = tree_to_latex(tree)
    
    assert "\\" in latex_str  # Checks for standard LaTeX syntax (e.g. \frac or \sin)
    assert "t" in latex_str
    