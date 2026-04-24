import copy
import numpy as np
from typing import Callable, List, Tuple
from scipy.optimize import minimize
from gp.tree import Node

def _get_constant_nodes(node: Node) -> List[Node]:
    """
    Recursively traverse the AST to extract references to all terminal constant nodes.

    Args:
        node: The root or current node of the AST
    
    Returns:
        List[Node]: A list of nodes where the value is a float constant.
    """
    const_nodes = []
    if not node.children and isinstance(node.value, float):
        const_nodes.append(node)

    for child in node.children:
        const_nodes.extend(_get_constant_nodes(child))
    
    return const_nodes

def _clear_caches_recursively(node: Node) -> None:
    """
    Invokes the custom cache invalidation on the node and all children
    """
    if hasattr(node, 'invalidate_cache'):
        node.invalidate_cache()
    else:
        node._hash_cache = None
    
    for child in node.children:
        _clear_caches_recursively(child)

def hybrid_optimize_constants(
    tree: Node,
    fitness_fn: Callable[[Node], float],
    bounds: Tuple[float, float] = (1e-6, 10.0),
    maxiter: int = 15
) -> Node:
    """
    Extracts constants from an AST, uses L-BFGS-B to optimize them against a fitness function,
    and returns a newly optimized tree.
    
    Args:
        tree: The original AST to optimize.
        fitness_fn: A callback that evaluates the tree and returns a float loss.
        bounds: Optimization constraints to prevent mathematically absurd learning rates.
        maxiter: Maximum iterations for the L-BFGS-B solver to keep evolution fast.
        
    Returns:
        Node: A deepcopy of the original tree with optimized constants.
    """
    optimized_tree = copy.deepcopy(tree)
    constant_nodes = _get_constant_nodes(optimized_tree)

    # If the equation has no constants (e.g., just "t" or "sin(t)"), exit early.
    if not constant_nodes:
        return optimized_tree
    
    initial_guess = np.array([n.value for n in constant_nodes], dtype=np.float64)
    scipy_bounds = [bounds for _ in constant_nodes]

     # Define the objective function for SciPy
    def objective(x: np.ndarray) -> float:
        # 1. Inject the new SciPy-proposed constants back into the AST
        for n, val in zip(constant_nodes, x):
            n.value = float(val)
            
        # 2. Clear tree cache so the fitness function sees the updated values
        _clear_caches_recursively(optimized_tree)
        
        # 3. Evaluate using the provided callback (e.g., ProbeTrainer)
        return fitness_fn(optimized_tree)

    # Execute bounded gradient descent
    res = minimize(
        objective,
        initial_guess,
        method='L-BFGS-B',
        bounds=scipy_bounds,
        options={'maxiter': maxiter}
    )
    
    # Ensure the tree definitively holds the absolute best values found
    lower, upper = bounds

    for n, val in zip(constant_nodes, res.x):
        # Force value inside bounds
        val = max(min(val, upper), lower)

        # Snap exactly to boundary if very close
        if abs(val - lower) < 1e-12:
            val = lower
        elif abs(val - upper) < 1e-12:
            val = upper

        n.value = float(val)
    _clear_caches_recursively(optimized_tree)
    
    return optimized_tree
