import random
import copy
from typing import List, Tuple
from .tree import Node
from .population import generate_tree

def _get_all_nodes(node: Node) -> List[Node]:
    """Helper method to flatten the AST into a list of Node references."""
    nodes = [node]
    for child in node.children:
        nodes.extend(_get_all_nodes(child))
    return nodes

def _get_all_nodes_with_parent(root: Node) -> List[Tuple[Node, Node, int]]:
    """Returns list of (node, parent, child_index) tuples for subtree grafting."""
    result = [(root, None, -1)]
    stack = [(root, None, -1)]
    # Use iterative approach
    while stack:
        node, parent, idx = stack.pop()
        for i, child in enumerate(node.children):
            result.append((child, node, i))
            stack.append((child, node, i))
    return result

def _clear_all_caches(node: Node) -> None:
    """Recursively invokes the user-defined cache invalidation logic."""
    if hasattr(node, 'invalidate_cache'):
        node.invalidate_cache()

def tournament_selection(population: List[Node], fitnesses: List[float], k: int = 3) -> Node:
    """
    Selects the best individual from a randomly chosen subset (tournament).
    
    Args:
        population: The entire current generation of trees.
        fitnesses: A parallel list of fitness scores (lower is better).
        k: Tournament size.
        
    Returns:
        Node: A deep copy of the tournament winner.
    """
    selected_indices = random.sample(range(len(population)), k)
    best_idx = min(selected_indices, key=lambda idx: fitnesses[idx])
    return copy.deepcopy(population[best_idx])

def subtree_crossover(parent1: Node, parent2: Node, max_depth: int = 7) -> Node:
    """
    Performs standard GP subtree crossover by grafting a subtree from parent2 into parent1.
    
    Args:
        parent1, parent2: The root nodes of the parent ASTs.
        max_depth: Maximum allowable depth for the new tree.
        
    Returns:
        Node: A new offspring AST.
    """
    p1 = copy.deepcopy(parent1)
    
    # Select a random crossover point in p1
    nodes1 = _get_all_nodes_with_parent(p1)
    node1, parent1_ref, child_idx = random.choice(nodes1)
    
    # Select a random subtree from p2 to donate
    nodes2 = _get_all_nodes(parent2)
    donor = copy.deepcopy(random.choice(nodes2))
    
    # Graft: replace node1 with donor in p1's tree
    if parent1_ref is None:
        p1 = donor  # replacing root
    else:
        parent1_ref.children[child_idx] = donor
        
    _clear_all_caches(p1)
    
    if p1.depth() > max_depth:
        return copy.deepcopy(parent1)
    return p1

def subtree_mutation(parent: Node, max_mutation_depth: int = 4, max_depth: int = 7) -> Node:
    """
    Replaces a randomly chosen node with a newly generated random subtree.
    """
    p = copy.deepcopy(parent)
    nodes = _get_all_nodes(p)
    n = random.choice(nodes)
    
    new_tree = generate_tree(1, max_mutation_depth, method='grow')
    
    n.value = new_tree.value
    n.children = new_tree.children
    
    _clear_all_caches(p)
    if p.depth() > max_depth: 
        return copy.deepcopy(parent)
    return p

def hoist_mutation(parent: Node) -> Node:
    """
    Anti-bloat mutation: Replaces a subtree with one of its own descendant nodes.
    This guarantees the offspring will be strictly smaller than the parent.
    """
    p = copy.deepcopy(parent)
    nodes = _get_all_nodes(p)
    
    # Only operators can have children to hoist from
    internal_nodes = [n for n in nodes if n.children]
    if not internal_nodes:
        return p
        
    n = random.choice(internal_nodes)
    
    # Get descendants of the chosen node (excluding itself)
    descendants = _get_all_nodes(n)[1:]
    if not descendants:
        return p
        
    chosen_descendant = random.choice(descendants)
    
    n.value = chosen_descendant.value
    n.children = chosen_descendant.children
    
    _clear_all_caches(p)
    return p

def point_mutation(parent: Node) -> Node:
    """Changes a single node's value to another of the same type (operator->operator, terminal->terminal)."""
    p = copy.deepcopy(parent)
    nodes = _get_all_nodes(p)
    n = random.choice(nodes)
    
    if n.children:  # operator node
        # Replace with operator of same arity
        current_arity = len(n.children)
        from .operators import OPERATORS
        candidates = [name for name, op in OPERATORS.items() if op.arity == current_arity and name != n.value]
        if candidates:
            n.value = random.choice(candidates)
    else:  # terminal node
        if isinstance(n.value, (int, float)):
            # Replace constant with different constant
            from .population import _random_terminal
            new = _random_terminal()
            n.value = new.value
        else:
            # Variable -> keep as 't' (only variable)
            pass
    
    _clear_all_caches(p)
    return p

def constant_perturbation(parent: Node, sigma: float = 0.15) -> Node:
    """Adds Gaussian noise to a random constant terminal. Enables fine-tuning."""
    p = copy.deepcopy(parent)
    nodes = _get_all_nodes(p)
    const_nodes = [n for n in nodes if isinstance(n.value, (int, float)) and not n.children]
    if not const_nodes:
        return p
    n = random.choice(const_nodes)
    perturbation = random.gauss(0, sigma)
    n.value = round(max(0.001, float(n.value) + perturbation), 4)
    _clear_all_caches(p)
    return p
