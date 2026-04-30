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


def _clear_all_caches(node: Node) -> None:
    """Recursively invokes the user-defined cache invalidation logic."""
    if hasattr(node, 'invalidate_cache'):
        node.invalidate_cache()
    else:
        node._hash_cache = None
        if hasattr(node, 'fitness'):
            node.fitness = None
        for child in node.children:
            _clear_all_caches(child)


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


def subtree_crossover(parent1: Node, parent2: Node, max_depth: int = 7) -> Tuple[Node, Node]:
    """
    Performs standard GP subtree crossover by swapping two random subtrees.
    
    Args:
        parent1, parent2: The root nodes of the parent ASTs.
        
    Returns:
        Tuple[Node, Node]: Two new offspring ASTs.
    """
    p1 = copy.deepcopy(parent1)
    p2 = copy.deepcopy(parent2)
    
    nodes1 = _get_all_nodes(p1)
    nodes2 = _get_all_nodes(p2)
    
    n1 = random.choice(nodes1)
    n2 = random.choice(nodes2)
    
    n1.value, n2.value = n2.value, n1.value
    n1.children, n2.children = n2.children, n1.children
    
    _clear_all_caches(p1)
    _clear_all_caches(p2)
    
    if p1.depth() > max_depth: p1 = copy.deepcopy(parent1)
    if p2.depth() > max_depth: p2 = copy.deepcopy(parent2)
    return p1, p2


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
    if p.depth() > max_depth: return copy.deepcopy(parent)
    return p


def hoist_mutation(parent: Node) -> Node:
    """
    Anti-bloat mutation: Replaces a subtree with one of its own descendant nodes.
    This guarantees the offspring will be strictly smaller than the parent.
    """
    p = copy.deepcopy(parent)
    nodes = _get_all_nodes(p)
    
    # Only operators can have children to hoist from
    internal_nodes =[n for n in nodes if n.children]
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
