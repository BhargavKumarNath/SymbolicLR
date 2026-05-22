import random
import math
from typing import List, Optional, Dict
from .tree import Node
from .operators import OPERATORS, get_active_operators, Operator

# Seed constants that appear in known good LR schedules
SEED_CONSTANTS = [0.0, 0.5, 1.0, 2.0]

def _random_terminal() -> Node:
    """Generate a terminal node: variable 't' or an ephemeral random constant."""
    r = random.random()
    if r < 0.3:
        return Node('t')  # 30% chance of variable
    elif r < 0.5:
        return Node(random.choice(SEED_CONSTANTS))  # 20% chance of seed constant
    else:
        # 50% ephemeral random constant in [0.001, 0.5] log-uniform
        # Log-uniform distribution makes finding small learning rates much easier
        val = round(math.exp(random.uniform(math.log(0.001), math.log(0.5))), 5)
        return Node(val)

def generate_tree(depth: int, max_depth: int, method: str = 'grow', operators: Optional[Dict[str, Operator]] = None) -> Node:
    """
    Recursively generates a random Abstract Syntax Tree.
    
    Args:
        depth: Current depth of the recursion.
        max_depth: Maximum allowable depth for the tree.
        method: 'grow' (asymmetric trees) or 'full' (perfectly symmetric trees).
        operators: Optional dict of operators to use. Uses all active operators if None.
        
    Returns:
        Node: The root node of the generated AST.
    """
    if depth == max_depth:
        return _random_terminal()
        
    ops = operators if operators is not None else get_active_operators()
    all_ops = list(ops.keys())
        
    # 'full' forces operators until max_depth. 'grow' gives a 40% chance of a terminal early.
    if method == 'full' or (method == 'grow' and random.random() > 0.4):
        if not all_ops:
            return _random_terminal()
            
        op_name = random.choice(all_ops)
        node = Node(op_name)
        
        arity = ops[op_name].arity
        node.children = [
            generate_tree(depth + 1, max_depth, method, operators) for _ in range(arity)
        ]
        return node
    else:
        return _random_terminal()

def ramped_half_and_half(pop_size: int, min_depth: int, max_depth: int, operators: Optional[Dict[str, Operator]] = None) -> List[Node]:
    """
    Generates a diverse initial population using the ramped half-and-half method.
    
    Args:
        pop_size: Total number of individuals in the population.
        min_depth: Minimum maximum-depth of generated trees.
        max_depth: Absolute maximum depth of generated trees.
        operators: Optional dict of operators to use.
        
    Returns:
        List[Node]: A list of generated AST root nodes.
    """
    population = []
    depths = range(min_depth, max_depth + 1)
    
    for i in range(pop_size):
        # Evenly distribute the depth limits
        depth = depths[i % len(depths)]
        # Alternate between 'full' and 'grow'
        method = 'full' if i % 2 == 0 else 'grow'
        population.append(generate_tree(1, depth, method, operators))
        
    return population
