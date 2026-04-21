import random
from typing import List
from .tree import Node
from .operators import OPERATORS

TERMINALS = ['t'] +[round(float(i) / 10.0, 1) for i in range(1, 10)]

# Categorise operators by arity for correct tree generation
BIN_OPS = [name for name, op in OPERATORS.items() if op.arity == 2]
UN_OPS =[name for name, op in OPERATORS.items() if op.arity == 1]
ALL_OPS = BIN_OPS + UN_OPS


def generate_tree(depth: int, max_depth: int, method: str = 'grow') -> Node:
    """
    Recursively generates a random Abstract Syntax Tree.
    
    Args:
        depth: Current depth of the recursion.
        max_depth: Maximum allowable depth for the tree.
        method: 'grow' (asymmetric trees) or 'full' (perfectly symmetric trees).
        
    Returns:
        Node: The root node of the generated AST.
    """
    if depth == max_depth:
        return Node(random.choice(TERMINALS))
        
    # 'full' forces operators until max_depth. 'grow' gives a 50% chance of a terminal early.
    if method == 'full' or (method == 'grow' and random.random() > 0.5):
        op_name = random.choice(ALL_OPS)
        node = Node(op_name)
        
        arity = OPERATORS[op_name].arity
        node.children =[
            generate_tree(depth + 1, max_depth, method) for _ in range(arity)
        ]
        return node
    else:
        return Node(random.choice(TERMINALS))


def ramped_half_and_half(pop_size: int, min_depth: int, max_depth: int) -> List[Node]:
    """
    Generates a diverse initial population using the ramped half-and-half method.
    
    Args:
        pop_size: Total number of individuals in the population.
        min_depth: Minimum maximum-depth of generated trees.
        max_depth: Absolute maximum depth of generated trees.
        
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
        population.append(generate_tree(1, depth, method))
        
    return population

