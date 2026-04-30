import pytest
from gp.tree import Node
from gp.population import generate_tree, ramped_half_and_half
from gp.evolution import (tournament_selection, subtree_crossover, subtree_mutation, hoist_mutation)

def test_generate_tree():
    """Validates boundary constraints of random tree generation."""
    tree = generate_tree(1, 3, method='full')
    
    assert isinstance(tree, Node)
    assert tree.depth() <= 3, "Generated tree exceeds maximum depth."
    
    # A full tree of depth 1 shouldn't have children
    shallow_tree = generate_tree(3, 3, method='full')
    assert shallow_tree.depth() == 1
    assert not shallow_tree.children


def test_ramped_half_and_half():
    """Validates population scaling and method diversity."""
    pop_size = 10
    population = ramped_half_and_half(pop_size, 2, 4)
    
    assert len(population) == pop_size
    
    depths = [tree.depth() for tree in population]
    # RHH should produce a spread of depths
    assert max(depths) <= 4
    assert min(depths) >= 1


def test_tournament_selection():
    """Tournament must reliably pick the tree with the lowest loss."""
    population = [Node("t"), Node(0.1), Node(0.5)]
    fitnesses = [1.5, 0.2, 5.0]  # 0.2 is the best
    
    # If k = 3 (entire population), it MUST pick index 1
    winner = tournament_selection(population, fitnesses, k=3)
    assert winner.value == 0.1
    
    # Winner must be a deep copy, not a reference
    assert winner is not population[1]


def test_subtree_crossover():
    """Crossover must yield new trees without mutating the original parents."""
    parent1 = generate_tree(1, 4, method='full')
    parent2 = generate_tree(1, 4, method='full')
    
    hash_p1_before = parent1.get_hash()
    hash_p2_before = parent2.get_hash()
    
    off1, off2 = subtree_crossover(parent1, parent2)
    
    # Original parents must remain perfectly unmodified
    assert parent1.get_hash() == hash_p1_before
    assert parent2.get_hash() == hash_p2_before
    
    # Offspring must be valid Node instances
    assert isinstance(off1, Node)
    assert isinstance(off2, Node)
    assert off1.fitness is None
    assert off2.fitness is None


def test_hoist_mutation_bloat_control():
    """Hoist mutation must return a tree that is structurally smaller or equal."""
    # Build a specifically bloated tree: (t + (0.5 * t))
    parent = Node("+",[
        Node("t"),
        Node("*", [Node(0.5), Node("t")])
    ])
    
    original_size = parent.size()
    
    # Perform hoist mutation
    offspring = hoist_mutation(parent)
    
    # The new size MUST be strictly less than or equal to original size
    assert offspring.size() <= original_size, "Hoist mutation failed to compress or maintain bloat."
    assert isinstance(offspring, Node)
    assert offspring.fitness is None


def test_mutation_clears_inherited_fitness_cache():
    """Mutated descendants must not retain stale cached fitness from archive parents."""
    parent = Node("+", [Node("t"), Node(0.5)])
    parent.fitness = 0.123
    parent.children[0].fitness = 0.456
    parent.children[1].fitness = 0.789

    offspring = subtree_mutation(parent, max_mutation_depth=3)

    assert offspring.fitness is None
    assert all(node.fitness is None for node in [offspring, *offspring.children])
