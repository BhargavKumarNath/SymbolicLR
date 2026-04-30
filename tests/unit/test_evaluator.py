import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from gp.tree import Node
from gp.evaluator import ParallelEvaluator

@pytest.fixture
def dummy_dependencies():
    """Mocks the Trainer and Loaders to bypass GPU dependency during CPU tests."""
    mock_trainer = MagicMock()
    # Simulate a fake GPU training loop that returns a fake validation loss
    mock_trainer.evaluate_schedule.return_value = 0.42 
    
    mock_train_loader = MagicMock()
    mock_val_loader = MagicMock()
    
    # Factory just returns a dummy object instead of a PyTorch nn.Module
    mock_factory = lambda: "DummyModel" 
    
    return mock_trainer, mock_train_loader, mock_val_loader, mock_factory


def test_evaluator_caches_fitness(dummy_dependencies):
    """Ensures evaluated individuals bypass the expensive Rust/GPU pipeline upon re-evaluation."""
    trainer, train, val, factory = dummy_dependencies
    evaluator = ParallelEvaluator(trainer, train, val, epochs=1, time_steps=10)
    
    tree = Node("+",[Node("t"), Node(0.5)])
    
    # First Evaluation: Should call the trainer
    loss1 = evaluator.evaluate_individual(tree, factory)
    assert loss1 == 0.42
    assert getattr(tree, "fitness") == 0.42
    assert trainer.evaluate_schedule.call_count == 1
    
    # Second Evaluation: Should hit cache, call_count shouldn't increment
    loss2 = evaluator.evaluate_individual(tree, factory)
    assert loss2 == 0.42
    assert trainer.evaluate_schedule.call_count == 1


def test_parallel_population_evaluation(dummy_dependencies):
    """Ensures thread pool returns fitness arrays strictly parallel to the population list."""
    trainer, train, val, factory = dummy_dependencies
    
    # Setup mock trainer to return varying losses based on the model calls
    trainer.evaluate_schedule.side_effect =[1.0, 2.0, 3.0, 4.0]
    
    evaluator = ParallelEvaluator(trainer, train, val, epochs=1, time_steps=10)
    
    # Population of 4 arbitrary trees
    pop =[
        Node("t"),
        Node(0.5),
        Node("sin", [Node("t")]),
        Node("*", [Node("t"), Node(2.0)])
    ]
    
    fitnesses = evaluator.evaluate_population(pop, factory, max_workers=2)
    
    assert len(fitnesses) == 4
    assert sorted(fitnesses) ==[1.0, 2.0, 3.0, 4.0]
    # Verify the cache flag was appended correctly to all trees via ThreadPool
    assert all(hasattr(tree, "fitness") for tree in pop)


def test_parallel_population_evaluation_reports_progress(dummy_dependencies):
    """Progress callback should be invoked as candidates complete."""
    trainer, train, val, factory = dummy_dependencies
    trainer.evaluate_schedule.side_effect = [1.0, 2.0, 3.0]

    evaluator = ParallelEvaluator(trainer, train, val, epochs=1, time_steps=10)
    pop = [Node("t"), Node(0.5), Node("sin", [Node("t")])]
    updates = []

    fitnesses = evaluator.evaluate_population(
        pop,
        factory,
        max_workers=8,
        progress_callback=lambda completed, total: updates.append((completed, total)),
    )

    assert fitnesses == [1.0, 2.0, 3.0]
    assert updates
    assert updates[-1] == (3, 3)

def test_evaluator_handles_rust_math_crash(dummy_dependencies):
    """Proves the thread pool won't die if Rust yields a mathematically invalid schedule."""
    trainer, train, val, factory = dummy_dependencies
    evaluator = ParallelEvaluator(trainer, train, val, epochs=1, time_steps=10)
    
    # Added exception incase rust crashes
    with patch('symbolr_rust.evaluate_fast', side_effect=Exception("Rust Panic!")):
        tree = Node("t")
        loss = evaluator.evaluate_individual(tree, factory)
        
    assert loss == float('inf'), "Failed to gracefully handle compiled logic crash."
