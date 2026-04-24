import pytest
import numpy as np
from unittest.mock import patch
from gp.tree import Node
from gp.map_elites import MAPElitesArchive

@pytest.fixture
def archive():
    return MAPElitesArchive(size_bins=10, com_bins=10, time_steps=100)

def test_behavioral_descriptors_front_loaded(archive):
    """A learning rate that decays immediately should have a Center of Mass near 0"""
    tree = Node("t")
    # Mock evalute_fast to return an array that drops from 1.0 to 0.0 instantly
    fake_lr = np.zeros(100)
    fake_lr[0] = 1.0

    with patch('symbolr_rust.evaluate_fast', return_value=fake_lr):
        size_idx, com_idx = archive._compute_descriptors(tree)
        
    assert size_idx == 1 # Tree size is 1
    assert com_idx == 0  # CoM should be exactly at bin 0

def test_behavioral_descriptors_back_loaded(archive):
    """A warmup learning rate should have a center of mass near 1.0"""
    tree = Node("+", [Node("t"), Node("t")])

    fake_lr = np.zeros(100)
    fake_lr[-1] = 1.0

    with patch('symbolr_rust.evaluate_fast', return_value=fake_lr):
        size_idx, com_idx = archive._compute_descriptors(tree)
        
    assert size_idx == 3
    assert com_idx == 9  # CoM should be at the absolute max bin (10-1)

def test_try_add_new_niche(archive):
    """An evaluated individual should successfully populate an empty niche."""
    tree = Node("t")
    
    with patch.object(archive, '_compute_descriptors', return_value=(1, 5)):
        success = archive.try_add(tree, loss=0.5)
        
    assert success is True
    assert (1, 5) in archive.archive
    assert archive.archive[(1, 5)][0] == 0.5

def test_try_add_competitive_niche(archive):
    """An individual should overwrite a niche ONLY if its loss is strictly lower."""
    tree1 = Node("t")
    tree2 = Node("sin", [Node("t")])
    tree3 = Node("cos", [Node("t")])
    
    # Force all to map to the exact same niche
    with patch.object(archive, '_compute_descriptors', return_value=(2, 5)):
        # First tree populates empty niche
        assert archive.try_add(tree1, loss=1.0) is True
        
        # Second tree is WORSE (higher loss), should be rejected
        assert archive.try_add(tree2, loss=1.5) is False
        assert archive.archive[(2, 5)][0] == 1.0
        
        # Third tree is BETTER (lower loss), should overwrite
        assert archive.try_add(tree3, loss=0.2) is True
        assert archive.archive[(2, 5)][0] == 0.2
        assert archive.archive[(2, 5)][1].value == "cos"

def test_sample_parents(archive):
    """Parents must be correctly sampled from occupied niches."""
    tree1 = Node("t")
    tree2 = Node(0.5)
    
    with patch.object(archive, '_compute_descriptors', side_effect=[(1, 1), (5, 5)]):
        archive.try_add(tree1, loss=0.1)
        archive.try_add(tree2, loss=0.2)
        
    parents = archive.sample_parents(batch_size=10)
    
    assert len(parents) == 10
    # Make sure we get actual deep copies of the nodes
    assert isinstance(parents[0], Node)
    
    # Verify diversity
    values = [p.value for p in parents]
    assert "t" in values
    assert 0.5 in values
