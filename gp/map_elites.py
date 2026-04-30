import random
import numpy as np
from typing import Dict, Tuple, List, Optional
import copy
from gp.tree import Node
try:
    import symbolr_rust
except ImportError:
    class MockSymbolrRust:
        @staticmethod
        def evaluate_fast(prefix, t_array):
            # Return a strictly positive decay as fallback for Mock Mode
            # Formula: 0.01 * (1.0 + 0.5 * sin(len(prefix)) - 0.2 * t_array)
            # This gives some variety based on formula length
            var = 0.5 * np.sin(len(prefix))
            return 0.01 * (1.0 + var - 0.2 * t_array)
    symbolr_rust = MockSymbolrRust()

class MAPElitesArchive:
    """
    Maintains a 2D grid of elite individuals based on behavioral descriptors. Optimizes for Quality-Diversity: Finding the best loss in every possible niche.
    """
    def __init__(self, size_bins: int = 30, com_bins: int = 20, time_steps: int = 100):
        self.size_bins = size_bins
        self.com_bins = com_bins

        # Grid dimensions: [Size, Center_of_Mass]
        # Stores tuples of (loss: float, tree: Node)
        self.archive: Dict[Tuple[int, int], Tuple[float, Node]] = {}
        
        # Pre-compute time array for Rust behavior extraction
        self.t_array = np.linspace(0.0, 1.0, time_steps, dtype=np.float64)

    def _compute_descriptors(self, tree: Node) -> Tuple[Optional[int], Optional[int]]:
        """
        Computes (Complexity, Center of Mass) descriptors. Returns (Node, Node) if the formula evaluates to invalid math.
        """
        size = tree.size()
        try:
            prefix = tree.to_prefix()
            lr_schedule = symbolr_rust.evaluate_fast(prefix, self.t_array)
            
            # Prevent sum(0) division errors
            total_lr = np.sum(lr_schedule)
            if total_lr == 0 or not np.isfinite(total_lr):
                return None, None
                
            # Center of Mass = sum(t * LR(t)) / sum(LR(t))
            com = np.sum(self.t_array * lr_schedule) / total_lr
            
            # Clamp COM strictly between 0.0 and 1.0
            com = max(0.0, min(1.0, float(com)))
            
        except Exception:
            return None, None
            
        # Discretize into bins
        size_idx = min(size, self.size_bins - 1)
        com_idx = int(com * (self.com_bins - 1))
        
        return size_idx, com_idx
    
    def try_add(self, tree: Node, loss: float) -> bool:
        """
        Attempts to place a tree into the archive. 
        Returns True if it occupies an empty niche or beats the current niche champion.
        """
        if not np.isfinite(loss):
            return False
            
        size_idx, com_idx = self._compute_descriptors(tree)
        if size_idx is None or com_idx is None:
            return False
            
        niche = (size_idx, com_idx)
        
        # If niche is empty, or new tree has strictly better loss
        if niche not in self.archive or loss < self.archive[niche][0]:
            # Always deepcopy to prevent future genetic mutations from corrupting the archive
            self.archive[niche] = (loss, copy.deepcopy(tree))
            return True
            
        return False

    def sample_parents(self, batch_size: int) -> List[Node]:
        """
        Uniformly samples parents directly from the occupied niches in the archive.
        This strongly protects diverse configurations and acts as the selection mechanism.
        """
        if not self.archive:
            return []
            
        parents =[]
        occupied_niches = list(self.archive.keys())
        
        for _ in range(batch_size):
            chosen_niche = random.choice(occupied_niches)
            elite_tree = self.archive[chosen_niche][1]
            parents.append(copy.deepcopy(elite_tree))
            
        return parents

    def get_hall_of_fame(self, top_k: int = 5) -> List[Tuple[float, Node]]:
        """Returns the absolute best globally performing formulas across all niches."""
        all_elites = list(self.archive.values())
        all_elites.sort(key=lambda x: x[0])  # Sort by loss ascending
        return all_elites[:top_k]