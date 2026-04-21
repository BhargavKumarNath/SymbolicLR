import hashlib
import numpy as np
from typing import List, Union, Optional
from gp.operators import OPERATORS

class Node:
    """
    An Abstract Syntax Tree (AST) node representing a mathematical schedule. Can we an operator (Function), a variable (Terminal), or a constant
    """
    def __init__(self, value: Union[str, float], children: Optional[List['Node']] = None):
        """
        Args:
            value: Operator string (e.g. '+'), variable 't', or a float constant.
            children: List of child Nodes (if an operator).
        """
        self.value = value
        self.children = children if children is not None else []
        self._hash_cache: Optional[str] = None
    
    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """
        Recursively evaluates the tree over an array of timesteps using NumPy.
        Args:
            t: A 1D numpy array of normalized time steps 

        Returns:
            np.ndarray: The computed learning rates of identical shape to `t`.
        """
        # 1. Variable Terminal
        if isinstance(self.value, str) and self.value == "t":
            return t
        
        # 2. Operator Function
        if isinstance(self.value, str) and self.value in OPERATORS:
            op = OPERATORS.get(self.value)
            if op is None:
                return str(self.value)
            child_results = [child.evaluate(t) for child in self.children]
            result = op.func(*child_results)
            return np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)
        
        # 3. Float Constant
        if isinstance(self.value, (float, int)):
            return np.full_like(t, float(self.value))
        
        raise ValueError(f"Unknown node value encountered: {self.value}")
    
    def __str__(self) -> str:
        """Generates the algebric string representation of the tree"""
        if not self.children:
            if isinstance(self.value, float):
                return f"{self.value:.4f}"
            return str(self.value)
        
        op = OPERATORS.get(self.value)
        if op is None:
            return str(self.value)
        if op.arity == 1:
            return f"{self.value}({str(self.children[0])})"
        elif op.arity == 2:
            return f"({str(self.children[0])} {self.value} {str(self.children[1])})"
        
        raise ValueError(f"Unsupported arity for string conversion: {op.arity}")
    
    def get_hash(self) -> str:
        if self._hash_cache is None:
            parts = [str(self.value)]
            for child in self.children:
                parts.append(child.get_hash())
            raw = "|".join(parts)
            self._hash_cache = hashlib.md5(raw.encode()).hexdigest()
        return self._hash_cache
    
    def depth(self) -> int:
        """Returns the maximum depth of the tree (1-indexed)"""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def size(self) -> int:
        """Returns the total number of nodes in the tree (bloat proxy)"""
        return 1 + sum(child.size() for child in self.children)
    
    def invalidate_cache(self):
        self._hash_cache = None
        for child in self.children:
            child.invalidate_cache()
    
    


    