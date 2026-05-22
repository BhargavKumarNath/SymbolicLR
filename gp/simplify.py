import copy 
import sympy
import numpy as np
from typing import Optional
from gp.tree import Node

def _node_to_sympy(node: Node) -> sympy.Expr:
    """
    Recursively converts a SymboLR AST Node into a SymPy expression.
    """
    if not node.children:
        if isinstance(node.value, str):
            return sympy.Symbol(node.value)
        # Convert float constants to SymPy Floats
        return sympy.Float(node.value)
        
    op = node.value
    
    # Binary Operators
    if op == '+':
        return _node_to_sympy(node.children[0]) + _node_to_sympy(node.children[1])
    if op == '-':
        return _node_to_sympy(node.children[0]) - _node_to_sympy(node.children[1])
    if op == '*':
        return _node_to_sympy(node.children[0]) * _node_to_sympy(node.children[1])
    if op == '/':
        # Use SymPy's true division to preserve structure
        return _node_to_sympy(node.children[0]) / _node_to_sympy(node.children[1])
        
    # Unary Operators
    arg = _node_to_sympy(node.children[0])
    if op == 'sin': return sympy.sin(arg)
    if op == 'cos': return sympy.cos(arg)
    if op == 'exp': return sympy.exp(arg)
    if op == 'log': return sympy.log(arg)
    if op == 'sqrt': return sympy.sqrt(arg)
    if op == 'abs': return sympy.Abs(arg)
    
    raise ValueError(f"Unknown operator in SymboLR AST during SymPy conversion: {op}")


def _sympy_to_node(expr: sympy.Expr) -> Node:
    """
    Recursively converts a simplified SymPy expression back into a SymboLR AST.
    Handles n-ary to binary folding (SymPy treats Add/Mul as n-ary).
    """
    if isinstance(expr, sympy.Symbol):
        return Node(str(expr.name))
    if isinstance(expr, sympy.Number):
        return Node(float(expr))
        
    # SymPy n-ary Add/Mul require folding back into binary operations
    if expr.is_Add:
        args = list(expr.args)
        node = _sympy_to_node(args[0])
        for arg in args[1:]:
            node = Node("+", [node, _sympy_to_node(arg)])
        return node
        
    if expr.is_Mul:
        args = list(expr.args)
        node = _sympy_to_node(args[0])
        for arg in args[1:]:
            node = Node("*",[node, _sympy_to_node(arg)])
        return node
        
    # SymPy converts Division and Roots into Powers
    if isinstance(expr, sympy.Pow):
        exp = expr.exp
        
        if exp == -1:
            return Node("/",[Node(1.0), _sympy_to_node(expr.base)])
        if exp == 0.5:
            return Node("sqrt", [_sympy_to_node(expr.base)])
        if exp == -0.5:
            return Node("/",[Node(1.0), Node("sqrt", [_sympy_to_node(expr.base)])])
        if exp == 2:
            return Node("*", [_sympy_to_node(expr.base), _sympy_to_node(expr.base)])
            
        # Generic fallback for Integer Powers
        if isinstance(exp, sympy.Integer) and exp > 2:
            res = _sympy_to_node(expr.base)
            for _ in range(int(exp) - 1):
                res = Node("*",[res, _sympy_to_node(expr.base)])
            return res
            
        # Mathematical absolute fallback to prevent crashing on weird fractional powers
        return Node("exp", [Node("*", [Node(float(exp)), Node("log", [_sympy_to_node(expr.base)])])])
        
    # Standard Math Functions
    if isinstance(expr, sympy.sin): return Node("sin",[_sympy_to_node(expr.args[0])])
    if isinstance(expr, sympy.cos): return Node("cos",[_sympy_to_node(expr.args[0])])
    if isinstance(expr, sympy.exp): return Node("exp", [_sympy_to_node(expr.args[0])])
    if isinstance(expr, sympy.log): return Node("log", [_sympy_to_node(expr.args[0])])
    if isinstance(expr, sympy.Abs): return Node("abs",[_sympy_to_node(expr.args[0])])
    
    raise ValueError(f"Unsupported SymPy expression encountered: {type(expr)}")


def _contains_variable(node: Node) -> bool:
    """Check if a subtree contains the variable 't'."""
    if isinstance(node.value, str) and node.value == 't':
        return True
    return any(_contains_variable(c) for c in node.children)


def _constant_fold(node: Node) -> Node:
    """If a subtree contains no variable 't', evaluate it and replace with a constant."""
    if not _contains_variable(node):
        try:
            t_dummy = np.array([0.5])  # doesn't matter since no 't'
            val = float(node.evaluate(t_dummy)[0])
            if np.isfinite(val):
                return Node(round(val, 6))
        except Exception:
            pass
    # Recurse into children
    if node.children:
        node.children = [_constant_fold(child) for child in node.children]
    return node


def simplify_tree(tree: Node) -> Node:
    """
    Public method to prune bloat from a SymboLR AST algebraically.
    Has a strict fallback: if SymPy encounters a translation error, 
    it logs and returns the original tree to prevent GP loop crashes.
    """
    original = copy.deepcopy(tree)
    # Phase 1: Constant folding (cheap, always do)
    original = _constant_fold(original)
    
    # Phase 2: Algebraic simplification (expensive, skip large trees)
    if original.size() > 30:
        if hasattr(original, 'invalidate_cache'):
            original.invalidate_cache()
        return original
        
    try:
        # Translate to mathematical engine
        sp_expr = _node_to_sympy(original)
        
        # Perform extreme algebraic reduction
        sp_expr_simplified = sympy.simplify(sp_expr)
        
        # Translate back to native format
        pruned_tree = _sympy_to_node(sp_expr_simplified)
        
        # Invalidate cache on the new tree
        if hasattr(pruned_tree, "invalidate_cache"):
            pruned_tree.invalidate_cache()
        else:
            pruned_tree._hash_cache = None
            
        return pruned_tree
        
    except Exception as e:
        # Silently fail closed: return the unsimplified tree if SymPy creates unsupported structures
        return original


def tree_to_latex(tree: Node) -> str:
    """Generates beautiful LaTeX for UI rendering."""
    try:
        expr = _node_to_sympy(tree)
        return sympy.latex(sympy.simplify(expr))
    except Exception:
        return str(tree)
