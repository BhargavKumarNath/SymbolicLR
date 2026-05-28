import re

def export_to_pytorch(prefix_str: str) -> str:
    """
    Converts a prefix notation formula into a native PyTorch LambdaLR function string.
    """
    tokens = prefix_str.strip().split()
    
    def _parse(tokens, pos):
        if pos >= len(tokens): return "0.1", pos + 1
        tok = tokens[pos]
        if tok in ['+', '-', '*', '/']:
            left, pos = _parse(tokens, pos + 1)
            right, pos = _parse(tokens, pos)
            if tok == '/':
                return f"({left} / (torch.abs({right}) + 1e-6))", pos
            return f"({left} {tok} {right})", pos
        elif tok == 'sin':
            arg, pos = _parse(tokens, pos + 1)
            return f"torch.sin({arg})", pos
        elif tok == 'cos':
            arg, pos = _parse(tokens, pos + 1)
            return f"torch.cos({arg})", pos
        elif tok == 'exp':
            arg, pos = _parse(tokens, pos + 1)
            return f"torch.exp(torch.clamp({arg}, max=20.0))", pos
        elif tok == 'log':
            arg, pos = _parse(tokens, pos + 1)
            return f"torch.log(torch.abs({arg}) + 1e-6)", pos
        elif tok == 'abs':
            arg, pos = _parse(tokens, pos + 1)
            return f"torch.abs({arg})", pos
        elif tok in ['x', 't']:
            return "epoch_normalized", pos + 1
        else:
            try:
                # Format to a float string, avoid trailing .0 if possible or keep standard notation
                val = float(tok)
                return f"torch.tensor({val}, dtype=torch.float32)", pos + 1
            except ValueError:
                return "torch.tensor(0.01, dtype=torch.float32)", pos + 1

    try:
        expr, _ = _parse(tokens, 0)
    except Exception:
        expr = "torch.tensor(0.01, dtype=torch.float32)"

    code = f'''import torch
from torch.optim.lr_scheduler import LambdaLR

def get_symbolr_schedule(optimizer, total_epochs: int):
    """
    Learning rate schedule discovered by SymboLR.
    """
    def lr_lambda(epoch: int) -> float:
        epoch_normalized = torch.tensor(epoch / max(1, total_epochs), dtype=torch.float32)
        multiplier = {expr}
        return float(multiplier)
        
    return LambdaLR(optimizer, lr_lambda)
'''
    return code

def export_to_latex(prefix_str: str) -> str:
    """
    Converts a prefix notation formula into LaTeX.
    """
    tokens = prefix_str.strip().split()
    
    def _parse(tokens, pos):
        if pos >= len(tokens): return "0.1", pos + 1
        tok = tokens[pos]
        if tok == '+':
            left, pos = _parse(tokens, pos + 1)
            right, pos = _parse(tokens, pos)
            return f"\\left( {left} + {right} \\right)", pos
        elif tok == '-':
            left, pos = _parse(tokens, pos + 1)
            right, pos = _parse(tokens, pos)
            return f"\\left( {left} - {right} \\right)", pos
        elif tok == '*':
            left, pos = _parse(tokens, pos + 1)
            right, pos = _parse(tokens, pos)
            return f"\\left( {left} \\cdot {right} \\right)", pos
        elif tok == '/':
            left, pos = _parse(tokens, pos + 1)
            right, pos = _parse(tokens, pos)
            return f"\\frac{{{left}}}{{{right}}}", pos
        elif tok == 'sin':
            arg, pos = _parse(tokens, pos + 1)
            return f"\\sin\\left( {arg} \\right)", pos
        elif tok == 'cos':
            arg, pos = _parse(tokens, pos + 1)
            return f"\\cos\\left( {arg} \\right)", pos
        elif tok == 'exp':
            arg, pos = _parse(tokens, pos + 1)
            return f"e^{{{arg}}}", pos
        elif tok == 'log':
            arg, pos = _parse(tokens, pos + 1)
            return f"\\ln\\left( {arg} \\right)", pos
        elif tok == 'abs':
            arg, pos = _parse(tokens, pos + 1)
            return f"\\left| {arg} \\right|", pos
        elif tok in ['x', 't']:
            return "t", pos + 1
        else:
            try:
                return str(round(float(tok), 3)), pos + 1
            except ValueError:
                return "0.01", pos + 1

    try:
        expr, _ = _parse(tokens, 0)
    except Exception:
        expr = "0.01"

    return f"$$ {expr} $$"
