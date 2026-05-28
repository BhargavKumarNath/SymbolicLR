"""
SymboLR MNIST Example 
---------------------
This script demonstrates how a researcher can plug their own custom neural network 
and dataset into SymboLR to discover an optimal learning rate schedule!

1. Subclass `BaseEvaluator`
2. Implement `evaluate()` to train your PyTorch model using the GP schedules
3. Pass your evaluator to the `RustEvolutionBridge`
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Import SymboLR Core
from src.symbolr.engine.evaluator import BaseEvaluator
from src.symbolr.engine.bridge import RustEvolutionBridge
from src.symbolr.cli.artifacts import export_to_pytorch

class MNISTEvaluator(BaseEvaluator):
    def __init__(self, epochs=3, subset_size=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        
        # 1. Load MNIST Dataset (using a small subset for fast GP evaluation)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        valset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # Use a subset of the dataset so GP evolution is fast!
        self.trainloader = DataLoader(Subset(trainset, range(subset_size)), batch_size=64, shuffle=True)
        self.valloader = DataLoader(Subset(valset, range(500)), batch_size=64, shuffle=False)
        
        # Pre-allocate normalized time tensor `t` for mapping ASTs to LR arrays
        self.total_steps = len(self.trainloader) * self.epochs
        self.t = torch.linspace(0.0, 1.0, self.total_steps, device=self.device)
        print(f"MNIST Evaluator Initialized on {self.device}. Steps per eval: {self.total_steps}")

    def _parse_and_evaluate_ast(self, prefix_str: str) -> torch.Tensor:
        """Helper to convert SymboLR Prefix strings into a Tensor of learning rates over time."""
        tokens = prefix_str.strip().split()
        stack = []
        for token in reversed(tokens):
            if token in ['x', 't']: stack.append(self.t)
            elif token == '+': stack.append(stack.pop() + stack.pop())
            elif token == '-': stack.append(stack.pop() - stack.pop())
            elif token == '*': stack.append(stack.pop() * stack.pop())
            elif token == '/':
                left, right = stack.pop(), stack.pop()
                stack.append(left / (torch.abs(right) + 1e-6))
            elif token == 'sin': stack.append(torch.sin(stack.pop()))
            elif token == 'cos': stack.append(torch.cos(stack.pop()))
            elif token == 'exp': stack.append(torch.exp(torch.clamp(stack.pop(), max=20.0)))
            elif token == 'log': stack.append(torch.log(torch.abs(stack.pop()) + 1e-6))
            elif token == 'abs': stack.append(torch.abs(stack.pop()))
            else:
                try: stack.append(torch.full_like(self.t, float(token)))
                except ValueError: stack.append(torch.full_like(self.t, 0.01))
        
        # Clamp learning rates to sane bounds [1e-6, 10.0]
        return torch.clamp(stack[0], 1e-6, 10.0)

    def evaluate(self, formulas: list[str]) -> list[float]:
        """
        SymboLR will pass batches of generated AST formulas here.
        We must train a model for each formula and return its final Validation Loss!
        """
        losses = []
        
        for formula in formulas:
            try:
                # 1. Map the mathematical formula to a concrete LR array
                lr_array = self._parse_and_evaluate_ast(formula)
                
                # 2. Initialize a fresh model for this evaluation
                model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(28 * 28, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                ).to(self.device)
                
                optimizer = optim.SGD(model.parameters(), lr=1.0)
                criterion = nn.CrossEntropyLoss()
                
                # 3. Train the model using the generated LR schedule
                step_idx = 0
                for epoch in range(self.epochs):
                    model.train()
                    for inputs, labels in self.trainloader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        
                        # Apply the GP-discovered learning rate for this specific step
                        current_lr = lr_array[step_idx].item()
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr
                            
                        optimizer.zero_grad()
                        loss = criterion(model(inputs), labels)
                        loss.backward()
                        optimizer.step()
                        step_idx += 1
                
                # 4. Calculate final Validation Loss
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in self.valloader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        val_loss += criterion(model(inputs), labels).item()
                        
                avg_val_loss = val_loss / len(self.valloader)
                if not math.isfinite(avg_val_loss):
                    avg_val_loss = float('inf')
                    
                # Add a tiny parsimony penalty to prefer shorter equations
                parsimony = 0.001 * len(formula.split())
                losses.append(avg_val_loss + parsimony)
                
            except Exception as e:
                # If the formula causes an error (e.g. exploding gradients), it dies
                losses.append(float('inf'))
                
        return losses

if __name__ == "__main__":
    print("Initializing MNIST Evaluator...")
    evaluator = MNISTEvaluator(epochs=2, subset_size=2000)
    
    print("Booting SymboLR Rust Engine...")
    bridge = RustEvolutionBridge(
        eval_callback=evaluator.evaluate,
        max_generations=5,
        pop_size=20,  # Small population for this quick demo
        seed=42
    )
    
    best_formula = ""
    for result in bridge.stream():
        print(f"Gen {result.generation_number:02d} | Best Val Loss: {result.best_mse:.4f} | Formula: {result.top_formula_prefix}")
        best_formula = result.top_formula_prefix
        
    print("\n--- EVOLUTION COMPLETE ---")
    print("Exporting best schedule to PyTorch Code:")
    print(export_to_pytorch(best_formula))
