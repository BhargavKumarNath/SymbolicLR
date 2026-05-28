"""
experiments/mnist_example.py — Minimal MNIST evaluator demonstration.

Shows how to subclass BaseEvaluator and plug a custom PyTorch training loop
into the RustEvolutionBridge. This is the entry point for researchers who
want to evaluate discovered schedules on their own models.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from src.symbolr.core.evaluator import BaseEvaluator
from src.symbolr.core.bridge import RustEvolutionBridge
from src.symbolr.artifacts.pytorch_export import export_to_pytorch
from src.symbolr.artifacts.prefix_parser import evaluate_formula


class MNISTEvaluator(BaseEvaluator):
    """
    Trains a small MLP on MNIST using each formula as an LR schedule.

    NOTE: This evaluator is intentionally simple and serial. For high-throughput
    evaluation, use GradientAwareEvaluator (Phase 3) with batched parallel training.
    """

    def __init__(self, epochs: int = 3, subset_size: int = 1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        valset   = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        self.trainloader = DataLoader(Subset(trainset, range(subset_size)), batch_size=64, shuffle=True)
        self.valloader   = DataLoader(Subset(valset, range(500)), batch_size=64, shuffle=False)
        self.total_steps = len(self.trainloader) * self.epochs

        print(f"MNISTEvaluator ready on {self.device}. Steps/eval: {self.total_steps}")

    def evaluate(self, formulas: list[str]) -> list[float]:
        losses = []
        for formula in formulas:
            try:
                model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(28 * 28, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10),
                ).to(self.device)
                optimizer = optim.SGD(model.parameters(), lr=1.0)
                criterion = nn.CrossEntropyLoss()

                step_idx = 0
                for epoch in range(self.epochs):
                    model.train()
                    for inputs, labels in self.trainloader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        t_norm = step_idx / max(1, self.total_steps - 1)
                        lr = evaluate_formula(formula, t=t_norm)
                        for pg in optimizer.param_groups:
                            pg['lr'] = lr
                        optimizer.zero_grad()
                        criterion(model(inputs), labels).backward()
                        optimizer.step()
                        step_idx += 1

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in self.valloader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        val_loss += criterion(model(inputs), labels).item()
                avg = val_loss / len(self.valloader)
                losses.append(avg if math.isfinite(avg) else float('inf'))
            except Exception:
                losses.append(float('inf'))
        return losses


if __name__ == "__main__":
    print("Initializing MNIST Evaluator...")
    evaluator = MNISTEvaluator(epochs=2, subset_size=2000)

    print("Booting SymboLR Rust Engine...")
    bridge = RustEvolutionBridge(
        eval_callback=evaluator.evaluate,
        max_generations=5,
        pop_size=20,
        seed=42,
    )

    best_formula = ""
    for result in bridge.stream():
        print(f"Gen {result.generation_number:02d} | Val Loss: {result.best_mse:.4f} | {result.top_formula_prefix}")
        best_formula = result.top_formula_prefix

    print("\n--- EVOLUTION COMPLETE ---")
    print(export_to_pytorch(best_formula))
