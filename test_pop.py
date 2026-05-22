import torch
import numpy as np
from data.fidelity import FidelityManager
from models.probe import create_compiled_model, ProbeTrainer
from gp.fitness import evaluate_fitness
from gp.tree import Node
from gp.evaluator import ParallelEvaluator

def main():
    device = torch.device('cuda')
    fidelity = FidelityManager()
    train_loader, val_loader = fidelity.get_low_fidelity(device, batch_size=256)
    trainer = ProbeTrainer(device)
    model_factory = lambda init_seed=None: create_compiled_model(device, in_channels=1, init_seed=init_seed)
    
    evaluator = ParallelEvaluator(trainer, train_loader, val_loader, epochs=5)
    
    # Test LR 0.94 exactly
    population = [Node(0.94), Node(0.942088)]
    
    fitnesses = evaluator.evaluate_population(population, model_factory, max_workers=1)
    
    for tree, fit in zip(population, fitnesses):
        print(f"LR {tree.value:.3f} -> Loss {fit}")

if __name__ == "__main__":
    main()
