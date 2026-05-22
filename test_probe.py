import torch
from data.fidelity import FidelityManager
from models.probe import create_compiled_model, ProbeTrainer
from gp.fitness import evaluate_fitness
from gp.tree import Node
import numpy as np
from config.settings import get_config

def main():
    device = torch.device('cuda')
    fidelity = FidelityManager()
    train_loader, val_loader = fidelity.get_low_fidelity(device, batch_size=256)
    
    trainer = ProbeTrainer(device)
    
    # Needs to match benchmark.py exactly
    model_factory = lambda init_seed=None: create_compiled_model(device, in_channels=1, init_seed=init_seed)
    
    tree = Node(0.94)  # constant 0.94 LR
    t_array = np.linspace(0.0, 1.0, 100)
    
    loss = evaluate_fitness(
        tree=tree,
        t_array=t_array,
        trainer=trainer,
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5
    )
    print(f"Validation Loss via evaluate_fitness: {loss}")

if __name__ == "__main__":
    main()
