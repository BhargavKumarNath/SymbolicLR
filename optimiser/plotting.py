"""
optimiser/plotting.py - Lightweight static plotting utilities for the CLI.
No interactive GUIs. Exports to disk only.
"""
import os
from typing import List, Dict, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def export_convergence_plot(
    generations: List[int],
    best_losses: List[float],
    median_losses: List[float],
    output_path: str,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(generations, best_losses, label="Best Loss", color="green", linewidth=2)
    plt.plot(generations, median_losses, label="Median Loss", color="orange", linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Loss (Log Scale)")
    plt.yscale("log")
    plt.title("Convergence History")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def export_diversity_plot(
    generations: List[int],
    structural: List[float],
    behavioral: List[float],
    output_path: str,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(generations, structural, label="Structural Diversity", color="blue", linewidth=2)
    plt.plot(generations, behavioral, label="Behavioral Diversity", color="purple", linewidth=2)
    plt.axhline(0.3, color="red", linestyle=":", label="Collapse Threshold")
    plt.xlabel("Generation")
    plt.ylabel("Diversity Score")
    plt.ylim(0, 1.05)
    plt.title("Archive Diversity Trends")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def export_schedules_plot(
    schedules: List[Tuple[str, np.ndarray]], 
    output_path: str
) -> None:
    """Plot multiple learning rate schedules (Time vs LR)"""
    if not MATPLOTLIB_AVAILABLE or not schedules:
        return

    plt.figure(figsize=(8, 5))
    t = np.linspace(0, 1, len(schedules[0][1]))
    
    for label, lr_array in schedules:
        plt.plot(t, lr_array, label=label, linewidth=2)
        
    plt.xlabel("Normalized Training Time (t)")
    plt.ylabel("Learning Rate")
    plt.title("Hall of Fame - Discovered Schedules")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
