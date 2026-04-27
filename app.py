import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import torch
from data.fidelity import FidelityManager
from models.probe import ProbeTrainer, create_compiled_model
from gp.population import ramped_half_and_half
from gp.evolution import subtree_crossover, subtree_mutation, hoist_mutation
from gp.evaluator import ParallelEvaluator
from gp.map_elites import MAPElitesArchive
from gp.simplify import simplify_tree, tree_to_latex
import symbolr_rust

st.set_page_config(page_title="SymboLR Evolution", page_icon="🧬", layout="wide")

def plot_archive(archive: MAPElitesArchive):
    """Creates a 2D scatter plot of the MAP Elites Archive"""
    if not archive.archive:
        return st.empty
    
    data = []
    for (size_idx, com_idx), (loss, tree) in archive.archive.items():
        data.append({
            "Size": size_idx,
            "Center of Mass": com_idx / archive.com_bins,
            "Loss": loss,
            "Formula": str(tree)
        })
    df = pd.DataFrame(data)
    chart = alt.Chart(df).mark_circle(size=100).encode(
        x=alt.X('Center of Mass', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Size', title='Tree Complexity'),
        color=alt.Color('Loss', scale=alt.Scale(scheme='viridis', reverse=True)),
        tooltip=['Loss', 'Size', 'Center of Mass', 'Formula']
    ).properties(title="MAP-Elites 2D Archive Niche Map", height=400)

    return chart

def plot_schedules(trees, t_array):
    """Plots the learning rate curves of the Hall of Fame."""
    data =[]
    for i, (loss, tree) in enumerate(trees):
        try:
            lrs = symbolr_rust.evaluate_fast(tree.to_prefix(), t_array)
            for t_val, lr in zip(t_array, lrs):
                data.append({"Time": t_val, "LR": lr, "Rank": f"Rank {i+1} (Loss: {loss:.3f})"})
        except Exception:
            continue
            
    if not data:
        return st.empty()
        
    df = pd.DataFrame(data)
    chart = alt.Chart(df).mark_line().encode(
        x='Time',
        y=alt.Y('LR', title="Learning Rate"),
        color='Rank'
    ).properties(title="Top Discovered LR Schedules", height=400)
    
    return chart

# UI Layout
st.title("🧬 SymboLR: Symbolic Auto-ML Dashboard")
st.markdown("Discovering mathematical learning rate schedules via Quality-Diversity Genetic Programming.")

with st.sidebar:
    st.header("Hyperparameters")
    gen_count = st.slider("Generations", 1, 50, 5)
    pop_size = st.slider("Population Size", 10, 200, 50)
    epochs = st.slider("Epochs per Eval", 1, 5, 1)
    run_btn = st.button("🚀 Start Evolution", width='stretch')

if run_btn:
    # 1. Setup Backend
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.toast(f"Hardware utilized: {device.type.upper()}", icon="⚡")
    
    fidelity = FidelityManager(seed=42)
    train_loader, val_loader = fidelity.get_low_fidelity(device, batch_size=256)
    trainer = ProbeTrainer(device=device, patience=2, amp_enabled=True)
    evaluator = ParallelEvaluator(trainer, train_loader, val_loader, epochs=epochs, time_steps=100)
    archive = MAPElitesArchive(size_bins=30, com_bins=20, time_steps=100)
    model_factory = lambda: create_compiled_model(device, in_channels=1)

    # UI Placeholder Setup
    progress_bar = st.progress(0, text="Initializing Generation 0...")
    col1, col2, col3 = st.columns(3)
    metric_loss = col1.empty()
    metric_niches = col2.empty()
    metric_size = col3.empty()
    
    plot_col1, plot_col2 = st.columns(2)
    chart_archive = plot_col1.empty()
    chart_lrs = plot_col2.empty()
    table_hof = st.empty()

    # 2. Initial Population
    population = ramped_half_and_half(pop_size, min_depth=2, max_depth=4)
    fitnesses = evaluator.evaluate_population(population, model_factory, max_workers=1)
    for tree, fit in zip(population, fitnesses):
        archive.try_add(simplify_tree(tree), fit)

    # 3. Evolution Loop
    for gen in range(1, gen_count + 1):
        progress_bar.progress(gen / gen_count, text=f"Evolving Generation {gen}/{gen_count}...")
        
        parents = archive.sample_parents(pop_size)
        offspring =[]
        for _ in range(pop_size // 2):
            p1, p2 = random.choice(parents), random.choice(parents)
            r = random.random()
            if r < 0.5:
                o1, o2 = subtree_crossover(p1, p2)
                offspring.extend([o1, o2])
            elif r < 0.75:
                offspring.extend([subtree_mutation(p1, 3), subtree_mutation(p2, 3)])
            else:
                offspring.extend([hoist_mutation(p1), hoist_mutation(p2)])
                
        offspring =[simplify_tree(ind) for ind in offspring]
        fitnesses = evaluator.evaluate_population(offspring, model_factory, max_workers=4)
        
        for ind, fit in zip(offspring, fitnesses):
            archive.try_add(ind, fit)
            
        # Update UI dynamically
        hof = archive.get_hall_of_fame(top_k=5)
        best_loss = hof[0][0] if hof else float('inf')
        
        metric_loss.metric("Best Val Loss", f"{best_loss:.4f}")
        metric_niches.metric("Active Niches", len(archive.archive))
        metric_size.metric("Hall of Fame Depth", hof[0][1].depth() if hof else 0)
        
        chart_archive.altair_chart(plot_archive(archive), width='stretch')
        chart_lrs.altair_chart(plot_schedules(hof, archive.t_array), width='stretch')
        
        # Build LaTeX Markdown Table
        md_table = "| Rank | Loss | Complexity | LaTeX Formula |\n|---|---|---|---|\n"
        for i, (loss, tree) in enumerate(hof):
            md_table += f"| {i+1} | {loss:.4f} | {tree.size()} | ${tree_to_latex(tree)}$ |\n"
        table_hof.markdown(md_table)

    progress_bar.progress(1.0, text="Evolution Complete! 🎉")
    st.balloons()