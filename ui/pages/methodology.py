"""
ui/pages/methodology.py
Methodology page — GP theory, MAP-Elites, evaluation stack, operator table.
"""

import pandas as pd
import streamlit as st
from ui.theme import page_header, section_header, info_card


def render():
    page_header(
        eyebrow="Technical Deep-Dive",
        title="Methodology",
        subtitle=(
            "A walk through the theoretical foundations, design decisions, and engineering "
            "trade-offs that make SymboLR work at production scale."
        ),
    )

    tab_gp, tab_me, tab_eval, tab_ops = st.tabs(
        ["Genetic Programming", "MAP-Elites", "Evaluation Stack", "Operator Design"]
    )

    # Tab 1: Genetic Programming
    with tab_gp:
        section_header("What is Symbolic Regression via GP?")
        info_card(
            "Overview",
            "Genetic Programming (GP) treats mathematical formulas as programs represented "
            "as Abstract Syntax Trees (ASTs). The evolutionary loop applies biologically-inspired "
            "operators — crossover, mutation, and selection — to a population of ASTs, iteratively "
            "improving their fitness against a defined objective (validation loss).",
            accent="green",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            info_card(
                "🔀  Subtree Crossover",
                "Selects one random subtree from each parent and swaps them, "
                "combining structural features from both. "
                "Probability: <span style='color:var(--accent-green);font-family:var(--font-mono);'>50%</span>",
            )
        with c2:
            info_card(
                "🌿  Subtree Mutation",
                "Replaces a randomly chosen node with a freshly generated random subtree, "
                "introducing structural novelty. "
                "Probability: <span style='color:var(--accent-blue);font-family:var(--font-mono);'>25%</span>",
            )
        with c3:
            info_card(
                "✂️  Hoist Mutation",
                "Anti-bloat operator: replaces a subtree with one of its own descendants, "
                "<em>guaranteeing</em> a strictly smaller offspring. "
                "Probability: <span style='color:var(--accent-amber);font-family:var(--font-mono);'>25%</span>",
            )

        section_header("Initial Population: Ramped Half-and-Half")
        info_card(
            "Diversity from Generation Zero",
            "The initial generation uses <em>ramped half-and-half</em>: population is split evenly "
            "across depth levels (2 to max_depth). Within each depth level, half use the <em>full</em> "
            "method (symmetric trees, all branches reach max_depth) and half use <em>grow</em> "
            "(asymmetric, 50% terminal probability at each node). This maximises structural diversity "
            "before a single evaluation is run.",
        )

    # Tab 2: MAP-Elites
    with tab_me:
        section_header("Quality-Diversity Optimization")
        info_card(
            "Why MAP-Elites?",
            "Instead of maintaining a single population optimized purely for fitness, MAP-Elites "
            "maintains a 2D grid of <em>elite individuals</em>, one per behavioral niche. "
            "This prevents premature convergence and guarantees the final archive contains "
            "high-quality solutions across the entire behavioral space.",
            accent="blue",
        )

        col1, col2 = st.columns(2)
        with col1:
            info_card(
                "📐  Dimension 1: Tree Complexity",
                "Measured as total AST node count. Binned into 30 buckets. "
                "Ensures discovery of both ultra-compact formulas (e.g. <code>sin(t)</code>) "
                "and complex multi-term schedules.",
            )
            info_card(
                "⚖️  Niche Competition",
                "A new individual enters the archive only if its niche is empty <em>or</em> "
                "its validation loss is strictly lower than the current occupant. "
                "Every niche holds exactly one elite at any time.",
            )
        with col2:
            info_card(
                "📊  Dimension 2: Center of Mass",
                "Computes <code>Σ(t · LR(t)) / Σ(LR(t))</code> over the schedule, "
                "quantifying whether a formula concentrates learning early (warmup) "
                "or late (cooldown). Binned into 20 buckets.",
            )
            info_card(
                "🎲  Parent Selection",
                "Parents are sampled <em>uniformly at random</em> from all occupied niches — "
                "not by fitness rank. This gives rare formulas equal reproductive opportunity, "
                "strongly protecting diversity across generations.",
            )

    # Tab 3: Evaluation Stack
    with tab_eval:
        section_header("Multi-Fidelity Evaluation Stack")
        c1, c2, c3 = st.columns(3)
        with c1:
            info_card(
                "🟢  Low Fidelity",
                "<strong>5% of MNIST</strong> (~3,000 samples). Used during GP evolution for "
                "rapid candidate screening. Entire dataset pinned to GPU VRAM via custom "
                "<code>VRAMDataLoader</code> — zero PCIe transfer overhead per batch.",
                accent="green",
            )
        with c2:
            info_card(
                "🟡  Medium Fidelity",
                "<strong>20% of CIFAR-10</strong> (~10,000 samples). Used for refinement of "
                "shortlisted candidates after the GP run. Harder task forces genuine "
                "generalization beyond MNIST noise.",
                accent="amber",
            )
        with c3:
            info_card(
                "🔴  High Fidelity",
                "<strong>100% of CIFAR-10</strong> (50,000 samples). Final evaluation of "
                "Hall of Fame elites. Full training budget at maximum task difficulty "
                "for definitive ranking.",
                accent="rose",
            )

        info_card(
            "🔄  Concurrent Evaluation Architecture",
            "A <code>ThreadPoolExecutor</code> (1–8 configurable workers) evaluates the population "
            "concurrently. Each worker executes: (1) Rust prefix-expression parsing → "
            "(2) element-wise schedule evaluation → (3) <code>FastConvNet</code> training on GPU. "
            "With VRAM-resident data, GPU utilization stays high while the ThreadPool overlaps "
            "Rust CPU work with GPU kernel launches. A fitness cache prevents re-evaluating "
            "identical ASTs across generations.",
        )

    # Tab 4: Operators
    with tab_ops:
        section_header("Operator Set Design")
        info_card(
            "Protected Numerics",
            "All operators include protection against mathematically undefined behavior. "
            "The Rust core mirrors these protections exactly, ensuring parity between Python "
            "evaluation (used in tests) and production Rust evaluation (used at runtime).",
        )

        ops_df = pd.DataFrame({
            "Operator":   ["+", "−", "×", "÷", "sin", "cos", "exp", "log", "√", "|x|"],
            "Arity":      [2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
            "Protection": [
                "—", "—", "—",
                "Returns 1.0 if |denom| < 1e−6",
                "—", "—",
                "Clamps exponent to [−100, 10]",
                "log(max(|x|, 1e−6))",
                "Operates on |x|",
                "—",
            ],
            "Role": [
                "Additive combination",
                "Subtractive / decay",
                "Scaling / modulation",
                "Ratio schedules",
                "Periodic oscillation",
                "Cosine-family schedules",
                "Exponential warmup / decay",
                "Logarithmic scaling",
                "Sublinear growth",
                "Non-negativity enforcement",
            ],
        })
        st.dataframe(ops_df, use_container_width=True, hide_index=True)

