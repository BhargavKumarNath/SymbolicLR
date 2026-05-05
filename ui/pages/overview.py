"""
ui/pages/overview.py — Page 1: The Problem & Our Approach.

Narrative flow:
  "Learning rate schedules matter → hand-crafting is limited → 
   evolution can discover novel formulas → here's our system"

Includes:
  - Interactive baseline schedule gallery (real curves, not static)
  - System pipeline diagram
  - Core capabilities with engineering context
  - Clear call-to-action to the Evolution Lab
"""

import streamlit as st
from ui.theme import page_header, section_header, info_card
from ui.charts import baseline_gallery_chart
from config.settings import get_config


def render():
    cfg = get_config()

    page_header(
        eyebrow="Research System · Symbolic AutoML",
        title="Symbolic Learning Rate Discovery",
        subtitle=(
            "SymboLR evolves mathematical formulas for neural network training schedules "
            "using Quality-Diversity Genetic Programming — discovering symbolic expressions "
            "that outperform hand-crafted schedules without any human intuition."
        ),
    )

    # ── The Problem ──────────────────────────────────────────────────
    section_header("Why Learning Rate Schedules Matter", "The Problem")

    col_problem, col_viz = st.columns([2, 3])

    with col_problem:
        info_card(
            "📉  The Core Challenge",
            "The learning rate is the single most important hyperparameter in neural network "
            "training. A poorly chosen schedule wastes compute, diverges, or stagnates. "
            "Practitioners spend hours hand-tuning schedules like cosine annealing and step "
            "decay — but these represent only a tiny fraction of all possible mathematical formulas.",
            accent="rose",
        )
        info_card(
            "🧬  Our Approach",
            "Instead of hand-crafting, SymboLR uses <strong>Genetic Programming</strong> to search "
            "the space of all mathematical expressions <code>η(t)</code> built from "
            "<code>{+, -, ×, ÷, sin, cos, exp, log, √, |·|}</code> and the time variable "
            "<code>t ∈ [0, 1]</code>. A <strong>MAP-Elites</strong> archive maintains diverse "
            "elites across schedule timing and formula complexity.",
            accent="green",
        )

    with col_viz:
        st.altair_chart(baseline_gallery_chart(), width='stretch')

    # ── Hero stat strip ──────────────────────────────────────────────
    st.markdown("""
    <div class="stat-row">
        <div class="stat-pill">
            <div class="sp-label">Approach</div>
            <div class="sp-value" style="font-size:14px;color:var(--accent-green);font-family:var(--font-mono);">MAP-Elites GP</div>
        </div>
        <div class="stat-pill">
            <div class="sp-label">Operator Set</div>
            <div class="sp-value">10</div>
        </div>
        <div class="stat-pill">
            <div class="sp-label">Behavioral Dims</div>
            <div class="sp-value">2</div>
        </div>
        <div class="stat-pill">
            <div class="sp-label">Probe Model</div>
            <div class="sp-value" style="font-size:14px;color:var(--accent-blue);font-family:var(--font-mono);">FastConvNet</div>
        </div>
        <div class="stat-pill">
            <div class="sp-label">Runtime</div>
            <div class="sp-value" style="font-size:14px;color:var(--accent-amber);font-family:var(--font-mono);">""" + ("GPU + Rust" if cfg.is_gpu else ("CPU" if cfg.torch_available else "Cloud (Sim)")) + """</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline diagram ─────────────────────────────────────────────
    section_header("System Pipeline", "Architecture")
    st.markdown("""
    <div class="pipeline">
        <div class="pipeline-step active">
            <div class="step-icon">🌱</div>
            <div class="step-label">Init</div>
            <div class="step-name">Ramped H&amp;H</div>
        </div>
        <div class="pipeline-step">
            <div class="step-icon">⚡</div>
            <div class="step-label">Evaluate</div>
            <div class="step-name">""" + ("Rust + GPU" if cfg.is_gpu else "Synthetic Sim") + """</div>
        </div>
        <div class="pipeline-step">
            <div class="step-icon">🗺️</div>
            <div class="step-label">Archive</div>
            <div class="step-name">MAP-Elites</div>
        </div>
        <div class="pipeline-step">
            <div class="step-icon">🔀</div>
            <div class="step-label">Evolve</div>
            <div class="step-name">3 Operators</div>
        </div>
        <div class="pipeline-step">
            <div class="step-icon">✂️</div>
            <div class="step-label">Simplify</div>
            <div class="step-name">SymPy CAS</div>
        </div>
        <div class="pipeline-step">
            <div class="step-icon">🎯</div>
            <div class="step-label">Refine</div>
            <div class="step-name">L-BFGS-B</div>
        </div>
        <div class="pipeline-step active">
            <div class="step-icon">🏆</div>
            <div class="step-label">Output</div>
            <div class="step-name">Hall of Fame</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Core Capabilities ────────────────────────────────────────────
    section_header("Core Capabilities", "Features")
    col1, col2 = st.columns(2)

    with col1:
        info_card(
            "🧬  Quality-Diversity Evolution",
            "MAP-Elites maintains a 2D behavioral archive indexed by <em>schedule timing</em> "
            "(center-of-mass) and <em>formula complexity</em>, ensuring the evolved population "
            "explores diverse regions of the search space rather than converging prematurely.",
            accent="green",
        )
        info_card(
            "⚡  Multi-Backend Evaluation",
            "Formula evaluation routes through Rust (PyO3) when available for production speed, "
            "or falls back to NumPy-based evaluation for cloud deployment. The Python AST evaluator "
            "uses identical protected operators, ensuring consistent behavior across backends.",
            accent="blue",
        )

    with col2:
        info_card(
            "🎯  Hybrid Memetic Optimization",
            "After evolution completes, numeric constants in the top-k formulas are refined via "
            "L-BFGS-B gradient descent bounded to valid learning-rate ranges — combining the "
            "global search of GP with local gradient efficiency.",
            accent="amber",
        )
        info_card(
            "✂️  Algebraic Simplification",
            "Every offspring passes through SymPy CAS to prune bloat "
            "(<code>t + 0</code> → <code>t</code>, <code>t / t</code> → <code>1</code>), "
            "keeping the archive compact and formulas human-readable as LaTeX.",
            accent="rose",
        )

    # ── Getting started callout ──────────────────────────────────────
    mode_note = ""
    if cfg.is_cloud:
        mode_note = (
            " <strong>Note:</strong> This cloud deployment uses a synthetic fitness simulation. "
            "For real GPU-accelerated training, clone the repo and run locally."
        )

    st.markdown(f"""
    <div class="info-card" style="margin-top:24px;background:rgba(0,229,160,0.04);border-color:rgba(0,229,160,0.3);">
        <h4 style="color:var(--accent-green);">→  Getting Started</h4>
        <p>Read the <strong>Methodology</strong> page to understand the technical foundations, then
        navigate to <strong>Evolution Lab</strong> to configure parameters and run your own
        experiment. After evolution completes, <strong>Results & Analysis</strong> will show
        real convergence curves and baseline comparisons.{mode_note}</p>
    </div>
    """, unsafe_allow_html=True)
