"""
ui/pages/overview.py
Overview page — system purpose, pipeline diagram, and feature cards.
"""

import streamlit as st
from ui.theme import page_header, section_header, info_card


def render():
    page_header(
        eyebrow="Research System · v1.0",
        title="Symbolic Learning Rate Discovery",
        subtitle=(
            "SymboLR evolves mathematical formulas for neural network training schedules "
            "using Quality-Diversity Genetic Programming — discovering symbolic expressions "
            "that outperform hand-crafted schedules without any human intuition."
        ),
    )

    # Hero stat strip
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
            <div class="sp-label">Fidelity Tiers</div>
            <div class="sp-value">3</div>
        </div>
        <div class="stat-pill">
            <div class="sp-label">Eval Engine</div>
            <div class="sp-value" style="font-size:14px;color:var(--accent-amber);font-family:var(--font-mono);">Rust + GPU</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline diagram
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
            <div class="step-name">Rust + GPU</div>
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

    # Feature cards
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
            "⚡  Rust-Accelerated Evaluation",
            "Formula evaluation is compiled to native Rust via PyO3, bypassing Python overhead "
            "entirely. The Rust core mirrors protected NumPy semantics (guarded ÷, log, √, exp) "
            "and achieves near-C performance for AST traversal.",
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

    # Getting started callout
    st.markdown("""
    <div class="info-card" style="margin-top:24px;background:rgba(0,229,160,0.04);border-color:rgba(0,229,160,0.3);">
        <h4 style="color:var(--accent-green);">→  Getting Started</h4>
        <p>Configure evolution parameters in the sidebar (generations, population size, evaluation epochs,
        parallel workers), then click <strong>🚀 Start Evolution</strong>. Navigate to
        <strong>Evolution Lab</strong> to watch the archive grow in real time, or review
        <strong>Results &amp; Analysis</strong> after the run completes.</p>
    </div>
    """, unsafe_allow_html=True)

