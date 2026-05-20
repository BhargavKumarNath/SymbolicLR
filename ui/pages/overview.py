"""
ui/pages/overview.py - Page 1: The Problem & Our Approach.

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
            """
            • Evolves mathematical learning rate schedules for neural network training<br>
            • Powered by Quality Diversity Genetic Programming<br>
            • Discovers interpretable symbolic expressions automatically<br>
            • Outperforms traditional hand designed schedules<br>
            • Requires no human intuition or manual tuning<br>
            """
        ),
    )

    # The Problem
    section_header("Why Learning Rate Schedules Matter", "The Problem")

    col_problem, col_viz = st.columns([2, 3])

    with col_problem:
        info_card(
            "📉  The Core Challenge",
            "The learning rate is the single most important hyperparameter in neural network "
            "training. A poorly chosen schedule wastes compute, diverges, or stagnates. "
            "Practitioners spend hours hand-tuning schedules like cosine annealing and step "
            "decay but these represent only a tiny fraction of all possible mathematical formulas.",
            accent="rose",
        )
        info_card(
            "🧬  Our Approach",
            "Instead of hand-crafting, SymboLR uses <strong>Genetic Programming</strong> to search "
            "the space of all mathematical expressions <code>η(t)</code> built from "
            "<code>{+, -, x, ÷, sin, cos, exp, log, √, |·|}</code> and the time variable "
            "<code>t ∈ [0, 1]</code>. A <strong>MAP-Elites</strong> archive maintains diverse "
            "elites across schedule timing and formula complexity.",
            accent="green",
        )

    with col_viz:
        st.altair_chart(baseline_gallery_chart(), width='stretch')

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
            <div class="sp-label">Runtime</div>
            <div class="sp-value" style="font-size:14px;color:var(--accent-amber);font-family:var(--font-mono);">""" + ("GPU + Rust" if cfg.is_gpu else ("CPU" if cfg.torch_available else "Cloud (Sim)")) + """</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline diagram
    section_header("System Pipeline", "Architecture")
    
    pipeline_html = """
<style>
.pipeline-tabs input[type="radio"] { display: none; }
.pipeline-content {
    display: none;
    padding: 24px;
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-top: 16px;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }

.pipeline-content h4 { color: var(--text-primary); margin: 0 0 12px 0; font-size: 18px; }
.pipeline-content p { color: var(--text-secondary); margin: 0 0 12px 0; font-family: var(--font-body); font-size: 14.5px; line-height: 1.6; }
.pipeline-content p:last-child { margin-bottom: 0; }

/* Show content based on checked radio */
#step1:checked ~ .content-area #content1,
#step2:checked ~ .content-area #content2,
#step3:checked ~ .content-area #content3,
#step4:checked ~ .content-area #content4,
#step5:checked ~ .content-area #content5,
#step6:checked ~ .content-area #content6,
#step7:checked ~ .content-area #content7 { display: block; }

/* Highlight active pipeline step */
.pipeline label { cursor: pointer; flex: 1; display: flex; }
.pipeline label .pipeline-step { width: 100%; transition: all 0.2s ease; border-left: none; }
.pipeline label:first-child .pipeline-step { border-left: 1px solid var(--border); border-radius: 10px 0 0 10px; }
.pipeline label:last-child .pipeline-step { border-radius: 0 10px 10px 0; }
.pipeline label:hover .pipeline-step { background: rgba(255,255,255,0.03); }

#step1:checked ~ .pipeline label[for="step1"] .pipeline-step,
#step2:checked ~ .pipeline label[for="step2"] .pipeline-step,
#step3:checked ~ .pipeline label[for="step3"] .pipeline-step,
#step4:checked ~ .pipeline label[for="step4"] .pipeline-step,
#step5:checked ~ .pipeline label[for="step5"] .pipeline-step,
#step6:checked ~ .pipeline label[for="step6"] .pipeline-step,
#step7:checked ~ .pipeline label[for="step7"] .pipeline-step {
    background: var(--bg-overlay);
    border-color: var(--accent-green);
    box-shadow: inset 0 2px 0 var(--accent-green);
}

/* Remove original active class since it's driven by radio now */
.pipeline-step.active { background: inherit; border-color: inherit; }
</style>

<div class="pipeline-tabs">
    <input type="radio" name="pl" id="step1" checked>
    <input type="radio" name="pl" id="step2">
    <input type="radio" name="pl" id="step3">
    <input type="radio" name="pl" id="step4">
    <input type="radio" name="pl" id="step5">
    <input type="radio" name="pl" id="step6">
    <input type="radio" name="pl" id="step7">
    
    <div class="pipeline">
        <label for="step1">
            <div class="pipeline-step">
                <div class="step-icon">🌱</div>
                <div class="step-label">Init</div>
                <div class="step-name">Ramped H&amp;H</div>
            </div>
        </label>
        <label for="step2">
            <div class="pipeline-step">
                <div class="step-icon">⚡</div>
                <div class="step-label">Evaluate</div>
                <div class="step-name">""" + ("Rust + GPU" if cfg.is_gpu else "Synthetic Sim") + """</div>
            </div>
        </label>
        <label for="step3">
            <div class="pipeline-step">
                <div class="step-icon">🗺️</div>
                <div class="step-label">Archive</div>
                <div class="step-name">MAP-Elites</div>
            </div>
        </label>
        <label for="step4">
            <div class="pipeline-step">
                <div class="step-icon">🔀</div>
                <div class="step-label">Evolve</div>
                <div class="step-name">3 Operators</div>
            </div>
        </label>
        <label for="step5">
            <div class="pipeline-step">
                <div class="step-icon">✂️</div>
                <div class="step-label">Simplify</div>
                <div class="step-name">SymPy CAS</div>
            </div>
        </label>
        <label for="step6">
            <div class="pipeline-step">
                <div class="step-icon">🎯</div>
                <div class="step-label">Refine</div>
                <div class="step-name">L-BFGS-B</div>
            </div>
        </label>
        <label for="step7">
            <div class="pipeline-step">
                <div class="step-icon">🏆</div>
                <div class="step-label">Output</div>
                <div class="step-name">Hall of Fame</div>
            </div>
        </label>
    </div>
    
    <div class="content-area">
        <div id="content1" class="pipeline-content">
            <h4 style="color:var(--accent-green);">🌱 Initialization: Ramped Half-and-Half</h4>
            <p>The evolutionary process begins by generating a highly diverse initial population of mathematical formulas. SymboLR uses the <strong>Ramped Half-and-Half</strong> method.</p>
            <p>Half the population is generated using the "full" method (symmetric, deep trees) and half using the "grow" method (asymmetric, variable depth). This guarantees structural variety across the search space right from generation zero.</p>
        </div>
        <div id="content2" class="pipeline-content">
            <h4 style="color:var(--accent-blue);">⚡ Evaluation: Multi-Fidelity Fitness</h4>
            <p>Each schedule must be evaluated to determine how well it trains a neural network. In full <strong>GPU mode</strong>, SymboLR passes the formula to a Rust engine which compiles it for lightning-fast execution, and then trains a FastConvNet probe on a subset of CIFAR-10.</p>
            <p>In <strong>Cloud mode</strong>, to respect memory limits, it evaluates using a <em>Synthetic Loss Landscape Simulator</em> which mathematically models gradient descent over a complex quadratic surface, providing realistic convergence dynamics without needing PyTorch.</p>
        </div>
        <div id="content3" class="pipeline-content">
            <h4 style="color:var(--accent-amber);">🗺️ Archive: MAP-Elites Grid</h4>
            <p>Instead of keeping just the single "best" formula, SymboLR stores formulas in a 2D <strong>Quality-Diversity Archive</strong> based on their behavior: their AST tree complexity and their temporal center-of-mass.</p>
            <p>This prevents premature convergence. If a formula is uniquely compact, or concentrates its learning rate unusually early in the training process, it is preserved even if its overall fitness isn't perfect yet.</p>
        </div>
        <div id="content4" class="pipeline-content">
            <h4 style="color:var(--text-primary);">🔀 Evolution: Genetic Operators</h4>
            <p>The system iteratively improves the archive using three biologically-inspired operators:</p>
            <ul style="color:var(--text-secondary); font-family:var(--font-body); font-size:14.5px; line-height:1.6; margin:0 0 12px 0;">
                <li><strong>Subtree Crossover:</strong> Swaps random mathematical sub-expressions between two parent schedules.</li>
                <li><strong>Subtree Mutation:</strong> Replaces a random node with entirely new math.</li>
                <li><strong>Hoist Mutation:</strong> Replaces a tree with one of its own subtrees, aggressively fighting bloat.</li>
            </ul>
        </div>
        <div id="content5" class="pipeline-content">
            <h4 style="color:var(--accent-rose);">✂️ Simplification: SymPy CAS</h4>
            <p>Genetic programming is notorious for creating mathematically bloated, unreadable formulas (e.g., <code>(t * 1) + 0</code>).</p>
            <p>Before any formula enters the archive, it is passed through the <strong>SymPy Computer Algebra System</strong>. SymPy applies algebraic reduction rules to completely eliminate redundant terms, ensuring the final formulas are as clean and interpretable as if a human mathematician wrote them.</p>
        </div>
        <div id="content6" class="pipeline-content">
            <h4 style="color:var(--accent-green);">🎯 Refinement: Memetic Optimization</h4>
            <p>Genetic Programming is excellent at global search (finding the right mathematical structure), but poor at local search (finding the perfect numeric constants like <code>0.142</code> vs <code>0.100</code>).</p>
            <p>Once evolution completes, SymboLR extracts the top formulas and applies <strong>L-BFGS-B gradient descent</strong> directly to their scalar constants. This hybrid approach combines the structural discovery of GP with the numerical precision of gradient descent.</p>
        </div>
        <div id="content7" class="pipeline-content">
            <h4 style="color:var(--accent-amber);">🏆 Output: Hall of Fame</h4>
            <p>The final, simplified, and refined formulas are ranked strictly by their validation loss.</p>
            <p>The system automatically translates the internal Abstract Syntax Trees (ASTs) into clean <strong>LaTeX equations</strong>, allowing researchers and engineers to instantly copy and implement the newly discovered learning rate schedules into their own PyTorch or TensorFlow projects.</p>
        </div>
    </div>
</div>
"""
    st.html(pipeline_html)

    # Core Capabilities
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
            "L-BFGS-B gradient descent bounded to valid learning-rate ranges - combining the "
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
