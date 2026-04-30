"""
ui/pages/insights.py
Insights & Interpretation page — key research findings, engineering rationale,
and future roadmap cards.
"""

import streamlit as st
from ui.theme import page_header, section_header, info_card


# Static content data
_FINDINGS = [
    (
        "📉  Early Warmup is Almost Universal",
        "Across all discovered elite formulas, schedules with center-of-mass below 0.5 "
        "consistently outperform late-heavy schedules. The model needs stable gradients "
        "early — aggressive initial learning rates cause irreversible damage to the loss landscape.",
        "green",
    ),
    (
        "📐  Compact Beats Complex",
        "The best-performing formulas almost never exceed 7–9 AST nodes. Simplicity correlates "
        "strongly with generalization: complex formulas tend to overfit to low-fidelity MNIST noise "
        "rather than capturing universal optimization geometry.",
        "blue",
    ),
    (
        "🔄  Oscillatory Schedules Surprise",
        "MAP-Elites consistently discovers periodic sin/cos-based formulas that achieve competitive "
        "validation loss. These emerge as structurally equivalent to warmup-with-restarts — but "
        "arise purely from the fitness signal, without any human prior.",
        "amber",
    ),
    (
        "🧮  The t Variable is Non-Negotiable",
        "Formulas containing no reference to the time variable <code>t</code> (i.e. constant "
        "schedules) cluster in the worst-performing niches. Even a single multiplication by "
        "<code>t</code> is sufficient to produce meaningful adaptive decay.",
        "rose",
    ),
]

_ENGINEERING = [
    (
        "Why Rust for evaluation?",
        "Python AST traversal with NumPy incurs significant per-call overhead when evaluating "
        "hundreds of formulas per generation. The Rust PyO3 extension parses prefix notation once "
        "and evaluates element-wise in a single tight loop — achieving ~10–50× throughput "
        "improvement over pure Python for complex expressions.",
    ),
    (
        "Why VRAM-resident DataLoader?",
        "Standard PyTorch DataLoaders involve CPU→GPU PCIe transfers every batch. With 8 GB VRAM "
        "and a 3,000-sample low-fidelity dataset, the entire training corpus fits in GPU memory. "
        "The custom VRAMDataLoader eliminates PCIe entirely, reducing per-batch time to pure "
        "kernel compute.",
    ),
    (
        "Why SymPy for simplification?",
        "GP bloat is a known failure mode: trees grow arbitrarily large without improving fitness, "
        "wasting archive slots and slowing Rust evaluation. SymPy CAS reduces algebraic "
        "identities (e.g. <code>(t + 0) × 1</code> → <code>t</code>) keeping the search space "
        "compact and final formulas human-readable.",
    ),
    (
        "Why L-BFGS-B post-processing?",
        "GP excels at discovering structure (e.g. 'the formula should involve sin(t)') but is "
        "poor at fine-tuning scalar constants. L-BFGS-B injects gradient information locally after "
        "GP terminates, combining global structure search with local numeric precision.",
    ),
    (
        "Why uniform parent sampling in MAP-Elites?",
        "Fitness-proportional selection collapses behavioral diversity by letting dominant niches "
        "monopolize reproduction. Uniform sampling from occupied niches gives equal reproductive "
        "probability to a simple <code>sin(t)</code> and a complex multi-term formula, "
        "preserving archive breadth across generations.",
    ),
]

_ROADMAP = [
    ("🔬", "Multi-Fidelity Pipeline",
     "Integrate medium and high-fidelity evaluation tiers: CIFAR-10 subset for shortlisted "
     "candidates, full CIFAR-10 for final ranking. This cascades compute toward the "
     "most promising formulas."),
    ("📈", "Adaptive Operator Rates",
     "Use a sliding window of per-operator offspring fitness improvements to adaptively "
     "reweight crossover vs. mutation vs. hoist probabilities during evolution."),
    ("🌐", "Transfer to Transformers",
     "Evaluate discovered schedules on language model fine-tuning tasks to test "
     "cross-architecture generalization. A formula discovered on MNIST convnets may "
     "transfer surprisingly well."),
    ("🧪", "MLflow Experiment Tracking",
     "Add MLflow logging to systematically compare evolution runs, enabling reproducible "
     "research comparisons across hyperparameter configurations and random seeds."),
    ("🤖", "Neural-Guided GP",
     "Train a lightweight surrogate model (LSTM or GNN) on the archive to predict fitness "
     "from AST structure, enabling low-cost pre-screening of offspring before GPU evaluation."),
]


# Render
def render():
    page_header(
        eyebrow="Research Findings",
        title="Insights & Interpretation",
        subtitle=(
            "Key takeaways from the SymboLR research system, engineering decisions that matter, "
            "and the road ahead for symbolic AutoML."
        ),
    )

    # Findings grid
    section_header("What Makes a Good LR Schedule?", "Findings")
    col1, col2 = st.columns(2)
    for i, (title, body, accent) in enumerate(_FINDINGS):
        with (col1 if i % 2 == 0 else col2):
            info_card(title, body, accent=accent)

    # Engineering decisions
    section_header("Engineering Decisions", "Design Rationale")
    for title, body in _ENGINEERING:
        with st.expander(title):
            st.markdown(
                f'<p style="font-family:var(--font-body);font-size:14px;'
                f'color:var(--text-secondary);line-height:1.65;">{body}</p>',
                unsafe_allow_html=True,
            )

    # Roadmap
    section_header("Future Directions", "Roadmap")
    cols = st.columns(len(_ROADMAP))
    for col, (icon, title, desc) in zip(cols, _ROADMAP):
        with col:
            st.markdown(f"""
            <div class="info-card" style="height:100%;">
                <div style="font-size:24px;margin-bottom:8px;">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

