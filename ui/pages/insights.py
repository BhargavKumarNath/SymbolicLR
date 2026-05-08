"""
ui/pages/insights.py - Page 5: Insights, Deployment Guide, and Engineering Rationale.

Combines research findings with practical deployment information.
Dynamic insights derived from run data when available.
"""

import streamlit as st
from ui.theme import page_header, section_header, info_card
from config.settings import get_config


_FINDINGS = [
    ("📉  Early Warmup is Almost Universal",
     "Across all discovered elite formulas, schedules with center-of-mass below 0.5 "
     "consistently outperform late-heavy schedules. The model needs stable gradients "
     "early - aggressive initial learning rates cause irreversible damage to the loss landscape.",
     "green"),
    ("📐  Compact Beats Complex",
     "The best-performing formulas almost never exceed 7–9 AST nodes. Simplicity correlates "
     "strongly with generalization: complex formulas tend to overfit to low-fidelity noise "
     "rather than capturing universal optimization geometry.",
     "blue"),
    ("🔄  Oscillatory Schedules Surprise",
     "MAP-Elites consistently discovers periodic sin/cos-based formulas that achieve competitive "
     "validation loss. These emerge as structurally equivalent to warmup-with-restarts - but "
     "arise purely from the fitness signal, without any human prior.",
     "amber"),
    ("🧮  The t Variable is Non-Negotiable",
     "Formulas containing no reference to the time variable <code>t</code> (i.e. constant "
     "schedules) cluster in the worst-performing niches. Even a single multiplication by "
     "<code>t</code> is sufficient to produce meaningful adaptive decay.",
     "rose"),
]

_ENGINEERING = [
    ("Why Rust for evaluation?",
     "Python AST traversal with NumPy incurs significant per-call overhead when evaluating "
     "hundreds of formulas per generation. The Rust PyO3 extension parses prefix notation once "
     "and evaluates element-wise in a single tight loop - achieving ~10–50× throughput "
     "improvement over pure Python for complex expressions."),
    ("Why VRAM-resident DataLoader?",
     "Standard PyTorch DataLoaders involve CPU→GPU PCIe transfers every batch. With 8 GB VRAM "
     "and a 3,000-sample low-fidelity dataset, the entire training corpus fits in GPU memory. "
     "The custom VRAMDataLoader eliminates PCIe entirely."),
    ("Why SymPy for simplification?",
     "GP bloat is a known failure mode. SymPy CAS reduces algebraic identities "
     "(e.g. <code>(t + 0) × 1</code> → <code>t</code>) keeping the search space compact."),
    ("Why synthetic fitness in cloud mode?",
     "Streamlit Cloud has strict RAM limits and no GPU. Instead of a broken mock, "
     "the synthetic mode simulates SGD on a quadratic loss surface with realistic dynamics. "
     "This produces meaningful evolution without requiring PyTorch."),
]

_ROADMAP = [
    ("🔬", "Multi-Fidelity Pipeline",
     "Cascade compute from MNIST to CIFAR-10 for promising candidates."),
    ("📈", "Adaptive Operator Rates",
     "Reweight crossover/mutation/hoist based on fitness improvement."),
    ("🌐", "Transfer to Transformers",
     "Test discovered schedules on language model fine-tuning tasks."),
    ("🧪", "MLflow Tracking",
     "Systematic reproducibility analysis across seeds and configs."),
    ("🤖", "Neural-Guided GP",
     "Surrogate model to pre-screen offspring before evaluation."),
]


def _dynamic_insights():
    """Generate insights from actual run data when available."""
    hof = st.session_state.get("hof", [])
    gen_log = st.session_state.get("gen_log", [])
    archive = st.session_state.get("archive_snapshot", [])

    if not hof or not gen_log:
        return

    section_header("Run-Specific Insights", "From Your Experiment")

    best_loss = hof[0][0]
    best_tree = hof[0][1]
    total_niches = gen_log[-1]["niches"] if gen_log else 0

    col1, col2 = st.columns(2)
    with col1:
        info_card("🏆  Best Discovered Formula",
            f"Loss: <strong>{best_loss:.4f}</strong> · "
            f"Size: {best_tree.size()} nodes · Depth: {best_tree.depth()}<br/>"
            f"Formula: <code>{str(best_tree)[:80]}</code>",
            accent="green")
    with col2:
        improvement = gen_log[0]["best_loss"] - gen_log[-1]["best_loss"] if len(gen_log) > 1 else 0
        info_card("📊  Evolution Progress",
            f"Generations: {len(gen_log)} · Niches discovered: {total_niches}<br/>"
            f"Loss improvement: <strong>{improvement:.4f}</strong> from Gen 1 to final",
            accent="blue")


def render():
    cfg = get_config()

    page_header("Research Findings", "Insights & Deployment",
        "Key takeaways, engineering decisions, and how to run SymboLR at full power.")

    # Dynamic insights from actual run
    _dynamic_insights()

    # Static findings
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

    # Deployment guide
    section_header("Run Locally with Full GPU Power", "Deployment")

    if cfg.is_cloud:
        st.warning("☁️ You're viewing the cloud demo. For full performance, run locally.")

    st.markdown("""
    <div class="info-card accent-green">
        <h4>🚀  Local GPU Setup (3 commands)</h4>
        <p>
            <code style="color:var(--accent-green);font-family:var(--font-mono);font-size:12px;">
            git clone https://github.com/BhargavKumarNath/SymbolicLR.git<br/>
            pip install -r requirements-gpu.txt<br/>
            streamlit run app.py
            </code>
        </p>
        <p style="margin-top:8px;">Requires: Python 3.10+, CUDA 12.x, 8GB+ VRAM (RTX 3060+)</p>
    </div>
    """, unsafe_allow_html=True)

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
