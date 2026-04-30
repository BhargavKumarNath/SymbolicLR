"""
app.py - SymboLR dashboard entry point.

Responsibilities:
  1. Streamlit page config
  2. Theme injection
  3. Sidebar - logo, navigation radio, parameter sliders, run controls, live stats
  4. Route to the correct page module
  5. Launch background evolution runs
  6. Non-blocking live progress via st.fragment(run_every=...)
"""

import streamlit as st

st.set_page_config(
    page_title="SymboLR",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.theme import inject_theme
from ui.state import init_state, sync_state, start_evolution
from ui.pages import overview, methodology, evolution_lab, results, insights


inject_theme()
init_state()
sync_state()

# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="logo-mark">🧬 Symbo<span>LR</span></div>
            <div class="logo-sub">Symbolic LR Discovery</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nav-section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "nav",
        [
            "🏠  Overview",
            "🔬  Methodology",
            "🚀  Evolution Lab",
            "📊  Results & Analysis",
            "⚡  Insights",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="nav-section-label">Configuration</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div style="padding:0 10px;">', unsafe_allow_html=True)
        gen_count = st.slider("Generations", 1, 50, 5)
        pop_size  = st.slider("Population Size", 10, 200, 50)
        epochs    = st.slider("Epochs per Eval", 1, 5, 1)
        workers   = st.slider("Parallel Workers", 1, 8, 4)
        st.markdown("</div>", unsafe_allow_html=True)

    is_running = st.session_state.get("run_status") in {"queued", "running"}
    st.markdown('<div style="padding:10px 10px 0;">', unsafe_allow_html=True)
    run_btn = st.button("🚀  Start Evolution", disabled=is_running)
    st.markdown("</div>", unsafe_allow_html=True)

    if is_running:
        # Live progress panel
        @st.fragment(run_every=1.5)
        def _live_progress():
            sync_state()
            ratio = float(st.session_state.get("progress_ratio", 0.0))
            label = st.session_state.get("progress_label", "Running…")
            phase = st.session_state.get("phase_label", "")
            completed = st.session_state.get("eval_completed", 0)
            total     = st.session_state.get("eval_total", 0)

            st.progress(ratio)
            st.markdown(
                f"""
                <div class="sidebar-stat">
                    <div class="ss-label">Run Status</div>
                    <div class="ss-value green">{label}</div>
                    <div style="font-family:var(--font-body);font-size:12px;
                                color:var(--text-secondary);margin-top:4px;">
                        {phase}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if total > 0:
                st.caption(f"Candidates: {completed} / {total}")

            # When the run finishes, trigger a full-page rerun once so the
            # completed results render on the active page.
            status = st.session_state.get("run_status", "idle")
            if status not in {"queued", "running"}:
                st.rerun()

        _live_progress()

    elif st.session_state.get("run_status") == "failed":
        st.error(st.session_state.get("run_error", "Evolution run failed."))

    if st.session_state.get("evolution_done"):
        log  = st.session_state["gen_log"]
        best = log[-1]["best_loss"] if log else 0.0
        niches = log[-1]["niches"] if log else 0
        st.markdown(
            f"""
            <div class="sidebar-stat">
                <div class="ss-label">Best Val Loss</div>
                <div class="ss-value green">{best:.4f}</div>
            </div>
            <div class="sidebar-stat">
                <div class="ss-label">Niches Discovered</div>
                <div class="ss-value">{niches}</div>
            </div>
            <div class="sidebar-stat">
                <div class="ss-label">Total Runtime</div>
                <div class="ss-value">{st.session_state["total_time"]:.1f}s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Run trigger
if run_btn:
    started = start_evolution(gen_count, pop_size, epochs, workers)
    if not started:
        st.warning("An evolution run is already active for this session.")
    st.rerun()

# Page routing
if "Overview" in page:
    overview.render()
elif "Methodology" in page:
    methodology.render()
elif "Evolution Lab" in page:
    evolution_lab.render()
elif "Results" in page:
    results.render()
elif "Insights" in page:
    insights.render()