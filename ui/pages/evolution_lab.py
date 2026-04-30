"""
ui/pages/evolution_lab.py
Evolution Lab page — live MAP-Elites archive, discovered schedules,
Hall of Fame formulas, and generation-by-generation log.
"""

import streamlit as st
from ui.theme import page_header, section_header, formula_block, gen_log_row
from ui.charts import archive_chart, lr_schedule_chart
from gp.simplify import tree_to_latex


def _empty_state_notice():
    st.markdown("""
    <div class="info-card accent-green" style="margin-bottom:20px;">
        <h4>⏳  Waiting for Evolution Run</h4>
        <p>No evolution data yet. Configure your parameters in the sidebar and click
        <strong>🚀 Start Evolution</strong> to populate this page with live results.</p>
    </div>
    """, unsafe_allow_html=True)


def _running_notice():
    st.markdown(
        f"""
        <div class="info-card accent-blue" style="margin-bottom:20px;">
            <h4>⏱️  Evolution Running</h4>
            <p>
                <strong>{st.session_state.get("progress_label", "Running")}</strong><br/>
                {st.session_state.get("phase_label", "Evaluating candidates")}<br/>
                Navigation is safe now — use any page or click <strong>Refresh Run Status</strong> in the sidebar to pull the latest snapshot.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render():
    page_header(
        eyebrow="Live Execution",
        title="Evolution Lab",
        subtitle=(
            "Real-time view of the MAP-Elites archive, discovered schedules, and "
            "generation-by-generation metrics. Configure parameters in the sidebar "
            "and click Start Evolution to begin."
        ),
    )

    is_running = st.session_state.get("run_status") in {"queued", "running"}
    if is_running:
        _running_notice()

    if not st.session_state.get("evolution_done") and not st.session_state.get("gen_log") and not st.session_state.get("archive_snapshot"):
        _empty_state_notice()
        return

    log = st.session_state["gen_log"]
    hof = st.session_state["hof"]

    # KPI row
    best_loss = log[-1]["best_loss"] if log else 0.0
    niches    = log[-1]["niches"]    if log else 0
    best_depth = hof[0][1].depth()  if hof else 0
    runtime    = st.session_state["total_time"]

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Best Val Loss",       f"{best_loss:.4f}")
    with k2: st.metric("Niches Discovered",   niches)
    with k3: st.metric("Best Formula Depth",  best_depth)
    with k4: st.metric("Total Runtime",       f"{runtime:.1f}s")

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

    # Main charts
    col_arch, col_lr = st.columns(2)

    with col_arch:
        if st.session_state["archive_snapshot"]:
            st.altair_chart(
                archive_chart(st.session_state["archive_snapshot"]),
                width='stretch',
            )

    with col_lr:
        if st.session_state["lr_curves"]:
            st.altair_chart(
                lr_schedule_chart(st.session_state["lr_curves"]),
                width='stretch',
            )

    # Hall of Fame
    section_header("Hall of Fame", "Top Formulas")

    for i, (loss, tree) in enumerate(hof):
        formula_block(
            rank=i + 1,
            loss=loss,
            size=tree.size(),
            depth=tree.depth(),
            latex=tree_to_latex(tree),
        )

    # Generation log
    section_header("Generation Log", "Per-Gen Stats")

    st.markdown("""
    <div style="font-family:var(--font-mono);font-size:10px;color:var(--text-muted);
                padding:6px 14px;display:flex;gap:12px;
                border-bottom:1px solid var(--border);margin-bottom:4px;">
        <span style="width:52px;">GEN</span>
        <span style="width:70px;">BEST LOSS</span>
        <span style="width:60px;">NICHES</span>
        <span>NEW THIS GEN</span>
    </div>
    """, unsafe_allow_html=True)

    prev_best = float("inf")
    for entry in log:
        is_best = entry["best_loss"] < prev_best
        if is_best:
            prev_best = entry["best_loss"]
        gen_log_row(entry, is_best)

