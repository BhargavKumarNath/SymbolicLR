"""
ui/pages/evolution_lab.py - Page 3: Live Evolution Lab.
"""

import streamlit as st
from ui.theme import page_header, section_header, formula_block, gen_log_row, info_card
from ui.charts import archive_chart, lr_schedule_chart
from gp.simplify import tree_to_latex
from config.settings import get_config


def _empty_state_notice():
    cfg = get_config()
    mode_hint = ""
    if cfg.is_cloud:
        mode_hint = (
            " The cloud version uses a synthetic fitness simulation - "
            "evolution will produce meaningful results in seconds."
        )
    st.markdown(f"""
    <div class="info-card accent-green" style="margin-bottom:20px;">
        <h4>🧬  Ready to Evolve</h4>
        <p>Configure parameters in the sidebar and click <strong>🚀 Start Evolution</strong>.{mode_hint}</p>
    </div>
    """, unsafe_allow_html=True)
    if cfg.is_cloud:
        st.markdown("""
        <div class="info-card accent-blue" style="margin-bottom:20px;">
            <h4>💡  Recommended Cloud Settings</h4>
            <p><strong>Generations:</strong> 5–10  ·  <strong>Population:</strong> 30–50  ·  
            <strong>Epochs:</strong> 1  ·  <strong>Workers:</strong> 1</p>
        </div>
        """, unsafe_allow_html=True)


def render():
    page_header("Live Execution", "Evolution Lab",
        "Real-time MAP-Elites archive, discovered schedules, and generation metrics.")

    is_running = st.session_state.get("run_status") in {"queued", "running"}
    if is_running:
        st.markdown(f"""
        <div class="info-card accent-blue" style="margin-bottom:20px;">
            <h4>⏱️  Evolution Running</h4>
            <p><strong>{st.session_state.get("progress_label", "Running")}</strong><br/>
            {st.session_state.get("phase_label", "Evaluating candidates")}</p>
        </div>""", unsafe_allow_html=True)

    has_data = st.session_state.get("evolution_done") or st.session_state.get("gen_log") or st.session_state.get("archive_snapshot")
    if not has_data:
        _empty_state_notice()
        return

    log = st.session_state.get("gen_log", [])
    hof = st.session_state.get("hof", [])

    best_loss = log[-1]["best_loss"] if log else 0.0
    niches = log[-1]["niches"] if log else 0
    best_depth = hof[0][1].depth() if hof else 0
    runtime = st.session_state.get("total_time", 0.0)

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Best Val Loss", f"{best_loss:.4f}")
    with k2: st.metric("Niches Discovered", niches)
    with k3: st.metric("Best Formula Depth", best_depth)
    with k4: st.metric("Total Runtime", f"{runtime:.1f}s")

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

    col_arch, col_lr = st.columns(2)
    with col_arch:
        snapshot = st.session_state.get("archive_snapshot", [])
        if snapshot:
            st.altair_chart(archive_chart(snapshot), width='stretch')
    with col_lr:
        curves = st.session_state.get("lr_curves", [])
        if curves:
            st.altair_chart(lr_schedule_chart(curves), width='stretch')

    if hof:
        section_header("Hall of Fame", "Top Formulas")
        for i, (loss, tree) in enumerate(hof):
            formula_block(rank=i+1, loss=loss, size=tree.size(), depth=tree.depth(), latex=tree_to_latex(tree))

    if log:
        section_header("Generation Log", "Per-Gen Stats")
        st.markdown("""<div style="font-family:var(--font-mono);font-size:10px;color:var(--text-muted);
            padding:6px 14px;display:flex;gap:12px;border-bottom:1px solid var(--border);margin-bottom:4px;">
            <span style="width:52px;">GEN</span><span style="width:70px;">BEST LOSS</span>
            <span style="width:60px;">NICHES</span><span>NEW THIS GEN</span></div>""", unsafe_allow_html=True)
        prev_best = float("inf")
        for entry in log:
            is_best = entry["best_loss"] < prev_best
            if is_best: prev_best = entry["best_loss"]
            gen_log_row(entry, is_best)
