"""
ui/pages/results.py - Page 4: Results & Analysis.
Real convergence curves, computed baseline comparison, and Hall of Fame detail.
"""

import pandas as pd
import streamlit as st
from ui.theme import page_header, section_header, baseline_row
from ui.charts import loss_convergence_chart, niche_growth_chart, baseline_comparison_chart
from gp.simplify import tree_to_latex
from optimiser.compare import get_comparison_data


def render():
    page_header("Post-Run Analysis", "Results & Analysis",
        "Convergence curves, niche growth, baseline comparisons, and discovered formulas.")

    is_running = st.session_state.get("run_status") in {"queued", "running"}
    gen_log = st.session_state.get("gen_log", [])
    hof = st.session_state.get("hof", [])

    if is_running:
        st.markdown(f"""
        <div class="info-card accent-blue" style="margin-bottom:20px;">
            <h4>⏱️  Run In Progress</h4>
            <p><strong>{st.session_state.get("progress_label", "Running")}</strong><br/>
            {st.session_state.get("phase_label", "Waiting for results")}</p>
        </div>""", unsafe_allow_html=True)

    # Convergence + Niche charts
    if len(gen_log) >= 2:
        col_conv, col_niche = st.columns([3, 2])
        with col_conv:
            st.altair_chart(loss_convergence_chart(gen_log), width='stretch')
        with col_niche:
            st.altair_chart(niche_growth_chart(gen_log), width='stretch')
    else:
        st.markdown("""
        <div class="info-card" style="margin-bottom:20px;">
            <h4>No evolution data yet</h4>
            <p>Run an evolution from the sidebar. Convergence curves will appear automatically.</p>
        </div>""", unsafe_allow_html=True)

    # Baseline comparison (REAL DATA)
    section_header("Baseline Comparison", "Benchmark")
    symbolr_loss = hof[0][0] if hof else None
    comparison_data = get_comparison_data(symbolr_loss)
    st.altair_chart(baseline_comparison_chart(comparison_data), width='stretch')

    worst = max(b["Val Loss"] for b in comparison_data)
    for row in comparison_data:
        baseline_row(
            name=row["Schedule"],
            loss=row["Val Loss"],
            worst=worst,
            is_winner=(row["Type"] == "Discovered"),
        )

    # Hall of Fame detail table
    if hof:
        section_header("Hall of Fame - Full Detail", "Discovered Formulas")
        md_table = "| Rank | Val Loss | Nodes | Depth | Formula | Prefix |\n|---|---|---|---|---|---|\n"
        for i, (loss, tree) in enumerate(hof):
            prefix = tree.to_prefix()
            prefix_trunc = prefix[:60] + ("..." if len(prefix) > 60 else "")
            latex = tree_to_latex(tree)
            md_table += f"| {i + 1} | {loss:.5f} | {tree.size()} | {tree.depth()} | ${latex}$ | `{prefix_trunc}` |\n"
            
        st.markdown(md_table)
