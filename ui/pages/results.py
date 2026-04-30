"""
ui/pages/results.py
Results & Analysis page - convergence curves, niche growth, baseline
comparison, and full Hall of Fame detail table.
"""

import pandas as pd
import streamlit as st
from ui.theme import page_header, section_header, baseline_row
from ui.charts import (
    loss_convergence_chart,
    niche_growth_chart,
    baseline_comparison_chart,
    BASELINE_DATA,
)
from gp.simplify import tree_to_latex


def render():
    page_header(
        eyebrow="Post-Run Analysis",
        title="Results & Analysis",
        subtitle=(
            "Convergence curves, niche growth statistics, baseline comparisons, "
            "and the final Hall of Fame formulas presented for in-depth evaluation."
        ),
    )

    is_running = st.session_state.get("run_status") in {"queued", "running"}
    gen_log = st.session_state.get("gen_log", [])
    hof = st.session_state.get("hof", [])

    if is_running:
        st.markdown(
            f"""
            <div class="info-card accent-blue" style="margin-bottom:20px;">
                <h4>⏱️  Run In Progress</h4>
                <p>
                    <strong>{st.session_state.get("progress_label", "Running")}</strong><br/>
                    {st.session_state.get("phase_label", "Waiting for more results")}<br/>
                    This page reads from the active background run, so navigation will not interrupt evolution.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if len(gen_log) >= 2:
        col_conv, col_niche = st.columns([3, 2])
        with col_conv:
            st.altair_chart(
                loss_convergence_chart(gen_log),
                width='stretch',
            )
        with col_niche:
            st.altair_chart(
                niche_growth_chart(gen_log),
                width='stretch',
            )
    else:
        st.markdown(
            """
            <div class="info-card" style="margin-bottom:20px;">
                <h4>No evolution data yet</h4>
                <p>Run an evolution from the sidebar. Convergence curves and niche statistics
                will appear here automatically.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_header("Baseline Comparison", "Benchmark")
    st.altair_chart(baseline_comparison_chart(), width='stretch')

    worst = max(b["Val Loss"] for b in BASELINE_DATA)
    for row in BASELINE_DATA:
        baseline_row(
            name=row["Schedule"],
            loss=row["Val Loss"],
            worst=worst,
            is_winner=(row["Type"] == "Discovered"),
        )

    if hof:
        section_header("Hall of Fame - Full Detail", "Discovered Formulas")
        rows = []
        for i, (loss, tree) in enumerate(hof):
            prefix = tree.to_prefix()
            rows.append(
                {
                    "Rank": i + 1,
                    "Val Loss": round(loss, 5),
                    "Nodes": tree.size(),
                    "Depth": tree.depth(),
                    "LaTeX": f"${tree_to_latex(tree)}$",
                    "Prefix": prefix[:60] + ("..." if len(prefix) > 60 else ""),
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            width='stretch',
            hide_index=True,
        )
