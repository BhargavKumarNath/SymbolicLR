"""
ui/charts.py — Altair chart factories for SymboLR dashboard.
Every function accepts plain Python data structures and returns an alt.Chart.

"""

import altair as alt
import numpy as np
import pandas as pd
from ui.theme import VEGA_CONFIG

# Accent palette shared across charts
_RANK_COLORS = ["#00e5a0", "#0ea5e9", "#f59e0b", "#f43f5e", "#a78bfa"]
_BASELINE_COLOR = "#4a6080"
_DISCOVERED_COLOR = "#00e5a0"


# Archive scatter
def archive_chart(snapshot: list) -> alt.Chart:
    """
    2D scatter of the MAP-Elites behavioral archive.
    X = Center of Mass (schedule timing), Y = Tree Complexity, Color = Val Loss.
    """
    df = pd.DataFrame(snapshot)
    return (
        alt.Chart(df)
        .mark_circle(opacity=0.85, stroke="transparent")
        .encode(
            x=alt.X(
                "Center of Mass:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(title="Center of Mass  →  schedule timing (0 = early, 1 = late)"),
            ),
            y=alt.Y("Size:Q", axis=alt.Axis(title="Tree Complexity (nodes)")),
            size=alt.Size(
                "Loss:Q",
                scale=alt.Scale(range=[40, 280], reverse=True),
                legend=None,
            ),
            color=alt.Color(
                "Loss:Q",
                scale=alt.Scale(scheme="viridis", reverse=True),
                legend=alt.Legend(title="Val Loss"),
            ),
            tooltip=[
                alt.Tooltip("Loss:Q", format=".4f"),
                alt.Tooltip("Size:Q", title="Nodes"),
                alt.Tooltip("Center of Mass:Q", format=".3f"),
                alt.Tooltip("Formula:N"),
            ],
        )
        .properties(title="MAP-Elites Behavioral Archive  ·  Each dot = 1 discovered niche", height=380)
        .configure(**VEGA_CONFIG)
    )


# LR Schedule Lines
def lr_schedule_chart(curves: list) -> alt.Chart:
    """
    Multi-line plot of discovered LR schedules.
    curves: list of dicts with keys Time, LR, Rank.
    """
    df = pd.DataFrame(curves)
    return (
        alt.Chart(df)
        .mark_line(strokeWidth=2.5, opacity=0.9)
        .encode(
            x=alt.X("Time:Q", axis=alt.Axis(title="Normalized Training Time  t ∈ [0, 1]")),
            y=alt.Y("LR:Q", axis=alt.Axis(title="Learning Rate  η(t)")),
            color=alt.Color(
                "Rank:N",
                scale=alt.Scale(range=_RANK_COLORS),
                legend=alt.Legend(title="Hall of Fame"),
            ),
            tooltip=["Time:Q", alt.Tooltip("LR:Q", format=".6f"), "Rank:N"],
        )
        .properties(title="Discovered Learning Rate Schedules  ·  Hall of Fame", height=380)
        .configure(**VEGA_CONFIG)
    )


# Convergence Line
def loss_convergence_chart(gen_log: list) -> alt.Chart:
    """
    Line + dot chart of best val loss per generation.
    """
    df = pd.DataFrame(gen_log)
    base = alt.Chart(df)
    line = base.mark_line(color="#00e5a0", strokeWidth=2).encode(
        x=alt.X("gen:Q", axis=alt.Axis(title="Generation")),
        y=alt.Y("best_loss:Q", scale=alt.Scale(zero=False), axis=alt.Axis(title="Best Validation Loss", format=".4f")),
    )
    dots = base.mark_circle(color="#00e5a0", size=60, opacity=0.9).encode(
        x="gen:Q",
        y="best_loss:Q",
        tooltip=[
            alt.Tooltip("gen:Q", title="Gen"),
            alt.Tooltip("best_loss:Q", format=".4f", title="Loss"),
        ],
    )
    return (
        (line + dots)
        .properties(title="Best Loss Trajectory  ·  across generations", height=280)
        .configure(**VEGA_CONFIG)
    )


# Niche Growth
def niche_growth_chart(gen_log: list) -> alt.Chart:
    """Bar chart of total occupied niches per generation."""
    df = pd.DataFrame(gen_log)
    return (
        alt.Chart(df)
        .mark_bar(
            color="#0ea5e9",
            opacity=0.7,
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3,
        )
        .encode(
            x=alt.X("gen:Q", axis=alt.Axis(title="Generation")),
            y=alt.Y("niches:Q", axis=alt.Axis(title="Total Active Niches")),
            tooltip=[
                "gen:Q",
                alt.Tooltip("niches:Q", title="Total Niches"),
                alt.Tooltip("new_niches:Q", title="New This Gen"),
            ],
        )
        .properties(title="Niche Discovery Growth  ·  diversity expansion", height=240)
        .configure(**VEGA_CONFIG)
    )


# Baseline Comparison (REAL DATA)
def baseline_comparison_chart(
    comparison_data: list = None,
    symbolr_loss: float = None,
) -> alt.Chart:
    """
    Horizontal bar chart comparing SymboLR against hand-crafted baselines.
    Uses REAL computed data from optimiser/compare.py.
    """
    if comparison_data is None:
        from optimiser.compare import get_comparison_data
        comparison_data = get_comparison_data(symbolr_loss)

    df = pd.DataFrame(comparison_data)

    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("Schedule:N", sort="-x", axis=alt.Axis(title=None)),
            x=alt.X(
                "Val Loss:Q",
                axis=alt.Axis(title="Validation Loss (lower is better)", format=".4f"),
            ),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["Hand-crafted", "Discovered"],
                    range=[_BASELINE_COLOR, _DISCOVERED_COLOR],
                ),
                legend=alt.Legend(title="Method"),
            ),
            tooltip=["Schedule:N", alt.Tooltip("Val Loss:Q", format=".4f")],
        )
        .properties(
            title="Performance vs. Hand-crafted Baselines  ·  synthetic probe task",
            height=300,
        )
        .configure(**VEGA_CONFIG)
    )


# Baseline LR Curves Gallery
def baseline_gallery_chart() -> alt.Chart:
    """
    Multi-line chart showing all baseline LR schedules overlaid.
    Used in the Overview page to introduce the concept.
    """
    from optimiser.compare import get_baseline_curves

    curves = get_baseline_curves()
    t_array = np.linspace(0.0, 1.0, 100)

    records = []
    for name, lr_values in curves.items():
        for t_val, lr_val in zip(t_array, lr_values):
            records.append({"Time": float(t_val), "LR": float(lr_val), "Schedule": name})

    df = pd.DataFrame(records)

    palette = ["#f43f5e", "#0ea5e9", "#f59e0b", "#a78bfa", "#8899bb", "#00e5a0", "#ef4444"]

    return (
        alt.Chart(df)
        .mark_line(strokeWidth=2, opacity=0.85)
        .encode(
            x=alt.X("Time:Q", axis=alt.Axis(title="Normalized Training Time t ∈ [0, 1]")),
            y=alt.Y("LR:Q", axis=alt.Axis(title="Learning Rate η(t)")),
            color=alt.Color(
                "Schedule:N",
                scale=alt.Scale(range=palette),
                legend=alt.Legend(title="Schedule Type"),
            ),
            tooltip=["Schedule:N", "Time:Q", alt.Tooltip("LR:Q", format=".6f")],
        )
        .properties(title="Standard LR Schedules  ·  The baselines SymboLR evolves against", height=320)
        .configure(**VEGA_CONFIG)
    )