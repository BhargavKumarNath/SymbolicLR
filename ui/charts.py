"""
ui/charts.py
Altair chart factories for SymboLR dashboard.
Every function accepts plain Python data structures and returns an alt.Chart.
"""

import altair as alt
import pandas as pd
from ui.theme import VEGA_CONFIG

# Accent palette shared across charts
_RANK_COLORS = ["#00e5a0", "#0ea5e9", "#f59e0b", "#f43f5e", "#a78bfa"]

# Archive
def archive_chart(snapshot: list) -> alt.Chart:
    """
    2D scatter of the MAP-Elites behavioral archive.
    X = Center of Mass (schedule timing), Y = Tree Complexity, Color = Val Loss.
    snapshot: list of dicts with keys Size, Center of Mass, Loss, Formula.
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


# LR Schedules
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


# Convergence
def loss_convergence_chart(gen_log: list) -> alt.Chart:
    """
    Line + dot chart of best val loss per generation.
    gen_log: list of dicts with keys gen, best_loss.
    """
    df = pd.DataFrame(gen_log)
    base = alt.Chart(df)
    line = base.mark_line(color="#00e5a0", strokeWidth=2).encode(
        x=alt.X("gen:Q", axis=alt.Axis(title="Generation")),
        y=alt.Y("best_loss:Q", scale=alt.Scale(zero=False), axis=alt.Axis(title="Best Validation Loss")),
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
    """
    Bar chart of total occupied niches per generation.
    gen_log: list of dicts with keys gen, niches, new_niches.
    """
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


# Baseline Comparison
BASELINE_DATA = [
    {"Schedule": "CosineAnnealing", "Val Loss": 0.312, "Type": "Hand-crafted"},
    {"Schedule": "StepDecay",       "Val Loss": 0.328, "Type": "Hand-crafted"},
    {"Schedule": "WarmRestarts",    "Val Loss": 0.306, "Type": "Hand-crafted"},
    {"Schedule": "LinearDecay",     "Val Loss": 0.341, "Type": "Hand-crafted"},
    {"Schedule": "Constant LR",     "Val Loss": 0.389, "Type": "Hand-crafted"},
    {"Schedule": "SymboLR Elite",   "Val Loss": 0.289, "Type": "Discovered"},
]


def baseline_comparison_chart() -> alt.Chart:
    """
    Horizontal bar chart comparing SymboLR against hand-crafted baselines.
    Uses illustrative pre-computed values; swap for real results post-run.
    """
    df = pd.DataFrame(BASELINE_DATA)
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("Schedule:N", sort="-x", axis=alt.Axis(title=None)),
            x=alt.X(
                "Val Loss:Q",
                scale=alt.Scale(domain=[0.25, 0.42]),
                axis=alt.Axis(title="Validation Loss (lower is better)"),
            ),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["Hand-crafted", "Discovered"],
                    range=["#4a6080", "#00e5a0"],
                ),
                legend=alt.Legend(title="Method"),
            ),
            tooltip=["Schedule:N", alt.Tooltip("Val Loss:Q", format=".3f")],
        )
        .properties(
            title="Performance vs. Hand-crafted Baselines  ·  MNIST probe task",
            height=280,
        )
        .configure(**VEGA_CONFIG)
    )