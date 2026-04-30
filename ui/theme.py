"""
ui/theme.py
Design system for SymboLR dashboard.
Injects CSS variables, global overrides, and reusable HTML component builders.
"""

import streamlit as st

#CSS
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
  --bg-base:       #080c12;
  --bg-surface:    #0d1520;
  --bg-raised:     #121d2e;
  --bg-overlay:    #192640;
  --border:        #1e3050;
  --border-bright: #2a4570;
  --accent-green:  #00e5a0;
  --accent-blue:   #0ea5e9;
  --accent-amber:  #f59e0b;
  --accent-rose:   #f43f5e;
  --text-primary:  #e8f0fe;
  --text-secondary:#8899bb;
  --text-muted:    #4a6080;
  --font-display:  'Syne', sans-serif;
  --font-mono:     'Space Mono', monospace;
  --font-body:     'DM Sans', sans-serif;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, [data-testid="stHeader"] { display: none !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"],
[data-testid="stHeaderActionElements"],
[data-testid="stDecoration"] { display: none !important; }

[data-testid="stHeader"] {
    height: 0 !important;
    min-height: 0 !important;
}

[data-testid="stAppViewContainer"] > .main {
    padding-top: 0 !important;
}

/* ── Main container ── */
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
.stMainBlockContainer,
.block-container {
    padding: 28px 40px 60px !important;
    max-width: 100% !important;
}

/* ── Page wrapper ── */
@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"],
    .stMainBlockContainer,
    .block-container {
        padding: 20px 20px 40px !important;
    }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarUserContent"] {
    padding-top: 1rem !important;
}
[data-testid="stSidebarHeader"] {
    display: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* ── Sidebar logo ── */
.sidebar-logo {
    padding: 28px 20px 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}
.sidebar-logo .logo-mark {
    font-family: var(--font-display);
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 10px;
}
.sidebar-logo .logo-mark span { color: var(--accent-green); }
.sidebar-logo .logo-sub {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Nav ── */
.nav-section-label {
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 16px 20px 6px;
}
.stRadio > label { display: none !important; }
.stRadio > div { gap: 2px !important; padding: 0 10px; }
.stRadio > div > label {
    display: flex !important;
    align-items: center;
    padding: 9px 14px !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    font-size: 13.5px !important;
    font-weight: 400 !important;
    color: var(--text-secondary) !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    border: 1px solid transparent !important;
}
.stRadio > div > label:hover {
    background: var(--bg-raised) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
}
.stRadio > div > label[data-baseweb="radio"] > div:first-child { display: none !important; }

/* ── Sliders ── */
[data-testid="stSidebar"] .stSlider > div > div > div { background: var(--border-bright) !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div { background: var(--accent-green) !important; }
.stSlider p, .stSlider label {
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    color: var(--text-secondary) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent-green) !important;
    color: #080c12 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.5px !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #00ffb3 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(0, 229, 160, 0.3) !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--accent-green), var(--accent-blue)) !important;
    border-radius: 4px !important;
}
[data-testid="stProgressBar"] > div {
    background: var(--bg-overlay) !important;
    border-radius: 4px !important;
    height: 6px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    color: var(--accent-green) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    color: var(--text-muted) !important;
    padding: 10px 20px !important;
    border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-green) !important;
    border-bottom: 2px solid var(--accent-green) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    color: var(--text-secondary) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Toast ── */
[data-testid="stToast"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--accent-green) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

/* ── Page header ── */
.page-header { margin-bottom: 32px; padding-bottom: 24px; border-bottom: 1px solid var(--border); }
.page-header .eyebrow {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent-green);
    margin-bottom: 8px;
}
.page-header h1 {
    font-family: var(--font-display) !important;
    font-size: 36px !important;
    font-weight: 800 !important;
    letter-spacing: -1px !important;
    color: var(--text-primary) !important;
    margin: 0 0 8px !important;
    line-height: 1.1 !important;
}
.page-header .subtitle {
    font-family: var(--font-body);
    font-size: 15px;
    color: var(--text-secondary);
    font-weight: 300;
    max-width: 640px;
    line-height: 1.6;
}

/* ── Section header ── */
.section-header { display: flex; align-items: baseline; gap: 12px; margin: 28px 0 16px; }
.section-header h3 {
    font-family: var(--font-display) !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
}
.section-header .section-tag {
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    background: var(--bg-overlay);
    border: 1px solid var(--border);
    padding: 2px 8px;
    border-radius: 4px;
}

/* ── Info card ── */
.info-card {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.info-card.accent-green { border-left: 3px solid var(--accent-green); }
.info-card.accent-blue  { border-left: 3px solid var(--accent-blue); }
.info-card.accent-amber { border-left: 3px solid var(--accent-amber); }
.info-card.accent-rose  { border-left: 3px solid var(--accent-rose); }
.info-card h4 {
    font-family: var(--font-display) !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0 0 6px !important;
}
.info-card p {
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    color: var(--text-secondary) !important;
    margin: 0 !important;
    line-height: 1.55 !important;
}

/* ── Pipeline ── */
.pipeline {
    display: flex;
    align-items: stretch;
    gap: 0;
    margin: 20px 0;
    overflow-x: auto;
    padding-bottom: 4px;
}
.pipeline-step {
    flex: 1;
    min-width: 120px;
    background: var(--bg-raised);
    border: 1px solid var(--border);
    padding: 16px 14px;
    text-align: center;
    position: relative;
}
.pipeline-step:first-child { border-radius: 10px 0 0 10px; }
.pipeline-step:last-child  { border-radius: 0 10px 10px 0; }
.pipeline-step + .pipeline-step { border-left: none; }
.pipeline-step.active { background: var(--bg-overlay); border-color: var(--accent-green); }
.pipeline-step .step-icon { font-size: 20px; margin-bottom: 6px; }
.pipeline-step .step-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-muted);
}
.pipeline-step .step-name {
    font-family: var(--font-display);
    font-size: 12px;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 2px;
}

/* ── Stat row ── */
.stat-row { display: flex; gap: 12px; margin: 16px 0; }
.stat-pill {
    background: var(--bg-overlay);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    flex: 1;
    text-align: center;
}
.stat-pill .sp-label {
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
}
.stat-pill .sp-value {
    font-family: var(--font-display);
    font-size: 20px;
    font-weight: 700;
    color: var(--accent-blue);
    margin-top: 2px;
}

/* ── Formula block ── */
.formula-block {
    background: var(--bg-base);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 22px;
    margin: 8px 0;
}
.formula-block .fb-rank {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.formula-block .fb-meta { display: flex; gap: 16px; margin-top: 8px; }
.formula-block .fb-tag { font-family: var(--font-mono); font-size: 10px; color: var(--text-muted); }
.formula-block .fb-tag span { color: var(--accent-green); }

/* ── Generation log ── */
.gen-log-entry {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 14px;
    border-radius: 6px;
    margin-bottom: 4px;
    border-left: 2px solid var(--border);
    background: var(--bg-raised);
    font-family: var(--font-mono);
    font-size: 11px;
}
.gen-log-entry .gle-gen   { color: var(--text-muted);  width: 52px; flex-shrink: 0; }
.gen-log-entry .gle-loss  { color: var(--accent-green); width: 70px; flex-shrink: 0; }
.gen-log-entry .gle-niches{ color: var(--accent-blue);  width: 60px; flex-shrink: 0; }
.gen-log-entry .gle-new   { color: var(--accent-amber); flex-shrink: 0; }
.gen-log-entry.best {
    border-left-color: var(--accent-green);
    background: rgba(0, 229, 160, 0.05);
}

/* ── Baseline rows ── */
.baseline-row {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 6px;
    border: 1px solid var(--border);
    background: var(--bg-raised);
}
.baseline-row.winner {
    border-color: var(--accent-green);
    background: rgba(0, 229, 160, 0.06);
}
.baseline-row .br-name {
    font-family: var(--font-body);
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
    width: 200px;
}
.baseline-row .br-loss { font-family: var(--font-mono); font-size: 14px; color: var(--text-secondary); flex: 1; }
.baseline-row .br-bar-wrap { flex: 2; height: 6px; background: var(--bg-overlay); border-radius: 3px; overflow: hidden; }
.baseline-row .br-bar { height: 100%; border-radius: 3px; background: var(--border-bright); }
.baseline-row.winner .br-bar { background: var(--accent-green); }
.baseline-row .br-badge {
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--accent-green);
    background: rgba(0, 229, 160, 0.1);
    border: 1px solid rgba(0, 229, 160, 0.3);
    padding: 3px 8px;
    border-radius: 4px;
    margin-left: 12px;
    white-space: nowrap;
}

/* ── Sidebar live stats ── */
.sidebar-stat { padding: 10px 20px; border-bottom: 1px solid var(--border); }
.sidebar-stat .ss-label {
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
}
.sidebar-stat .ss-value {
    font-family: var(--font-display);
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 2px;
}
.sidebar-stat .ss-value.green { color: var(--accent-green); }

/* ── Vega / Altair ── */
.vega-embed { background: transparent !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
"""

# Altair config dict (pass to .configure(**VEGA_CONFIG))

VEGA_CONFIG = {
    "background": "transparent",
    "view": {"fill": "transparent", "stroke": "transparent"},
    "axis": {
        "domainColor": "#1e3050",
        "gridColor": "#1e3050",
        "tickColor": "#1e3050",
        "labelColor": "#8899bb",
        "titleColor": "#8899bb",
        "labelFont": "Space Mono",
        "titleFont": "Space Mono",
        "labelFontSize": 10,
        "titleFontSize": 10,
    },
    "legend": {
        "labelColor": "#8899bb",
        "titleColor": "#8899bb",
        "labelFont": "Space Mono",
        "titleFont": "Space Mono",
        "labelFontSize": 10,
    },
    "title": {
        "color": "#e8f0fe",
        "font": "Syne",
        "fontSize": 13,
        "fontWeight": 700,
        "anchor": "start",
        "offset": 12,
    },
}


# Component helpers
def inject_theme():
    """Call once at app startup to inject global CSS."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str):
    st.markdown(f"""
    <div class="page-header">
        <div class="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p class="subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, tag: str = ""):
    tag_html = f'<span class="section-tag">{tag}</span>' if tag else ""
    st.markdown(f"""
    <div class="section-header">
        <h3>{title}</h3>
        {tag_html}
    </div>
    """, unsafe_allow_html=True)


def info_card(title: str, body: str, accent: str = ""):
    cls = f"info-card accent-{accent}" if accent else "info-card"
    st.markdown(f"""
    <div class="{cls}">
        <h4>{title}</h4>
        <p>{body}</p>
    </div>
    """, unsafe_allow_html=True)


def formula_block(rank: int, loss: float, size: int, depth: int, latex: str):
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    medal = medals[rank - 1] if rank <= len(medals) else f"#{rank}"
    st.markdown(f"""
    <div class="formula-block">
        <div>
            <div class="fb-rank">{medal}  Rank {rank}</div>

$$
{latex}
$$

            <div class="fb-meta">
                <span class="fb-tag">LOSS <span>{loss:.4f}</span></span>
                <span class="fb-tag">NODES <span>{size}</span></span>
                <span class="fb-tag">DEPTH <span>{depth}</span></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def baseline_row(name: str, loss: float, worst: float, is_winner: bool = False):
    bar_pct = int((loss / worst) * 100)
    cls = "baseline-row winner" if is_winner else "baseline-row"
    badge = '<span class="br-badge">BEST</span>' if is_winner else ""
    st.markdown(f"""
    <div class="{cls}">
        <span class="br-name">{name}</span>
        <span class="br-loss">{loss:.3f}</span>
        <div class="br-bar-wrap"><div class="br-bar" style="width:{bar_pct}%"></div></div>
        {badge}
    </div>
    """, unsafe_allow_html=True)


def gen_log_row(entry: dict, is_best: bool):
    cls = "gen-log-entry best" if is_best else "gen-log-entry"
    badge = '<span style="color:var(--accent-amber);">▲ new best</span>' if is_best else ""
    st.markdown(f"""
    <div class="{cls}">
        <span class="gle-gen">Gen {entry['gen']:02d}</span>
        <span class="gle-loss">{entry['best_loss']:.4f}</span>
        <span class="gle-niches">{entry['niches']}</span>
        <span class="gle-new">+{entry['new_niches']}</span>
        {badge}
    </div>
    """, unsafe_allow_html=True)
