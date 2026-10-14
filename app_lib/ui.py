"""Shared look, data loading and small HTML components for the Streamlit app
(same design system as the sentiment-classifier app)."""
import html
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data import PROCESSED
from src.train import MODELS, REPORTS

# ---- design tokens -------------------------------------------------------------------
INK = "#1C2333"
INK_SOFT = "#5A6478"
RULE = "#E3E6EC"
PAPER = "#F7F8FA"
FRAUD, LEGIT = "#D4562E", "#2A6FD6"
# Decision tiers: an ordered scale from "let it through" to "stop and verify".
TIER_COLOR = {"approve": "#8A94A6", "review": "#B07D00", "hold_verify": "#D4562E"}
TIER_NAME = {"approve": "Approve", "review": "Analyst review", "hold_verify": "Hold + verify customer"}
TIER_SHORT = {"approve": "approve", "review": "review", "hold_verify": "hold + verify"}
MODEL_COLOR = {"improved": "#13976A", "baseline": "#6250D6"}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&display=swap');
html, body, .stApp, p, li, label, h1, h2, h3, h4, td, th, textarea, input, select,
button p, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {{
  font-family: 'IBM Plex Sans Arabic', system-ui, sans-serif !important;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ max-width: 1120px; padding-top: 4.2rem; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 600; font-size: 2.35rem; line-height: 1.15; }}
h2 {{ font-weight: 600; font-size: 1.45rem; margin-top: 1.6rem; }}
h3 {{ font-weight: 600; font-size: 1.1rem; }}
p, li {{ line-height: 1.6; }}
.lede {{ color: {INK_SOFT}; font-size: 1.08rem; max-width: 72ch; margin-top: -0.4rem; }}
.muted {{ color: {INK_SOFT}; font-size: 0.92rem; }}
.chip {{ display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 2px 10px;
         font-size: 0.82rem; font-weight: 500; border: 1px solid {RULE}; background: #fff; color: {INK}; }}
.dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}

/* one alert, opened: tier, score bar with thresholds, reasons */
.verdict {{ background: #fff; border: 1px solid {RULE}; border-radius: 12px; padding: 18px 20px; margin-bottom: 12px; }}
.verdict .head {{ display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; font-size: 0.9rem; color: {INK_SOFT}; }}
.verdict .label {{ font-size: 1.6rem; font-weight: 600; line-height: 1.15; margin-top: 6px; }}
.verdict .swatch {{ display: inline-block; width: 18px; height: 18px; border-radius: 5px; margin-inline-end: 10px; vertical-align: -1px; }}
.verdict .label small {{ font-size: 1rem; font-weight: 400; color: {INK_SOFT}; margin-inline-start: 8px; }}
.scorebar {{ position: relative; height: 14px; border-radius: 7px; background: #EEF0F4; margin: 16px 0 26px; }}
.scorebar .fill {{ position: absolute; inset: 0 auto 0 0; border-radius: 7px; }}
.scorebar .tick {{ position: absolute; top: -4px; width: 2px; height: 22px; background: {INK}; }}
.scorebar .tick span {{ position: absolute; top: 24px; left: -30px; width: 60px; text-align: center;
                        font-size: 0.75rem; color: {INK_SOFT}; white-space: nowrap; }}
.reasons {{ margin: 4px 0 0; padding: 0; list-style: none; }}
.reasons li {{ padding: 7px 0 7px 26px; border-top: 1px solid {RULE}; position: relative; }}
.reasons li::before {{ content: ""; position: absolute; left: 4px; top: 15px; width: 10px; height: 10px;
                       border-radius: 3px; background: {FRAUD}; }}
.truth {{ margin-top: 12px; font-size: 0.88rem; padding: 8px 12px; border-radius: 8px; }}
.slip {{ background: #fff; border: 1px solid {RULE}; border-radius: 10px; padding: 14px 18px; margin: 6px 0 12px; }}
.slip .meta {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; font-size: 0.85rem; color: {INK_SOFT}; }}

.figure-row {{ display: flex; gap: 28px; flex-wrap: wrap; margin: 10px 0 4px; }}
.figure {{ min-width: 150px; }}
.figure .n {{ font-size: 2rem; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1.1; }}
.figure .l {{ font-size: 0.88rem; color: {INK_SOFT}; max-width: 22ch; }}
.callout {{ border-inline-start: 4px solid {INK}; background: #fff; padding: 14px 18px; border-radius: 0 10px 10px 0; margin: 12px 0; }}
table.plain {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 0.95rem; }}
table.plain th {{ text-align: start; font-weight: 500; color: {INK_SOFT}; border-bottom: 1px solid {RULE}; padding: 8px 10px; }}
table.plain td {{ border-bottom: 1px solid {RULE}; padding: 8px 10px; }}
table.plain tr.hl td {{ font-weight: 600; background: #fff; }}
@media (max-width: 640px) {{ h1 {{ font-size: 1.8rem; }} .figure .n {{ font-size: 1.6rem; }} }}
</style>
"""


def setup():
    st.html(CSS)


def esc(s) -> str:
    return html.escape(str(s))


def dot(color: str) -> str:
    return f'<span class="dot" style="background:{color}"></span>'


def tier_chip(tier: str) -> str:
    return f'<span class="chip">{dot(TIER_COLOR[tier])}{TIER_NAME[tier]}</span>'


def figures(items) -> str:
    cells = "".join(f'<div class="figure"><div class="n">{esc(n)}</div><div class="l">{esc(l)}</div></div>'
                    for n, l in items)
    return f'<div class="figure-row">{cells}</div>'


def table(df: pd.DataFrame, highlight=None) -> str:
    """Plain HTML table; `highlight` = index label of the row to bold. Values must be pre-formatted."""
    head = "".join(f"<th>{c}</th>" for c in df.columns)  # headers are trusted HTML
    rows = "".join(f'<tr class="{"hl" if i == highlight else ""}">' + "".join(f"<td>{v}</td>" for v in r) + "</tr>"
                   for i, r in zip(df.index, df.itertuples(index=False)))
    return f'<table class="plain"><tr>{head}</tr>{rows}</table>'


def alert_card(row: pd.Series, reasons: list[str], review_thr: float, hold_thr: float, show_truth: bool) -> str:
    tier, score = row["tier"], float(row["score"])
    # log scale from 0.001 to 1, so the review and hold thresholds (0.02, 0.11) are visibly apart
    pos = lambda t: f"{min(max((np.log10(max(t, 1e-3)) + 3) / 3, 0), 1) * 100:.1f}%"
    items = "".join(f"<li>{esc(r)}</li>" for r in reasons) or '<li>No single strong driver</li>'
    truth = ""
    if show_truth:
        is_f = int(row["is_fraud"]) == 1
        truth = (f'<div class="truth" style="background:{"#FCEDE7" if is_f else "#EAF1FC"};'
                 f'color:{"#7A2A10" if is_f else "#123F80"}">In hindsight: '
                 f'<b>{"fraud" if is_f else "a genuine transaction"}</b></div>')
    return f"""
    <div class="verdict">
      <div class="head"><span>{esc(row['cc_num'])} · {row['ts']:%b %d, %H:%M} · {esc(row['category'])}</span>
        <span>{DAYS[int(row['day_of_week'])]}</span></div>
      <div class="label"><span class="swatch" style="background:{TIER_COLOR[tier]}"></span>{TIER_NAME[tier]}</div>
      <div class="muted" style="margin-top:4px">${row['amt']:,.2f} · risk score {score:.3f} (log scale below)</div>
      <div class="scorebar"><div class="fill" style="width:{pos(score)};background:{TIER_COLOR[tier]}"></div>
        <div class="tick" style="left:{pos(review_thr)}"><span>review</span></div>
        <div class="tick" style="left:{pos(hold_thr)}"><span>hold</span></div></div>
      <div class="muted">Why it was flagged</div>
      <ul class="reasons">{items}</ul>
      {truth}
    </div>"""


# ---- data ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    path = MODELS / "fraud_model.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_meta() -> dict:
    path = MODELS / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def load_scored() -> pd.DataFrame | None:
    path = PROCESSED / "test_scored.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    d["day"] = d["ts"].dt.date
    return d


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_json(name: str) -> dict:
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def need_model():
    st.info("This view needs the trained model. Run `python -m src.train` once (see README, *How to run*).")


def altair_theme():
    font = "IBM Plex Sans Arabic, system-ui, -apple-system, Helvetica, Arial, sans-serif"
    return {"config": {
        "font": font, "background": "transparent", "view": {"stroke": None},
        "axis": {"labelColor": INK_SOFT, "titleColor": INK_SOFT, "gridColor": RULE, "domainColor": RULE,
                 "tickColor": RULE, "labelFontSize": 12, "titleFontSize": 12, "titleFontWeight": 500},
        "legend": {"labelColor": INK, "titleColor": INK_SOFT, "labelFontSize": 12, "titleFontWeight": 500},
        "text": {"font": font},
        "title": {"color": INK, "fontWeight": 600, "fontSize": 14, "anchor": "start"},
    }}


def use_altair_theme():
    alt.theme.register("plex", enable=True)(altair_theme)


def money(x: float) -> str:
    return f"${x / 1e3:,.0f}k" if abs(x) >= 1e4 else f"${x:,.0f}"


def pct(x: float, d=1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}%}"
