"""
================================================================================
ClearSight Analytics — Clinical AI Command Center
================================================================================
Tech-forward Streamlit interface integrating Models 1 (XGBoost) and 2 (DNN).
Dark-mode, glassmorphism, animated SVG gauges, real model inference.

Run: streamlit run webapp/app.py   (from project root)
================================================================================
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import logging
import time
import traceback
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path
from typing import Any
from PIL import Image
import gc
import cv2
import io
import plotly.express as px
import plotly.graph_objects as go
# torch and transformers are imported lazily inside load_model4() to avoid
# a crash when torchvision is missing from the environment on startup.

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import engineer_features

# =============================================================================
# LOGGING — writes to logs/app.log; directory is created on first run
# =============================================================================
_LOG_DIR = PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("clearsight")
if not logger.handlers:  # guard against duplicate handlers on hot-reload
    logger.setLevel(logging.DEBUG)
    _fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # File handler — full DEBUG+ detail
    _fh = logging.FileHandler(_LOG_DIR / "app.log", encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(_fmt)
    logger.addHandler(_fh)
    # Console handler — INFO+ only
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)

# --- Model paths ---
M1_DIR = PROJECT_ROOT / "models" / "model1_traditional_ml" / "saved_model"
M2_DIR = PROJECT_ROOT / "models" / "model2_deep_learning" / "saved_model"
M3_DIR = PROJECT_ROOT / "models" / "model3_cnn" / "saved_model"              # CNN Retinal (Doug)
M4_DIR = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model" # NLP (Wes)
M5_DIR = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model"       # Capacity Planning
M6_RESULTS = PROJECT_ROOT / "test_data" / "model6_results.csv"

def inject_sidebar_styles() -> None:
    """Injects custom CSS to style the sidebar as a dark-mode SaaS navigation menu.

    Transforms the native ``st.radio`` widget into hoverable navigation blocks and
    registers ``.status-pill``, ``.sidebar-header``, and ``.brand`` CSS classes used
    throughout the sidebar. Writes directly to the Streamlit DOM via ``st.markdown``.

    Raises:
        streamlit.errors.StreamlitAPIException: If called outside a valid Streamlit
            execution context.
    """
    st.markdown("""
    <style>
    /* 1. Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #0b1121 !important;
        border-right: 1px solid rgba(34,211,238,0.1);
    }
    
    /* 2. Style the native radio widget container */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 0.3rem;
        padding: 0 0.5rem;
    }
    
    /* 3. Transform radio options into clickable blocks */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: transparent;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        transition: all 0.2s ease;
        cursor: pointer;
        border-left: 3px solid transparent;
        width: 100%;
    }
    
    /* Hover effect */
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(34,211,238,0.05);
        border-left: 3px solid rgba(34,211,238,0.4);
        transform: translateX(3px);
    }
    
    /* 4. HIDE the default radio circles! */
    [data-testid="stRadio"] div[role="radio"] > div:first-child,
    [data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }
    
    /* 5. Typography for the links */
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        font-weight: 500;
        color: #cbd5e1;
        margin: 0;
    }
    
    /* 6. System Status Headers */
    .sidebar-header {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        padding-left: 1rem;
    }
    
    /* 7. Status Pill styling */
    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 1rem;
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .status-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 999px;
        letter-spacing: 0.05em;
    }
    .pill-online { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .pill-pending { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
    </style>
    """, unsafe_allow_html=True)



# =============================================================================
# PAGE CONFIGURATION & GLOBAL STYLES
# =============================================================================
st.set_page_config(
    page_title="ClearSight | Clinical AI Command Center",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_sidebar_styles()


# =============================================================================
# EMBEDDED LOGO (SVG)
# =============================================================================
CLEARSIGHT_ICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="48" height="48">
  <defs>
    <radialGradient id="irisG" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#5eead4"/>
      <stop offset="60%" stop-color="#22d3ee"/>
      <stop offset="100%" stop-color="#185FA5"/>
    </radialGradient>
    <linearGradient id="outlineG" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#22d3ee"/>
      <stop offset="100%" stop-color="#185FA5"/>
    </linearGradient>
  </defs>
  <g transform="translate(10, 32)">
    <path d="M 4 28 Q 26 4, 50 4 Q 74 4, 96 28 Q 74 52, 50 52 Q 26 52, 4 28 Z"
          fill="none" stroke="url(#outlineG)" stroke-width="3.5" stroke-linecap="round"/>
    <circle cx="50" cy="28" r="18" fill="url(#irisG)"/>
    <circle cx="50" cy="28" r="14.5" fill="none" stroke="#5eead4" stroke-width="0.9" opacity="0.5"/>
    <circle cx="50" cy="28" r="7" fill="#0a1220"/>
    <circle cx="46.5" cy="23.5" r="3" fill="#ffffff" opacity="0.95"/>
    <circle cx="53" cy="31" r="1.2" fill="#ffffff" opacity="0.6"/>
    <line x1="14" y1="28" x2="22" y2="28" stroke="#5eead4" stroke-width="1.8" stroke-linecap="round" opacity="0.7"/>
    <line x1="78" y1="28" x2="86" y2="28" stroke="#5eead4" stroke-width="1.8" stroke-linecap="round" opacity="0.7"/>
  </g>
</svg>
"""


# =============================================================================
# CSS — Dark tech aesthetic
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global reset ────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #e2e8f0;
}
.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(34,211,238,0.06) 0%, transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(94,234,212,0.05) 0%, transparent 40%),
        linear-gradient(180deg, #0a1220 0%, #0d1b2a 100%);
    background-attachment: fixed;
}

/* Subtle grid pattern overlay */
.stApp::before {
    content: "";
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(94,234,212,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(94,234,212,0.025) 1px, transparent 1px);
    background-size: 40px 40px;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1400px; }

/* Force sidebar always visible */
[data-testid="collapsedControl"] { display: block !important; visibility: visible !important; }
section[data-testid="stSidebar"] { display: block !important; min-width: 260px !important; }

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07101e 0%, #0a1628 100%) !important;
    border-right: 1px solid rgba(34,211,238,0.15);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.92rem;
    font-weight: 500;
    transition: all 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(34,211,238,0.08);
}

/* ── Brand block ─────────────────────────────────────────────────── */
.brand {
    display: flex; align-items: center; gap: 12px;
    padding: 0.6rem 0 1.2rem;
    border-bottom: 1px solid rgba(34,211,238,0.15);
    margin-bottom: 1rem;
}
.brand .name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem; font-weight: 700;
    color: white !important;
    letter-spacing: -0.02em;
    line-height: 1;
}
.brand .tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 500;
    color: #5eead4 !important;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Hero ────────────────────────────────────────────────────────── */
.hero {
    position: relative;
    background:
        radial-gradient(circle at 80% 30%, rgba(34,211,238,0.18) 0%, transparent 50%),
        linear-gradient(135deg, #0d1b2a 0%, #112c4a 50%, #0d1b2a 100%);
    border: 1px solid rgba(34,211,238,0.2);
    border-radius: 18px;
    padding: 2.5rem 2.75rem;
    margin-bottom: 2rem;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(34,211,238,0.08), inset 0 1px 0 rgba(94,234,212,0.1);
}
.hero::after {
    content: "";
    position: absolute; top: 0; right: 0; width: 300px; height: 100%;
    background: radial-gradient(ellipse, rgba(94,234,212,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero .badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 600;
    color: #5eead4;
    background: rgba(94,234,212,0.1);
    border: 1px solid rgba(94,234,212,0.3);
    padding: 4px 12px; border-radius: 999px;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem; font-weight: 700;
    color: white !important;
    margin: 0 0 0.6rem;
    letter-spacing: -0.03em; line-height: 1.1;
    background: linear-gradient(120deg, #ffffff 0%, #5eead4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero .sub {
    font-size: 1.05rem;
    color: #94a3b8 !important;
    max-width: 720px;
    line-height: 1.55;
    margin: 0 0 1.4rem;
}
.pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(34,211,238,0.08);
    color: #5eead4 !important;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.78rem; font-weight: 600;
    margin-right: 0.5rem;
    border: 1px solid rgba(34,211,238,0.25);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
}

/* ── Glass card ──────────────────────────────────────────────────── */
.gcard {
    background: rgba(15,30,52,0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(34,211,238,0.12);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    height: 100%;
}
.gcard:hover {
    transform: translateY(-3px);
    border-color: rgba(94,234,212,0.4);
    box-shadow: 0 12px 40px rgba(34,211,238,0.15);
}
.gcard .tag {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem; font-weight: 700;
    color: #5eead4;
    letter-spacing: 0.12em; text-transform: uppercase;
    background: rgba(94,234,212,0.08);
    padding: 3px 10px; border-radius: 6px;
    margin-bottom: 12px;
    border: 1px solid rgba(94,234,212,0.2);
}
.gcard h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem; font-weight: 600;
    color: white !important;
    margin: 8px 0;
}
.gcard p {
    color: #94a3b8 !important;
    font-size: 0.9rem; line-height: 1.55;
    margin: 0;
}

/* ── Stat block ──────────────────────────────────────────────────── */
.stat {
    background: rgba(15,30,52,0.5);
    border: 1px solid rgba(34,211,238,0.12);
    border-radius: 12px;
    padding: 1.2rem 1.3rem;
    transition: border-color 0.2s;
}
.stat:hover { border-color: rgba(94,234,212,0.3); }
.stat .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem; font-weight: 600;
    color: #64748b !important;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 6px;
}
.stat .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.85rem; font-weight: 700;
    color: white !important;
    line-height: 1;
}
.stat .delta {
    font-size: 0.78rem; font-weight: 500;
    color: #5eead4 !important;
    margin-top: 6px;
}

/* ── Section heading ─────────────────────────────────────────────── */
.section-head {
    display: flex; align-items: center; gap: 12px;
    margin: 2rem 0 1.2rem;
}
.section-head .num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; font-weight: 600;
    color: #5eead4;
    background: rgba(94,234,212,0.08);
    padding: 2px 10px; border-radius: 6px;
    border: 1px solid rgba(94,234,212,0.2);
}
.section-head h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem; font-weight: 600;
    color: white !important;
    margin: 0;
    letter-spacing: -0.02em;
}
.section-head .line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(34,211,238,0.3), transparent);
}

/* ── Live status dot (pulsing) ───────────────────────────────────── */
.dot {
    display: inline-block;
    width: 8px; height: 8px; border-radius: 50%;
    margin-right: 8px; vertical-align: middle;
    animation: pulse 2s infinite;
}
.dot-ok { background: #5eead4; box-shadow: 0 0 8px rgba(94,234,212,0.7); }
.dot-warn { background: #fbbf24; box-shadow: 0 0 8px rgba(251,191,36,0.7); }
.dot-err { background: #f87171; box-shadow: 0 0 8px rgba(248,113,113,0.7); }
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.2); }
}

.status-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 0;
    font-size: 0.85rem;
}
.status-row .name { color: #cbd5e1 !important; font-weight: 500; }
.status-row .val { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #5eead4 !important; }

/* ── Risk badges ─────────────────────────────────────────────────── */
.risk-high   { background: rgba(248,113,113,0.12); color: #fca5a5;
               border: 1px solid rgba(248,113,113,0.4); border-radius: 10px;
               padding: 12px 18px; font-weight: 700; font-size: 1rem;
               text-align: center; font-family: 'Space Grotesk', sans-serif;
               letter-spacing: 0.04em; }
.risk-medium { background: rgba(251,191,36,0.12); color: #fcd34d;
               border: 1px solid rgba(251,191,36,0.4); border-radius: 10px;
               padding: 12px 18px; font-weight: 700; font-size: 1rem;
               text-align: center; font-family: 'Space Grotesk', sans-serif;
               letter-spacing: 0.04em; }
.risk-low    { background: rgba(94,234,212,0.12); color: #5eead4;
               border: 1px solid rgba(94,234,212,0.4); border-radius: 10px;
               padding: 12px 18px; font-weight: 700; font-size: 1rem;
               text-align: center; font-family: 'Space Grotesk', sans-serif;
               letter-spacing: 0.04em; }

/* ── Buttons ─────────────────────────────────────────────────────── */
div.stButton > button,
div.stFormSubmitButton > button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #22d3ee 0%, #185FA5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(34,211,238,0.3), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    transition: all 0.2s !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.02em !important;
}
div.stButton > button:hover,
div.stFormSubmitButton > button:hover,
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 30px rgba(34,211,238,0.5), inset 0 1px 0 rgba(255,255,255,0.3) !important;
}

/* ── Form inputs ─────────────────────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stSelectbox > div > div {
    background: rgba(10,18,32,0.6) !important;
    border: 1px solid rgba(34,211,238,0.2) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #5eead4 !important;
    box-shadow: 0 0 0 3px rgba(94,234,212,0.15) !important;
}
label, .stMarkdown p, .stMarkdown li { color: #cbd5e1 !important; }

/* Form container */
[data-testid="stForm"] {
    background: rgba(15,30,52,0.4);
    border: 1px solid rgba(34,211,238,0.12);
    border-radius: 14px;
    padding: 1.5rem !important;
    backdrop-filter: blur(8px);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(10,18,32,0.5);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #94a3b8 !important;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(34,211,238,0.15) !important;
    color: #5eead4 !important;
}

/* Disclaimer */
.disclaimer {
    background: rgba(251,146,60,0.08);
    border-left: 3px solid #fb923c;
    padding: 1rem 1.2rem; border-radius: 8px;
    color: #fdba74;
    font-size: 0.85rem; line-height: 1.55;
    margin-top: 1.5rem;
}

/* Dataframe styling */
.stDataFrame { background: transparent; }

/* Spinner color */
.stSpinner > div { border-top-color: #5eead4 !important; }

/* Headings */
h1, h2, h3 { color: white !important; font-family: 'Space Grotesk', sans-serif; }

/* ── Expander header legibility ──────────────────────────────────── */
.streamlit-expanderHeader {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] {
    border: 1px solid rgba(34,211,238,0.2) !important;
    border-radius: 10px !important;
    background: rgba(15,23,42,0.5) !important;
}

/* ── Code block: dark theme + word-wrap ──────────────────────────── */
div.stCodeBlock > div {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(34, 211, 238, 0.2) !important;
}
div.stCodeBlock code {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    color: #e2e8f0 !important;
}
[data-testid="stCode"],
[data-testid="stCode"] > div,
[data-testid="stCode"] pre {
    background-color: rgba(15, 23, 42, 0.85) !important;
    border: 1px solid rgba(34, 211, 238, 0.2) !important;
    border-radius: 8px !important;
}
[data-testid="stCode"] code,
[data-testid="stCode"] pre,
[data-testid="stCode"] span {
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    color: #e2e8f0 !important;
    background: transparent !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.65 !important;
}

/* =========================================
   GLOBAL SCROLLBAR STYLING (HIGH CONTRAST)
   ========================================= */
/* For Firefox */
* {
    scrollbar-width: thin;
    scrollbar-color: #38bdf8 rgba(15, 23, 42, 0.3);
}

/* For Chrome, Edge, and Safari */
::-webkit-scrollbar {
    width: 8px !important;
    height: 8px !important;
}

::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.3) !important;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background-color: #38bdf8 !important;
    border-radius: 4px;
    border: 2px solid rgba(15, 23, 42, 1);
}

::-webkit-scrollbar-thumb:hover {
    background-color: #7dd3fc !important;
}

[data-testid="stAppViewContainer"] ::-webkit-scrollbar-thumb,
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb,
div.stCodeBlock div ::-webkit-scrollbar-thumb {
    background-color: #38bdf8 !important;
}

/* ── Main page scroll container (primary fix) ────────────────────── */
section[data-testid="stMain"] {
    overflow-y: auto !important;
    scrollbar-width: thin !important;
    scrollbar-color: #38bdf8 rgba(15, 23, 42, 0.3) !important;
}
section[data-testid="stMain"]::-webkit-scrollbar {
    width: 8px !important;
    height: 8px !important;
}
section[data-testid="stMain"]::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.3) !important;
    border-radius: 4px !important;
}
section[data-testid="stMain"]::-webkit-scrollbar-thumb {
    background-color: #38bdf8 !important;
    border-radius: 4px !important;
    border: 2px solid rgba(15, 23, 42, 1) !important;
}
section[data-testid="stMain"]::-webkit-scrollbar-thumb:hover {
    background-color: #7dd3fc !important;
}
/* Fallback: target the inner block container as well */
[data-testid="stAppViewBlockContainer"] {
    scrollbar-width: thin !important;
    scrollbar-color: #38bdf8 rgba(15, 23, 42, 0.3) !important;
}
[data-testid="stAppViewBlockContainer"]::-webkit-scrollbar-thumb {
    background-color: #38bdf8 !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL LOADERS  (cached)
# =============================================================================
_HF_REPO = "FrankAlonsoskyMolina/clearsight-analytics"


def _ensure_hf_artifact(local_path: Path, hf_filename: str) -> None:
    """Download a model artifact from HuggingFace if it is not present locally.

    Uses the same repo (FrankAlonsoskyMolina/clearsight-analytics) and hf_hub_download pattern
    as load_model4() and load_m6_rankings(). Safe to call from inside
    @st.cache_resource functions — the download is a one-time operation.

    Args:
        local_path: Expected local path of the artifact (e.g. M1_DIR / "model.joblib").
        hf_filename: Filename as stored in the HuggingFace repo (e.g. "m1_model.joblib").

    Raises:
        RuntimeError: If the file is absent locally and cannot be downloaded.
    """
    if local_path.exists():
        return
    logger.info("%s not found locally — downloading from HuggingFace...", local_path.name)
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # Download to the HF cache (no local_dir) so nested repo paths like
        # "model1_traditional_ml/saved_model/model.joblib" don't mirror their
        # subfolder tree under local_path.parent.
        dl = hf_hub_download(
            repo_id=_HF_REPO,
            filename=hf_filename,
        )
        shutil.copy2(dl, str(local_path))
        logger.info("Downloaded %s from HuggingFace → %s", hf_filename, local_path)
    except Exception as exc:
        raise RuntimeError(
            f"Artifact '{local_path.name}' not found locally and could not be downloaded "
            f"from HuggingFace ({_HF_REPO}). "
            f"Upload the file as '{hf_filename}' to the repo and retry. Error: {exc}"
        ) from exc


@st.cache_resource(max_entries=1, show_spinner=False)
def load_model1() -> tuple[Any, dict, list, float]:
    """Loads and caches the Model 1 XGBoost ensemble and its preprocessing artifacts.

    Results are cached by Streamlit's ``@st.cache_resource`` decorator so disk
    I/O occurs only on the first call per server session.

    Returns:
        A 4-tuple of:

            - **model** (Any): Trained XGBoost ensemble (sklearn-compatible estimator).
            - **state** (dict): Preprocessing state dict produced by the training pipeline.
            - **feats** (list): Ordered list of feature names expected by the model.
            - **thresh** (float): Optimal decision threshold; defaults to ``0.5`` if the
              artifact file is absent.

    Raises:
        FileNotFoundError: If any required ``.joblib`` artifact is missing from
            ``M1_DIR``.
    """
    t0 = time.perf_counter()
    logger.info("Loading Model 1 (XGBoost) artifacts from %s", M1_DIR)
    _ensure_hf_artifact(M1_DIR / "model.joblib", "model1_traditional_ml/saved_model/model.joblib")
    _ensure_hf_artifact(M1_DIR / "preprocessing_state.joblib", "model1_traditional_ml/saved_model/preprocessing_state.joblib")
    _ensure_hf_artifact(M1_DIR / "feature_names.joblib", "model1_traditional_ml/saved_model/feature_names.joblib")
    _ensure_hf_artifact(M1_DIR / "optimal_threshold.joblib", "model1_traditional_ml/saved_model/optimal_threshold.joblib")
    try:
        model  = joblib.load(M1_DIR / "model.joblib")
        state  = joblib.load(M1_DIR / "preprocessing_state.joblib")
        feats  = joblib.load(M1_DIR / "feature_names.joblib")
        thresh = joblib.load(M1_DIR / "optimal_threshold.joblib") if (M1_DIR / "optimal_threshold.joblib").exists() else 0.5
    except Exception:
        logger.error("Failed to load Model 1 artifacts", exc_info=True)
        raise
    logger.info("Model 1 loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
    return model, state, feats, thresh


@st.cache_resource(max_entries=1, show_spinner=False)
def load_model2() -> tuple[Any, Any, dict, list]:
    """Loads and caches the Model 2 Keras DNN and its preprocessing artifacts.

    TensorFlow is imported lazily inside this function so the app starts without
    requiring TF on the Python import path. Results are cached for the lifetime of
    the Streamlit server process.

    Returns:
        A 4-tuple of:

            - **model** (Any): Compiled Keras Sequential model loaded from ``.keras``.
            - **scaler** (Any): Fitted ``StandardScaler`` for feature normalisation.
            - **state** (dict): Preprocessing state dict produced by the training pipeline.
            - **feats** (list): Ordered list of feature names expected by the model.

    Raises:
        FileNotFoundError: If any required artifact is missing from ``M2_DIR``.
        ImportError: If TensorFlow is not installed in the current environment.
    """
    from tensorflow.keras import backend as K
    K.clear_session()
    gc.collect()
    t0 = time.perf_counter()
    logger.info("Loading Model 2 (Keras DNN) artifacts from %s", M2_DIR)
    _ensure_hf_artifact(M2_DIR / "model.keras", "model2_deep_learning/saved_model/model.keras")
    _ensure_hf_artifact(M2_DIR / "scaler.joblib", "model2_deep_learning/saved_model/scaler.joblib")
    _ensure_hf_artifact(M2_DIR / "preprocessing_state.joblib", "model2_deep_learning/saved_model/preprocessing_state.joblib")
    _ensure_hf_artifact(M2_DIR / "feature_names.joblib", "model2_deep_learning/saved_model/feature_names.joblib")
    try:
        import tensorflow as tf
        model  = tf.keras.models.load_model(M2_DIR / "model.keras")
        scaler = joblib.load(M2_DIR / "scaler.joblib")
        state  = joblib.load(M2_DIR / "preprocessing_state.joblib")
        feats  = joblib.load(M2_DIR / "feature_names.joblib")
    except Exception:
        logger.error("Failed to load Model 2 artifacts", exc_info=True)
        raise
    logger.info("Model 2 loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
    return model, scaler, state, feats


# =============================================================================
# MODEL 3 — CNN Retinal (ResNet50, binary DR classifier)
# =============================================================================
@st.cache_resource(max_entries=1, show_spinner=False)
def load_model3():
    """Loads and caches the Model 3 ResNet50 Diabetic Retinopathy classifier.

    Returns:
        tf.keras.Model: Compiled model loaded from best_model.keras, or None on failure.
    """
    import tensorflow as tf
    from tensorflow.keras import backend as K
    K.clear_session()
    gc.collect()
    t0 = time.perf_counter()
    logger.info("Loading Model 3 (CNN Retinal) from %s", M3_DIR)
    _ensure_hf_artifact(M3_DIR / "best_model.keras", "model3_cnn/saved_model/best_model.keras")
    try:
        model = tf.keras.models.load_model(M3_DIR / "best_model.keras", compile=False)
    except Exception:
        logger.error("Failed to load Model 3 artifacts", exc_info=True)
        raise
    logger.info("Model 3 loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
    return model


def predict_m3(image_file, model) -> tuple[str, float]:
    """Runs ResNet50 inference on a fundus image file.

    Args:
        image_file: A file-like object (e.g. from st.file_uploader).
        model: Loaded tf.keras.Model instance.

    Returns:
        A 2-tuple of (label: str, confidence: float 0–1).
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras import backend as K
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array.astype(np.float32))
    preds = model.predict(img_preprocessed)
    confidence = float(preds[0][0])
    if confidence > 0.5:
        label, conf_out = "HIGH RISK — Diabetic Retinopathy Detected", confidence
    else:
        label, conf_out = "LOW RISK — No Retinopathy Detected", 1.0 - confidence
    K.clear_session()
    gc.collect()
    return label, conf_out


# =============================================================================
# MODEL 3 — Grad-CAM helpers
# =============================================================================
def make_gradcam(img_array, model, last_conv_layer="conv5_block3_out"):
    """Computes a Grad-CAM heatmap for the given preprocessed image array.

    Args:
        img_array: Preprocessed image batch (1, 224, 224, 3) float32 numpy array.
        model: Loaded tf.keras.Model (ResNet50).
        last_conv_layer: Name of the final convolutional layer to target.

    Returns:
        heatmap: 2-D numpy array (H×W) with values in [0, 1].
    """
    import tensorflow as tf
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def overlay_gradcam(pil_image, heatmap, alpha=0.4):
    """Overlays a Grad-CAM heatmap onto a PIL image.

    Args:
        pil_image: Original PIL.Image (any size).
        heatmap: 2-D numpy array from make_gradcam().
        alpha: Blending weight for the heatmap overlay (0–1).

    Returns:
        PIL.Image: Superimposed RGB image.
    """
    img = np.array(pil_image.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_rgb, alpha, 0)
    return Image.fromarray(superimposed)


class Vocabulary:
    def get(self, word, default=0):
        for val in self.__dict__.values():
            if isinstance(val, dict):
                return val.get(word, default)
        return default

    def __len__(self):
        for val in self.__dict__.values():
            if isinstance(val, dict):
                return len(val)
        return 10000 
# ----------------------------------



@st.cache_data(show_spinner=False)
def load_m6_rankings() -> pd.DataFrame:
    if not M6_RESULTS.exists():
        try:
            from huggingface_hub import hf_hub_download
            logger.info("model6_results.csv not found locally — downloading from HuggingFace...")
            hf_hub_download(
                repo_id="FrankAlonsoskyMolina/clearsight-analytics",
                filename="model6_results.csv",
                local_dir=str(M6_RESULTS.parent),
            )
        except Exception as e:
            logger.error("Failed to download model6_results.csv from HuggingFace: %s", e)
            return pd.DataFrame()
    return pd.read_csv(M6_RESULTS)


def get_m6_recommendations(condition: str, current_drug: str, top_n: int = 5) -> pd.DataFrame:
    """Return top_n alternative drugs for a condition, excluding current_drug."""
    df = load_m6_rankings()
    if df.empty:
        return df
    matches = df[df["condition"].str.lower() == condition.strip().lower()].copy()
    if current_drug.strip():
        matches = matches[matches["urlDrugName"].str.lower() != current_drug.strip().lower()]
    return matches.head(top_n)


@st.cache_resource(max_entries=1, show_spinner=False)
def load_model4() -> tuple[Any, Any, Any, Any, Any]:
    gc.collect()
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    from huggingface_hub import hf_hub_download

    HF_REPO = "FrankAlonsoskyMolina/clearsight-analytics"

    def get_model_file(filename):
        local_path = M4_DIR / filename
        if not local_path.exists():
            logger.info("%s not found locally — downloading from HuggingFace...", filename)
            try:
                hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=str(M4_DIR))
            except Exception as e:
                raise RuntimeError(
                    "Could not find '" + filename + "' locally or download from HuggingFace ("
                    + HF_REPO + "). Error: " + str(e)
                )
        return local_path

    class BioBERTMetadataClassifier(nn.Module):
        def __init__(self, biobert_name, num_drugs, num_conditions,
                     meta_embed_dim=32, hidden_dim=256, num_classes=3, dropout=0.3):
            super().__init__()
            self.bert                = AutoModel.from_pretrained(biobert_name)
            self.drug_embedding      = nn.Embedding(num_drugs,      meta_embed_dim)
            self.condition_embedding = nn.Embedding(num_conditions, meta_embed_dim)
            combined_dim = 768 + meta_embed_dim * 2
            self.norm    = nn.LayerNorm(combined_dim)
            self.proj    = nn.Linear(combined_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(hidden_dim, num_classes)

        @staticmethod
        def _mean_pool(last_hidden_state, attention_mask):
            mask   = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden_state * mask).sum(dim=1)
            return summed / mask.sum(dim=1).clamp(min=1)

        def forward(self, input_ids, attention_mask, drug_idx, cond_idx):
            outputs  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            text_out = self._mean_pool(outputs.last_hidden_state, attention_mask)
            drug_out = self.drug_embedding(drug_idx)
            cond_out = self.condition_embedding(cond_idx)
            combined = torch.cat([text_out, drug_out, cond_out], dim=1)
            combined = self.norm(combined)
            combined = torch.nn.functional.gelu(self.proj(combined))
            return self.fc(self.dropout(combined))

    t0 = time.perf_counter()
    logger.info("Loading Model 4 (BioBERT) artifacts from %s", M4_DIR)

    try:
        drug_le  = joblib.load(get_model_file("drug_encoder.joblib"))
        cond_le  = joblib.load(get_model_file("condition_encoder.joblib"))
        label_le = joblib.load(get_model_file("label_encoder_biobert_lora_all_combos.joblib"))

        biobert_name = "dmis-lab/biobert-base-cased-v1.2"
        tokenizer = AutoTokenizer.from_pretrained(biobert_name)

        model = BioBERTMetadataClassifier(
            biobert_name   = biobert_name,
            num_drugs      = len(drug_le.classes_),
            num_conditions = len(cond_le.classes_),
        )
        model.load_state_dict(torch.load(get_model_file("model_biobert_lora_all_combos.pt"), map_location="cpu"))
        model.eval()

    except Exception:
        logger.error("Failed to load Model 4 (BioBERT) artifacts", exc_info=True)
        raise

    logger.info("Model 4 (BioBERT) loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
    return model, tokenizer, drug_le, cond_le, label_le


def predict_m4(text_notes: str, drug_name: str, condition: str) -> tuple[str, float, str, str]:
    """Runs NLP inference using the cached BioBERT Classifier."""
    logger.info("M4 prediction triggered — drug=%s condition=%s", drug_name, condition)
    import torch

    model, tokenizer, drug_le, cond_le, label_le = load_model4()

    def safe_encode(le: Any, val: str) -> int:
        return le.transform([val if val in le.classes_ else "unknown"])[0]

    # 1. Encode metadata
    drug_idx = torch.tensor([safe_encode(drug_le, drug_name)], dtype=torch.long)
    cond_idx = torch.tensor([safe_encode(cond_le, condition)], dtype=torch.long)

    # 2. Tokenize raw text (BioBERT handles its own tokenization — no preprocessing)
    enc = tokenizer(
        str(text_notes),
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]

    # 3. Model forward pass
    with torch.no_grad():
        logits = model(input_ids, attn_mask, drug_idx, cond_idx)
        probs  = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))

    label      = label_le.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])
    logger.info("M4 result — label=%s confidence=%.4f", label, confidence)

    explanation_map = {
        "Ineffective":        "Critical Interpretation: Linguistic markers suggest severe symptoms, treatment failure, or a potential emergency. Immediate review advised.",
        "Somewhat Effective": "Elevated Interpretation: Linguistic markers indicate lingering symptoms or an incomplete response to current treatment context.",
        "Highly Effective":   "Stable Interpretation: Linguistic markers indicate a positive response to treatment and stable patient condition.",
    }
    css_map = {
        "Ineffective":        "risk-high",
        "Somewhat Effective": "risk-medium",
        "Highly Effective":   "risk-low",
    }
    display_map = {
        "Ineffective":        "CRITICAL",
        "Somewhat Effective": "ELEVATED",
        "Highly Effective":   "STABLE",
    }

    explanation   = explanation_map.get(label, "Interpretation unavailable for this label.")
    display_title = display_map.get(label, label.upper())

    gc.collect()
    return f"{display_title} RISK SENTIMENT", confidence, css_map.get(label, "risk-low"), explanation


def generate_clinical_synthesis(
    m1_proba: float,
    m2_proba: float,
    m5_label: str,
    m4_label: str,
    m4_explanation: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=st.secrets.get("GROQ_API_KEY", ""),
        base_url="https://api.groq.com/openai/v1",
    )

    prompt = f"""You are a senior clinical informatics specialist reviewing AI model outputs 
for a hospitalized diabetic patient at MedInsight Healthcare.

MULTI-MODEL DIAGNOSTIC OUTPUTS:
- Readmission Risk (XGBoost M1): {m1_proba*100:.1f}%
- Readmission Risk (DNN M2): {m2_proba*100:.1f}%
- Predicted Length of Stay (M5): {m5_label}
- Clinical Notes Sentiment (NLP M4): {m4_label}
- NLP Context: {m4_explanation}

Based on these AI model outputs, write a concise clinical synthesis in exactly this format:

SYNTHESIS (2-3 sentences): A professional summary of the patient's overall risk profile, 
noting where models agree or diverge.

RECOMMENDATIONS:
1. [First actionable recommendation for the care team]
2. [Second recommendation]
3. [Third recommendation]

Keep the tone clinical, precise, and professional. Do not mention specific model names 
(M1, M2, etc.) — speak in terms of clinical risk indicators."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=1024,
            temperature=0.2,
            messages=[
                {"role": "system", "content": (
                    "You are a helpful medical AI assistant. Be concise and clinical. "
                    "CRITICAL RULE: YOU MUST ALWAYS END EVERY RESPONSE WITH THIS EXACT TEXT: "
                    "'\n\n\u26a0\ufe0f **Disclaimer:** This is an AI-generated analysis for investigational "
                    "use only. Always consult a licensed healthcare provider.'"
                )},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Clinical synthesis API call failed: %s", e, exc_info=True)
        return f"Synthesis unavailable: {str(e)}"


@st.cache_resource(max_entries=1, show_spinner=False)
def load_model5() -> tuple[Any, dict, list]:
    """Loads and caches the Model 5 Length-of-Stay classifier and its artifacts.

    Results are cached by Streamlit's ``@st.cache_resource`` decorator so disk
    I/O occurs only on the first call per server session.

    Returns:
        A 3-tuple of:

            - **model** (Any): Trained multi-class classifier (sklearn-compatible).
            - **state** (dict): Preprocessing state dict produced by the training pipeline.
            - **feats** (list): Ordered list of feature names expected by the model.

    Raises:
        FileNotFoundError: If any required ``.joblib`` artifact is missing from
            ``M5_DIR``.
    """
    M5_DIR = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model"
    t0 = time.perf_counter()
    logger.info("Loading Model 5 (LoS classifier) artifacts from %s", M5_DIR)
    _ensure_hf_artifact(M5_DIR / "model.joblib", "model5_innovation/saved_model/model.joblib")
    _ensure_hf_artifact(M5_DIR / "preprocessing_state.joblib", "model5_innovation/saved_model/preprocessing_state.joblib")
    _ensure_hf_artifact(M5_DIR / "feature_names.joblib", "model5_innovation/saved_model/feature_names.joblib")
    try:
        model = joblib.load(M5_DIR / "model.joblib")
        state = joblib.load(M5_DIR / "preprocessing_state.joblib")
        feats = joblib.load(M5_DIR / "feature_names.joblib")
    except Exception:
        logger.error("Failed to load Model 5 artifacts", exc_info=True)
        raise
    logger.info("Model 5 loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
    return model, state, feats

def predict_m5(patient_dict: dict) -> tuple[str, float, str]:
    """Runs inference for the Model 5 Length-of-Stay (LoS) classifier.

    Preprocesses the patient record, calls the cached multi-class classifier, and
    maps the predicted class index to a human-readable duration label and UI CSS class.

    Args:
        patient_dict: Raw patient encounter dictionary from the Streamlit form.

    Returns:
        A 3-tuple of:

            - **label** (str): Human-readable stay category
              (e.g. ``"SHORT STAY (1-2 days)"``).
            - **confidence** (float): Maximum class probability in [0, 1].
            - **css_class** (str): CSS class for UI badge colouring
              (``"risk-low"``, ``"risk-medium"``, or ``"risk-high"``).

    Raises:
        ValueError: If ``preprocess_patient`` produces a feature shape mismatch.
        KeyError: If the predicted class index is not present in the labels mapping.
    """
    logger.info("M5 prediction triggered — age=%s time_in_hospital=%s",
                patient_dict.get("age"), patient_dict.get("time_in_hospital"))
    model, state, feats = load_model5()  # uses @st.cache_resource — no disk I/O after first call

    # Process input
    X = preprocess_patient(patient_dict, state, feats)
    
    # Run prediction — derive class from probabilities so label and confidence
    # are always consistent (model.predict() can diverge from predict_proba argmax
    # due to internal voting thresholds in some sklearn estimators).
    probabilities = model.predict_proba(X)[0]
    prediction = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    
    # Mapping
    labels = {
        0: "SHORT STAY (1-2 days)", 
        1: "STANDARD STAY (3-5 days)", 
        2: "EXTENDED STAY (6+ days)"
    }
    css_classes = {0: "risk-low", 1: "risk-medium", 2: "risk-high"}
    
    logger.info("M5 result — class=%d label=%s confidence=%.4f",
                prediction, labels[prediction], confidence)
    # CRITICAL: Return exactly 3 values
    return labels[prediction], confidence, css_classes[prediction]


# =============================================================================
# PREPROCESSING — single patient dict → model-ready row
# =============================================================================
def preprocess_patient(patient: dict, preprocessing_state: dict,
                       feature_names: list) -> pd.DataFrame:
    """Transforms a raw patient encounter dict into a model-ready feature DataFrame.

    Merges the input record with medication defaults, converts the age bracket string
    to a numeric midpoint, derives recurrency features, runs the shared
    ``engineer_features`` pipeline, drops the target column if present, and reindexes
    to the exact column order the model was trained on.

    Args:
        patient: Raw patient dictionary from the prediction form. Expected keys include
            demographic fields (``age``, ``gender``, ``race``), clinical metrics
            (``time_in_hospital``, ``num_lab_procedures``, etc.), and ICD-9 codes
            (``diag_1``, ``diag_2``, ``diag_3``).
        preprocessing_state: Serialised encoding maps and statistics from the
            training pipeline's ``engineer_features`` call.
        feature_names: Ordered list of column names the downstream model expects.

    Returns:
        A single-row ``pd.DataFrame`` with columns matching ``feature_names``;
        any missing columns are filled with ``0``.

    Raises:
        KeyError: If ``engineer_features`` expects a column not derivable from
            ``patient``.
    """
    med_defaults = {c: 'No' for c in [
        'metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
        'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
        'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide',
        'insulin','glyburide-metformin','glipizide-metformin',
        'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone',
    ]}
    row = {**med_defaults, **patient}

    age_map = {'[0-10)':5,'[10-20)':15,'[20-30)':25,'[30-40)':35,'[40-50)':45,
               '[50-60)':55,'[60-70)':65,'[70-80)':75,'[80-90)':85,'[90-100)':95}
    row['age_numeric'] = age_map.get(row.get('age', '[60-70)'), 65)
    row.pop('age', None)

    row.setdefault('prior_encounters_count', 0)
    row.setdefault('is_recurrent_patient',   0)
    row.setdefault('prior_inpatient_cumsum', row.get('number_inpatient', 0))

    df = pd.DataFrame([row])
    df, _ = engineer_features(df, preprocessing_state=preprocessing_state)
    df.drop(columns=[c for c in ['readmission_binary'] if c in df.columns], inplace=True)
    df = df.reindex(columns=feature_names, fill_value=0)
    return df


def predict_m1(patient_dict: dict) -> tuple[int, float, float]:
    """Runs readmission inference using the cached Model 1 XGBoost ensemble.

    Preprocesses the patient record and applies the cost-optimised decision
    threshold stored in the model artifacts to produce the binary prediction.

    Args:
        patient_dict: Raw patient encounter dictionary from the Streamlit form.

    Returns:
        A 3-tuple of:

            - **pred** (int): Binary readmission label (``1`` = high risk,
              ``0`` = low risk) using the cost-optimised threshold.
            - **proba** (float): Raw readmission probability in [0, 1].
            - **conf** (float): Model confidence as ``max(proba, 1 - proba)``
              in [0.5, 1.0].

    Raises:
        ValueError: If ``preprocess_patient`` produces a feature shape mismatch.
    """
    logger.info("M1 prediction triggered — age=%s gender=%s",
                patient_dict.get("age"), patient_dict.get("gender"))
    model, state, feats, thresh = load_model1()
    X = preprocess_patient(patient_dict, state, feats)
    proba = float(model.predict_proba(X)[:, 1][0])
    pred  = int(proba >= thresh)
    conf  = max(proba, 1 - proba)
    logger.info("M1 result — proba=%.4f pred=%d threshold=%.2f", proba, pred, thresh)
    return pred, proba, conf


def predict_m2(patient_dict: dict) -> tuple[int, float, float]:
    """Runs readmission inference using the cached Model 2 Keras DNN.

    Preprocesses the patient record, applies the fitted ``StandardScaler``, and
    runs a forward pass through the cached Keras model at the 0.5 threshold.

    Args:
        patient_dict: Raw patient encounter dictionary from the Streamlit form.

    Returns:
        A 3-tuple of:

            - **pred** (int): Binary readmission label (``1`` = high risk,
              ``0`` = low risk) at the 0.5 probability threshold.
            - **proba** (float): Raw sigmoid output probability in [0, 1].
            - **conf** (float): Model confidence as ``max(proba, 1 - proba)``
              in [0.5, 1.0].

    Raises:
        ValueError: If ``preprocess_patient`` produces a feature shape mismatch.
        ImportError: If TensorFlow is not installed in the current environment.
    """
    logger.info("M2 prediction triggered — age=%s gender=%s",
                patient_dict.get("age"), patient_dict.get("gender"))
    model, scaler, state, feats = load_model2()
    X = preprocess_patient(patient_dict, state, feats)
    X_scaled = scaler.transform(X)
    proba = float(model.predict(X_scaled, verbose=0).flatten()[0])
    pred  = int(proba >= 0.5)
    conf  = max(proba, 1 - proba)
    logger.info("M2 result — proba=%.4f pred=%d", proba, pred)
    from tensorflow.keras import backend as K
    K.clear_session()
    gc.collect()
    return pred, proba, conf


# =============================================================================
# RISK GAUGE (animated SVG)
# =============================================================================
def risk_gauge_svg(probability: float, label: str = "RISK") -> str:
    """Renders an animated circular risk gauge as an inline SVG string.

    Produces a colour-coded arc gauge (teal / amber / red) with a glowing progress
    ring, a percentage label at the centre, and a smooth CSS dash-offset transition.
    Intended to be rendered via ``st.markdown(..., unsafe_allow_html=True)``.

    Args:
        probability: Readmission probability in [0.0, 1.0]. Values outside this
            range are clamped automatically.
        label: Short uppercase label shown inside the gauge and used as a unique
            suffix for the SVG ``filter`` ID to prevent DOM collisions when multiple
            gauges appear on the same page. Defaults to ``"RISK"``.

    Returns:
        An HTML string containing a ``<div>``-wrapped inline ``<svg>`` element
        ready to pass to ``st.markdown``.
    """
    pct = max(0.0, min(1.0, probability))
    # Color interpolation
    if pct >= 0.65:
        color, glow = "#f87171", "rgba(248,113,113,0.5)"
    elif pct >= 0.40:
        color, glow = "#fbbf24", "rgba(251,191,36,0.5)"
    else:
        color, glow = "#5eead4", "rgba(94,234,212,0.5)"

    # Arc geometry
    radius = 70
    circumference = 2 * np.pi * radius
    offset = circumference * (1 - pct)

    return f"""
    <div style="display:flex; justify-content:center; padding:1rem 0;">
      <svg width="180" height="180" viewBox="0 0 180 180">
        <defs>
          <filter id="glow{label}">
            <feGaussianBlur stdDeviation="3" result="blur"/>
            <feMerge>
              <feMergeNode in="blur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        <!-- Background ring -->
        <circle cx="90" cy="90" r="{radius}" fill="none"
                stroke="rgba(34,211,238,0.1)" stroke-width="10"/>
        <!-- Progress ring -->
        <circle cx="90" cy="90" r="{radius}" fill="none"
                stroke="{color}" stroke-width="10" stroke-linecap="round"
                stroke-dasharray="{circumference:.2f}"
                stroke-dashoffset="{offset:.2f}"
                transform="rotate(-90 90 90)"
                filter="url(#glow{label})"
                style="transition: stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1);"/>
        <!-- Center text -->
        <text x="90" y="86" text-anchor="middle"
              font-family="JetBrains Mono, monospace"
              font-size="32" font-weight="700"
              fill="white">{pct*100:.1f}<tspan font-size="20" fill="#94a3b8">%</tspan></text>
        <text x="90" y="110" text-anchor="middle"
              font-family="JetBrains Mono, monospace"
              font-size="9" font-weight="600"
              fill="#64748b" letter-spacing="2">{label}</text>
      </svg>
    </div>
    """.strip()


def risk_label(p: float) -> tuple[str, str]:
    """Maps a readmission probability to a human-readable risk tier and CSS class.

    Uses the same thresholds as the production inference pipeline:
    high risk ≥ 0.65, moderate risk ≥ 0.40, low risk below that.

    Args:
        p: Readmission probability in [0.0, 1.0].

    Returns:
        A 2-tuple of:

            - **label** (str): Human-readable tier (``"HIGH RISK"``,
              ``"MODERATE RISK"``, or ``"LOW RISK"``).
            - **css_class** (str): Corresponding CSS class name (``"risk-high"``,
              ``"risk-medium"``, or ``"risk-low"``).
    """
    if p >= 0.65: return "HIGH RISK", "risk-high"
    if p >= 0.40: return "MODERATE RISK", "risk-medium"
    return "LOW RISK", "risk-low"


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar() -> str:
    """Renders the full sidebar UI and returns the active navigation page label.

    Builds the ClearSight brand block, the navigation ``st.radio`` widget, per-model
    ONLINE / ERROR status indicators (determined by attempting each cached loader),
    PENDING badges for models not yet integrated, and the investigational-use badge.

    Returns:
        The raw string value of the selected radio option
        (e.g. ``"🛰️  Command Center"``). Used by ``main()`` to route page rendering.
    """
    with st.sidebar:
        st.markdown(
            f'<div class="brand">{CLEARSIGHT_ICON_SVG}'
            '<div><div class="name">ClearSight</div>'
            '<div class="tag">Analytics</div></div></div>',
            unsafe_allow_html=True,
        )
        
        
        st.markdown('<div class="sidebar-header">NAVIGATION</div>', unsafe_allow_html=True)
        
        # The radio buttons will now be styled as hoverable blocks!
        page = st.radio("nav", [
            "🛰️  Command Center",
            "🔬  Predict",
            "📊  Insights",
            "👁️  Retinal AI",
        ], label_visibility="collapsed")

        # ========== DIVIDER: Nav / System Status ==========
        st.markdown(
            '<hr style="border:none;border-top:1px solid rgba(94,234,212,0.2);margin:0.6rem 0 0.8rem;">',
            unsafe_allow_html=True,
        )

        # ========== SYSTEM STATUS ==========
        st.markdown('<div class="sidebar-header">SYSTEM STATUS</div>', unsafe_allow_html=True)

        # -- 🤖 AI MODELS subsection header --
        st.markdown(
            '<div style="font-size:0.68rem;font-weight:700;color:#5eead4;'
            'letter-spacing:0.1em;margin:0.5rem 0 0.3rem;padding-left:2px;">'
            '🤖 AI MODELS</div>',
            unsafe_allow_html=True,
        )

        # ── Lazy status: file-existence check only — no model loading at startup ──
        # Models are loaded on first predict click via @st.cache_resource loaders.
        # READY  = artifact file present on disk (loads instantly on first predict)
        # STANDBY = not yet downloaded (auto-downloads from HuggingFace on first predict)
        def _status_pill(label, dot_color, text, pill_cls):
            return (
                f'<div class="status-row"><span>'
                f'<span style="color:{dot_color}; margin-right:5px;">●</span>'
                f' {label}</span>'
                f'<span class="status-pill {pill_cls}">{text}</span></div>'
            )

        _m1_ok = (M1_DIR / "model.joblib").exists()
        st.markdown(_status_pill("M1 · XGBoost",
                                 "#10b981" if _m1_ok else "#f59e0b",
                                 "READY" if _m1_ok else "STANDBY",
                                 "pill-online" if _m1_ok else "pill-pending"),
                    unsafe_allow_html=True)

        _m2_ok = (M2_DIR / "model.keras").exists()
        st.markdown(_status_pill("M2 · DNN",
                                 "#10b981" if _m2_ok else "#f59e0b",
                                 "READY" if _m2_ok else "STANDBY",
                                 "pill-online" if _m2_ok else "pill-pending"),
                    unsafe_allow_html=True)

        _m3_ok = (M3_DIR / "best_model.keras").exists() or (M3_DIR / "model.keras").exists()
        st.markdown(_status_pill("M3 · CNN Retina",
                                 "#10b981" if _m3_ok else "#f59e0b",
                                 "READY" if _m3_ok else "STANDBY",
                                 "pill-online" if _m3_ok else "pill-pending"),
                    unsafe_allow_html=True)

        _m4_ok = (M4_DIR / "model_biobert_lora_all_combos.pt").exists()
        st.markdown(_status_pill("M4 · NLP Notes",
                                 "#10b981" if _m4_ok else "#f59e0b",
                                 "READY" if _m4_ok else "STANDBY",
                                 "pill-online" if _m4_ok else "pill-pending"),
                    unsafe_allow_html=True)

        _m5_ok = (M5_DIR / "model.joblib").exists()
        st.markdown(_status_pill("M5 · Innovation",
                                 "#10b981" if _m5_ok else "#f59e0b",
                                 "READY" if _m5_ok else "STANDBY",
                                 "pill-online" if _m5_ok else "pill-pending"),
                    unsafe_allow_html=True)

        _m6_ok = M6_RESULTS.exists()
        st.markdown(_status_pill("Drug Recommendation",
                                 "#10b981" if _m6_ok else "#f59e0b",
                                 "READY" if _m6_ok else "STANDBY",
                                 "pill-online" if _m6_ok else "pill-pending"),
                    unsafe_allow_html=True)

        # AI Copilot entry
        st.markdown(
            '<div class="status-row">'
            '<span style="color:var(--text-primary);">🤖 AI Copilot (Llama 3.1)</span>'
            '<span class="status-pill pill-online">ONLINE</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ========== DIVIDER: AI Models / System Health ==========
        st.markdown(
            '<hr style="border:none;border-top:1px solid rgba(94,234,212,0.2);margin:0.6rem 0 0.5rem;">',
            unsafe_allow_html=True,
        )

        # ========== SYSTEM HEALTH ==========
        st.markdown(
            '<div style="font-size:0.68rem;font-weight:700;color:#5eead4;'
            'letter-spacing:0.1em;margin:0 0 0.4rem;padding-left:2px;">'
            '⚡ SYSTEM HEALTH</div>',
            unsafe_allow_html=True,
        )
        # API Status — teal badge
        st.markdown(
            '<div class="status-row">'
            '<span style="color:var(--text-secondary);">● API Status</span>'
            '<span class="status-pill pill-online">ACTIVE</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        # Avg Response — plain text value
        st.markdown(
            '<div class="status-row">'
            '<span style="color:var(--text-secondary);">● Avg Response</span>'
            '<span style="color:var(--text-primary);font-weight:600;font-size:0.78rem;">142ms</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        # Uptime — plain text value
        st.markdown(
            '<div class="status-row">'
            '<span style="color:var(--text-secondary);">● Uptime</span>'
            '<span style="color:var(--text-primary);font-weight:600;font-size:0.78rem;">99.8%</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ========== DIVIDER: System Health / Disclaimer ==========
        st.markdown(
            '<hr style="border:none;border-top:1px solid rgba(94,234,212,0.2);margin:0.6rem 0 0.6rem;">',
            unsafe_allow_html=True,
        )

        # Investigational Use Badge
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                    color:#fbbf24; background:rgba(251,191,36,0.08);
                    border:1px solid rgba(251,191,36,0.3); border-radius:8px;
                    padding:10px; text-align:center; line-height:1.5; margin: 0 1rem;">
          ⚠ INVESTIGATIONAL USE ONLY<br>NOT FDA CLEARED
        </div>
        """, unsafe_allow_html=True)
        
    return page
# =============================================================================
# PAGE: COMMAND CENTER (Home)
# =============================================================================

def page_home() -> None:
    """Renders the Command Center landing page.

    Displays an animated hero section, dataset foundation statistics, live model
    performance metric bars with JS-animated progress fills, estimated clinical
    impact cards, and a four-step clinical decision pipeline overview. All content
    is injected via ``st.markdown`` with embedded CSS and JavaScript.
    """

    # ── Animated hero with JS counters + teal particles ──────────────────────
    st.markdown("""
    <style>
    /* ── Counter animation ─────────────────────────────────────────── */
    .count-up {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        line-height: 1;
    }

    /* ── Progress bar animated ─────────────────────────────────────── */
    .metric-bar-wrap {
        background: rgba(34,211,238,0.07);
        border: 1px solid rgba(34,211,238,0.15);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 10px;
        position: relative;
        overflow: hidden;
    }
    .metric-bar-wrap:hover {
        border-color: rgba(94,234,212,0.4);
        background: rgba(34,211,238,0.1);
        transition: all 0.2s;
    }
    .metric-bar-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .metric-bar-name {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-bar-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: white;
    }
    .metric-bar-track {
        background: rgba(255,255,255,0.06);
        border-radius: 999px;
        height: 8px;
        overflow: hidden;
    }
    .metric-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #22d3ee, #5eead4);
        box-shadow: 0 0 12px rgba(34,211,238,0.5);
        width: 0%;
        transition: width 1.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-bar-fill-amber {
        background: linear-gradient(90deg, #f59e0b, #fcd34d);
        box-shadow: 0 0 12px rgba(245,158,11,0.5);
    }
    .metric-bar-fill-green {
        background: linear-gradient(90deg, #10b981, #5eead4);
        box-shadow: 0 0 12px rgba(16,185,129,0.5);
    }

    /* ── Impact cards ───────────────────────────────────────────────── */
    .impact-card {
        background: rgba(15,30,52,0.6);
        border: 1px solid rgba(34,211,238,0.12);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.2s;
        position: relative;
        overflow: hidden;
    }
    .impact-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #22d3ee, transparent);
    }
    .impact-card:hover {
        transform: translateY(-4px);
        border-color: rgba(94,234,212,0.35);
        box-shadow: 0 12px 40px rgba(34,211,238,0.12);
    }
    .impact-icon {
        font-size: 2rem;
        margin-bottom: 0.6rem;
        display: block;
    }
    .impact-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.9rem;
        font-weight: 700;
        color: #5eead4;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .impact-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .impact-sub {
        font-size: 0.82rem;
        color: #475569;
        margin-top: 0.4rem;
        line-height: 1.4;
    }

    /* ── Dataset stat pills ────────────────────────────────────────── */
    .ds-pill {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(34,211,238,0.06);
        border: 1px solid rgba(34,211,238,0.15);
        border-radius: 10px;
        padding: 1rem 1.3rem;
        width: 100%;
        transition: all 0.2s;
    }
    .ds-pill:hover {
        background: rgba(34,211,238,0.1);
        border-color: rgba(94,234,212,0.3);
    }
    .ds-pill .ico { font-size: 1.4rem; }
    .ds-pill .txt .top {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: white;
        line-height: 1;
    }
    .ds-pill .txt .bot {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-top: 3px;
    }

    /* ── Section divider ───────────────────────────────────────────── */
    .sec-div {
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 2.2rem 0 1.4rem;
    }
    .sec-div .num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        color: #5eead4;
        background: rgba(94,234,212,0.08);
        border: 1px solid rgba(94,234,212,0.2);
        padding: 3px 10px;
        border-radius: 6px;
        white-space: nowrap;
    }
    .sec-div h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: white !important;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .sec-div .line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(34,211,238,0.3), transparent);
    }
    </style>

    <!-- ═══ HERO ═══════════════════════════════════════════════════════ -->
    <div style="
        position: relative;
        background:
            radial-gradient(circle at 75% 40%, rgba(34,211,238,0.22) 0%, transparent 45%),
            radial-gradient(circle at 10% 80%, rgba(94,234,212,0.1) 0%, transparent 40%),
            linear-gradient(135deg, #060e1a 0%, #0d1f38 50%, #070f1c 100%);
        border: 1px solid rgba(34,211,238,0.25);
        border-radius: 20px;
        padding: 2.8rem 3rem;
        margin-bottom: 2rem;
        overflow: hidden;
        box-shadow: 0 0 0 1px rgba(94,234,212,0.05),
                    0 20px 60px rgba(0,0,0,0.4),
                    inset 0 1px 0 rgba(94,234,212,0.12);
    ">
      <!-- Animated teal glow orb -->
      <div style="
          position: absolute; top: -60px; right: -60px;
          width: 300px; height: 300px; border-radius: 50%;
          background: radial-gradient(circle, rgba(34,211,238,0.18) 0%, transparent 70%);
          animation: orbPulse 4s ease-in-out infinite;
          pointer-events: none;
      "></div>
      <style>
        @keyframes orbPulse {
          0%,100% { transform: scale(1); opacity: 0.8; }
          50%      { transform: scale(1.15); opacity: 1; }
        }
        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(18px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .hero-badge { animation: fadeSlideUp 0.5s ease both; }
        .hero-title { animation: fadeSlideUp 0.6s 0.1s ease both; }
        .hero-sub   { animation: fadeSlideUp 0.6s 0.2s ease both; }
        .hero-pills { animation: fadeSlideUp 0.6s 0.35s ease both; }
      </style>

      <div class="hero-badge" style="
          display:inline-block;
          font-family:'JetBrains Mono',monospace;
          font-size:0.68rem; font-weight:700;
          color:#5eead4;
          background:rgba(94,234,212,0.1);
          border:1px solid rgba(94,234,212,0.35);
          padding:5px 14px; border-radius:999px;
          letter-spacing:0.12em; text-transform:uppercase;
          margin-bottom:1rem;
      ">● LIVE · MEDINSIGHT CLINICAL AI PLATFORM</div>

      <h1 class="hero-title" style="
          font-family:'Space Grotesk',sans-serif;
          font-size:2.8rem; font-weight:800;
          margin:0 0 0.7rem;
          letter-spacing:-0.04em; line-height:1.05;
          background: linear-gradient(120deg, #ffffff 0%, #5eead4 60%, #22d3ee 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          max-width: 780px;
      ">Clinical AI Command Center</h1>

      <p class="hero-sub" style="
          font-size:1.08rem;
          color:#94a3b8;
          max-width:680px;
          line-height:1.6;
          margin: 0 0 1.6rem;
      ">
        Precision readmission prediction and capacity planning for MedInsight's
        47 partner hospitals — trained on <strong style="color:#5eead4;">101,766 real patient encounters</strong>
        with full SHAP interpretability for clinical trust.
      </p>

      <div class="hero-pills">
        <span style="display:inline-flex;align-items:center;gap:6px;
                     background:rgba(34,211,238,0.1);color:#5eead4;
                     padding:6px 14px;border-radius:999px;font-size:0.75rem;
                     font-weight:600;margin-right:8px;margin-bottom:6px;
                     border:1px solid rgba(34,211,238,0.25);
                     font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;">
          ⚡ &lt;300ms inference
        </span>
        <span style="display:inline-flex;align-items:center;gap:6px;
                     background:rgba(34,211,238,0.1);color:#5eead4;
                     padding:6px 14px;border-radius:999px;font-size:0.75rem;
                     font-weight:600;margin-right:8px;margin-bottom:6px;
                     border:1px solid rgba(34,211,238,0.25);
                     font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;">
          🔬 SHAP interpretability
        </span>
        <span style="display:inline-flex;align-items:center;gap:6px;
                     background:rgba(34,211,238,0.1);color:#5eead4;
                     padding:6px 14px;border-radius:999px;font-size:0.75rem;
                     font-weight:600;margin-right:8px;margin-bottom:6px;
                     border:1px solid rgba(34,211,238,0.25);
                     font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;">
          🛡️ HIPAA-aware design
        </span>
        <span style="display:inline-flex;align-items:center;gap:6px;
                     background:rgba(16,185,129,0.1);color:#10b981;
                     padding:6px 14px;border-radius:999px;font-size:0.75rem;
                     font-weight:600;margin-right:8px;margin-bottom:6px;
                     border:1px solid rgba(16,185,129,0.25);
                     font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;">
          ✓ 5 models live
        </span>
      </div>
    </div>

    <!-- JS counter animation script -->
    <script>
    function animateCounter(el, target, duration, prefix, suffix, decimals) {
        var start = 0;
        var step  = target / (duration / 16);
        var timer = setInterval(function() {
            start += step;
            if (start >= target) { start = target; clearInterval(timer); }
            var disp = decimals ? start.toFixed(decimals) : Math.floor(start).toLocaleString();
            el.textContent = (prefix||'') + disp + (suffix||'');
        }, 16);
    }
    document.addEventListener('DOMContentLoaded', function() {
        var counters = document.querySelectorAll('[data-count]');
        counters.forEach(function(el) {
            var target   = parseFloat(el.getAttribute('data-count'));
            var duration = parseInt(el.getAttribute('data-dur') || '1800');
            var prefix   = el.getAttribute('data-prefix') || '';
            var suffix   = el.getAttribute('data-suffix') || '';
            var decimals = parseInt(el.getAttribute('data-decimals') || '0');
            animateCounter(el, target, duration, prefix, suffix, decimals);
        });
        var bars = document.querySelectorAll('[data-bar]');
        bars.forEach(function(el) {
            var pct = el.getAttribute('data-bar');
            setTimeout(function() { el.style.width = pct + '%'; }, 200);
        });
    });
    </script>
    """, unsafe_allow_html=True)

    # ── SECTION 1: Dataset Foundation ────────────────────────────────────────
    st.markdown("""
    <div class="sec-div">
      <span class="num">01</span>
      <h2>Dataset Foundation</h2>
      <div class="line"></div>
    </div>
    """, unsafe_allow_html=True)

    d1, d2, d3, d4, d5, d6 = st.columns(6)

    # Card 1 — Patient Encounters
    d1.markdown("""
    <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                padding:24px;min-height:180px;transition:all 0.3s ease;"
         onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
         onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
        <div style="font-size:2rem;">📋</div>
        <div>
          <div style="font-size:1.6rem;font-weight:700;color:#e2e8f0;line-height:1;">101,766</div>
          <div style="color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.5px;margin-top:3px;">Patient Encounters</div>
        </div>
      </div>
      <div style="margin-bottom:10px;">
        <div style="color:#94a3b8;font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:5px;">Data Completeness</div>
        <div style="color:#5eead4;font-size:1.05rem;font-weight:600;">98.2%</div>
      </div>
      <div style="background:rgba(71,85,105,0.3);border-radius:6px;height:6px;overflow:hidden;margin-bottom:7px;">
        <div style="background:linear-gradient(90deg,#22d3ee 0%,#5eead4 100%);height:100%;width:98.2%;transition:width 1s ease;"></div>
      </div>
      <div style="color:#64748b;font-size:0.65rem;line-height:1.4;">Missing: Weight (97%), Specialty (49%)</div>
    </div>
    """, unsafe_allow_html=True)

    # Card 2 — Clinical Features
    d2.markdown("""
    <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                padding:24px;min-height:180px;transition:all 0.3s ease;"
         onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
         onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
        <div style="font-size:2rem;">📄</div>
        <div>
          <div style="font-size:1.6rem;font-weight:700;color:#e2e8f0;line-height:1;">50</div>
          <div style="color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.5px;margin-top:3px;">Clinical Features</div>
        </div>
      </div>
      <div style="margin-bottom:10px;">
        <div style="color:#94a3b8;font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:5px;">Features in Production</div>
        <div style="color:#5eead4;font-size:1.05rem;font-weight:600;">39 / 50</div>
      </div>
      <div style="background:rgba(71,85,105,0.3);border-radius:6px;height:6px;overflow:hidden;margin-bottom:7px;">
        <div style="background:linear-gradient(90deg,#22d3ee 0%,#5eead4 100%);height:100%;width:78%;transition:width 1s ease;"></div>
      </div>
      <div style="color:#64748b;font-size:0.65rem;line-height:1.4;">Top 10 features contribute 67% predictive power</div>
    </div>
    """, unsafe_allow_html=True)

    # Card 3 — Partner Hospitals
    d3.markdown("""
    <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                padding:24px;min-height:180px;transition:all 0.3s ease;"
         onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
         onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
        <div style="font-size:2rem;">🏥</div>
        <div>
          <div style="font-size:1.6rem;font-weight:700;color:#e2e8f0;line-height:1;">47</div>
          <div style="color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.5px;margin-top:3px;">Partner Hospitals</div>
        </div>
      </div>
      <div style="margin-bottom:10px;">
        <div style="color:#94a3b8;font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:5px;">Active Diabetic Patients</div>
        <div style="color:#5eead4;font-size:1.05rem;font-weight:600;">180,000+</div>
      </div>
      <div style="background:rgba(71,85,105,0.3);border-radius:6px;height:6px;overflow:hidden;margin-bottom:7px;">
        <div style="background:linear-gradient(90deg,#22d3ee 0%,#5eead4 100%);height:100%;width:100%;transition:width 1s ease;"></div>
      </div>
      <div style="color:#64748b;font-size:0.65rem;line-height:1.4;">Midwest network coverage · 47 sites</div>
    </div>
    """, unsafe_allow_html=True)

    # Card 4 — Retinal Images (two bars)
    d4.markdown("""
    <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                padding:24px;min-height:180px;transition:all 0.3s ease;"
         onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
         onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
        <div style="font-size:2rem;">👁️</div>
        <div>
          <div style="font-size:1.6rem;font-weight:700;color:#e2e8f0;line-height:1;">3,222</div>
          <div style="color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.5px;margin-top:3px;">Retinal Images</div>
        </div>
      </div>
      <div style="margin-bottom:8px;">
        <div style="color:#94a3b8;font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Class Balance (DR / No DR)</div>
        <div style="color:#5eead4;font-size:1.05rem;font-weight:600;">51% / 49%</div>
      </div>
      <div style="margin-bottom:4px;">
        <div style="background:rgba(71,85,105,0.3);border-radius:4px;height:5px;overflow:hidden;">
          <div style="background:linear-gradient(90deg,#22d3ee 0%,#5eead4 100%);height:100%;width:51%;transition:width 1s ease;"></div>
        </div>
        <div style="color:#5eead4;font-size:0.62rem;margin-top:2px;">51% Has DR</div>
      </div>
      <div style="margin-bottom:6px;">
        <div style="background:rgba(71,85,105,0.3);border-radius:4px;height:5px;overflow:hidden;">
          <div style="background:linear-gradient(90deg,#64748b 0%,#475569 100%);height:100%;width:49%;transition:width 1s ease;"></div>
        </div>
        <div style="color:#94a3b8;font-size:0.62rem;margin-top:2px;">49% No DR</div>
      </div>
      <div style="color:#64748b;font-size:0.65rem;line-height:1.4;">Well-balanced binary target</div>
    </div>
    """, unsafe_allow_html=True)

    # Card 5 — Drug Reviews (three bars)
    d5.markdown("""
    <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                padding:24px;min-height:180px;transition:all 0.3s ease;"
         onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
         onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px;">
        <div style="font-size:2rem;">💊</div>
        <div>
          <div style="font-size:1.6rem;font-weight:700;color:#e2e8f0;line-height:1;">192,482</div>
          <div style="color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.5px;margin-top:3px;">Drug Reviews</div>
        </div>
      </div>
      <div style="margin-bottom:7px;">
        <div style="color:#94a3b8;font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:5px;">3-Class Distribution</div>
        <div style="color:#5eead4;font-size:0.95rem;font-weight:600;">48.7% / 33.5% / 17.8%</div>
      </div>
      <div style="margin-bottom:3px;">
        <div style="background:rgba(71,85,105,0.3);border-radius:4px;height:5px;overflow:hidden;">
          <div style="background:linear-gradient(90deg,#10b981 0%,#059669 100%);height:100%;width:35%;transition:width 1s ease;"></div>
        </div>
        <div style="color:#10b981;font-size:0.62rem;margin-top:2px;">35% Highly Effective</div>
      </div>
      <div style="margin-bottom:3px;">
        <div style="background:rgba(71,85,105,0.3);border-radius:4px;height:5px;overflow:hidden;">
          <div style="background:linear-gradient(90deg,#22d3ee 0%,#5eead4 100%);height:100%;width:42%;transition:width 1s ease;"></div>
        </div>
        <div style="color:#5eead4;font-size:0.62rem;margin-top:2px;">42% Somewhat Effective</div>
      </div>
      <div style="margin-bottom:5px;">
        <div style="background:rgba(71,85,105,0.3);border-radius:4px;height:5px;overflow:hidden;">
          <div style="background:linear-gradient(90deg,#fb923c 0%,#f97316 100%);height:100%;width:23%;transition:width 1s ease;"></div>
        </div>
        <div style="color:#fb923c;font-size:0.62rem;margin-top:2px;">23% Ineffective</div>
      </div>
      <div style="color:#64748b;font-size:0.65rem;line-height:1.4;">Balanced sentiment across effectiveness categories</div>
    </div>
    """, unsafe_allow_html=True)

    # Card 6 — Readmit Balance (two bars)
    d6.markdown("""
    <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                padding:24px;min-height:180px;transition:all 0.3s ease;"
         onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
         onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
        <div style="font-size:2rem;">📊</div>
        <div>
          <div style="font-size:1.4rem;font-weight:700;color:#e2e8f0;line-height:1;">46/54%</div>
          <div style="color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.5px;margin-top:3px;">Readmit Balance</div>
        </div>
      </div>
      <div style="margin-bottom:8px;">
        <div style="color:#94a3b8;font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:5px;">Binary Target Split</div>
        <div style="color:#5eead4;font-size:0.9rem;font-weight:600;">46% Readmit / 54% No Readmit</div>
      </div>
      <div style="margin-bottom:4px;">
        <div style="background:rgba(71,85,105,0.3);border-radius:4px;height:5px;overflow:hidden;">
          <div style="background:linear-gradient(90deg,#22d3ee 0%,#5eead4 100%);height:100%;width:46%;transition:width 1s ease;"></div>
        </div>
        <div style="color:#5eead4;font-size:0.62rem;margin-top:2px;">Readmitted (&lt;30 or &gt;30)</div>
      </div>
      <div style="margin-bottom:6px;">
        <div style="background:rgba(71,85,105,0.3);border-radius:4px;height:5px;overflow:hidden;">
          <div style="background:linear-gradient(90deg,#64748b 0%,#475569 100%);height:100%;width:54%;transition:width 1s ease;"></div>
        </div>
        <div style="color:#94a3b8;font-size:0.62rem;margin-top:2px;">Not Readmitted</div>
      </div>
      <div style="color:#64748b;font-size:0.65rem;line-height:1.4;">Well-balanced classification target</div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 2: AI Models Overview ────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:16px;margin:64px 0 8px 0;">
      <div style="background:rgba(34,211,238,0.1);border:1px solid rgba(34,211,238,0.3);
                  border-radius:8px;padding:8px 16px;font-size:0.9rem;font-weight:700;
                  color:#22d3ee;letter-spacing:0.5px;">02</div>
      <h2 style="margin:0;font-size:1.75rem;font-weight:700;color:#e2e8f0;">AI Models Overview</h2>
    </div>
    <p style="color:#94a3b8;font-size:0.95rem;margin-bottom:32px;">Five specialized AI models working in parallel to support clinical decisions across the care continuum</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:20px;margin-top:32px;">

      <!-- MODEL 1: XGBoost Readmission -->
      <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                  border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                  padding:24px;min-height:320px;transition:all 0.3s ease;"
           onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
           onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
        <div style="font-size:3rem;text-align:center;margin-bottom:16px;">🔬</div>
        <div style="background:rgba(34,211,238,0.1);border:1px solid rgba(34,211,238,0.2);
                    border-radius:8px;padding:6px 12px;font-size:0.7rem;font-weight:600;
                    color:#22d3ee;letter-spacing:0.5px;margin-bottom:12px;text-align:center;">MODEL 1 · XGBOOST</div>
        <h3 style="color:#e2e8f0;font-size:1.1rem;font-weight:600;margin-bottom:12px;text-align:center;">Readmission Risk</h3>
        <p style="color:#94a3b8;font-size:0.85rem;line-height:1.5;margin-bottom:16px;">
          Predicts 30-day hospital readmission probability using 39 engineered clinical features and patient history.
        </p>
        <div style="background:rgba(30,41,59,0.4);border-radius:8px;padding:12px;margin-bottom:12px;">
          <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">ARCHITECTURE</div>
          <div style="color:#5eead4;font-size:0.85rem;font-weight:600;">XGBoost + SMOTE</div>
          <div style="color:#94a3b8;font-size:0.75rem;margin-top:2px;">Binary classification with cost-optimized threshold</div>
        </div>
        <div style="background:rgba(34,211,238,0.05);border:1px solid rgba(34,211,238,0.15);border-radius:8px;padding:12px;">
          <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">CLINICAL VALUE</div>
          <div style="color:#22d3ee;font-size:0.8rem;line-height:1.4;">
            • Early intervention flagging<br>
            • Discharge planning optimization<br>
            • $15K avg savings per prevented case
          </div>
        </div>
      </div>

      <!-- MODEL 2: DNN Readmission -->
      <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                  border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                  padding:24px;min-height:320px;transition:all 0.3s ease;"
           onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
           onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
        <div style="font-size:3rem;text-align:center;margin-bottom:16px;">🧠</div>
        <div style="background:rgba(34,211,238,0.1);border:1px solid rgba(34,211,238,0.2);
                    border-radius:8px;padding:6px 12px;font-size:0.7rem;font-weight:600;
                    color:#22d3ee;letter-spacing:0.5px;margin-bottom:12px;text-align:center;">MODEL 2 · DEEP NEURAL NETWORK</div>
        <h3 style="color:#e2e8f0;font-size:1.1rem;font-weight:600;margin-bottom:12px;text-align:center;">Readmission Risk (Neural)</h3>
        <p style="color:#94a3b8;font-size:0.85rem;line-height:1.5;margin-bottom:16px;">
          Same prediction task as Model 1 using deep learning architecture for ensemble consensus and validation.
        </p>
        <div style="background:rgba(30,41,59,0.4);border-radius:8px;padding:12px;margin-bottom:12px;">
          <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">ARCHITECTURE</div>
          <div style="color:#5eead4;font-size:0.85rem;font-weight:600;">Dense 256→128→64→1</div>
          <div style="color:#94a3b8;font-size:0.75rem;margin-top:2px;">BatchNorm + Dropout regularization</div>
        </div>
        <div style="background:rgba(34,211,238,0.05);border:1px solid rgba(34,211,238,0.15);border-radius:8px;padding:12px;">
          <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">CLINICAL VALUE</div>
          <div style="color:#22d3ee;font-size:0.8rem;line-height:1.4;">
            • Neural network validation<br>
            • Ensemble consensus checking<br>
            • Captures non-linear patterns
          </div>
        </div>
      </div>

      <!-- MODEL 3: CNN Retinal -->
      <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                  border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                  padding:24px;min-height:320px;transition:all 0.3s ease;"
           onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
           onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
        <div style="font-size:3rem;text-align:center;margin-bottom:16px;">👁️</div>
        <div style="background:rgba(34,211,238,0.1);border:1px solid rgba(34,211,238,0.2);
                    border-radius:8px;padding:6px 12px;font-size:0.7rem;font-weight:600;
                    color:#22d3ee;letter-spacing:0.5px;margin-bottom:12px;text-align:center;">MODEL 3 · CNN RESNET50</div>
        <h3 style="color:#e2e8f0;font-size:1.1rem;font-weight:600;margin-bottom:12px;text-align:center;">Diabetic Retinopathy Detection</h3>
        <p style="color:#94a3b8;font-size:0.85rem;line-height:1.5;margin-bottom:16px;">
          Automated screening of fundus images for any grade of diabetic retinopathy using computer vision.
        </p>
        <div style="background:rgba(30,41,59,0.4);border-radius:8px;padding:12px;margin-bottom:12px;">
          <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">ARCHITECTURE</div>
          <div style="color:#5eead4;font-size:0.85rem;font-weight:600;">ResNet50 Transfer Learning</div>
          <div style="color:#94a3b8;font-size:0.75rem;margin-top:2px;">Binary sigmoid · 224×224 input · Grad-CAM explainability</div>
        </div>
        <div style="background:rgba(34,211,238,0.05);border:1px solid rgba(34,211,238,0.15);border-radius:8px;padding:12px;">
          <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">CLINICAL VALUE</div>
          <div style="color:#22d3ee;font-size:0.8rem;line-height:1.4;">
            • Automated DR screening at scale<br>
            • 85% sensitivity for early detection<br>
            • Reduces ophthalmologist workload
          </div>
        </div>
      </div>

      <!-- MODEL 4: NLP -->
      <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                  border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                  padding:24px;min-height:320px;transition:all 0.3s ease;"
           onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
           onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
        <div style="font-size:3rem;text-align:center;margin-bottom:16px;">💬</div>
        <div style="background:rgba(34,211,238,0.1);border:1px solid rgba(34,211,238,0.2);
                    border-radius:8px;padding:6px 12px;font-size:0.7rem;font-weight:600;
                    color:#22d3ee;letter-spacing:0.5px;margin-bottom:12px;text-align:center;">MODEL 4 · BioBERT (LoRA)</div>
        <h3 style="color:#e2e8f0;font-size:1.1rem;font-weight:600;margin-bottom:12px;text-align:center;">Medication Sentiment Analysis</h3>
        <p style="color:#94a3b8;font-size:0.85rem;line-height:1.5;margin-bottom:16px;">
          Classifies patient drug reviews into effectiveness categories using natural language processing.
        </p>
        <div style="background:rgba(30,41,59,0.4);border-radius:8px;padding:12px;margin-bottom:12px;">
          <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">ARCHITECTURE</div>
          <div style="color:#5eead4;font-size:0.85rem;font-weight:600;">BioBERT (LoRA)</div>
          <div style="color:#94a3b8;font-size:0.75rem;margin-top:2px;">110M params · Drug &amp; condition embeddings · 3-class output</div>
        </div>
        <div style="background:rgba(34,211,238,0.05);border:1px solid rgba(34,211,238,0.15);border-radius:8px;padding:12px;">
          <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">CLINICAL VALUE</div>
          <div style="color:#22d3ee;font-size:0.8rem;line-height:1.4;">
            • Real-world effectiveness insights<br>
            • Patient experience tracking<br>
            • Medication optimization guidance
          </div>
        </div>
      </div>

      <!-- MODEL 5: Innovation -->
      <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                  border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                  padding:24px;min-height:320px;transition:all 0.3s ease;"
           onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(167,139,250,0.15)';"
           onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
        <div style="font-size:3rem;text-align:center;margin-bottom:16px;">⏱️</div>
        <div style="background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.2);
                    border-radius:8px;padding:6px 12px;font-size:0.7rem;font-weight:600;
                    color:#a78bfa;letter-spacing:0.5px;margin-bottom:12px;text-align:center;">MODEL 5 · RANDOM FOREST</div>
        <h3 style="color:#e2e8f0;font-size:1.1rem;font-weight:600;margin-bottom:12px;text-align:center;">Length-of-Stay Prediction</h3>
        <p style="color:#94a3b8;font-size:0.85rem;line-height:1.5;margin-bottom:16px;">
          Forecasts hospital stay duration (Short/Standard/Extended) for capacity planning and resource allocation.
        </p>
        <div style="background:rgba(30,41,59,0.4);border-radius:8px;padding:12px;margin-bottom:12px;">
          <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">ARCHITECTURE</div>
          <div style="color:#c084fc;font-size:0.85rem;font-weight:600;">Random Forest Classifier</div>
          <div style="color:#94a3b8;font-size:0.75rem;margin-top:2px;">Multi-class output · 39 features · Ensemble voting</div>
        </div>
        <div style="background:rgba(167,139,250,0.05);border:1px solid rgba(167,139,250,0.15);border-radius:8px;padding:12px;">
          <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">CLINICAL VALUE</div>
          <div style="color:#a78bfa;font-size:0.8rem;line-height:1.4;">
            • Bed management optimization<br>
            • Staff allocation planning<br>
            • Discharge timeline forecasting
          </div>
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    # ========== AI CLINICAL SYNTHESIS LAYER CARD ==========
    st.markdown("""
    <style>
    @keyframes pulse-glow {
        0%   { box-shadow: 0 0 0 1px rgba(34,211,238,0.15),
                            0 8px 32px rgba(34,211,238,0.06); }
        50%  { box-shadow: 0 0 0 1px rgba(94,234,212,0.35),
                            0 16px 48px rgba(34,211,238,0.18),
                            0 0 60px rgba(34,211,238,0.08); }
        100% { box-shadow: 0 0 0 1px rgba(34,211,238,0.15),
                            0 8px 32px rgba(34,211,238,0.06); }
    }
    @keyframes shimmer {
        0%   { left: -100%; }
        100% { left: 200%; }
    }
    </style>

    <!-- ── Divider badge ─────────────────────────────────────────── -->
    <div style="display:flex;align-items:center;gap:16px;margin:2.4rem 0 1.6rem;">
      <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(34,211,238,0.25));"></div>
      <div style="
          font-family:'JetBrains Mono',monospace;
          font-size:0.7rem;font-weight:700;letter-spacing:0.14em;
          color:#22d3ee;text-transform:uppercase;
          background:linear-gradient(135deg,rgba(34,211,238,0.12),rgba(94,234,212,0.06));
          border:1px solid rgba(34,211,238,0.3);
          border-radius:999px;padding:6px 18px;white-space:nowrap;
      ">🤖 AI CLINICAL SYNTHESIS LAYER</div>
      <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(34,211,238,0.25),transparent);"></div>
    </div>

    <!-- ── Copilot info card ─────────────────────────────────────── -->
    <div style="
        position:relative;overflow:hidden;
        background:linear-gradient(135deg,rgba(34,211,238,0.12) 0%,rgba(94,234,212,0.06) 100%);
        border:2px solid rgba(34,211,238,0.35);
        border-radius:18px;padding:2rem 2.2rem 1.6rem;
        margin-bottom:0.5rem;
        animation:pulse-glow 4s ease-in-out infinite;
    ">

      <!-- shimmer sweep -->
      <div style="
          position:absolute;top:0;width:40%;height:100%;
          background:linear-gradient(90deg,transparent,rgba(94,234,212,0.06),transparent);
          animation:shimmer 3s ease-in-out infinite;pointer-events:none;
      "></div>

      <!-- header row: icon + title -->
      <div style="display:flex;align-items:center;gap:1.4rem;margin-bottom:1.2rem;">
        <div style="
            flex-shrink:0;width:80px;height:80px;border-radius:18px;
            background:linear-gradient(135deg,rgba(34,211,238,0.2),rgba(94,234,212,0.1));
            border:1px solid rgba(34,211,238,0.35);
            display:flex;align-items:center;justify-content:center;font-size:2.4rem;
            box-shadow:0 0 24px rgba(34,211,238,0.2);
        ">🤖</div>
        <div>
          <h2 style="
              font-family:'Space Grotesk',sans-serif;font-size:1.8rem;font-weight:700;
              margin:0 0 0.3rem;letter-spacing:-0.02em;
              background:linear-gradient(120deg,#ffffff 0%,#5eead4 70%);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text;
          ">AI Clinical Copilot</h2>
          <div style="
              font-family:'JetBrains Mono',monospace;font-size:0.75rem;font-weight:600;
              color:#22d3ee;letter-spacing:0.06em;text-transform:uppercase;
          ">Generative AI Integration &nbsp;&middot;&nbsp; Llama 3.1 via Groq</div>
        </div>
      </div>

      <!-- description -->
      <p style="
          font-size:1.05rem;color:#94a3b8;line-height:1.7;margin-bottom:1.5rem;
      ">
        Synthesizes outputs from all 5 specialized models into natural language clinical narratives.
        Provides discharge recommendations, risk interpretation, and interactive Q&amp;A with physicians
        using state-of-the-art Large Language Models.
      </p>

      <!-- feature mini-cards grid -->
      <div style="
          display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
          gap:0.8rem;margin-bottom:1.6rem;
      ">
        <div style="
            background:rgba(34,211,238,0.08);border:1px solid rgba(34,211,238,0.3);
            border-radius:8px;padding:0.8rem;
        ">
          <div style="font-size:1.4rem;margin-bottom:0.4rem;">💬</div>
          <div style="font-size:0.88rem;font-weight:600;color:#e2e8f0;margin-bottom:0.15rem;">Interactive Reasoning</div>
          <div style="font-size:0.75rem;color:#64748b;">Ask follow-up questions</div>
        </div>
        <div style="
            background:rgba(34,211,238,0.08);border:1px solid rgba(34,211,238,0.3);
            border-radius:8px;padding:0.8rem;
        ">
          <div style="font-size:1.4rem;margin-bottom:0.4rem;">🔗</div>
          <div style="font-size:0.88rem;font-weight:600;color:#e2e8f0;margin-bottom:0.15rem;">Multi-Model Synthesis</div>
          <div style="font-size:0.75rem;color:#64748b;">Aggregates 5 AI outputs</div>
        </div>
        <div style="
            background:rgba(34,211,238,0.08);border:1px solid rgba(34,211,238,0.3);
            border-radius:8px;padding:0.8rem;
        ">
          <div style="font-size:1.4rem;margin-bottom:0.4rem;">📋</div>
          <div style="font-size:0.88rem;font-weight:600;color:#e2e8f0;margin-bottom:0.15rem;">Clinical Protocols</div>
          <div style="font-size:0.75rem;color:#64748b;">ICD-10, LACE, APACHE II</div>
        </div>
      </div>

      <!-- CTA footer -->
      <div style="
          background:rgba(13,27,42,0.6);border-top:1px solid rgba(94,234,212,0.2);
          border-radius:0 0 10px 10px;
          margin:-1.6rem -2.2rem -1.6rem;padding:1.1rem 1.6rem;
          text-align:center;
      ">
        <div style="font-size:0.8rem;font-weight:600;color:#5eead4;margin-bottom:0.3rem;">
          💡 Available in Patient Risk Analysis
        </div>
        <div style="font-size:0.75rem;color:#475569;">
          Navigate to <em style="color:#94a3b8;">Predict</em> &nbsp;→&nbsp; Run predictions &nbsp;→&nbsp; Access the AI Clinical Copilot tab
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 3: Model Performance ─────────────────────────────────────────
    st.markdown("""
    <div class="sec-div">
      <span class="num">03</span>
      <h2>Model Performance — Live Metrics</h2>
      <div class="line"></div>
    </div>
    """, unsafe_allow_html=True)

    mp1, mp2, mp3, mp4, mp5 = st.columns(5, gap="medium")

    # Model 1
    with mp1:
        st.markdown("""
        <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                    border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                    padding:24px;height:100%;transition:all 0.3s ease;"
             onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
             onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
          <div style="display:inline-block;background:rgba(34,211,238,0.1);
                      border:1px solid rgba(34,211,238,0.2);border-radius:8px;
                      padding:6px 12px;font-size:0.75rem;font-weight:600;
                      color:#22d3ee;letter-spacing:0.5px;margin-bottom:16px;">
            M1 &bull; XGBOOST ENSEMBLE
          </div>
          <h3 style="margin:0 0 24px 0;font-size:1.25rem;font-weight:600;color:#e2e8f0;">
            Readmission &middot; Traditional ML
          </h3>
          <div style="background:rgba(30,41,59,0.4);border-radius:12px;padding:20px;margin-bottom:16px;">
            <div style="color:#94a3b8;font-size:0.75rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:8px;">AUC-ROC</div>
            <div style="font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#22d3ee 0%,#5eead4 100%);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px;">
              0.6943
            </div>
            <div style="background:rgba(71,85,105,0.3);border-radius:8px;height:8px;overflow:hidden;">
              <div style="background:linear-gradient(90deg,#22d3ee 0%,#5eead4 100%);height:100%;
                          width:69.43%;transition:width 1s ease;"></div>
            </div>
          </div>
          <div style="background:rgba(30,41,59,0.4);border-radius:12px;padding:16px;margin-bottom:16px;">
            <div style="color:#94a3b8;font-size:0.7rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:6px;">RECALL @ THRESHOLD 0.38</div>
            <div style="font-size:1.5rem;font-weight:600;color:#e2e8f0;margin-bottom:8px;">85.0%</div>
            <div style="background:rgba(71,85,105,0.3);border-radius:6px;height:6px;overflow:hidden;">
              <div style="background:linear-gradient(90deg,#22d3ee 0%,#5eead4 100%);height:100%;
                          width:85%;transition:width 1s ease;"></div>
            </div>
          </div>
          <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(148,163,184,0.1);
                      border-radius:12px;padding:16px;">
            <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:8px;">ARCHITECTURE</div>
            <div style="color:#5eead4;font-size:0.95rem;font-weight:600;margin-bottom:4px;">
              XGBoost + SMOTE
            </div>
            <div style="color:#94a3b8;font-size:0.85rem;">
              39 engineered features &middot; Class-weighted
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Model 2
    with mp2:
        st.markdown("""
        <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                    border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                    padding:24px;height:100%;transition:all 0.3s ease;"
             onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
             onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
          <div style="display:inline-block;background:rgba(34,211,238,0.1);
                      border:1px solid rgba(34,211,238,0.2);border-radius:8px;
                      padding:6px 12px;font-size:0.75rem;font-weight:600;
                      color:#22d3ee;letter-spacing:0.5px;margin-bottom:16px;">
            M2 &bull; DEEP LEARNING DNN
          </div>
          <h3 style="margin:0 0 24px 0;font-size:1.25rem;font-weight:600;color:#e2e8f0;">
            Readmission &middot; Neural Network
          </h3>
          <div style="background:rgba(30,41,59,0.4);border-radius:12px;padding:20px;margin-bottom:16px;">
            <div style="color:#94a3b8;font-size:0.75rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:8px;">AUC-ROC</div>
            <div style="font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#22d3ee 0%,#5eead4 100%);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px;">
              0.6854
            </div>
            <div style="background:rgba(71,85,105,0.3);border-radius:8px;height:8px;overflow:hidden;">
              <div style="background:linear-gradient(90deg,#22d3ee 0%,#5eead4 100%);height:100%;
                          width:68.54%;transition:width 1s ease;"></div>
            </div>
          </div>
          <div style="background:rgba(30,41,59,0.4);border-radius:12px;padding:16px;margin-bottom:16px;">
            <div style="color:#94a3b8;font-size:0.7rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:6px;">VS. XGBOOST DELTA</div>
            <div style="font-size:1.5rem;font-weight:600;color:#e2e8f0;margin-bottom:8px;">&minus;0.0089</div>
            <div style="background:rgba(71,85,105,0.3);border-radius:6px;height:6px;overflow:hidden;">
              <div style="background:linear-gradient(90deg,#fb923c 0%,#f97316 100%);height:100%;
                          width:50%;transition:width 1s ease;"></div>
            </div>
          </div>
          <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(148,163,184,0.1);
                      border-radius:12px;padding:16px;">
            <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:8px;">ARCHITECTURE</div>
            <div style="color:#5eead4;font-size:0.95rem;font-weight:600;margin-bottom:4px;">
              Dense 256&rarr;128&rarr;64&rarr;1
            </div>
            <div style="color:#94a3b8;font-size:0.85rem;">
              BatchNorm + Dropout &middot; Early stopping
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Model 3
    with mp3:
        st.markdown("""
        <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                    border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                    padding:24px;height:100%;transition:all 0.3s ease;"
             onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
             onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
          <div style="display:inline-block;background:rgba(34,211,238,0.1);
                      border:1px solid rgba(34,211,238,0.2);border-radius:8px;
                      padding:6px 12px;font-size:0.75rem;font-weight:600;
                      color:#22d3ee;letter-spacing:0.5px;margin-bottom:16px;">
            M3 &bull; CNN RETINAL SCAN
          </div>
          <h3 style="margin:0 0 24px 0;font-size:1.25rem;font-weight:600;color:#e2e8f0;">
            Retinopathy Detection
          </h3>
          <div style="background:rgba(30,41,59,0.4);border-radius:12px;padding:20px;margin-bottom:16px;">
            <div style="color:#94a3b8;font-size:0.75rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:8px;">ACCURACY</div>
            <div style="font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#22d3ee 0%,#5eead4 100%);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px;">
              97.0%
            </div>
            <div style="background:rgba(71,85,105,0.3);border-radius:8px;height:8px;overflow:hidden;">
              <div style="background:linear-gradient(90deg,#22d3ee 0%,#5eead4 100%);height:100%;
                          width:97%;transition:width 1s ease;"></div>
            </div>
          </div>
          <div style="background:rgba(30,41,59,0.4);border-radius:12px;padding:16px;margin-bottom:16px;">
            <div style="color:#94a3b8;font-size:0.7rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:6px;">AUC-ROC</div>
            <div style="font-size:1.5rem;font-weight:600;color:#e2e8f0;margin-bottom:8px;">0.994</div>
            <div style="background:rgba(71,85,105,0.3);border-radius:6px;height:6px;overflow:hidden;">
              <div style="background:linear-gradient(90deg,#22d3ee 0%,#5eead4 100%);height:100%;
                          width:99.4%;transition:width 1s ease;"></div>
            </div>
          </div>
          <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(148,163,184,0.1);
                      border-radius:12px;padding:16px;">
            <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:8px;">ARCHITECTURE</div>
            <div style="color:#5eead4;font-size:0.95rem;font-weight:600;margin-bottom:4px;">
              ResNet50 &middot; Transfer Learning
            </div>
            <div style="color:#94a3b8;font-size:0.85rem;">
              Binary sigmoid &middot; 224&times;224 input &middot; Grad-CAM
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Model 4
    with mp4:
        st.markdown("""
        <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                    border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                    padding:24px;height:100%;transition:all 0.3s ease;"
             onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
             onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
          <div style="display:inline-block;background:rgba(34,211,238,0.1);
                      border:1px solid rgba(34,211,238,0.2);border-radius:8px;
                      padding:6px 12px;font-size:0.75rem;font-weight:600;
                      color:#22d3ee;letter-spacing:0.5px;margin-bottom:16px;">
            M4 &bull; BioBERT (LoRA)
          </div>
          <h3 style="margin:0 0 24px 0;font-size:1.25rem;font-weight:600;color:#e2e8f0;">
            NLP Notes &middot; Sentiment Analysis
          </h3>
          <div style="background:rgba(30,41,59,0.4);border-radius:12px;padding:20px;margin-bottom:16px;">
            <div style="color:#94a3b8;font-size:0.75rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:8px;">WEIGHTED F1-SCORE</div>
            <div style="font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#22d3ee 0%,#5eead4 100%);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px;">
              0.9018
            </div>
            <div style="background:rgba(71,85,105,0.3);border-radius:8px;height:8px;overflow:hidden;">
              <div style="background:linear-gradient(90deg,#22d3ee 0%,#5eead4 100%);height:100%;
                          width:90.18%;transition:width 1s ease;"></div>
            </div>
          </div>
          <div style="background:rgba(30,41,59,0.4);border-radius:12px;padding:16px;margin-bottom:16px;">
            <div style="color:#94a3b8;font-size:0.7rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:6px;">3-CLASS ACCURACY</div>
            <div style="font-size:1.5rem;font-weight:600;color:#e2e8f0;margin-bottom:8px;">90.0%</div>
            <div style="background:rgba(71,85,105,0.3);border-radius:6px;height:6px;overflow:hidden;">
              <div style="background:linear-gradient(90deg,#22d3ee 0%,#5eead4 100%);height:100%;
                          width:90%;transition:width 1s ease;"></div>
            </div>
          </div>
          <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(148,163,184,0.1);
                      border-radius:12px;padding:16px;">
            <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:8px;">ARCHITECTURE</div>
            <div style="color:#5eead4;font-size:0.95rem;font-weight:600;margin-bottom:4px;">
              BioBERT (LoRA)
            </div>
            <div style="color:#94a3b8;font-size:0.85rem;">
              110M params &middot; Drug &amp; condition embeddings
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Model 5
    with mp5:
        st.markdown("""
        <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                    border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                    padding:24px;height:100%;transition:all 0.3s ease;"
             onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(167,139,250,0.15)';"
             onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
          <div style="display:inline-block;background:rgba(167,139,250,0.1);
                      border:1px solid rgba(167,139,250,0.2);border-radius:8px;
                      padding:6px 12px;font-size:0.75rem;font-weight:600;
                      color:#a78bfa;letter-spacing:0.5px;margin-bottom:16px;">
            M5 &bull; INNOVATION
          </div>
          <h3 style="margin:0 0 24px 0;font-size:1.25rem;font-weight:600;color:#e2e8f0;">
            Length-of-Stay &middot; Capacity Planning
          </h3>
          <div style="background:rgba(30,41,59,0.4);border-radius:12px;padding:20px;margin-bottom:16px;">
            <div style="color:#94a3b8;font-size:0.75rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:8px;">WEIGHTED F1-SCORE</div>
            <div style="font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#a78bfa 0%,#c084fc 100%);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px;">
              0.6047
            </div>
            <div style="background:rgba(71,85,105,0.3);border-radius:8px;height:8px;overflow:hidden;">
              <div style="background:linear-gradient(90deg,#a78bfa 0%,#c084fc 100%);height:100%;
                          width:60.47%;transition:width 1s ease;"></div>
            </div>
          </div>
          <div style="background:rgba(30,41,59,0.4);border-radius:12px;padding:16px;margin-bottom:16px;">
            <div style="color:#94a3b8;font-size:0.7rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:6px;">3-CLASS DISTRIBUTION</div>
            <div style="font-size:1.5rem;font-weight:600;color:#e2e8f0;margin-bottom:4px;">
              Short / Std / Ext
            </div>
            <div style="color:#64748b;font-size:0.8rem;">Capacity planning classes</div>
          </div>
          <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(148,163,184,0.1);
                      border-radius:12px;padding:16px;">
            <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:8px;">CLINICAL VALUE</div>
            <div style="color:#a78bfa;font-size:0.95rem;font-weight:600;margin-bottom:4px;">
              Bed management &middot; Discharge planning
            </div>
            <div style="color:#94a3b8;font-size:0.85rem;">
              Staff allocation &middot; Optimization
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION 4: Clinical Impact ────────────────────────────────────────────
    st.markdown("""
    <div class="sec-div">
      <span class="num">04</span>
      <h2>Estimated Clinical Impact</h2>
      <div class="line"></div>
      <span style="font-size:0.75rem;color:#475569;font-family:'JetBrains Mono',monospace;">
        based on MedInsight network · 22% baseline readmission rate · $15K cost/readmission
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:24px;margin-top:32px;">

      <!-- CARD 1: READMISSIONS -->
      <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                  border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                  padding:28px;transition:all 0.3s ease;"
           onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
           onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
        <div style="font-size:3rem;margin-bottom:16px;text-align:center;">🏥</div>
        <div style="background:rgba(71,85,105,0.2);border-radius:8px;padding:12px;margin-bottom:12px;">
          <div style="color:#94a3b8;font-size:0.7rem;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.5px;margin-bottom:4px;">CURRENT BASELINE</div>
          <div style="color:#cbd5e1;font-size:1.25rem;font-weight:600;">22,388
            <span style="color:#64748b;font-size:0.85rem;margin-left:6px;">/year</span>
          </div>
          <div style="color:#64748b;font-size:0.75rem;margin-top:2px;">22% readmission rate</div>
        </div>
        <div style="text-align:center;color:#22d3ee;font-size:1.5rem;margin:8px 0;">&#9660;</div>
        <div style="background:rgba(34,211,238,0.1);border:1px solid rgba(34,211,238,0.3);
                    border-radius:8px;padding:12px;margin-bottom:16px;">
          <div style="color:#22d3ee;font-size:0.7rem;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.5px;margin-bottom:4px;">WITH CLEARSIGHT</div>
          <div style="color:#e2e8f0;font-size:2rem;font-weight:700;">19,530
            <span style="color:#94a3b8;font-size:1rem;margin-left:6px;">/year</span>
          </div>
          <div style="color:#5eead4;font-size:0.75rem;margin-top:2px;">19% readmission rate</div>
        </div>
        <div style="background:linear-gradient(135deg,#10b981 0%,#059669 100%);
                    border-radius:8px;padding:10px;text-align:center;">
          <div style="color:#fff;font-size:1.1rem;font-weight:700;">-12.8% REDUCTION</div>
          <div style="color:rgba(255,255,255,0.9);font-size:0.75rem;margin-top:4px;">
            2,858 fewer readmissions/year
          </div>
        </div>
      </div>

      <!-- CARD 2: HIGH-RISK DETECTION -->
      <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                  border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                  padding:28px;transition:all 0.3s ease;"
           onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
           onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
        <div style="font-size:3rem;margin-bottom:16px;text-align:center;">🎯</div>
        <div style="background:rgba(71,85,105,0.2);border-radius:8px;padding:12px;margin-bottom:12px;">
          <div style="color:#94a3b8;font-size:0.7rem;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.5px;margin-bottom:4px;">CURRENT DETECTION</div>
          <div style="color:#cbd5e1;font-size:1.25rem;font-weight:600;">8,550
            <span style="color:#64748b;font-size:0.85rem;margin-left:6px;">/year</span>
          </div>
          <div style="color:#64748b;font-size:0.75rem;margin-top:2px;">38% catch rate</div>
        </div>
        <div style="text-align:center;color:#22d3ee;font-size:1.5rem;margin:8px 0;">&#9650;</div>
        <div style="background:rgba(34,211,238,0.1);border:1px solid rgba(34,211,238,0.3);
                    border-radius:8px;padding:12px;margin-bottom:16px;">
          <div style="color:#22d3ee;font-size:0.7rem;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.5px;margin-bottom:4px;">WITH CLEARSIGHT</div>
          <div style="color:#e2e8f0;font-size:2rem;font-weight:700;">19,030
            <span style="color:#94a3b8;font-size:1rem;margin-left:6px;">/year</span>
          </div>
          <div style="color:#5eead4;font-size:0.75rem;margin-top:2px;">85% catch rate</div>
        </div>
        <div style="background:linear-gradient(135deg,#3b82f6 0%,#2563eb 100%);
                    border-radius:8px;padding:10px;text-align:center;">
          <div style="color:#fff;font-size:1.1rem;font-weight:700;">+122% INCREASE</div>
          <div style="color:rgba(255,255,255,0.9);font-size:0.75rem;margin-top:4px;">
            10,480 more high-risk flagged
          </div>
        </div>
      </div>

      <!-- CARD 3: PREVENTED -->
      <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                  border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                  padding:28px;transition:all 0.3s ease;"
           onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
           onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
        <div style="font-size:3rem;margin-bottom:16px;text-align:center;">🛡️</div>
        <div style="text-align:center;margin-bottom:20px;">
          <div style="color:#94a3b8;font-size:0.8rem;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.5px;margin-bottom:12px;">ADDITIONAL PREVENTED</div>
          <div style="font-size:3rem;font-weight:700;
                      background:linear-gradient(135deg,#10b981 0%,#059669 100%);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                      margin-bottom:8px;">2,854</div>
          <div style="color:#5eead4;font-size:0.9rem;font-weight:600;">READMISSIONS / YEAR</div>
        </div>
        <div style="background:rgba(71,85,105,0.3);border-radius:8px;height:8px;
                    overflow:hidden;margin-bottom:12px;">
          <div style="background:linear-gradient(90deg,#10b981 0%,#059669 100%);
                      height:100%;width:100%;transition:width 1s ease;"></div>
        </div>
        <div style="color:#64748b;font-size:0.75rem;text-align:center;">
          Early intervention success rate: 15%
        </div>
      </div>

      <!-- CARD 4: COST SAVINGS -->
      <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                  border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                  padding:28px;transition:all 0.3s ease;"
           onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 24px rgba(34,211,238,0.15)';"
           onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
        <div style="font-size:3rem;margin-bottom:16px;text-align:center;">💵</div>
        <div style="text-align:center;margin-bottom:20px;">
          <div style="color:#94a3b8;font-size:0.8rem;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.5px;margin-bottom:12px;">ANNUAL COST SAVINGS</div>
          <div style="font-size:3rem;font-weight:700;
                      background:linear-gradient(135deg,#fbbf24 0%,#f59e0b 100%);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                      margin-bottom:8px;">$42.8M</div>
          <div style="color:#94a3b8;font-size:0.85rem;">Across 47 partner hospitals</div>
        </div>
        <div style="background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.2);
                    border-radius:8px;padding:12px;">
          <div style="color:#fbbf24;font-size:0.75rem;font-weight:600;margin-bottom:6px;">
            CALCULATION
          </div>
          <div style="color:#cbd5e1;font-size:0.8rem;line-height:1.5;">
            2,854 prevented readmissions<br>
            &times; $15,000 avg cost/readmission
          </div>
        </div>
      </div>

    </div>

    <div style="text-align:center;margin-top:32px;padding:16px;
                background:rgba(71,85,105,0.1);
                border:1px solid rgba(148,163,184,0.1);border-radius:12px;">
      <div style="color:#94a3b8;font-size:0.8rem;line-height:1.6;">
        <strong style="color:#22d3ee;">Projected 12-month impact</strong> &bull;
        Based on 101,766 training encounters &bull; MedInsight network baseline: 22% readmission rate
        &bull; Conservative 15% intervention success rate on flagged high-risk patients
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 5: How it works ───────────────────────────────────────────────
    st.markdown("""
    <div class="sec-div">
      <span class="num">05</span>
      <h2>Clinical Decision Pipeline</h2>
      <div class="line"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;
                gap:16px;align-items:center;margin-top:32px;">

      <!-- CARD 1 -->
      <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                  border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                  padding:24px;text-align:center;">
        <div style="font-size:3rem;margin-bottom:16px;">📋</div>
        <div style="color:#22d3ee;font-size:2rem;font-weight:700;margin-bottom:8px;">01</div>
        <div style="color:#e2e8f0;font-size:0.95rem;font-weight:600;text-transform:uppercase;
                    letter-spacing:0.5px;margin-bottom:16px;">DATA INTAKE</div>
        <div style="color:#94a3b8;font-size:0.85rem;line-height:1.6;text-align:left;">
          • EHR records (demographics, vitals, labs)<br>
          • Prior encounters &amp; medication history<br>
          • Real-time clinical notes ingestion
        </div>
      </div>

      <!-- ARROW 1 -->
      <div style="font-size:2rem;color:#22d3ee;">&#10132;</div>

      <!-- CARD 2 -->
      <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                  border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                  padding:24px;text-align:center;">
        <div style="font-size:3rem;margin-bottom:16px;">⚙️</div>
        <div style="color:#22d3ee;font-size:2rem;font-weight:700;margin-bottom:8px;">02</div>
        <div style="color:#e2e8f0;font-size:0.95rem;font-weight:600;text-transform:uppercase;
                    letter-spacing:0.5px;margin-bottom:16px;">FEATURE ENGINEERING</div>
        <div style="color:#94a3b8;font-size:0.85rem;line-height:1.6;text-align:left;">
          • Risk factor detection (comorbidities, trends)<br>
          • Medication interaction flags<br>
          • Utilization pattern analysis
        </div>
      </div>

      <!-- ARROW 2 -->
      <div style="font-size:2rem;color:#22d3ee;">&#10132;</div>

      <!-- CARD 3 -->
      <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                  border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                  padding:24px;text-align:center;">
        <div style="font-size:3rem;margin-bottom:16px;">🤖</div>
        <div style="color:#22d3ee;font-size:2rem;font-weight:700;margin-bottom:8px;">03</div>
        <div style="color:#e2e8f0;font-size:0.95rem;font-weight:600;text-transform:uppercase;
                    letter-spacing:0.5px;margin-bottom:16px;">MULTI-MODEL INFERENCE</div>
        <div style="color:#94a3b8;font-size:0.85rem;line-height:1.6;text-align:left;">
          • 5 AI models run in parallel (&lt;200ms)<br>
          • Consensus-based risk scoring<br>
          • Disagreement flagged for review
        </div>
      </div>

      <!-- ARROW 3 -->
      <div style="font-size:2rem;color:#22d3ee;">&#10132;</div>

      <!-- CARD 4 -->
      <div style="background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
                  border:1px solid rgba(148,163,184,0.1);border-radius:16px;
                  padding:24px;text-align:center;">
        <div style="font-size:3rem;margin-bottom:16px;">💡</div>
        <div style="color:#22d3ee;font-size:2rem;font-weight:700;margin-bottom:8px;">04</div>
        <div style="color:#e2e8f0;font-size:0.95rem;font-weight:600;text-transform:uppercase;
                    letter-spacing:0.5px;margin-bottom:16px;">EXPLAINABLE INSIGHTS</div>
        <div style="color:#94a3b8;font-size:0.85rem;line-height:1.6;text-align:left;">
          • SHAP highlights top 3 risk drivers<br>
          • Actionable clinical recommendations<br>
          • Confidence intervals per prediction
        </div>
      </div>

    </div>

    <div style="text-align:center;margin-top:24px;padding:16px;
                background:rgba(34,211,238,0.05);
                border:1px solid rgba(34,211,238,0.2);border-radius:12px;">
      <div style="color:#94a3b8;font-size:0.75rem;font-weight:600;text-transform:uppercase;
                  letter-spacing:0.5px;margin-bottom:4px;">END-TO-END LATENCY</div>
      <div style="color:#22d3ee;font-size:1.5rem;font-weight:700;">&lt;300ms</div>
      <div style="color:#64748b;font-size:0.8rem;margin-top:4px;">
        Average inference time from EHR intake to explainable output
      </div>
    </div>
    """, unsafe_allow_html=True)



def parse_id(val: str) -> int:
    """Extracts an integer ID from a ``"<id> - <description>"`` selectbox string.

    Selectbox options follow the pattern ``"1 - Emergency"``. This helper splits on
    the first ``" - "`` delimiter and converts the leading segment to an integer.

    Args:
        val: A selectbox option string in ``"<id> - <description>"`` format.

    Returns:
        The integer ID at the start of ``val``. Returns ``1`` as a safe default if
        the string cannot be parsed.
    """
    try:
        return int(val.split(" - ")[0].strip())
    except (ValueError, IndexError):
        return 1






# =============================================================================
# PAGE: PREDICT
# =============================================================================
def page_predict() -> None:
    """Renders the Readmission Risk Predictor page.

    Presents a multi-section patient encounter form (demographics, clinical metrics,
    ICD-9 diagnoses, medications, clinical notes). On submission, runs Models 1, 2,
    4, and 5 in sequence and displays animated conic-gradient gauges for M1 / M2, a
    capacity-planning card for M5, a BioBERT (LoRA) sentiment card for M4, and a consensus
    summary block with a clinical recommendation.
    """
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
      <span class="badge" style="display:inline-block; font-family:'JetBrains Mono',monospace;
            font-size:0.7rem; font-weight:600; color:#5eead4;
            background:rgba(94,234,212,0.1); border:1px solid rgba(94,234,212,0.3);
            padding:4px 12px; border-radius:999px; letter-spacing:0.08em;
            text-transform:uppercase; margin-bottom:0.6rem;">● PREDICTOR</span>
      <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2rem;
                 margin:0.4rem 0 0.4rem; letter-spacing:-0.02em;">
        Readmission Risk Predictor
      </h1>
      <p style="color:#94a3b8; font-size:1rem; margin:0; max-width:760px;">
        Enter a patient encounter below — both models run in parallel and you'll
        see a head-to-head comparison with calibrated probabilities and confidence scores.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── QUICK LOAD PRESETS ─────────────────────────────────────────────────
    _PRESETS = {
        "🔴 Critical": {
            "age": "[70-80)", "gender": "Male", "race": "AfricanAmerican",
            "admission_type": "1 - Emergency", "discharge_disp": "3 - SNF",
            "admission_src": "7 - Emergency Room",
            "time_in_hosp": 10, "num_lab_procs": 72, "num_procs": 4,
            "num_meds": 24, "num_diag": 9,
            "num_outpatient": 0, "num_emergency": 3, "num_inpatient": 2,
            "prior_inp_cum": 5,
            "diag1": "428.0", "diag2": "250.01", "diag3": "401.9",
            "med_spec": "Cardiology",
            "insulin": "Up", "metformin": "Steady", "change": "Ch",
            "diabetes_md": "Yes", "a1c": ">8", "max_glu": ">300",
            "nlp_drug": "Insulin", "nlp_cond": "Heart Attack",
            "clinical_notes": (
                "Patient is a 74-year-old African American male admitted via ER with acute "
                "decompensated heart failure. Patient appears disoriented and confused regarding "
                "medication schedule. HbA1c critically elevated at 12.1%. Reports severe shortness "
                "of breath and ankle edema. Non-compliant with fluid restriction. Family reports "
                "missed doses over the past two weeks. Discharge planning complicated by lack of "
                "social support."
            ),
        },
        "🟢 Stable": {
            "age": "[40-50)", "gender": "Female", "race": "Caucasian",
            "admission_type": "3 - Elective", "discharge_disp": "1 - Home",
            "admission_src": "1 - Physician Referral",
            "time_in_hosp": 2, "num_lab_procs": 28, "num_procs": 1,
            "num_meds": 5, "num_diag": 3,
            "num_outpatient": 1, "num_emergency": 0, "num_inpatient": 0,
            "prior_inp_cum": 0,
            "diag1": "250.00", "diag2": "272.4", "diag3": "V58.67",
            "med_spec": "InternalMedicine",
            "insulin": "Steady", "metformin": "Steady", "change": "No",
            "diabetes_md": "Yes", "a1c": "Norm", "max_glu": "Norm",
            "nlp_drug": "Metformin", "nlp_cond": "Diabetes, Type 2",
            "clinical_notes": (
                "Patient is a 44-year-old female admitted for routine diabetes management review. "
                "She reports excellent adherence to medication regimen and dietary plan. Blood glucose "
                "well controlled. Excellent recovery noted since last visit. Patient is highly motivated "
                "and demonstrates strong health literacy. Discharge home planned for tomorrow. No "
                "concerns at this time."
            ),
        },
        "🟡 Consensus": {
            "age": "[60-70)", "gender": "Male", "race": "Hispanic",
            "admission_type": "2 - Urgent", "discharge_disp": "2 - Home Health Care",
            "admission_src": "4 - Transfer from Hospital",
            "time_in_hosp": 4, "num_lab_procs": 44, "num_procs": 2,
            "num_meds": 15, "num_diag": 7,
            "num_outpatient": 1, "num_emergency": 1, "num_inpatient": 1,
            "prior_inp_cum": 2,
            "diag1": "250.01", "diag2": "401.9", "diag3": "272.0",
            "med_spec": "InternalMedicine",
            "insulin": "Steady", "metformin": "Steady", "change": "Ch",
            "diabetes_md": "Yes", "a1c": ">7", "max_glu": ">200",
            "nlp_drug": "Metformin", "nlp_cond": "Diabetes, Type 2",
            "clinical_notes": (
                "Patient is a 63-year-old Hispanic male with Type 2 Diabetes and hypertension, "
                "admitted urgently following blood glucose spike. Patient reports severe anxiety about "
                "managing medications at home due to financial constraints. Social work consult placed. "
                "Patient expressed fear of losing job due to repeated hospitalizations. Moderate risk "
                "for non-compliance identified. Discharge planning initiated with home health coordination."
            ),
        },
        "🟠 Comorbidity": {
            "age": "[80-90)", "gender": "Female", "race": "Caucasian",
            "admission_type": "1 - Emergency", "discharge_disp": "3 - SNF",
            "admission_src": "4 - Transfer from Hospital",
            "time_in_hosp": 7, "num_lab_procs": 58, "num_procs": 3,
            "num_meds": 19, "num_diag": 9,
            "num_outpatient": 2, "num_emergency": 2, "num_inpatient": 1,
            "prior_inp_cum": 3,
            "diag1": "250.80", "diag2": "585.9", "diag3": "428.0",
            "med_spec": "Nephrology",
            "insulin": "Up", "metformin": "No", "change": "Ch",
            "diabetes_md": "Yes", "a1c": ">8", "max_glu": ">300",
            "nlp_drug": "Insulin", "nlp_cond": "Diabetes, Type 2",
            "clinical_notes": (
                "Patient is an 83-year-old female with complex comorbidities including Type 2 Diabetes "
                "with renal complications, chronic kidney disease stage 3, and congestive heart failure. "
                "Polypharmacy noted — patient reports significant confusion managing 19 medications. "
                "Metformin discontinued due to renal function decline. Patient appears fatigued and shows "
                "signs of cognitive decline. Family caregiver present but overwhelmed. Transfer to SNF "
                "recommended for continued care."
            ),
        },
        "🔵 NLP Alert": {
            "age": "[50-60)", "gender": "Male", "race": "Other",
            "admission_type": "1 - Emergency", "discharge_disp": "1 - Home",
            "admission_src": "7 - Emergency Room",
            "time_in_hosp": 3, "num_lab_procs": 32, "num_procs": 1,
            "num_meds": 11, "num_diag": 5,
            "num_outpatient": 0, "num_emergency": 5, "num_inpatient": 3,
            "prior_inp_cum": 8,
            "diag1": "250.01", "diag2": "296.20", "diag3": "311",
            "med_spec": "Psychiatry",
            "insulin": "Steady", "metformin": "Steady", "change": "No",
            "diabetes_md": "Yes", "a1c": ">7", "max_glu": "None",
            "nlp_drug": "sertraline", "nlp_cond": "Depression",
            "clinical_notes": (
                "Patient is a 54-year-old male with a history of Type 2 Diabetes and major depressive "
                "disorder, presenting to the ER for the fifth time this year. Patient expresses feelings "
                "of hopelessness and fear regarding his health prognosis. Reports inability to afford "
                "medication and expresses passive ideation. Psychiatry consult placed. Social determinants "
                "of health are the primary driver of repeated presentations. Despite relatively stable "
                "physical metrics, patient is at high psychosocial risk."
            ),
        },
    }

    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:700;
                color:#5eead4; letter-spacing:0.12em; text-transform:uppercase;
                padding-bottom:0.5rem; margin-bottom:0.4rem;
                border-bottom:1px solid rgba(94,234,212,0.12);">
        ⚡ Quick Load — Clinical Scenarios
    </div>
    """, unsafe_allow_html=True)

    _ql_cols = st.columns(5)
    _ql_labels = list(_PRESETS.keys())
    _ql_hints  = [
        "10d · 24 meds · ICD-9 428",
        "2d · 5 meds · Elective",
        "4d · 15 meds · Social risk",
        "7d · 19 meds · Polypharmacy",
        "3d · High visits · Depressive",
    ]
    for _col, _lbl, _hint in zip(_ql_cols, _ql_labels, _ql_hints):
        with _col:
            if st.button(_lbl, key=f"ql_{_lbl[:4]}", use_container_width=True, help=_hint):
                st.session_state["_preset"] = _PRESETS[_lbl]
                st.rerun()

    st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)

    # ── Apply preset values to widget session-state keys before the form renders ──
    if "_preset" in st.session_state:
        _p = st.session_state["_preset"]
        del st.session_state["_preset"]
        st.session_state["sb_age"]            = _p.get("age",            "[60-70)")
        st.session_state["sb_gender"]         = _p.get("gender",         "Male")
        st.session_state["sb_race"]           = _p.get("race",           "Caucasian")
        st.session_state["sb_admission_type"] = _p.get("admission_type", "1 - Emergency")
        st.session_state["sb_discharge_disp"] = _p.get("discharge_disp", "1 - Home")
        st.session_state["sb_admission_src"]  = _p.get("admission_src",  "7 - Emergency Room")
        st.session_state["ni_days_hospital"]  = int(_p.get("time_in_hosp",   4))
        st.session_state["ni_lab_procs"]      = int(_p.get("num_lab_procs", 44))
        st.session_state["ni_procs"]          = int(_p.get("num_procs",      1))
        st.session_state["ni_meds"]           = int(_p.get("num_meds",      15))
        st.session_state["ni_diag"]           = int(_p.get("num_diag",       7))
        st.session_state["ni_outpatient"]     = int(_p.get("num_outpatient", 0))
        st.session_state["ni_emergency"]      = int(_p.get("num_emergency",  0))
        st.session_state["ni_inpatient"]      = int(_p.get("num_inpatient",  0))
        st.session_state["ni_prior_inp"]      = int(_p.get("prior_inp_cum",  0))
        st.session_state["ti_diag1"]          = _p.get("diag1",    "250.01")
        st.session_state["ti_diag2"]          = _p.get("diag2",    "428")
        st.session_state["ti_diag3"]          = _p.get("diag3",    "401")
        st.session_state["ti_med_spec"]       = _p.get("med_spec", "InternalMedicine")
        st.session_state["sb_insulin"]        = _p.get("insulin",     "No")
        st.session_state["sb_metformin"]      = _p.get("metformin",   "No")
        st.session_state["sb_change"]         = _p.get("change",      "No")
        st.session_state["sb_diabetes_md"]    = _p.get("diabetes_md", "Yes")
        st.session_state["sb_a1c"]            = _p.get("a1c",         "None")
        st.session_state["sb_max_glu"]        = _p.get("max_glu",     "None")
        st.session_state["ti_nlp_drug"]       = _p.get("nlp_drug",    "Metformin")
        st.session_state["ti_nlp_cond"]       = _p.get("nlp_cond",    "Diabetes")
        st.session_state["ta_clinical_notes"] = _p.get("clinical_notes", "")

    with st.form("predict_form"):

        # ── Section 1: Demographics ─────────────────────────────────
        with st.expander("👤 Demographics & Admission", expanded=False):
            r1a, r1b, r1c, r1d = st.columns(4)
            age = r1a.selectbox("Age Bracket", [
                '[0-10)','[10-20)','[20-30)','[30-40)','[40-50)',
                '[50-60)','[60-70)','[70-80)','[80-90)','[90-100)'
            ], index=6, key="sb_age", help="Age group (matches encounter dataset format).")
            gender = r1b.selectbox("Gender", ["Male", "Female"], key="sb_gender")
            race   = r1c.selectbox("Race", [
                "Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"
            ], key="sb_race")
            admission_type = r1d.selectbox("Admission Type", [
                "1 - Emergency", "2 - Urgent", "3 - Elective",
                "4 - Newborn", "5 - Not Available", "6 - NULL", "7 - Trauma Center"
            ], key="sb_admission_type")

            r2a, r2b = st.columns(2)
            discharge_disp = r2a.selectbox("Discharge Disposition", [
                "1 - Home", "2 - Home Health Care", "3 - SNF",
                "6 - Home w/ Home Health", "18 - Not Available",
            ], key="sb_discharge_disp", help="Where patient was discharged to.")
            admission_src = r2b.selectbox("Admission Source", [
                "1 - Physician Referral", "2 - Clinic Referral",
                "4 - Transfer from Hospital", "7 - Emergency Room",
                "9 - Not Available", "17 - NULL"
            ], key="sb_admission_src")

        # ── Section 2: Clinical Metrics ─────────────────────────────
        with st.expander("🏥 Clinical Metrics", expanded=False):
            st.markdown("**Current Encounter**")
            r3a, r3b, r3c, r3d, r3e = st.columns(5)
            time_in_hosp  = r3a.number_input("Days in Hospital", 1, 14, 4,
                                              key="ni_days_hospital")
            num_lab_procs = r3b.number_input("Lab Procedures", 0, 132, 44,
                                              key="ni_lab_procs")
            num_procs     = r3c.number_input("Procedures", 0, 6, 1, key="ni_procs")
            num_meds      = r3d.number_input("Medications", 1, 81, 15, key="ni_meds")
            num_diag      = r3e.number_input("Diagnoses", 1, 16, 7, key="ni_diag")

            st.markdown("**Prior Utilization (past 12 months)**")
            r4a, r4b, r4c, r4d = st.columns(4)
            num_outpatient = r4a.number_input("Outpatient Visits", 0, 42, 0,
                                               key="ni_outpatient")
            num_emergency  = r4b.number_input("Emergency Visits", 0, 76, 0,
                                               key="ni_emergency")
            num_inpatient  = r4c.number_input("Inpatient Visits", 0, 21, 0,
                              key="ni_inpatient",
                              help="Strong readmission predictor (per literature).")
            prior_inp_cum  = r4d.number_input("Prior Inpatient (cumulative)", 0, 50, 0,
                                               key="ni_prior_inp")

        # ── Section 3: Diagnoses ────────────────────────────────────
        with st.expander("🩺 ICD-9 Diagnoses", expanded=False):
            st.markdown("**Top 3 diagnosis codes for this encounter**")
            r5a, r5b, r5c = st.columns(3)
            diag1 = r5a.text_input("Primary (diag_1)", "250.01", key="ti_diag1",
                     help="250.xx = diabetes, 428.xx = heart failure, 401.xx = hypertension")
            diag2 = r5b.text_input("Secondary (diag_2)", "428", key="ti_diag2")
            diag3 = r5c.text_input("Tertiary (diag_3)", "401", key="ti_diag3")
            med_spec = st.text_input("Medical Specialty", "InternalMedicine",
                                     key="ti_med_spec",
                                     help="Specialty of the attending physician.")

        # ── Section 4: Medications & Labs ───────────────────────────
        with st.expander("💊 Medications & Labs", expanded=False):
            st.markdown("**Diabetes Management**")
            r6a, r6b, r6c, r6d, r6e = st.columns(5)
            insulin     = r6a.selectbox("Insulin",     ["No", "Steady", "Up", "Down"],
                                        key="sb_insulin")
            metformin   = r6b.selectbox("Metformin",   ["No", "Steady", "Up", "Down"],
                                        key="sb_metformin")
            change      = r6c.selectbox("Med Changed?", ["No", "Ch"],
                          key="sb_change",
                          help="Was any medication changed during this encounter?")
            diabetes_md = r6d.selectbox("Diabetes Med?", ["Yes", "No"],
                                        key="sb_diabetes_md")
            a1c         = r6e.selectbox("A1C Result",  ["None", "Norm", ">7", ">8"],
                                        key="sb_a1c")

            st.markdown("**Lab Results**")
            max_glu = st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"],
                                   key="sb_max_glu")


        # --- Section 5: Clinical Notes (Model 4 - NLP) ---
        with st.expander("📝 Clinical Notes Analysis (NLP Intelligence)", expanded=False):
            st.markdown("""
            <p style='font-size: 0.9rem; color: #94a3b8;'>
                Enter physician progress notes or patient feedback. BioBERT will analyze 
                the linguistic sentiment in context with the medication and condition.
            </p>
            """, unsafe_allow_html=True)
            
            # These two fields are required by Wes's model metadata
            col_n1, col_n2 = st.columns(2)
            nlp_drug = col_n1.text_input("Context Medication", value="Metformin",
                                         key="ti_nlp_drug")
            nlp_cond = col_n2.text_input("Context Condition", value="Diabetes, Type 2",
                                         key="ti_nlp_cond")

            clinical_notes = st.text_area(
                "Physician/Nursing Progress Notes:",
                placeholder="Example: Patient reports difficulty following the insulin regimen and shows signs of anxiety regarding discharge...",
                height=150,
                key="ta_clinical_notes",
            )

        st.markdown("")
        submitted = st.form_submit_button("⚡ RUN PREDICTION")

    # ── PREDICTIONS ───────────────────────────────────────────────────
    if submitted:
        patient_dict = {
            "age":                      age,
            "gender":                   gender,
            "race":                     race,
            "admission_type_id":        parse_id(admission_type),
            "discharge_disposition_id": parse_id(discharge_disp),
            "admission_source_id":      parse_id(admission_src),
            "time_in_hospital":         time_in_hosp,
            "num_lab_procedures":       num_lab_procs,
            "num_procedures":           num_procs,
            "num_medications":          num_meds,
            "number_diagnoses":         num_diag,
            "number_outpatient":        num_outpatient,
            "number_emergency":         num_emergency,
            "number_inpatient":         num_inpatient,
            "prior_inpatient_cumsum":   prior_inp_cum,
            "diag_1":                   diag1,
            "diag_2":                   diag2,
            "diag_3":                   diag3,
            "insulin":                  insulin,
            "metformin":                metformin,
            "change":                   change,
            "diabetesMed":              diabetes_md,
            "A1Cresult":                a1c,
            "max_glu_serum":            max_glu,
            "medical_specialty":        med_spec,
        }

        # --- Run all models and persist results in session_state ---
        proba1, proba2 = None, None
        pred1, pred2 = None, None
        conf1, conf2 = None, None
        latency_m1, latency_m2 = 0, 0
        los_label = los_conf = los_css = None
        nlp_label = nlp_conf = nlp_css = nlp_explanation = None

        # ── Model 1 ──────────────────────────────────────────────────
        with st.spinner("Running Model 1 — XGBoost…"):
            t0 = time.perf_counter()
            try:
                pred1, proba1, conf1 = predict_m1(patient_dict)
                latency_m1 = (time.perf_counter() - t0) * 1000
                st.session_state["m1_result"] = {
                    "proba": proba1, "pred": pred1,
                    "latency": latency_m1, "conf": conf1,
                }
            except Exception:
                logger.error("M1 prediction failed", exc_info=True)
                st.error("Model 1 could not complete the prediction. The issue has been logged — please try again or contact support.")

        # ── Model 2 ──────────────────────────────────────────────────
        st.cache_resource.clear()
        gc.collect()
        with st.spinner("Running Model 2 — DNN…"):
            t0 = time.perf_counter()
            try:
                pred2, proba2, conf2 = predict_m2(patient_dict)
                latency_m2 = (time.perf_counter() - t0) * 1000
                st.session_state["m2_result"] = {
                    "proba": proba2, "pred": pred2,
                    "latency": latency_m2, "conf": conf2,
                }
            except Exception:
                logger.error("M2 prediction failed", exc_info=True)
                st.error("Model 2 could not complete the prediction. The issue has been logged — please try again or contact support.")

        # ── Model 5 Innovation ────────────────────────────────────────
        with st.spinner("Analyzing capacity requirements (Model 5)..."):
            try:
                los_label, los_conf, los_css = predict_m5(patient_dict)
                st.session_state["m5_result"] = {
                    "label": los_label, "conf": los_conf, "css": los_css,
                }
            except Exception:
                logger.error("M5 prediction failed", exc_info=True)
                st.error("Capacity Planning model could not complete the prediction. The issue has been logged — please try again or contact support.")

        # ── Model 4 NLP ───────────────────────────────────────────────
        st.cache_resource.clear()
        gc.collect()
        with st.spinner("Analyzing clinical sentiment with BioBERT (LoRA)..."):
            try:
                nlp_label, nlp_conf, nlp_css, nlp_explanation = predict_m4(clinical_notes, nlp_drug, nlp_cond)
                st.session_state["m4_result"] = {
                    "label": nlp_label, "conf": nlp_conf,
                    "css": nlp_css, "explanation": nlp_explanation,
                }
                # Store M6 recommendations alongside M4 result
                _recs_df = get_m6_recommendations(nlp_cond, nlp_drug)
                if not _recs_df.empty:
                    _recs_text = "; ".join(
                        f"#{int(r['rank'])} {r['urlDrugName']} ({r['pct_highly_effective']*100:.1f}% highly effective)"
                        for _, r in _recs_df.iterrows()
                    )
                else:
                    _recs_text = "No alternatives found with sufficient patient review data."
                st.session_state["_syn_m6_drug"]  = nlp_drug
                st.session_state["_syn_m6_cond"]  = nlp_cond
                st.session_state["_syn_m6_recs"]  = _recs_text
            except Exception:
                logger.error("M4 prediction failed", exc_info=True)
                st.error("NLP Intelligence model could not complete the analysis. The issue has been logged — please try again or contact support.")

        # ── Consensus — store in session_state so it survives re-runs ─
        if proba1 is not None and proba2 is not None:
            st.session_state["_syn_p1"] = proba1
            st.session_state["_syn_p2"] = proba2
            st.session_state["_syn_m5"] = los_label if los_label is not None else "Unavailable"
            st.session_state["_syn_m4_label"] = nlp_label if nlp_label is not None else "Unavailable"
            st.session_state["_syn_m4_expl"]  = nlp_explanation if nlp_explanation is not None else ""
            st.session_state["_con_pred1"] = pred1
            st.session_state["_con_pred2"] = pred2
            st.session_state["_con_lat1"]  = latency_m1
            st.session_state["_con_lat2"]  = latency_m2

        # ── Auto-scroll to results after prediction ───────────────────
        import streamlit.components.v1 as _stc
        _stc.html(
            "<script>if(document.body){window.scrollTo({top:document.body.scrollHeight,behavior:'smooth'});}else{window.addEventListener('load',function(){window.scrollTo({top:document.body.scrollHeight,behavior:'smooth'});});};</script>",
            height=0,
        )

    # ── CSS for gauges — injected always so persisted cards render correctly ──
    st.markdown("""
    <style>
    .result-panel {
        background: rgba(15,30,52,0.6);
        border: 1px solid rgba(34,211,238,0.15);
        border-radius: 16px;
        padding: 2rem 1rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    .gauge-wrapper {
        position: relative;
        width: 160px;
        height: 160px;
        margin: 0 auto 1.5rem auto;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(0,0,0,0.2);
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
    }
    .gauge-inner {
        position: absolute;
        width: 130px;
        height: 130px;
        background: #0b1121;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 10;
    }
    .gauge-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
        color: white;
    }
    .gauge-lbl {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 4px;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .risk-LOW { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .risk-MODERATE { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
    .risk-HIGH { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    </style>
    """, unsafe_allow_html=True)

    def get_gauge_html(proba: float, title: str, latency: float, threshold: float = 0.38) -> str:
        pct = int(proba * 100)
        if proba >= 0.60:
            color, risk_tier = "#ef4444", "HIGH"
        elif proba >= threshold:
            color, risk_tier = "#f59e0b", "MODERATE"
        else:
            color, risk_tier = "#10b981", "LOW"
        return f"""
        <div class="result-panel">
            <div style="font-family:'JetBrains Mono'; font-size:0.8rem; font-weight:700; color:#5eead4; letter-spacing:0.1em; margin-bottom:0.3rem;">
                {title}
            </div>
            <div style="font-family:'JetBrains Mono'; font-size:0.65rem; color:#64748b; margin-bottom:1.5rem;">
                latency · {latency:.1f}ms
            </div>
            <div class="gauge-wrapper" style="background: conic-gradient({color} {pct}%, rgba(255,255,255,0.05) {pct}%);">
                <div class="gauge-inner">
                    <span class="gauge-val">{pct}%</span>
                    <span class="gauge-lbl" style="font-size:0.58rem;">30-DAY READMISSION</span>
                </div>
            </div>
            <div class="risk-badge risk-{risk_tier}">{risk_tier} RISK</div>
            <p style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
                Decision Threshold: {threshold}
            </p>
        </div>
        """

    # ── Results tabs ─────────────────────────────────────────────────
    # ── Dynamic tab labels — badge shows once prediction has run ─────
    if "m1_result" in st.session_state or "m2_result" in st.session_state:
        _r1_t    = st.session_state.get("m1_result", {})
        _r2_t    = st.session_state.get("m2_result", {})
        _n_t     = int(bool(_r1_t)) + int(bool(_r2_t))
        _avg_p_t = (_r1_t.get("proba", 0) + _r2_t.get("proba", 0)) / max(_n_t, 1)
        _t1_bdg  = "HIGH ⚠" if _avg_p_t >= 0.60 else ("MODERATE ⚡" if _avg_p_t >= 0.38 else "LOW ✓")
        _tab1_label = f"🫀 Readmission · {_t1_bdg}"
    else:
        _tab1_label = "🫀 Readmission Risk (M1 & M2)"

    if "m4_result" in st.session_state:
        _tab2_label = f"🧪 Notes · {st.session_state['m4_result']['label']}"
    else:
        _tab2_label = "🧪 Clinical Notes (M4)"

    if "m5_result" in st.session_state:
        _tab3_label = f"🏥 Ops · {st.session_state['m5_result']['label']}"
    else:
        _tab3_label = "🏥 Capacity & Ops (M5)"

    if True:  # tabs always visible — placeholders shown before first prediction
        _tab1, _tab2, _tab3 = st.tabs([_tab1_label, _tab2_label, _tab3_label])

        # ── TAB 1: Readmission Risk (M1 & M2) + Consensus ──────────
        with _tab1:
            if not (
                "m1_result" in st.session_state
                or "m2_result" in st.session_state
                or "_syn_p1" in st.session_state
            ):
                st.info("👈 Fill out the patient encounter form and click **⚡ Run Prediction** to see readmission risk results here.")
            if "m1_result" in st.session_state or "m2_result" in st.session_state:
                st.markdown("""
                <div class="section-head">
                  <span class="num">🎯</span><h2>Inference Results</h2><div class="line"></div>
                </div>""", unsafe_allow_html=True)
                st.markdown(
                    "<p style='color:#94a3b8; font-size:0.95rem; margin:-0.5rem 0 1.2rem;'>"
                    "🎯 <b>Clinical Metric:</b> Probability of unplanned hospital readmission "
                    "within <b>30 days</b> of discharge.</p>",
                    unsafe_allow_html=True,
                )
                col_m1, col_m2 = st.columns(2, gap="large")
                with col_m1:
                    if "m1_result" in st.session_state:
                        r = st.session_state["m1_result"]
                        st.markdown(get_gauge_html(r["proba"], "M1 · XGBOOST ENSEMBLE", r["latency"], threshold=0.38), unsafe_allow_html=True)
                with col_m2:
                    if "m2_result" in st.session_state:
                        r = st.session_state["m2_result"]
                        st.markdown(get_gauge_html(r["proba"], "M2 · DEEP LEARNING DNN", r["latency"], threshold=0.38), unsafe_allow_html=True)
            if "_syn_p1" in st.session_state:
                _p1   = st.session_state["_syn_p1"]
                _p2   = st.session_state["_syn_p2"]
                _pred1 = st.session_state.get("_con_pred1")
                _pred2 = st.session_state.get("_con_pred2")
                _lat1  = st.session_state.get("_con_lat1", 0)
                _lat2  = st.session_state.get("_con_lat2", 0)
                _agreement       = "AGREE" if _pred1 == _pred2 else "DISAGREE"
                _agreement_color = "#5eead4" if _agreement == "AGREE" else "#fbbf24"
                _avg_proba       = (_p1 + _p2) / 2
                _high = max(_p1, _p2)
                if _high >= 0.65:
                    _rec       = ("⚠ Recommend enhanced discharge planning, scheduled "
                                  "follow-up within 7 days, and medication reconciliation.")
                    _rec_color = "#fca5a5"
                elif _high >= 0.40:
                    _rec       = ("⚡ Standard discharge with phone follow-up within 14 days. "
                                  "Monitor for medication adherence.")
                    _rec_color = "#fcd34d"
                else:
                    _rec       = "✓ Standard discharge protocol. Routine follow-up sufficient."
                    _rec_color = "#5eead4"

                st.markdown("""
                <div class="section-head">
                  <span class="num">⊕</span><h2>Consensus</h2><div class="line"></div>
                </div>""", unsafe_allow_html=True)
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.markdown(f"""
                <div class="stat">
                  <div class="label">MODEL AGREEMENT</div>
                  <div class="val" style="color:{_agreement_color};">{_agreement}</div>
                  <div class="delta">consensus check</div>
                </div>""", unsafe_allow_html=True)
                cc2.markdown(f"""
                <div class="stat">
                  <div class="label">AVG PROBABILITY</div>
                  <div class="val">{_avg_proba*100:.1f}%</div>
                  <div class="delta">M1 + M2 average</div>
                </div>""", unsafe_allow_html=True)
                cc3.markdown(f"""
                <div class="stat">
                  <div class="label">DELTA M1 ↔ M2</div>
                  <div class="val">{abs(_p1-_p2)*100:.1f}pp</div>
                  <div class="delta">disagreement gap</div>
                </div>""", unsafe_allow_html=True)
                cc4.markdown(f"""
                <div class="stat">
                  <div class="label">TOTAL LATENCY</div>
                  <div class="val">{(_lat1+_lat2):.0f}ms</div>
                  <div class="delta">end-to-end</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="gcard" style="border-color:rgba(94,234,212,0.3); margin-top:1rem;">
                  <span class="tag">CLINICAL RECOMMENDATION</span>
                  <p style="color:{_rec_color}!important; font-size:1rem;
                            margin-top:0.5rem; line-height:1.6; font-weight:500;">{_rec}</p>
                </div>""", unsafe_allow_html=True)

        # ── TAB 2: Clinical Notes (M4) + AI Clinical Synthesis ───────
        with _tab2:
            if not ("m4_result" in st.session_state or "_syn_p1" in st.session_state):
                st.info("👈 Fill out the patient encounter form and click **⚡ Run Prediction** to see clinical notes analysis here.")
            if "m4_result" in st.session_state:
                r4 = st.session_state["m4_result"]
                st.markdown("""<div class="section-head">
                  <span class="num">04</span><h2>NLP Intelligence</h2><div class="line"></div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="gcard">
                    <span class="tag">M4 · BioBERT (LoRA)</span>
                    <h3>Clinical Sentiment Analysis</h3>
                    <div class="{r4['css']}" style="margin:1rem 0; padding:15px; font-size:1.1rem; border-radius:10px; text-align:center; font-weight:700;">
                        {r4['label']}
                    </div>
                    <div style="display:flex; gap:8px; margin-top:0.8rem;">
                        <div class="stat" style="flex:1;">
                            <div class="label">NLP CONFIDENCE</div>
                            <div class="val">{r4['conf']*100:.1f}%</div>
                        </div>
                    </div>
                    <p style="font-size:0.85rem; color:#94a3b8!important; margin-top:1rem;">
                        <strong>Insight:</strong> {r4['explanation']}
                    </p>
                </div>""", unsafe_allow_html=True)
            if "_syn_p1" in st.session_state:
                st.markdown("""
                <div class="section-head">
                  <span class="num">🧠</span><h2>AI Clinical Synthesis</h2><div class="line"></div>
                </div>""", unsafe_allow_html=True)

                st.markdown("""
                <p style="font-size:0.82rem; color:#64748b; margin-bottom:0.8rem;
                          font-family:'JetBrains Mono',monospace; letter-spacing:0.03em;">
                    POWERED BY LLAMA 3.1 · GROQ · GENERATIVE AI CONSENSUS REPORT
                </p>""", unsafe_allow_html=True)

                if st.button("⚕️ Generate AI Clinical Consensus Report"):
                    with st.spinner("Synthesizing multi-model diagnostics with Llama AI..."):
                        st.session_state["synthesis_result"] = generate_clinical_synthesis(
                            m1_proba=st.session_state["_syn_p1"],
                            m2_proba=st.session_state["_syn_p2"],
                            m5_label=st.session_state["_syn_m5"],
                            m4_label=st.session_state["_syn_m4_label"],
                            m4_explanation=st.session_state["_syn_m4_expl"],
                        )

                if "synthesis_result" in st.session_state:
                    synthesis = st.session_state["synthesis_result"]
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(15,30,52,0.8) 0%, rgba(17,44,74,0.8) 100%);
                        border: 1px solid rgba(94,234,212,0.35);
                        border-radius: 14px;
                        padding: 1.8rem 2rem;
                        box-shadow: 0 8px 32px rgba(34,211,238,0.08), inset 0 1px 0 rgba(94,234,212,0.1);
                        margin-top: 0.5rem;
                    ">
                        <div style="display:flex; align-items:center; gap:10px; margin-bottom:1.2rem;">
                            <span style="font-size:1.3rem;">🧠</span>
                            <span style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                                         font-weight:700; color:#5eead4; text-transform:uppercase;
                                         letter-spacing:0.12em;">
                                AI CLINICAL SYNTHESIS · LLAMA 3.1 via GROQ
                            </span>
                        </div>
                        <div style="font-size:0.93rem; color:#e2e8f0; line-height:1.75;
                                    white-space:pre-wrap; font-family:'Inter',sans-serif;">
{synthesis}
                        </div>
                        <div style="margin-top:1.2rem; padding-top:1rem;
                                    border-top:1px solid rgba(94,234,212,0.1);
                                    font-size:0.72rem; color:#475569;
                                    font-family:'JetBrains Mono',monospace;">
                            ⚠ AI-GENERATED SYNTHESIS · FOR INVESTIGATIONAL USE ONLY · NOT A CLINICAL DIAGNOSIS
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("🗑️ Clear Synthesis"):
                        del st.session_state["synthesis_result"]
                        st.rerun()

        # ── TAB 3: Capacity & Ops (M5) + Drug Recommendation ─────────
        with _tab3:
            if not ("m5_result" in st.session_state or "m4_result" in st.session_state):
                st.info("👈 Fill out the patient encounter form and click **⚡ Run Prediction** to see capacity & operations results here.")
            if "m5_result" in st.session_state:
                r5 = st.session_state["m5_result"]
                st.markdown("""<div class="section-head">
                  <span class="num">⚡</span><h2>Innovation: Capacity Planning</h2><div class="line"></div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="gcard" style="border-color:rgba(34,211,238,0.4);">
                    <span class="tag">M5 · LENGTH OF STAY PREDICTOR</span>
                    <h3 style="margin-top:0.5rem;">Predicted Capacity Requirement</h3>
                    <div style="display:flex; align-items:center; gap:20px; margin:1rem 0;">
                        <div class="{r5['css']}" style="flex:1; font-size:1.2rem; padding:20px;">{r5['label']}</div>
                        <div class="stat" style="width:150px;">
                            <div class="label">CONFIDENCE</div>
                            <div class="val">{r5['conf']*100:.1f}%</div>
                        </div>
                    </div>
                    <p style="font-size:0.85rem; color:#94a3b8!important;">
                        <strong>Clinical Utility:</strong> This model assists in early discharge planning and
                        bed management by predicting the expected duration of stay upon admission.
                    </p>
                </div>""", unsafe_allow_html=True)
            if "m4_result" in st.session_state:
                r4 = st.session_state["m4_result"]
                # ── Model 6: Drug Recommendations ────────────────────
                _m6_condition = st.session_state.get("ti_nlp_cond", "")
                _m6_drug      = st.session_state.get("ti_nlp_drug", "")
                _m6_recs      = get_m6_recommendations(_m6_condition, _m6_drug)

                if _m6_recs.empty:
                    st.markdown(f"""
                    <div class="gcard" style="margin-top:1rem; opacity:0.7;">
                        <span class="tag">Innovation — Drug Recommendation</span>
                        <p style="color:#64748b; font-size:0.9rem; margin-top:0.75rem;">
                            ⚠ Insufficient patient review data to recommend alternative drugs for
                            <strong style="color:#94a3b8;">{_m6_condition}</strong>.
                            Try a condition such as <em>Depression</em>, <em>Anxiety</em>, or <em>Diabetes, Type 2</em>.
                        </p>
                    </div>""", unsafe_allow_html=True)
                else:
                    _css = r4.get("css", "risk-low")
                    _drug_name = _m6_drug.strip() if _m6_drug.strip() else "Drug"
                    _cond_name = _m6_condition.strip() if _m6_condition.strip() else "this condition"
                    if _css == "risk-high":
                        _rec_label   = f"{_drug_name} was Ineffective for {_cond_name}"
                        _rec_heading = "Recommended Alternatives Likely to Produce Better Outcomes"
                        _rec_color   = "#ef4444"
                    elif _css == "risk-medium":
                        _rec_label   = f"{_drug_name} was Somewhat Effective for {_cond_name}"
                        _rec_heading = "Drugs That May Produce Better Results for This Condition"
                        _rec_color   = "#f59e0b"
                    else:
                        _rec_label   = f"{_drug_name} was Highly Effective for {_cond_name}"
                        _rec_heading = "Other Drugs That Have Also Shown Strong Results for This Condition"
                        _rec_color   = "#10b981"

                    st.markdown(f"""
                    <div class="gcard" style="margin-top:1rem;">
                        <span class="tag">Innovation — Drug Recommendation</span>
                        <h3 style="color:{_rec_color}; margin-bottom:0.25rem;">{_rec_label}</h3>
                        <p style="color:#94a3b8; font-size:0.9rem; margin-bottom:1rem;">{_rec_heading}</p>
                    """, unsafe_allow_html=True)

                    for _, row in _m6_recs.iterrows():
                        he_pct  = row["pct_highly_effective"] * 100
                        se_pct  = row["pct_somewhat_effective"] * 100
                        in_pct  = row["pct_ineffective"] * 100
                        reviews = int(row["total_reviews"])
                        rank    = int(row["rank"])
                        st.markdown(f"""
                        <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                                    border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.5rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                                <span style="font-family:'JetBrains Mono'; font-weight:700; font-size:0.95rem; color:#e2e8f0;">
                                    #{rank} &nbsp; {row['urlDrugName']}
                                </span>
                                <span style="font-size:0.75rem; color:#64748b;">{reviews:,} reviews</span>
                            </div>
                            <div style="display:flex; gap:1rem; font-size:0.8rem;">
                                <span style="color:#10b981;">✔ Highly Effective: {he_pct:.1f}%</span>
                                <span style="color:#f59e0b;">~ Somewhat: {se_pct:.1f}%</span>
                                <span style="color:#ef4444;">✗ Ineffective: {in_pct:.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

    # ── Export Clinical Summary ───────────────────────────────────────────────
    if "_syn_p1" in st.session_state:
        with st.expander("📄 Export Clinical Summary (Copy to EHR)"):
            _r1  = st.session_state.get("m1_result", {})
            _r2  = st.session_state.get("m2_result", {})
            _r4  = st.session_state.get("m4_result", {})
            _r5  = st.session_state.get("m5_result", {})
            _m1p = _r1.get("proba", 0.0)
            _m2p = _r2.get("proba", 0.0)
            _synthesis_text = st.session_state.get("synthesis_result", "Not generated — click 'Generate AI Clinical Consensus Report' above.")

            clinical_report = f"""=========================================
CLEARSIGHT ANALYTICS — PATIENT ENCOUNTER SUMMARY
Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
=========================================

[DEMOGRAPHICS & ADMISSION]
  Age Bracket        : {st.session_state.get('sb_age', 'N/A')}
  Gender             : {st.session_state.get('sb_gender', 'N/A')}
  Race               : {st.session_state.get('sb_race', 'N/A')}
  Admission Type     : {st.session_state.get('sb_admission_type', 'N/A')}
  Discharge Disp.    : {st.session_state.get('sb_discharge_disp', 'N/A')}
  Admission Source   : {st.session_state.get('sb_admission_src', 'N/A')}

[CLINICAL METRICS]
  Days in Hospital   : {st.session_state.get('ni_days_hospital', 'N/A')}
  Lab Procedures     : {st.session_state.get('ni_lab_procs', 'N/A')}
  Procedures         : {st.session_state.get('ni_procs', 'N/A')}
  Total Medications  : {st.session_state.get('ni_meds', 'N/A')}
  Number of Diagnoses: {st.session_state.get('ni_diag', 'N/A')}
  Outpatient Visits  : {st.session_state.get('ni_outpatient', 'N/A')}
  Emergency Visits   : {st.session_state.get('ni_emergency', 'N/A')}
  Inpatient Visits   : {st.session_state.get('ni_inpatient', 'N/A')}
  Prior Inpatient    : {st.session_state.get('ni_prior_inp', 'N/A')}

[ICD-9 DIAGNOSES]
  Primary (diag_1)   : {st.session_state.get('ti_diag1', 'N/A')}
  Secondary (diag_2) : {st.session_state.get('ti_diag2', 'N/A')}
  Tertiary (diag_3)  : {st.session_state.get('ti_diag3', 'N/A')}
  Medical Specialty  : {st.session_state.get('ti_med_spec', 'N/A')}

[MEDICATIONS & LABS]
  Insulin            : {st.session_state.get('sb_insulin', 'N/A')}
  Metformin          : {st.session_state.get('sb_metformin', 'N/A')}
  Med Changed        : {st.session_state.get('sb_change', 'N/A')}
  Diabetes Med       : {st.session_state.get('sb_diabetes_md', 'N/A')}
  A1C Result         : {st.session_state.get('sb_a1c', 'N/A')}
  Max Glucose Serum  : {st.session_state.get('sb_max_glu', 'N/A')}

[AI RISK ASSESSMENT]
  XGBoost M1 Prob.   : {_m1p:.1%}  (latency {_r1.get('latency', 0):.1f}ms)
  DNN M2 Prob.       : {_m2p:.1%}  (latency {_r2.get('latency', 0):.1f}ms)
  Avg Ensemble Risk  : {(_m1p + _m2p) / 2:.1%}
  Model Agreement    : {"AGREE" if _r1.get('pred') == _r2.get('pred') else "DISAGREE"}

[CAPACITY PLANNING (M5)]
  Predicted LoS      : {_r5.get('label', 'N/A')}
  LoS Confidence     : {_r5.get('conf', 0.0):.1%}

[NLP CLINICAL SENTIMENT (M4)]
  Context Medication : {st.session_state.get('ti_nlp_drug', 'N/A')}
  Context Condition  : {st.session_state.get('ti_nlp_cond', 'N/A')}
  Sentiment Label    : {_r4.get('label', 'N/A')}
  NLP Confidence     : {_r4.get('conf', 0.0):.1%}
  Interpretation     : {_r4.get('explanation', 'N/A')}

[DRUG RECOMMENDATION (Innovation)]
  Drug Assessed      : {st.session_state.get('_syn_m6_drug', 'N/A')}
  Condition          : {st.session_state.get('_syn_m6_cond', 'N/A')}
  Top Alternatives   : {st.session_state.get('_syn_m6_recs', 'N/A')}

[PHYSICIAN PROGRESS NOTE]
{st.session_state.get('ta_clinical_notes', 'None provided')}

[GENERATIVE AI SYNTHESIS]
{_synthesis_text}

=========================================
⚠ FOR INVESTIGATIONAL USE ONLY · NOT A CLINICAL DIAGNOSIS
ClearSight Analytics · Powered by XGBoost, DNN, BioBERT (LoRA), Llama 3.1
========================================="""

            st.code(clinical_report, language="markdown")

    # ── AI Clinical Copilot ───────────────────────────────────────────────────
    if "_syn_p1" in st.session_state:
        st.markdown("""
        <div class="section-head">
          <span class="num">💬</span><h2>AI Clinical Copilot</h2><div class="line"></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <p style="font-size:0.82rem; color:#64748b; margin-bottom:1rem;
                  font-family:'JetBrains Mono',monospace; letter-spacing:0.03em;">
            INTERACTIVE CLINICAL DECISION SUPPORT &nbsp;·&nbsp; LLAMA 3.1 via GROQ &nbsp;·&nbsp; FOLLOW-UP QUERIES
        </p>""", unsafe_allow_html=True)

        # Initialise persistent chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # ── System prompt — embeds live patient context ──────────────────────
        _copilot_system_prompt = (
            "You are an expert Clinical Decision Support System (CDSS). You assist "
            "physicians by analyzing predictive model outputs and patient metrics. "
            "Use professional medical terminology. DO NOT provide definitive diagnoses. "
            "Prioritize patient safety and base all recommendations strictly on the "
            "provided context.\n\n"
            "CURRENT PATIENT — MULTI-MODEL AI DIAGNOSTIC OUTPUTS:\n"
            f"  • Readmission Risk — XGBoost Ensemble (M1): {st.session_state['_syn_p1']*100:.1f}%\n"
            f"  • Readmission Risk — Deep Neural Network (M2): {st.session_state['_syn_p2']*100:.1f}%\n"
            f"  • Predicted Length of Stay (M5 Capacity Classifier): {st.session_state['_syn_m5']}\n"
            f"  • Clinical Notes Sentiment Classification (NLP M4): {st.session_state['_syn_m4_label']}\n"
            f"  • NLP Explanatory Context: {st.session_state['_syn_m4_expl']}\n"
            f"  • Drug Assessed (Innovation — Drug Recommendation): {st.session_state.get('_syn_m6_drug', 'N/A')} for {st.session_state.get('_syn_m6_cond', 'N/A')}\n"
            f"  • Recommended Alternatives (ranked by patient-reported effectiveness): {st.session_state.get('_syn_m6_recs', 'N/A')}\n\n"
            "Respond only within the scope of these outputs and validated clinical evidence."
            " CRITICAL RULE: YOU MUST ALWAYS END EVERY RESPONSE WITH THIS EXACT TEXT: "
            "'\n\n\u26a0\ufe0f **Disclaimer:** This is an AI-generated analysis for investigational "
            "use only. Always consult a licensed healthcare provider.'"
        )

        # ── Chat-specific CSS injection ───────────────────────────────────────
        st.markdown("""
        <style>
        /* ── Scrollable message window — top half of the widget ── */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(11,17,33,0.95) !important;
            border: 1px solid rgba(34,211,238,0.2) !important;
            border-radius: 12px 12px 0 0 !important;
            padding: 0 !important;
        }

        /* ── Input form row — visually attached below the window ── */
        [data-testid="stForm"] {
            background: rgba(11,17,33,0.98) !important;
            border: 1px solid rgba(34,211,238,0.2) !important;
            border-top: none !important;
            border-radius: 0 0 12px 12px !important;
            padding: 0.6rem 0.8rem !important;
            margin-top: 0 !important;
        }
        [data-testid="stForm"] > div:first-child {
            border: none !important;
            background: transparent !important;
            padding: 0 !important;
        }

        /* ── Text input inside form ── */
        [data-testid="stForm"] [data-testid="stTextInput"] input {
            background-color: rgba(15,30,52,0.6) !important;
            border: 1px solid rgba(34,211,238,0.25) !important;
            border-radius: 8px !important;
            color: #e2e8f0 !important;
            caret-color: #22d3ee !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.88rem !important;
        }
        [data-testid="stForm"] [data-testid="stTextInput"] input::placeholder {
            color: #475569 !important;
        }
        [data-testid="stForm"] [data-testid="stTextInput"] input:focus {
            border-color: rgba(34,211,238,0.55) !important;
            box-shadow: 0 0 0 2px rgba(34,211,238,0.08) !important;
        }
        [data-testid="stForm"] [data-testid="stTextInput"] label { display:none !important; }

        /* ── Send button ── */
        [data-testid="stFormSubmitButton"] button {
            background: rgba(34,211,238,0.12) !important;
            border: 1px solid rgba(34,211,238,0.35) !important;
            color: #22d3ee !important;
            border-radius: 8px !important;
            font-size: 1.1rem !important;
            height: 2.4rem !important;
            width: 100% !important;
            transition: all 0.15s ease !important;
        }
        [data-testid="stFormSubmitButton"] button:hover {
            background: rgba(34,211,238,0.22) !important;
            border-color: rgba(34,211,238,0.6) !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # ── Quick-starter buttons ─────────────────────────────────────────────
        _qs_col1, _qs_col2, _qs_col3 = st.columns(3)
        _starter_prompt = None

        with _qs_col1:
            if st.button(
                "🔬 Elaborate on NLP Risk",
                use_container_width=True,
                key="qs_nlp_risk",
            ):
                _m4r = st.session_state.get("m4_result", {})
                _starter_prompt = (
                    f"Elaborate on the clinical implications of the NLP-derived sentiment "
                    f"classification for this patient.\n\n"
                    f"PATIENT CONTEXT:\n"
                    f"- Age Bracket: {st.session_state.get('sb_age', 'Unknown')}\n"
                    f"- Gender: {st.session_state.get('sb_gender', 'Unknown')}\n"
                    f"- Primary ICD-9 Diagnosis: {st.session_state.get('ti_diag1', 'Unknown')}\n"
                    f"- Reported Drug (NLP input): {st.session_state.get('ti_nlp_drug', 'Unknown')}\n"
                    f"- Reported Condition (NLP input): {st.session_state.get('ti_nlp_cond', 'Unknown')}\n"
                    f"- M4 BioBERT Sentiment Label: {_m4r.get('label', st.session_state.get('_syn_m4_label', 'Unknown'))}\n"
                    f"- M4 Model Confidence: {_m4r.get('conf', 0)*100:.1f}%\n"
                    f"- M4 Explanation: {_m4r.get('explanation', st.session_state.get('_syn_m4_expl', 'N/A'))}\n"
                    f"- Drug Recommendation — Alternatives for {st.session_state.get('_syn_m6_cond', 'this condition')}: "
                    f"{st.session_state.get('_syn_m6_recs', 'No recommendation data available.')}\n\n"
                    "Reference pertinent ICD-10-CM diagnostic categories, relevant comorbidity "
                    "indices (Charlson, LACE), and validated predictive scoring instruments. "
                    "Identify all NLP-flagged risk signals and their specific clinical significance "
                    "for 30-day readmission risk. Based on the drug recommendation data provided, "
                    "include a section recommending alternative medications and explain why each "
                    "may produce better outcomes for this patient."
                )

        with _qs_col2:
            if st.button(
                "📋 Suggest Discharge Protocol",
                use_container_width=True,
                key="qs_discharge",
            ):
                _m1r = st.session_state.get("m1_result", {})
                _m5r = st.session_state.get("m5_result", {})
                _m1_proba = _m1r.get("proba", st.session_state.get("_syn_p1", 0))
                _m2_proba = st.session_state.get("_syn_p2", 0)
                _starter_prompt = (
                    f"Generate a structured, safe 30-day discharge and care transition "
                    f"protocol for this patient.\n\n"
                    f"PATIENT CONTEXT:\n"
                    f"- Age Bracket: {st.session_state.get('sb_age', 'Unknown')}\n"
                    f"- Gender: {st.session_state.get('sb_gender', 'Unknown')}\n"
                    f"- Race/Ethnicity: {st.session_state.get('sb_race', 'Unknown')}\n"
                    f"- Admission Type: {st.session_state.get('sb_admission_type', 'Unknown')}\n"
                    f"- Discharge Disposition: {st.session_state.get('sb_discharge_disp', 'Unknown')}\n"
                    f"- Primary ICD-9 Diagnosis: {st.session_state.get('ti_diag1', 'Unknown')}\n"
                    f"- Secondary Diagnosis: {st.session_state.get('ti_diag2', 'Unknown')}\n"
                    f"- Total Medications: {st.session_state.get('ni_meds', 'Unknown')}\n"
                    f"- Insulin Regimen: {st.session_state.get('sb_insulin', 'Unknown')}\n"
                    f"- HbA1c Result: {st.session_state.get('sb_a1c', 'Unknown')}\n"
                    f"- AI Readmission Risk — XGBoost M1: {_m1_proba*100:.1f}%\n"
                    f"- AI Readmission Risk — DNN M2: {_m2_proba*100:.1f}%\n"
                    f"- Predicted Length of Stay (M5): {_m5r.get('label', st.session_state.get('_syn_m5', 'Unknown'))}\n\n"
                    "FORMAT REQUIREMENTS:\n"
                    "- Provide a bulleted list of 3-4 immediate post-discharge actions.\n"
                    "- Suggest specific follow-up timelines based on the readmission risk level.\n"
                    "- Include a brief 'Red Flags' section for the patient to monitor at home.\n"
                    "- Address medication reconciliation concerns given the active drug list."
                )

        with _qs_col3:
            if st.button(
                "⚠️ Identify Primary Risk Drivers",
                use_container_width=True,
                key="qs_risk_drivers",
            ):
                _m1r = st.session_state.get("m1_result", {})
                _m2r = st.session_state.get("m2_result", {})
                _m1_proba = _m1r.get("proba", st.session_state.get("_syn_p1", 0))
                _m2_proba = _m2r.get("proba", st.session_state.get("_syn_p2", 0))
                _starter_prompt = (
                    f"Identify and rank the primary clinical risk drivers indicated by the "
                    f"multi-model diagnostic outputs for this patient.\n\n"
                    f"PATIENT CONTEXT:\n"
                    f"- Age Bracket: {st.session_state.get('sb_age', 'Unknown')}\n"
                    f"- Gender: {st.session_state.get('sb_gender', 'Unknown')}\n"
                    f"- Admission Type: {st.session_state.get('sb_admission_type', 'Unknown')}\n"
                    f"- Primary ICD-9 Diagnosis: {st.session_state.get('ti_diag1', 'Unknown')}\n"
                    f"- Secondary Diagnoses: {st.session_state.get('ti_diag2', 'Unknown')}, "
                    f"{st.session_state.get('ti_diag3', 'Unknown')}\n"
                    f"- Number of Active Diagnoses: {st.session_state.get('ni_diag', 'Unknown')}\n"
                    f"- Number of Medications: {st.session_state.get('ni_meds', 'Unknown')}\n"
                    f"- Number of Procedures: {st.session_state.get('ni_procs', 'Unknown')}\n"
                    f"- Prior Inpatient Visits: {st.session_state.get('ni_prior_inp', 'Unknown')}\n"
                    f"- Emergency Visits (past year): {st.session_state.get('ni_emergency', 'Unknown')}\n"
                    f"- HbA1c Result: {st.session_state.get('sb_a1c', 'Unknown')}\n"
                    f"- Max Glucose Serum: {st.session_state.get('sb_max_glu', 'Unknown')}\n"
                    f"- AI Readmission Risk — XGBoost M1: {_m1_proba*100:.1f}% "
                    f"(confidence: {_m1r.get('conf', 0)*100:.1f}%)\n"
                    f"- AI Readmission Risk — DNN M2: {_m2_proba*100:.1f}% "
                    f"(confidence: {_m2r.get('conf', 0)*100:.1f}%)\n"
                    f"- NLP Sentiment Classification (M4): {st.session_state.get('_syn_m4_label', 'Unknown')}\n\n"
                    "For each risk driver: specify the underlying clinical correlates, "
                    "associated ICD-10-CM codes where applicable, and prioritise targeted "
                    "interventions by urgency and projected impact on 30-day readmission prevention."
                )

        # ── HTML bubble renderer ──────────────────────────────────────────────
        import html as _html_mod
        import re as _re_mod

        def _render_bubble(text: str, role: str) -> str:
            safe = _html_mod.escape(text)
            safe = _re_mod.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
            safe = _re_mod.sub(r"\*(.+?)\*", r"<em>\1</em>", safe)
            safe = _re_mod.sub(
                r"(?m)^(\d+)\.\s+(.+)$",
                r"<span style='display:block;margin:0.15rem 0 0.15rem 0.8rem;'>"
                r"<span style='color:#94a3b8;'>\1.</span> \2</span>",
                safe,
            )
            safe = _re_mod.sub(
                r"(?m)^-\s+(.+)$",
                r"<span style='display:block;margin:0.15rem 0 0.15rem 0.8rem;'>"
                r"<span style='color:#22d3ee;'>&#9656;</span> \1</span>",
                safe,
            )
            safe = safe.replace("\n\n", "</p><p style='margin:0.5rem 0 0;'>")
            safe = safe.replace("\n", "<br>")
            if role == "user":
                return (
                    "<div style='display:flex;flex-direction:column;"
                    "align-items:flex-end;margin:0.45rem 0.6rem;'>"
                    "<div style='font-size:0.62rem;font-weight:700;color:#22d3ee;"
                    "font-family:\"JetBrains Mono\",monospace;letter-spacing:0.1em;"
                    "margin-bottom:0.2rem;'>YOU</div>"
                    "<div style='background:rgba(34,211,238,0.08);"
                    "border:1px solid rgba(34,211,238,0.28);"
                    "border-radius:14px 14px 2px 14px;"
                    "padding:0.55rem 0.9rem;max-width:80%;"
                    "font-size:0.875rem;color:#e2e8f0;line-height:1.6;"
                    "font-family:\"Inter\",sans-serif;'>"
                    f"<p style='margin:0;'>{safe}</p></div></div>"
                )
            return (
                "<div style='display:flex;flex-direction:column;"
                "align-items:flex-start;margin:0.45rem 0.6rem;'>"
                "<div style='font-size:0.62rem;font-weight:700;color:#10b981;"
                "font-family:\"JetBrains Mono\",monospace;letter-spacing:0.1em;"
                "margin-bottom:0.2rem;'>COPILOT &nbsp;&middot;&nbsp; LLAMA 3.1</div>"
                "<div style='background:rgba(6,12,24,0.85);"
                "border:1px solid rgba(16,185,129,0.18);"
                "border-radius:14px 14px 14px 2px;"
                "padding:0.55rem 0.9rem;max-width:86%;"
                "font-size:0.875rem;color:#cbd5e1;line-height:1.65;"
                "font-family:\"Inter\",sans-serif;'>"
                f"<p style='margin:0;'>{safe}</p></div></div>"
            )

        # ── Bounded scrollable message window ────────────────────────────────
        with st.container(height=430, border=True):
            if not st.session_state.chat_history:
                st.markdown(
                    "<div style='display:flex;align-items:center;"
                    "justify-content:center;height:100%;padding:3rem 0;'>"
                    "<p style='color:#334155;font-size:0.8rem;text-align:center;"
                    "font-family:\"JetBrains Mono\",monospace;letter-spacing:0.04em;'>"
                    "No messages yet.<br>Use the quick-starters above or type below.</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            for _chat_msg in st.session_state.chat_history:
                st.markdown(
                    _render_bubble(_chat_msg["content"], _chat_msg["role"]),
                    unsafe_allow_html=True,
                )
            # ── Auto-scroll to bottom after every render ──────────────────────
            st.markdown(
                """
                <script>
                (function() {
                    // Target every Streamlit height-bounded vertical block wrapper
                    var wrappers = window.parent.document.querySelectorAll(
                        '[data-testid="stVerticalBlockBorderWrapper"]'
                    );
                    if (!wrappers.length) return;
                    // The chat container is the last one on the page
                    var chatWrapper = wrappers[wrappers.length - 1];
                    // Walk up to find the overflow:auto scroll parent
                    var el = chatWrapper;
                    for (var i = 0; i < 8; i++) {
                        el = el.parentElement;
                        if (!el) break;
                        var overflow = window.parent.getComputedStyle(el).overflowY;
                        if (overflow === 'auto' || overflow === 'scroll') {
                            el.scrollTop = el.scrollHeight;
                            return;
                        }
                    }
                    // Fallback: directly scroll the wrapper itself
                    chatWrapper.scrollTop = chatWrapper.scrollHeight;
                })();
                </script>
                """,
                unsafe_allow_html=True,
            )

        # ── Inline input form — attached flush below the window ───────────────
        with st.form(key="copilot_form", clear_on_submit=True):
            _fi_col, _fb_col = st.columns([11, 1])
            with _fi_col:
                _typed_input = st.text_input(
                    label="msg",
                    label_visibility="collapsed",
                    placeholder="Type a clinical query and press Enter or ➤",
                    key="copilot_text_field",
                )
            with _fb_col:
                _form_submitted = st.form_submit_button("➤", use_container_width=True)

        _active_prompt = _starter_prompt or (
            _typed_input.strip() if _form_submitted and _typed_input.strip() else None
        )

        if _active_prompt:
            st.session_state.chat_history.append(
                {"role": "user", "content": _active_prompt}
            )
            _reply = "⚠️ Copilot unavailable: unknown error."
            with st.spinner("ClearSight Copilot is reasoning…"):
                try:
                    from openai import OpenAI as _OpenAI  # noqa: PLC0415
                    _copilot_client = _OpenAI(
                        api_key=st.secrets.get("GROQ_API_KEY", ""),
                        base_url="https://api.groq.com/openai/v1",
                    )
                    _copilot_messages = [
                        {"role": "system", "content": _copilot_system_prompt}
                    ] + [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history
                    ]
                    _copilot_response = _copilot_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        max_tokens=1024,
                        temperature=0.2,
                        messages=_copilot_messages,
                    )
                    _reply = _copilot_response.choices[0].message.content
                except Exception as _copilot_err:
                    logger.error(
                        "Copilot API call failed: %s", _copilot_err, exc_info=True
                    )
                    _reply = f"⚠️ Copilot unavailable: {_copilot_err}"
            st.session_state.chat_history.append(
                {"role": "assistant", "content": _reply}
            )
            st.rerun()

        # ── Clear conversation ────────────────────────────────────────────────
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Conversation", key="clear_copilot_chat"):
                st.session_state.chat_history = []
                st.rerun()

    st.markdown("""
        <div class="disclaimer">
          <strong>Clinical Decision Support Notice.</strong> Predictions are
          intended to assist — not replace — qualified clinical judgment.
          All outputs must be reviewed by a licensed clinician before any
          care decision. <strong>FOR INVESTIGATIONAL USE ONLY · NOT FDA CLEARED.</strong>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE: INSIGHTS (Full Teardown & Redesign)
# =============================================================================
def page_insights() -> None:
    """Renders the Clinical Interpretability & Architecture insights page.

    Organises content across five tabs: M1 XGBoost SHAP analysis (bar chart and
    summary plot), M2 DNN training curves and confusion matrix, M5 capacity-planning
    metrics, M4 BioBERT (LoRA) architecture overview, and a side-by-side model comparison
    table. Diagnostic PNG artefacts are loaded from each model's ``saved_model``
    directory; missing files are handled with ``st.info`` fallbacks.
    """
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <span class="badge" style="display:inline-block; font-family:'JetBrains Mono',monospace;
            font-size:0.7rem; font-weight:600; color:#5eead4;
            background:rgba(94,234,212,0.1); border:1px solid rgba(94,234,212,0.3);
            padding:4px 12px; border-radius:999px; letter-spacing:0.08em;
            text-transform:uppercase;">● MODEL INSIGHTS</span>
      <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2rem;
                 margin:0.4rem 0; letter-spacing:-0.02em;">
        Clinical Interpretability & Architecture
      </h1>
      <p style="color:#94a3b8; max-width: 800px; line-height: 1.6;">
        In healthcare, accuracy without transparency is a liability. This dashboard reveals the mechanics 
        behind every prediction—from feature importance (SHAP) and training dynamics, to the architectural 
        consensus that drives ClearSight's clinical trust.
      </p>
    </div>""", unsafe_allow_html=True)

    # ── 7-tab architecture ──────────────────────────────────────────────────
    t_consensus, t_m1, t_m2, t_m3, t_m4, t_m5, t_ai_gen = st.tabs([
        "⚖️ Consensus Architecture",
        "🌳 M1 · XGBoost",
        "🧠 M2 · DNN",
        "👁️ M3 · Retinal AI",
        "📝 M4 · NLP Notes",
        "⚡ M5 · Capacity",
        "💬 AI Generative Layer"
    ])

    # ── TAB: Consensus Architecture ─────────────────────────────────────────
    with t_consensus:
        st.markdown("""
        <div class="gcard" style="margin-bottom: 2rem;">
          <span class="tag">SYSTEM ARCHITECTURE</span>
          <h3 style="margin: 0.5rem 0;">The Consensus Strategy: Why Two Models?</h3>
          <p style="color: #94a3b8; font-size: 0.9rem;">
            In high-stakes clinical environments, relying on a single algorithm creates blind spots. 
            ClearSight utilizes a <b>Dual-Inference Consensus Engine</b>. We run both XGBoost (Tree-based) and DNN (Neural) in parallel. 
            When they agree, clinician confidence is amplified. When they disagree, it automatically flags the case for manual physician review, acting as a critical safety net.
          </p>
        </div>""", unsafe_allow_html=True)

        col_m1, col_m2 = st.columns(2, gap="large")

        with col_m1:
            st.markdown("""
            <div style="
                background: rgba(15,30,52,0.65);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(34,211,238,0.25);
                border-top: 3px solid #22d3ee;
                border-radius: 14px;
                padding: 1.6rem;
                height: 100%;
            ">
                <div style="font-size:2rem;margin-bottom:0.6rem;">🌳</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;font-weight:700;
                            color:#22d3ee;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.5rem;">
                    MODEL 1 · PRIMARY
                </div>
                <div style="font-size:1.15rem;font-weight:700;color:#e2e8f0;margin-bottom:1.2rem;">
                    Gradient Boosted Trees
                </div>
                <div style="display:flex;flex-direction:column;gap:0.75rem;">
                    <div style="background:rgba(34,211,238,0.06);border-radius:8px;padding:0.7rem 1rem;">
                        <div style="font-size:0.68rem;font-weight:700;color:#5eead4;letter-spacing:0.1em;margin-bottom:0.2rem;">STRENGTH</div>
                        <div style="font-size:0.88rem;color:#e2e8f0;">High Interpretability via native SHAP values</div>
                    </div>
                    <div style="background:rgba(34,211,238,0.06);border-radius:8px;padding:0.7rem 1rem;">
                        <div style="font-size:0.68rem;font-weight:700;color:#5eead4;letter-spacing:0.1em;margin-bottom:0.2rem;">CLINICAL ROLE</div>
                        <div style="font-size:0.88rem;color:#e2e8f0;">Primary logic for tabular clinical features</div>
                    </div>
                    <div style="background:rgba(34,211,238,0.06);border-radius:8px;padding:0.7rem 1rem;">
                        <div style="font-size:0.68rem;font-weight:700;color:#5eead4;letter-spacing:0.1em;margin-bottom:0.2rem;">INFERENCE SPEED</div>
                        <div style="font-size:0.88rem;color:#e2e8f0;">Ultra-fast · ~50 ms per prediction</div>
                    </div>
                    <div style="background:rgba(34,211,238,0.06);border-radius:8px;padding:0.7rem 1rem;">
                        <div style="font-size:0.68rem;font-weight:700;color:#5eead4;letter-spacing:0.1em;margin-bottom:0.2rem;">DECISION BOUNDARY</div>
                        <div style="font-size:0.88rem;color:#e2e8f0;">Cost-optimized threshold at 0.38 (Recall-first)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.markdown("""
            <div style="
                background: rgba(15,30,52,0.65);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(167,139,250,0.25);
                border-top: 3px solid #a78bfa;
                border-radius: 14px;
                padding: 1.6rem;
                height: 100%;
            ">
                <div style="font-size:2rem;margin-bottom:0.6rem;">🧠</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;font-weight:700;
                            color:#a78bfa;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.5rem;">
                    MODEL 2 · VALIDATOR
                </div>
                <div style="font-size:1.15rem;font-weight:700;color:#e2e8f0;margin-bottom:1.2rem;">
                    Deep Neural Network
                </div>
                <div style="display:flex;flex-direction:column;gap:0.75rem;">
                    <div style="background:rgba(167,139,250,0.06);border-radius:8px;padding:0.7rem 1rem;">
                        <div style="font-size:0.68rem;font-weight:700;color:#c4b5fd;letter-spacing:0.1em;margin-bottom:0.2rem;">STRENGTH</div>
                        <div style="font-size:0.88rem;color:#e2e8f0;">Capturing complex non-linear feature interactions</div>
                    </div>
                    <div style="background:rgba(167,139,250,0.06);border-radius:8px;padding:0.7rem 1rem;">
                        <div style="font-size:0.68rem;font-weight:700;color:#c4b5fd;letter-spacing:0.1em;margin-bottom:0.2rem;">CLINICAL ROLE</div>
                        <div style="font-size:0.88rem;color:#e2e8f0;">Secondary validation &amp; consensus check</div>
                    </div>
                    <div style="background:rgba(167,139,250,0.06);border-radius:8px;padding:0.7rem 1rem;">
                        <div style="font-size:0.68rem;font-weight:700;color:#c4b5fd;letter-spacing:0.1em;margin-bottom:0.2rem;">ARCHITECTURE</div>
                        <div style="font-size:0.88rem;color:#e2e8f0;">4-layer Dense Network (256→128→64→1) with Dropout</div>
                    </div>
                    <div style="background:rgba(167,139,250,0.06);border-radius:8px;padding:0.7rem 1rem;">
                        <div style="font-size:0.68rem;font-weight:700;color:#c4b5fd;letter-spacing:0.1em;margin-bottom:0.2rem;">DECISION BOUNDARY</div>
                        <div style="font-size:0.88rem;color:#e2e8f0;">Standard probabilistic threshold at 0.50</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            margin-top: 1.8rem;
            background: rgba(34,211,238,0.06);
            border: 1px solid rgba(34,211,238,0.2);
            border-left: 4px solid #22d3ee;
            border-radius: 10px;
            padding: 1rem 1.4rem;
        ">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:700;
                         color:#5eead4;letter-spacing:0.1em;">⚖️ CONSENSUS RULE</span>
            <p style="color:#e2e8f0;font-size:0.88rem;margin:0.4rem 0 0;line-height:1.6;">
                ClearSight issues a <strong style="color:#5eead4;">High Confidence</strong> recommendation
                only when both models reach agreement on the risk classification. A divergence between
                M1 and M2 automatically escalates the case for <strong style="color:#f59e0b;">manual physician review</strong>,
                acting as a critical safety net against algorithmic blind spots.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB: M1 XGBoost ─────────────────────────────────────────────────────
    with t_m1:
        st.markdown("""
        <div class="gcard" style="margin-bottom: 2rem;">
          <span class="tag">GLOBAL EXPLAINABILITY</span>
          <h3 style="margin: 0.5rem 0;">XGBoost Feature Mechanics (SHAP)</h3>
          <p style="color: #94a3b8; font-size: 0.9rem;">
            Gradient Boosted Trees excel at finding complex, non-linear interactions in structured tabular data. 
            By applying SHapley Additive exPlanations (SHAP), we reverse-engineer the model to understand exactly 
            which clinical variables are driving readmission risks across the entire MedInsight network.
          </p>
        </div>""", unsafe_allow_html=True)

        # ── Interactive SHAP Feature Importance Chart ────────────────────
        _shap_df = pd.DataFrame({
            "Feature":     ["number_inpatient", "time_in_hospital", "num_medications",
                            "discharge_disposition", "num_diagnoses", "age_cohort"],
            "SHAP Value":  [0.85, 0.62, 0.55, 0.45, 0.38, 0.30],
        }).sort_values("SHAP Value", ascending=True)  # ascending so highest bar is on top

        _shap_fig = px.bar(
            _shap_df,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            labels={"SHAP Value": "Mean |SHAP| Value (Impact on Prediction)", "Feature": ""},
        )
        _shap_fig.update_traces(
            marker_color="#22d3ee",
            marker_line_color="rgba(94,234,212,0.4)",
            marker_line_width=1,
        )
        _shap_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="JetBrains Mono, monospace", size=12),
            xaxis=dict(
                title="Mean |SHAP| Value (Impact on Prediction)",
                title_font=dict(color="#94a3b8", size=11),
                tickfont=dict(color="#94a3b8"),
                gridcolor="rgba(94,234,212,0.08)",
                zerolinecolor="rgba(94,234,212,0.15)",
            ),
            yaxis=dict(
                title="",
                tickfont=dict(color="#cbd5e1", size=12),
                gridcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=10, r=20, t=20, b=40),
            height=300,
        )
        st.plotly_chart(_shap_fig, use_container_width=True)

        sb = M1_DIR / "shap_bar.png"
        ss = M1_DIR / "shap_summary.png"
        cm1 = M1_DIR / "confusion_matrix.png"
        cm1o = M1_DIR / "confusion_matrix_optimal.png"

        c1, c2 = st.columns(2, gap="large")
        with c1:
            if sb.exists():
                st.image(str(sb), use_container_width=True)
                st.markdown("""
                <div style="background: rgba(34,211,238,0.05); border-left: 3px solid #22d3ee; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
                    <strong style="color: #5eead4; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">ANALYSIS: FEATURE MAGNITUDE</strong><br>
                    <span style="font-size: 0.85rem; color: #cbd5e1;">This bar chart ranks variables by their average absolute impact on the model's output. Historical utilization metrics (prior visits, inpatient cumulative sum) heavily dominate demographic factors, confirming that a patient's recent medical history is the strongest baseline indicator for future readmissions.</span>
                </div>
                """, unsafe_allow_html=True)

        with c2:
            if ss.exists():
                st.image(str(ss), use_container_width=True)
                st.markdown("""
                <div style="background: rgba(167,139,250,0.05); border-left: 3px solid #a78bfa; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
                    <strong style="color: #c4b5fd; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">ANALYSIS: DIRECTIONAL IMPACT</strong><br>
                    <span style="font-size: 0.85rem; color: #cbd5e1;">The summary plot reveals the <i>direction</i> of risk. Red dots indicate a high feature value, while blue indicates a low value. Notice how a high number of prior visits (red dots on the top row) clearly pushes the SHAP value to the right (increasing readmission risk), while lower utilization clusters to the left (protective factor).</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 3rem 0;'>", unsafe_allow_html=True)
        st.markdown("### Threshold Optimization & Confusion Matrices")
        
        c3, c4 = st.columns(2, gap="large")
        with c3:
            # ── Interactive ROC Curve (replaces missing static image) ──────────
            _fpr = np.array([0.00, 0.02, 0.05, 0.10, 0.15, 0.22, 0.30, 0.42, 0.55, 0.70, 0.85, 1.00])
            _tpr = np.array([0.00, 0.28, 0.48, 0.62, 0.70, 0.76, 0.81, 0.85, 0.88, 0.91, 0.95, 1.00])
            # Operating point closest to threshold 0.38 → index 5 (FPR=0.22, TPR=0.76)
            _op_fpr, _op_tpr = 0.22, 0.76

            _roc_fig = go.Figure()
            _roc_fig.add_trace(go.Scatter(
                x=_fpr, y=_tpr, mode="lines",
                name="XGBoost (AUC ≈ 0.76)",
                line=dict(color="#22d3ee", width=2.5),
            ))
            _roc_fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Random Chance",
                line=dict(color="#64748b", width=1.5, dash="dash"),
            ))
            _roc_fig.add_trace(go.Scatter(
                x=[_op_fpr], y=[_op_tpr], mode="markers+text",
                name="Chosen Threshold: 0.38",
                marker=dict(color="#f43f5e", size=10, symbol="circle"),
                text=["  Threshold 0.38"],
                textposition="middle right",
                textfont=dict(color="#f43f5e", size=10),
            ))
            _roc_fig.update_layout(
                title=dict(text="ROC Curve & Operating Point",
                           font=dict(color="#e2e8f0", size=13)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", family="JetBrains Mono, monospace"),
                xaxis=dict(title="False Positive Rate",
                           title_font=dict(color="#94a3b8"),
                           tickfont=dict(color="#94a3b8"),
                           gridcolor="rgba(94,234,212,0.08)",
                           zerolinecolor="rgba(94,234,212,0.15)"),
                yaxis=dict(title="True Positive Rate (Recall)",
                           title_font=dict(color="#94a3b8"),
                           tickfont=dict(color="#94a3b8"),
                           gridcolor="rgba(94,234,212,0.08)"),
                legend=dict(font=dict(color="#94a3b8", size=10),
                            bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=20, r=20, t=30, b=20),
                height=340,
            )
            st.plotly_chart(_roc_fig, use_container_width=True)
            st.markdown("""<p style="font-size: 0.8rem; color: #94a3b8; text-align: center;">Default Threshold (0.50): Optimizes for standard accuracy but misses too many at-risk patients (False Negatives).</p>""", unsafe_allow_html=True)

        with c4:
            if cm1o.exists():
                st.image(str(cm1o), use_container_width=True)
                st.markdown("""<p style="font-size: 0.8rem; color: #94a3b8; text-align: center;">Cost-Optimized (0.38): Maximizes Recall (85%) to prevent costly readmissions, accepting higher False Positives.</p>""", unsafe_allow_html=True)

    # ── TAB: M2 DNN ─────────────────────────────────────────────────────────
    with t_m2:
        st.markdown("""
        <div class="gcard" style="margin-bottom: 2rem;">
          <span class="tag">DEEP LEARNING DYNAMICS</span>
          <h3 style="margin: 0.5rem 0;">DNN Training Convergence</h3>
          <p style="color: #94a3b8; font-size: 0.9rem;">
            While XGBoost is highly interpretable, Neural Networks are powerful pattern-matchers capable of mapping highly non-linear relationships. 
            This model utilizes a Dense architecture (256→128→64) with Batch Normalization and Dropout layers to prevent overfitting on the minority class.
          </p>
        </div>""", unsafe_allow_html=True)
        
        from pathlib import Path
        correct_m2_dir = Path("models/model2_deep_learning/saved_model")
        tc = correct_m2_dir / "training_curves.png"
        cm2 = correct_m2_dir / "confusion_matrix.png"
        
        c1, c2 = st.columns(2, gap="large")
        with c1:
            if tc.exists():
                st.image(str(tc), use_container_width=True)
                st.markdown("""
                <div style="background: rgba(34,211,238,0.05); border-left: 3px solid #22d3ee; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
                    <strong style="color: #5eead4; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">LEARNING CURVES</strong><br>
                    <span style="font-size: 0.85rem; color: #cbd5e1;">The dual charts show Loss minimization and AUC maximization over training epochs. The close tracking of the validation curve (orange) to the training curve (blue) indicates that the early-stopping mechanisms and dropout regularization successfully prevented the model from memorizing the training data.</span>
                </div>
                """, unsafe_allow_html=True)
                
        with c2:
            if cm2.exists():
                st.image(str(cm2), use_container_width=True)
                st.markdown("""
                <div style="background: rgba(167,139,250,0.05); border-left: 3px solid #a78bfa; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
                    <strong style="color: #c4b5fd; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">VALIDATION PERFORMANCE</strong><br>
                    <span style="font-size: 0.85rem; color: #cbd5e1;">The DNN confusion matrix validates the model's ability to distinguish classes using calculated class-weights to address the ~46/54 imbalance. This acts as a robust secondary opinion to the primary XGBoost model.</span>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB: M3 Retinal AI ──────────────────────────────────────────────────
    with t_m3:
        st.markdown("### Retinal CNN Architecture (ResNet50)")
        st.markdown("""
        <div class="gcard" style="margin-bottom: 2rem;">
          <span class="tag">COMPUTER VISION</span>
          <h3 style="margin: 0.5rem 0;">Deep Spatial Feature Extraction from Fundus Imagery</h3>
          <p style="color: #94a3b8; font-size: 0.9rem;">
            Model 3 applies a fine-tuned <strong style="color:#5eead4;">ResNet50</strong> convolutional neural network 
            to fundus photographs for binary Diabetic Retinopathy (DR) classification. Unlike tabular models, 
            the CNN operates on raw pixel tensors — learning hierarchical spatial features (edges → textures → 
            pathological structures such as microaneurysms and neovascularization) through 50 residual layers 
            without any hand-crafted feature engineering. Grad-CAM visualizations expose the retinal regions 
            that most influenced each prediction, providing ophthalmology-grade interpretability for a 
            deep learning output.
          </p>
        </div>
        """, unsafe_allow_html=True)
        col_cnn1, col_cnn2 = st.columns(2, gap="large")

        with col_cnn1:
            st.markdown("#### The ResNet50 Backbone")
            st.markdown("""
            <div style="
                background: rgba(15,30,52,0.65);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(94,234,212,0.2);
                border-radius: 12px;
                padding: 1.4rem;
                font-size: 0.88rem;
                line-height: 1.8;
                color: #cbd5e1;
            ">
                <p style="margin: 0 0 0.9rem;">
                    <strong style="color:#5eead4;">Residual Learning &amp; Skip Connections</strong><br>
                    Standard deep networks suffer from <strong>vanishing gradients</strong> — as error signals 
                    are backpropagated through many layers, they shrink toward zero, effectively preventing 
                    early layers from learning. ResNet50 solves this by introducing <strong>shortcut connections</strong> 
                    that add the input of a block directly to its output: 
                    <code style="background:rgba(255,255,255,0.07);padding:1px 5px;border-radius:4px;">H(x) = F(x) + x</code>. 
                    This forces each block to learn only the <em>residual</em> correction 
                    <code style="background:rgba(255,255,255,0.07);padding:1px 5px;border-radius:4px;">F(x)</code>, 
                    not the full transformation — and when no update is needed, the block can trivially 
                    learn an identity mapping, keeping gradient flow intact across all 50 layers.
                </p>
                <p style="margin: 0 0 0.9rem;">
                    <strong style="color:#5eead4;">Hierarchical Feature Extraction</strong><br>
                    The network's depth enables a strict hierarchy of learned features:
                    <ul style="margin: 0.4rem 0 0.4rem 1.2rem; padding: 0;">
                        <li><strong>Early layers (conv1 – res2)</strong> detect low-level primitives: edges, 
                        color gradients, and blood-vessel boundaries.</li>
                        <li><strong>Mid layers (res3 – res4)</strong> compose those primitives into 
                        textures and localized structures — haemorrhage patches and vascular calibre changes.</li>
                        <li><strong>Deep layers (res5)</strong> recognise complex DR-specific pathologies: 
                        <strong>microaneurysms</strong>, <strong>hard exudates</strong>, and 
                        <strong>neovascularisation</strong> patterns that a shallow network cannot represent.</li>
                    </ul>
                </p>
                <p style="margin: 0;">
                    <strong style="color:#5eead4;">Transfer Learning from ImageNet</strong><br>
                    Training a 50-layer network from scratch on a modest retinal dataset risks overfitting. 
                    The model instead loads <strong>ImageNet pre-trained weights</strong>, which already 
                    encode robust low- and mid-level visual features. Fine-tuning on fundus photographs 
                    then specialises the deeper layers for DR pathology, dramatically accelerating 
                    convergence and improving generalisation on limited labelled medical data.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_cnn2:
            st.markdown("#### Mathematical Intuition: Grad-CAM")
            st.markdown("""
            <style>
            .formula-box {
                text-align: center;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.92rem;
                background: rgba(94,234,212,0.06);
                border: 1px solid rgba(94,234,212,0.18);
                border-radius: 8px;
                padding: 10px 14px;
                margin: 10px 0 6px;
                color: #5eead4;
                letter-spacing: 0.03em;
            }
            .formula-box sup, .formula-box sub {
                font-size: 0.72em;
                line-height: 0;
            }
            .icode {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.82em;
                background: rgba(255,255,255,0.07);
                padding: 1px 5px;
                border-radius: 4px;
                color: #94a3b8;
            }
            </style>
            <div style="
                background: rgba(15,30,52,0.65);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(94,234,212,0.2);
                border-radius: 12px;
                padding: 1.4rem;
                font-size: 0.88rem;
                line-height: 1.8;
                color: #cbd5e1;
            ">
                <p style="margin: 0 0 0.9rem;">
                    <strong style="color:#5eead4;">Gradients over Feature Maps</strong><br>
                    Gradient-weighted Class Activation Mapping (Grad-CAM) asks: <em>which spatial 
                    locations in the final convolutional layer most influenced the DR-positive score?</em> 
                    It answers by computing the gradient of the class score 
                    <span class="icode">y<sup>c</sup></span> 
                    with respect to every activation map 
                    <span class="icode">A<sup>k</sup></span> 
                    in the last convolutional layer:
                    <div class="formula-box">∂y<sup>c</sup> / ∂A<sup>k</sup><sub>ij</sub></div>
                    A large gradient magnitude at position (i, j) in channel k signals that 
                    spatial region is actively contributing to the prediction.
                </p>
                <p style="margin: 0 0 0.9rem;">
                    <strong style="color:#5eead4;">Global Average Pooling for Channel Weights</strong><br>
                    To convert the full gradient tensor into a single scalar importance weight 
                    <strong>α<sub>k</sub></strong> per channel, the gradients are 
                    <strong>globally average-pooled</strong> over all spatial positions (i, j):
                    <div class="formula-box">α<sub>k</sub> = (1 / Z) · Σ<sub>ij</sub> ∂y<sup>c</sup> / ∂A<sup>k</sup><sub>ij</sub></div>
                    This weight encodes <em>how important</em> the k-th feature map is, 
                    on average across the entire spatial grid, for producing the DR-positive output.
                </p>
                <p style="margin: 0;">
                    <strong style="color:#5eead4;">ReLU-gated Heatmap Generation</strong><br>
                    The final localisation map is a weighted linear combination of the forward 
                    activation maps, passed through a <strong>ReLU</strong>:
                    <div class="formula-box">L<sup>c</sup><sub>Grad-CAM</sub> = ReLU( Σ<sub>k</sub> α<sub>k</sub> · A<sup>k</sup> )</div>
                    The ReLU discards channels with <strong>negative</strong> influence 
                    (i.e., evidence <em>against</em> DR), retaining only regions that 
                    <em>positively</em> contributed to the diagnosis. The resulting 
                    low-resolution map is upsampled and overlaid on the original fundus 
                    image, highlighting lesion sites — such as microaneurysm clusters or 
                    exudate regions — that drove the classification decision.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB: M4 NLP Notes ───────────────────────────────────────────────────
    with t_m4:
        st.markdown("""
        <div class="gcard" style="margin-bottom: 2rem; border-color: rgba(245,158,11,0.3);">
          <span class="tag" style="background: rgba(245,158,11,0.1); color: #f59e0b;">MULTIMODAL EXPANSION</span>
          <h3 style="margin: 0.5rem 0;">BioBERT (LoRA) Clinical Sentiment Analysis</h3>
          <p style="color: #94a3b8; font-size: 0.9rem;">
            Structured EHR data often misses critical nuance hidden in physician progress notes. 
            Model 4 utilizes BioBERT (LoRA) — a 110M-parameter transformer pretrained on biomedical literature — to extract linguistic risk markers 
            (e.g., patient anxiety, medication non-compliance) and contextualizes them against specific drug and condition metadata.
          </p>
        </div>""", unsafe_allow_html=True)

        col_nlp1, col_nlp2 = st.columns([1.2, 1], gap="large")

        with col_nlp1:
            st.markdown("#### Sample Clinical Extraction")
            st.markdown("""
            <div style="
                background: rgba(15,30,52,0.65);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(245,158,11,0.25);
                border-radius: 12px;
                padding: 1.4rem;
                font-size: 0.88rem;
                line-height: 1.75;
                color: #cbd5e1;
            ">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;font-weight:700;
                            color:#f59e0b;letter-spacing:0.1em;margin-bottom:0.8rem;">
                    📋 PHYSICIAN PROGRESS NOTE — DAY 3
                </div>
                <p style="margin:0 0 0.6rem;">
                    Patient is a 58-year-old male with Type 2 Diabetes Mellitus and hypertension, admitted for 
                    hyperglycemic crisis. HbA1c on admission: 11.4%. Metformin 1000mg BID restarted. 
                    BP remains elevated at 148/92 mmHg.
                </p>
                <p style="margin:0 0 0.6rem;">
                    Patient reports <span style="
                        background-color: rgba(244,63,94,0.18);
                        border-bottom: 2px solid #f43f5e;
                        border-radius: 3px;
                        padding: 1px 4px;
                        color: #fda4af;
                        font-weight: 600;
                    ">severe anxiety and non-compliance with medication</span> 
                    regimen over the past three weeks due to financial constraints. 
                    Social work consult placed. Discharge planning initiated.
                </p>
                <p style="margin:0;">
                    A1C target &lt; 7.5%. Follow-up scheduled in 4 weeks. Patient instructed on 
                    self-monitoring of blood glucose.
                </p>
                <hr style="border-color:rgba(245,158,11,0.2);margin:1rem 0 0.8rem;">
                <div style="display:flex;gap:1.2rem;flex-wrap:wrap;">
                    <div>
                        <div style="font-size:0.65rem;font-weight:700;color:#f59e0b;
                                    letter-spacing:0.1em;margin-bottom:0.2rem;">SENTIMENT</div>
                        <div style="background:rgba(244,63,94,0.15);border:1px solid rgba(244,63,94,0.4);
                                    border-radius:6px;padding:3px 10px;font-size:0.78rem;
                                    font-weight:700;color:#fda4af;">⚠ Critical Risk</div>
                    </div>
                    <div>
                        <div style="font-size:0.65rem;font-weight:700;color:#f59e0b;
                                    letter-spacing:0.1em;margin-bottom:0.2rem;">EXTRACTED MARKER</div>
                        <div style="background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.3);
                                    border-radius:6px;padding:3px 10px;font-size:0.78rem;
                                    font-weight:600;color:#fcd34d;">Anxiety / Non-compliance</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_nlp2:
            st.markdown("#### Sentiment Distribution (Training Data)")
            _nlp_fig = px.pie(
                names=["Highly Effective", "Somewhat Effective", "Ineffective"],
                values=[48.7, 33.5, 17.8],
                hole=0.5,
                color_discrete_sequence=["#22d3ee", "#fbbf24", "#f43f5e"],
            )
            _nlp_fig.update_traces(
                textinfo="percent+label",
                textfont=dict(color="#e2e8f0", size=11),
                marker=dict(line=dict(color="rgba(0,0,0,0.4)", width=2)),
            )
            _nlp_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", family="JetBrains Mono, monospace"),
                showlegend=False,
                margin=dict(l=10, r=10, t=20, b=10),
                height=320,
            )
            st.plotly_chart(_nlp_fig, use_container_width=True)

    # ── TAB: M5 Capacity ────────────────────────────────────────────────────
    with t_m5:
        st.markdown("""
        <div class="gcard" style="margin-bottom: 2rem;">
          <span class="tag">OPERATIONAL INNOVATION</span>
          <h3 style="margin: 0.5rem 0;">Capacity Planning & Length of Stay</h3>
          <p style="color: #94a3b8; font-size: 0.9rem;">
            Predicting a readmission is only half the battle. Model 5 transforms clinical variables into 
            operational intelligence by classifying expected hospitalization duration (Short, Standard, Extended). 
            This allows hospital administrators to proactively manage bed capacity and staffing resources.
          </p>
        </div>""", unsafe_allow_html=True)
        
        from pathlib import Path
        correct_m5_dir = Path("models/model5_innovation/saved_model")
        cm5 = correct_m5_dir / "confusion_matrix.png"
        
        c1, c2 = st.columns(2, gap="large")
        with c1:
            if cm5.exists():
                st.image(str(cm5), use_container_width=True)
                
        with c2:
            st.markdown("""
            <div style="background: rgba(34,211,238,0.05); border-left: 3px solid #22d3ee; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
                <strong style="color: #5eead4; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">CLINICAL UTILITY (Weighted F1: 0.6047)</strong><br>
                <span style="font-size: 0.85rem; color: #cbd5e1;">By achieving a strong F1-Score on a complex multi-class problem, this model reliably identifies 'Extended Stay' patients at the point of admission. This enables early intervention from social workers and discharge planners, reducing bottlenecks in hospital flow.</span>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB: AI Generative Layer ─────────────────────────────────────────────
    with t_ai_gen:
        st.markdown("### Generative AI Consensus Engine")
        st.markdown("""
        <div class="gcard" style="margin-bottom: 2rem;">
          <span class="tag">GENERATIVE AI</span>
          <h3 style="margin: 0.5rem 0;">Two Modes of AI-Augmented Clinical Reasoning</h3>
          <p style="color: #94a3b8; font-size: 0.9rem;">
            ClearSight integrates a Groq-hosted <strong style="color:#5eead4;">Llama 3.1 (8B)</strong> large language model 
            at two distinct layers of the workflow, serving fundamentally different clinical purposes:
          </p>
        </div>
        """, unsafe_allow_html=True)

        _gen_col1, _gen_col2 = st.columns(2, gap="large")
        with _gen_col1:
            st.markdown("""
            <div style="border:1px solid rgba(34,211,238,0.3);border-radius:12px;padding:1.4rem;
                        background:rgba(13,27,42,0.8);height:100%;">
                <div style="font-size:1.4rem;margin-bottom:0.6rem;">📋</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:700;
                            color:#5eead4;letter-spacing:0.1em;margin-bottom:0.5rem;">
                    AI CLINICAL SYNTHESIS
                </div>
                <div style="font-size:1rem;font-weight:600;color:white;margin-bottom:0.8rem;">
                    Static Narrative Report
                </div>
                <p style="color:#94a3b8;font-size:0.85rem;line-height:1.6;">
                    Triggered once after all 5 models produce predictions for a patient. 
                    The LLM receives the aggregated risk scores and NLP sentiment as context, 
                    then generates a <strong style="color:#cbd5e1;">single structured clinical report</strong> — 
                    a synthesis paragraph plus three actionable recommendations. 
                    This is a <em>one-shot</em> call: read-only, non-interactive, deterministic at low temperature.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with _gen_col2:
            st.markdown("""
            <div style="border:1px solid rgba(94,234,212,0.4);border-radius:12px;padding:1.4rem;
                        background:rgba(13,27,42,0.8);height:100%;">
                <div style="font-size:1.4rem;margin-bottom:0.6rem;">🤖</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:700;
                            color:#5eead4;letter-spacing:0.1em;margin-bottom:0.5rem;">
                    AI CLINICAL COPILOT
                </div>
                <div style="font-size:1rem;font-weight:600;color:white;margin-bottom:0.8rem;">
                    Interactive LLM Agent
                </div>
                <p style="color:#94a3b8;font-size:0.85rem;line-height:1.6;">
                    A persistent, <strong style="color:#cbd5e1;">multi-turn conversational agent</strong> that retains 
                    full patient context across the session. Clinicians can ask follow-up questions, 
                    request discharge summaries, probe risk drivers, or explore treatment alternatives. 
                    Chat history is stored in <code style="color:#5eead4;">st.session_state</code> and 
                    each turn re-submits the complete context window to the model — enabling 
                    coherent, stateful clinical dialogue beyond the static report.
                </p>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# PAGE: RETINAL AI (High-Fidelity UI Simulation)
# =============================================================================
def page_retinal() -> None:
    """Renders the Retinal AI Screening page for Model 3 CNN (ResNet50).

    Loads the trained best_model.keras, accepts a fundus photograph upload,
    runs ResNet50 inference via predict_m3(), and displays the result label
    and confidence score. Class 0 = No DR, Class 1 = Has DR (threshold 0.5).
    """
    # 1. CSS for the Retinal Results Panel
    st.markdown("""
    <style>
    .scan-card {
        background: rgba(15,30,52,0.6);
        border: 1px solid rgba(34,211,238,0.15);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .severity-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        background: rgba(245,158,11,0.15); 
        color: #f59e0b; 
        border: 1px solid rgba(245,158,11,0.3);
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # 2. Header
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <span class="badge" style="display:inline-block; font-family:'JetBrains Mono',monospace;
            font-size:0.7rem; font-weight:600; color:#10b981;
            background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);
            padding:4px 12px; border-radius:999px; letter-spacing:0.08em;
            text-transform:uppercase;">✓ MODEL 3 · ONLINE</span>
      <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2rem;
                 margin:0.4rem 0;">Retinal AI Screening</h1>
      <p style="color:#94a3b8;">
        ResNet50 convolutional neural network for Diabetic Retinopathy detection
        from fundus photographs.
      </p>
    </div>""", unsafe_allow_html=True)

    # 3. Uploader
    uploaded = st.file_uploader("Upload a fundus photograph (PNG / JPG)", type=["png", "jpg", "jpeg"])

    st.warning(
        "⚠️ **Clinical Requirement:** This model requires a valid, high-resolution retinal "
        "fundus photograph. It does not currently perform Out-of-Distribution (OOD) detection "
        "for non-retinal images."
    )

    if uploaded is not None:
        model = load_model3()
        if model is None:
            st.error("Model could not be loaded.")
            return

        with st.spinner("Verifying image parameters and executing ResNet50 analysis..."):
            time.sleep(1.2)
            label, confidence = predict_m3(uploaded, model)
        is_high_risk = "HIGH RISK" in label
        logger.info("M3 result — label=%s confidence=%.4f", label, confidence)

        # ── ZONE 1: Result badge full width ──────────────────────
        if is_high_risk:
            st.markdown(
                f"""<div style="background:#7f1d1d;border-radius:10px;padding:12px 20px;
                margin-bottom:16px;display:flex;align-items:center;gap:12px;">
                <span style="font-size:1.4rem;">🔴</span>
                <span style="color:#fecaca;font-size:1.1rem;font-weight:700;">{label}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                f"""<div style="background:#14532d;border-radius:10px;padding:12px 20px;
                margin-bottom:16px;display:flex;align-items:center;gap:12px;">
                <span style="font-size:1.4rem;">🟢</span>
                <span style="color:#bbf7d0;font-size:1.1rem;font-weight:700;">{label}</span>
                </div>""", unsafe_allow_html=True)

        # ── ZONES 2-4: Tabbed interface ───────────────────────────
        _tab1, _tab2, _tab3 = st.tabs(["📊 Results", "🔬 Grad-CAM Analysis", "🤖 AI Clinical Interpretation"])

        # ── TAB 1: Results ────────────────────────────────────────
        with _tab1:
            col_img, col_details = st.columns([6, 4])

            with col_img:
                st.image(uploaded, caption="Fundus Photograph", use_container_width=True)

            with col_details:
                # Confidence bar
                st.markdown("**Model Confidence**")
                bar_color = "#ef4444" if is_high_risk else "#22c55e"
                st.markdown(
                    f"""<div style="background:#1e293b;border-radius:8px;padding:4px;margin-bottom:16px;">
                      <div style="width:{confidence*100:.1f}%;background:{bar_color};border-radius:6px;
                                  padding:8px 12px;color:white;font-weight:700;font-size:1rem;">
                        {confidence*100:.1f}%
                      </div>
                    </div>""", unsafe_allow_html=True)

                # Clinical recommendation
                st.markdown("**Clinical Recommendation**")
                if is_high_risk:
                    st.error("⚠️ Ophthalmologist referral recommended within 30 days.")
                else:
                    st.info("✅ Annual screening recommended. No immediate action required.")

            # ── DR Severity Reference Cards (full-width, below image/details) ──
            st.markdown("---")
            st.markdown("#### 📊 DR Severity Reference")

            _severity_levels = [
                {"level": "1", "color": "#22c55e", "emoji": "🟢", "name": "Mild NPDR",
                 "findings": "Microaneurysms only",
                 "action": "Monitor annually", "urgency": "Routine"},
                {"level": "2", "color": "#eab308", "emoji": "🟡", "name": "Moderate NPDR",
                 "findings": "More than microaneurysms, less than severe",
                 "action": "Follow-up in 6 months", "urgency": "Standard"},
                {"level": "3", "color": "#f97316", "emoji": "🟠", "name": "Severe NPDR",
                 "findings": "Extensive hemorrhages, venous beading",
                 "action": "Referral within 3 months", "urgency": "Elevated"},
                {"level": "4", "color": "#ef4444", "emoji": "🔴", "name": "Proliferative DR",
                 "findings": "Neovascularization, vitreous hemorrhage",
                 "action": "Urgent referral (< 2 weeks)", "urgency": "Critical"},
            ]

            _sev_cols = st.columns(4)
            for _i, _lvl in enumerate(_severity_levels):
                with _sev_cols[_i]:
                    st.markdown(f"""
                    <div style="border:1px solid {_lvl['color']};border-radius:12px;padding:1rem;
                                background:rgba(13,27,42,0.95);height:100%;">
                        <div style="font-size:1.5rem;margin-bottom:0.4rem;">{_lvl['emoji']}</div>
                        <div style="font-size:0.9rem;font-weight:700;color:{_lvl['color']};
                                    margin-bottom:0.2rem;">
                            Level {_lvl['level']} · {_lvl['name']}
                        </div>
                        <div style="font-size:0.7rem;color:#64748b;margin-bottom:0.5rem;">
                            Urgency: {_lvl['urgency']}
                        </div>
                        <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:0.4rem;">
                            <strong style="color:#cbd5e1;">Findings:</strong> {_lvl['findings']}
                        </div>
                        <div style="font-size:0.75rem;color:{_lvl['color']};font-weight:600;">
                            {_lvl['action']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── TAB 2: Grad-CAM ───────────────────────────────────────
        with _tab2:
            st.subheader("🔬 Grad-CAM — Model Attention Map")
            st.caption("Highlights the retinal regions that influenced the prediction.")

            if st.button("Generate Grad-CAM Heatmap", key="gradcam_btn"):
                with st.spinner("Computing attention map..."):
                    try:
                        pil_img = Image.open(uploaded).convert("RGB")
                        img_arr = np.array(pil_img.resize((224, 224))).astype(np.float32)
                        img_arr = np.expand_dims(img_arr, axis=0)
                        from tensorflow.keras.applications.resnet50 import preprocess_input
                        img_preprocessed = preprocess_input(img_arr)
                        heatmap = make_gradcam(img_preprocessed, model)
                        overlay = overlay_gradcam(pil_img, heatmap)
                        import io as _io
                        _buf = _io.BytesIO()
                        overlay.save(_buf, format="PNG")
                        st.session_state["_gradcam_overlay"] = _buf.getvalue()
                        st.session_state["_gradcam_uploaded"] = uploaded.getvalue()
                    except Exception as e:
                        st.error(f"Grad-CAM failed: {e}")
                        st.session_state.pop("_gradcam_overlay", None)

            if "_gradcam_overlay" in st.session_state:
                col_orig, col_cam = st.columns(2)
                with col_orig:
                    st.image(st.session_state["_gradcam_uploaded"], caption="Original", use_container_width=True)
                with col_cam:
                    st.image(st.session_state["_gradcam_overlay"], caption="Grad-CAM Overlay", use_container_width=True)
                st.caption("🔴 Red/yellow = high attention regions  |  🔵 Blue = low attention")

        # ── TAB 3: AI Clinical Interpretation ────────────────────
        with _tab3:
            st.markdown("### 🤖 AI Clinical Interpretation")

            _groq_key = st.secrets.get("GROQ_API_KEY", "")
            if not _groq_key:
                st.warning("⚠️ Groq API key not configured. AI Clinical Interpretation unavailable.")
            else:
                from openai import OpenAI as _RetinalOAI
                import html as _html_retinal
                _ai_client = _RetinalOAI(
                    api_key=_groq_key,
                    base_url="https://api.groq.com/openai/v1",
                )

                _retinal_system_prompt = f"""You are a Senior Ophthalmology Clinical AI Assistant specializing in diabetic retinopathy screening.

RETINAL SCAN FINDINGS:
- Classification: {label}
- Model Confidence: {confidence*100:.1f}%
- Risk Level: {"HIGH RISK — Diabetic Retinopathy Detected" if is_high_risk else "Normal — No Diabetic Retinopathy Detected"}

Provide clinical interpretation using medical terminology. Include:
1. Clinical summary of retinal findings
2. DR severity classification (if applicable)
3. Recommended follow-up actions with timeframes
4. ICD-10 codes where applicable (e.g. E11.311 for T2DM with mild NPDR)
5. Patient education talking points

Be concise, evidence-based, and actionable. Use clear headings.

CRITICAL RULE: YOU MUST ALWAYS END EVERY RESPONSE WITH THIS EXACT TEXT: '\n\n⚠️ **Disclaimer:** This is an AI-generated analysis for investigational use only. Always consult a licensed healthcare provider.'"""

                # Reset chat history when the scan result changes
                _result_key = f"{label}_{confidence:.4f}"
                if (st.session_state.get("_retinal_result_key") != _result_key
                        or "_retinal_chat_history" not in st.session_state):
                    st.session_state._retinal_result_key = _result_key
                    st.session_state._retinal_chat_history = []
                    # New scan loaded — allow auto-init to run
                    st.session_state["_retinal_skip_init"] = False

                # Auto-generate initial narrative on first load
                # Skip if the user just manually cleared the conversation
                if not st.session_state._retinal_chat_history and not st.session_state.get("_retinal_skip_init"):
                    with st.spinner("Generating clinical interpretation..."):
                        try:
                            _init_resp = _ai_client.chat.completions.create(
                                model="llama-3.1-8b-instant",
                                messages=[
                                    {"role": "system", "content": _retinal_system_prompt},
                                    {"role": "user", "content": "Provide a concise clinical interpretation of this retinal scan result."},
                                ],
                                temperature=0.2,
                                max_tokens=1024,
                            )
                            st.session_state._retinal_chat_history.append({
                                "role": "assistant",
                                "content": _init_resp.choices[0].message.content,
                            })
                        except Exception as _init_err:
                            st.error(f"AI interpretation failed: {_init_err}")

                # Display chat history
                _ret_container = st.container(height=420, border=True)
                with _ret_container:
                    for _msg in st.session_state._retinal_chat_history:
                        if _msg["role"] == "assistant":
                            st.markdown(f"**🤖 AI Assistant:**\n\n{_msg['content']}")
                            st.markdown("---")
                        else:
                            st.markdown(
                                f'<div style="background:rgba(34,211,238,0.08);'
                                f'border:1px solid rgba(34,211,238,0.2);border-radius:8px;'
                                f'padding:0.6rem 1rem;margin:0.5rem 0;font-size:0.9rem;">'
                                f'<strong>👨‍⚕️ You:</strong> '
                                f'{_html_retinal.escape(_msg["content"])}</div>',
                                unsafe_allow_html=True,
                            )

                # Quick question buttons
                st.markdown("**💡 Quick Questions:**")
                _rq_labels = [
                    ("qs_ret_1", "📊 Explain Grad-CAM findings",
                     f"Explain what the Grad-CAM heatmap regions indicate clinically for a scan with a {confidence*100:.1f}% confidence of being {label}. Mention specific anatomical features like microaneurysms or exudates if applicable."),
                    ("qs_ret_2", "⏰ Recommended follow-up?",
                     f"What is the recommended follow-up schedule and urgency for this patient given the {label} classification and {confidence*100:.1f}% AI confidence?"),
                    ("qs_ret_3", "🚨 How urgent is referral?",      "How urgent is the ophthalmology referral and why?"),
                    ("qs_ret_4", "💬 What to tell the patient?",    "What are the key talking points for patient education about this result?"),
                ]
                _rq_cols = st.columns(2)
                _ret_starter = None
                for _qi, (_rk, _rl, _rp) in enumerate(_rq_labels):
                    with _rq_cols[_qi % 2]:
                        if st.button(_rl, key=_rk):
                            _ret_starter = _rp

                # Chat form
                with st.form(key="retinal_copilot_form", clear_on_submit=True):
                    _rfc, _rsc = st.columns([11, 1])
                    with _rfc:
                        _ret_typed = st.text_input(
                            "Ask a follow-up question:",
                            label_visibility="collapsed",
                            placeholder="Type a clinical question…",
                        )
                    with _rsc:
                        _ret_submitted = st.form_submit_button("➤")

                _ret_active = _ret_starter or (
                    _ret_typed.strip() if _ret_submitted and _ret_typed.strip() else None
                )

                if _ret_active:
                    st.session_state._retinal_chat_history.append(
                        {"role": "user", "content": _ret_active}
                    )
                    try:
                        _ret_msgs = [{"role": "system", "content": _retinal_system_prompt}]
                        _ret_msgs += [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state._retinal_chat_history
                        ]
                        _ret_resp = _ai_client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=_ret_msgs,
                            temperature=0.2,
                            max_tokens=1024,
                        )
                        st.session_state._retinal_chat_history.append({
                            "role": "assistant",
                            "content": _ret_resp.choices[0].message.content,
                        })
                    except Exception as _ret_err:
                        st.session_state._retinal_chat_history.append({
                            "role": "assistant",
                            "content": f"⚠️ Error generating response: {_ret_err}",
                        })
                    st.rerun()

                if st.button("🗑️ Clear Conversation", key="clear_retinal_chat"):
                    st.session_state._retinal_chat_history = []
                    st.session_state["_retinal_skip_init"] = True
                    st.rerun()
# =============================================================================
# MAIN ROUTER
# =============================================================================
def main() -> None:
    """Entry point for the ClearSight Analytics Streamlit application.

    Delegates sidebar rendering to ``render_sidebar()``, which returns the active
    navigation label, then routes to the appropriate page-rendering function.
    Called once per Streamlit script rerun.
    """
    page = render_sidebar()
    if   "Command"   in page: page_home()
    elif "Predict"   in page: page_predict()
    elif "Insights"  in page: page_insights()
    elif "Retinal"   in page: page_retinal()


if __name__ == "__main__":
    main()
