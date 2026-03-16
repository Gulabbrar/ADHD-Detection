import os, sys
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import streamlit as st
import time
import datetime
import io
import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from database import get_connection, save_assessment, get_patient_history, init_db
from datetime import datetime as _dt
from auth import login_user, register_user

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wellness & Focus Platform",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM — CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Variables ─────────────────────────────────────────────────────────── */
:root {
  --navy:          #0c1f3f;
  --navy-mid:      #1e3a6e;
  --blue:          #2563eb;
  --blue-hover:    #1d4ed8;
  --blue-light:    #dbeafe;
  --blue-xlight:   #eff6ff;
  --teal:          #0d9488;
  --teal-light:    #ccfbf1;
  --green:         #16a34a;
  --green-light:   #dcfce7;
  --amber:         #d97706;
  --amber-light:   #fef3c7;
  --red:           #dc2626;
  --red-light:     #fee2e2;
  --purple:        #7c3aed;
  --purple-light:  #ede9fe;
  --cyan:          #0891b2;
  --cyan-light:    #cffafe;
  --slate:         #475569;
  --slate-light:   #94a3b8;
  --bg:            #f0f4f8;
  --surface:       #ffffff;
  --surface-2:     #f8fafc;
  --text-primary:  #0f172a;
  --text-secondary:#1e293b;
  --text-muted:    #64748b;
  --border:        #e2e8f0;
  --border-strong: #94a3b8;
  --radius-xs:     4px;
  --radius-sm:     7px;
  --radius:        11px;
  --radius-lg:     15px;
  --shadow-xs:     0 1px 3px rgba(15,23,42,0.08), 0 1px 2px rgba(15,23,42,0.04);
  --shadow-sm:     0 2px 8px rgba(15,23,42,0.10);
  --shadow:        0 4px 20px rgba(15,23,42,0.11);
  --shadow-lg:     0 12px 40px rgba(15,23,42,0.15);
  --transition:    all 0.18s ease;
}

/* ── Reset & Base ───────────────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  -webkit-font-smoothing: antialiased;
}
.main { background-color: var(--bg) !important; }
.main .block-container {
  padding: 2rem 2.5rem 3rem !important;
  max-width: 1280px !important;
}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface-2); }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 50px; }
::-webkit-scrollbar-thumb:hover { background: var(--slate); }

h1 { color: var(--navy) !important; font-weight: 800 !important; letter-spacing: -0.5px; }
h2 { color: var(--navy) !important; font-weight: 700 !important; letter-spacing: -0.3px; }
h3 { color: var(--text-primary) !important; font-weight: 600 !important; }
h4 { color: var(--text-secondary) !important; font-weight: 600 !important; }
p  { color: var(--text-secondary); line-height: 1.6; }

hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1rem 0 !important;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0c1f3f 0%, #0f2d5a 100%) !important;
  border-right: none !important;
  box-shadow: 3px 0 20px rgba(0,0,0,0.25) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

.sidebar-brand {
  padding: 1.6rem 1.25rem 1.1rem;
  border-bottom: 1px solid rgba(255,255,255,0.10);
  margin-bottom: 0.5rem;
  background: linear-gradient(135deg, rgba(37,99,235,0.25) 0%, transparent 100%);
}
.sidebar-brand-name {
  font-size: 1rem;
  font-weight: 800;
  color: #ffffff;
  letter-spacing: -0.3px;
  margin-bottom: 0.2rem;
}
.sidebar-brand-sub {
  font-size: 0.68rem;
  color: rgba(255,255,255,0.45);
  text-transform: uppercase;
  letter-spacing: 1px;
}
.sidebar-section-label {
  font-size: 0.62rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 1.4px;
  color: rgba(255,255,255,0.35);
  padding: 1rem 1.25rem 0.4rem;
}
.sidebar-stats {
  padding: 1rem 1.25rem;
  border-top: 1px solid rgba(255,255,255,0.10);
  margin-top: 0.5rem;
  background: rgba(0,0,0,0.15);
}
.sidebar-stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.32rem 0;
  font-size: 0.78rem;
}
.sidebar-stat-label { color: rgba(255,255,255,0.45); }
.sidebar-stat-value {
  color: #ffffff;
  font-weight: 700;
  background: rgba(37,99,235,0.3);
  padding: 0.1rem 0.5rem;
  border-radius: 50px;
  font-size: 0.72rem;
}
.sidebar-date {
  font-size: 0.75rem;
  color: rgba(255,255,255,0.4);
  padding: 0 1.25rem 0.75rem;
  font-weight: 500;
}

/* force all sidebar text/labels to be clearly visible at all times */
section[data-testid="stSidebar"] * {
  color: rgba(255,255,255,0.88) !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:first-child {
  color: rgba(255,255,255,0.42) !important;
  font-size: 0.62rem !important;
  font-weight: 800 !important;
  text-transform: uppercase !important;
  letter-spacing: 1.2px !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
  gap: 2px !important;
  padding: 0 0.5rem !important;
}
section[data-testid="stSidebar"] .stRadio label {
  padding: 0.58rem 0.85rem !important;
  border-radius: var(--radius-sm) !important;
  border-left: 3px solid transparent !important;
  transition: var(--transition) !important;
  display: block !important;
  cursor: pointer !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(255,255,255,0.09) !important;
  border-left-color: rgba(96,165,250,0.5) !important;
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
  background: rgba(37,99,235,0.38) !important;
  border-left-color: #60a5fa !important;
}
/* hide the actual radio circle dots — they look bad on dark bg */
section[data-testid="stSidebar"] .stRadio input[type="radio"] {
  display: none !important;
}

/* ── Page Header ────────────────────────────────────────────────────────── */
.page-header {
  background: linear-gradient(135deg, #ffffff 60%, #f0f6ff 100%);
  border: 1px solid var(--border);
  border-left: 5px solid var(--blue);
  border-radius: var(--radius);
  padding: 1.35rem 1.75rem;
  margin-bottom: 1.75rem;
  box-shadow: var(--shadow-sm);
}
.page-header-eyebrow {
  font-size: 0.67rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: var(--blue);
  margin-bottom: 0.25rem;
}
.page-header h1 {
  font-size: 1.55rem !important;
  margin: 0 0 0.25rem !important;
  color: var(--navy) !important;
}
.page-header p {
  font-size: 0.85rem;
  color: var(--text-muted);
  margin: 0;
}
.page-header-meta {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-top: 0.8rem;
  padding-top: 0.8rem;
  border-top: 1px solid var(--border);
}
.meta-badge {
  display: inline-flex;
  align-items: center;
  background: var(--blue-light);
  color: #1e40af;
  font-size: 0.67rem;
  font-weight: 700;
  padding: 0.28rem 0.75rem;
  border-radius: 50px;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  border: 1px solid #bfdbfe;
}
.meta-badge.green { background: var(--green-light);  color: #14532d; border-color: #86efac; }
.meta-badge.amber { background: var(--amber-light);  color: #78350f; border-color: #fcd34d; }
.meta-badge.red   { background: var(--red-light);    color: #7f1d1d; border-color: #fca5a5; }
.meta-badge.navy  { background: rgba(12,31,63,0.07); color: var(--navy); border-color: rgba(12,31,63,0.15); }

/* ── Metric / Stat Cards ────────────────────────────────────────────────── */
.kpi-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem 1.3rem;
  box-shadow: var(--shadow-xs);
  transition: var(--transition);
  position: relative;
  overflow: hidden;
}
.kpi-card::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 4px;
  border-radius: var(--radius) var(--radius) 0 0;
}
.kpi-card:hover { box-shadow: var(--shadow); transform: translateY(-1px); }
.kpi-label {
  font-size: 0.67rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--text-muted);
  margin-bottom: 0.45rem;
}
.kpi-value {
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: -1.5px;
  line-height: 1;
}
.kpi-sub {
  font-size: 0.73rem;
  color: var(--text-muted);
  margin-top: 0.25rem;
  font-weight: 500;
}
.kpi-card.accent-blue  { border-top: 4px solid var(--blue);   }
.kpi-card.accent-blue  .kpi-value { color: var(--blue); }
.kpi-card.accent-teal  { border-top: 4px solid var(--teal);   }
.kpi-card.accent-teal  .kpi-value { color: var(--teal); }
.kpi-card.accent-green { border-top: 4px solid var(--green);  }
.kpi-card.accent-green .kpi-value { color: var(--green); }
.kpi-card.accent-amber { border-top: 4px solid var(--amber);  }
.kpi-card.accent-amber .kpi-value { color: var(--amber); }
.kpi-card.accent-red   { border-top: 4px solid var(--red);    }
.kpi-card.accent-red   .kpi-value { color: var(--red); }
.kpi-card.accent-purple{ border-top: 4px solid var(--purple); }
.kpi-card.accent-purple .kpi-value { color: var(--purple); }
.kpi-card.accent-slate { border-top: 4px solid var(--slate);  }
.kpi-card.accent-slate .kpi-value { color: var(--slate); }

/* ── Section Label ──────────────────────────────────────────────────────── */
.section-label {
  font-size: 0.67rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: var(--slate);
  margin: 1.6rem 0 0.8rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── Cards ──────────────────────────────────────────────────────────────── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.5rem;
  box-shadow: var(--shadow-xs);
}
.card + .card { margin-top: 0.75rem; }

/* ── Feature Grid Cards ─────────────────────────────────────────────────── */
.feature-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.5rem 1.25rem;
  box-shadow: var(--shadow-xs);
  transition: var(--transition);
  height: 100%;
  position: relative;
  overflow: hidden;
}
.feature-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
}
.feature-card:nth-child(1)::before { background: var(--blue); }
.feature-card:nth-child(2)::before { background: var(--teal); }
.feature-card:nth-child(3)::before { background: var(--purple); }
.feature-card:nth-child(4)::before { background: var(--amber); }
.feature-card:nth-child(5)::before { background: var(--green); }
.feature-card:hover {
  box-shadow: var(--shadow);
  border-color: var(--border-strong);
  transform: translateY(-3px);
}
.feature-card-title {
  font-size: 0.9rem;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 0.4rem;
}
.feature-card-desc {
  font-size: 0.8rem;
  color: var(--text-muted);
  line-height: 1.6;
}

/* ── Timer ──────────────────────────────────────────────────────────────── */
.timer-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem 0;
}
.timer-ring {
  width: 230px; height: 230px;
  border-radius: 50%;
  border: 8px solid var(--blue-light);
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  background: radial-gradient(circle at 35% 35%, #f0f6ff 0%, #ffffff 60%);
  box-shadow: 0 0 0 3px var(--border), var(--shadow-lg);
  position: relative;
  transition: border-color 0.4s ease;
}
.timer-ring.mode-work   { border-color: var(--blue-light);  box-shadow: 0 0 0 3px #bfdbfe, 0 0 40px rgba(37,99,235,0.15), var(--shadow-lg); }
.timer-ring.mode-short  { border-color: var(--teal-light);  box-shadow: 0 0 0 3px #99f6e4, 0 0 40px rgba(13,148,136,0.15), var(--shadow-lg); }
.timer-ring.mode-long   { border-color: var(--amber-light); box-shadow: 0 0 0 3px #fde68a, 0 0 40px rgba(217,119,6,0.15),  var(--shadow-lg); }
.timer-ring::before {
  content: '';
  position: absolute;
  inset: 10px;
  border-radius: 50%;
  background: var(--surface);
}
.timer-digits {
  position: relative;
  font-size: 3.4rem;
  font-weight: 800;
  color: var(--navy);
  letter-spacing: -3px;
  font-variant-numeric: tabular-nums;
  text-shadow: 0 1px 2px rgba(0,0,0,0.06);
}
.timer-mode-label {
  position: relative;
  font-size: 0.68rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  margin-top: 0.25rem;
}
.timer-mode-label.mode-work  { color: var(--blue); }
.timer-mode-label.mode-short { color: var(--teal); }
.timer-mode-label.mode-long  { color: var(--amber); }

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
  background: var(--blue) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.5rem 1.4rem !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.84rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.1px !important;
  box-shadow: 0 1px 3px rgba(29,78,216,0.25) !important;
  transition: var(--transition) !important;
}
.stButton > button:hover {
  background: var(--blue-hover) !important;
  box-shadow: 0 4px 14px rgba(29,78,216,0.35) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stDownloadButton > button {
  background: var(--teal) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  box-shadow: 0 1px 3px rgba(15,118,110,0.25) !important;
}
.stDownloadButton > button:hover {
  background: #0d6b63 !important;
  box-shadow: 0 4px 14px rgba(15,118,110,0.35) !important;
}

/* ── Inputs ─────────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.875rem !important;
  color: var(--text-primary) !important;
  background: var(--surface) !important;
  transition: var(--transition) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stNumberInput > div > div > input:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(29,78,216,0.10) !important;
}
.stSelectbox > div > div {
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  background: var(--surface) !important;
}
.stRadio label {
  font-size: 0.85rem !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--text-secondary) !important;
}

/* ── Progress bar ───────────────────────────────────────────────────────── */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--blue) 0%, #60a5fa 100%) !important;
  border-radius: 50px !important;
}
.stProgress > div > div {
  border-radius: 50px !important;
  background: var(--blue-light) !important;
  height: 7px !important;
  box-shadow: inset 0 1px 2px rgba(0,0,0,0.06) !important;
}

/* ── Task list ──────────────────────────────────────────────────────────── */
.task-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.65rem 1rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  margin-bottom: 0.4rem;
  transition: var(--transition);
}
.task-row:hover { border-color: var(--border-strong); box-shadow: var(--shadow-xs); }
.task-row.done  { opacity: 0.5; }
.task-text-done { text-decoration: line-through; color: var(--text-muted); }
.priority-tag {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: var(--radius-xs);
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  flex-shrink: 0;
}
.tag-high   { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.tag-medium { background: #fef3c7; color: #78350f; border: 1px solid #fcd34d; }
.tag-low    { background: #dcfce7; color: #14532d; border: 1px solid #86efac; }

/* ── Mood ────────────────────────────────────────────────────────────────── */
.mood-option-card {
  background: var(--surface);
  border: 1.5px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 0.5rem;
  text-align: center;
  cursor: pointer;
  transition: var(--transition);
}
.mood-option-card:hover {
  border-color: var(--blue);
  box-shadow: var(--shadow-sm);
}
.mood-option-card.selected {
  border-color: var(--blue);
  background: var(--blue-light);
}
.mood-label-text {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-muted);
  margin-top: 0.3rem;
}

/* ── Sentiment / Voice ───────────────────────────────────────────────────── */
.voice-header {
  background: var(--navy);
  border-radius: var(--radius);
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
}
.voice-header-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: #ffffff;
  margin-bottom: 0.2rem;
}
.voice-header-sub {
  font-size: 0.78rem;
  color: rgba(255,255,255,0.5);
}
.sentiment-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem;
  box-shadow: var(--shadow-xs);
}
.sentiment-score-large {
  font-size: 2.6rem;
  font-weight: 800;
  letter-spacing: -1px;
  line-height: 1;
}
.sentiment-pos { color: var(--success); }
.sentiment-neu { color: var(--blue); }
.sentiment-neg { color: var(--danger); }
.sentiment-descriptor {
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-top: 0.3rem;
}
.bar-row {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.45rem;
}
.bar-label { font-size:0.75rem; font-weight:600; color:var(--text-muted); width:62px; flex-shrink:0; }
.bar-track { flex:1; background:var(--border); border-radius:50px; height:8px; overflow:hidden; }
.bar-fill  { height:8px; border-radius:50px; }
.bar-pct   { font-size:0.72rem; color:var(--text-muted); width:35px; text-align:right; flex-shrink:0; }
.suggested-mood-card {
  background: var(--blue-light);
  border: 1.5px solid #bfdbfe;
  border-radius: var(--radius);
  padding: 1rem 1.25rem;
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-top: 0.75rem;
}
.suggested-mood-label {
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: var(--blue);
}
.suggested-mood-value {
  font-size: 1rem;
  font-weight: 700;
  color: var(--navy);
}

/* ── Emotion Detection ───────────────────────────────────────────────────── */
.detection-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem;
  box-shadow: var(--shadow-xs);
}
.dominant-emotion {
  font-size: 1.75rem;
  font-weight: 800;
  color: var(--navy);
  text-transform: capitalize;
  letter-spacing: -0.5px;
}
.confidence-text {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-top: 0.2rem;
}
.emotion-bar-label {
  font-size:0.75rem;
  font-weight:600;
  color:var(--text-muted);
  text-transform:capitalize;
  width:62px;
  flex-shrink:0;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 3px;
  gap: 2px;
}
.stTabs [data-baseweb="tab"] {
  border-radius: var(--radius-xs) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  color: var(--text-muted) !important;
  background: transparent !important;
  border: none !important;
  padding: 0.4rem 0.85rem !important;
}
.stTabs [aria-selected="true"] {
  background: var(--surface) !important;
  color: var(--navy) !important;
  font-weight: 600 !important;
  box-shadow: var(--shadow-xs) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-top: none;
  border-radius: 0 0 var(--radius-sm) var(--radius-sm);
  padding: 1.25rem 1rem;
}

/* ── Clinical / ADHD ─────────────────────────────────────────────────────── */
.clinical-header {
  background: var(--navy);
  padding: 1.5rem 2rem;
  border-radius: var(--radius);
  margin-bottom: 1.75rem;
  box-shadow: var(--shadow);
}
.clinical-header-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: #ffffff;
  letter-spacing: -0.3px;
  margin-bottom: 0.2rem;
}
.clinical-header-sub {
  font-size: 0.8rem;
  color: rgba(255,255,255,0.5);
}
.clinical-badge {
  display: inline-block;
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.2);
  color: rgba(255,255,255,0.85);
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  padding: 0.25rem 0.75rem;
  border-radius: 50px;
  margin-top: 0.75rem;
  display: inline-block;
}
.question-row {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--blue);
  border-radius: var(--radius-sm);
  padding: 0.8rem 1rem;
  margin-bottom: 0.4rem;
}
.question-index {
  display: inline-block;
  background: var(--blue);
  color: white;
  font-size: 0.62rem;
  font-weight: 700;
  padding: 0.1rem 0.45rem;
  border-radius: 50px;
  margin-right: 0.5rem;
  vertical-align: middle;
}
.question-text {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-primary);
  line-height: 1.5;
}
.severity-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.45rem 1.1rem;
  border-radius: 50px;
  font-weight: 700;
  font-size: 0.875rem;
}
.sev-severe   { background: #fee2e2; color: #7f1d1d; border: 1.5px solid #fca5a5; }
.sev-moderate { background: #fef3c7; color: #78350f; border: 1.5px solid #fcd34d; }
.sev-mild     { background: #dcfce7; color: #14532d; border: 1.5px solid #86efac; }
.result-banner {
  background: var(--navy);
  padding: 1.25rem 1.75rem;
  border-radius: var(--radius);
  margin: 1.5rem 0 1.25rem;
  box-shadow: var(--shadow);
}
.result-banner-title {
  font-size: 1rem;
  font-weight: 700;
  color: #ffffff;
  margin-bottom: 0.2rem;
}
.result-banner-sub {
  font-size: 0.78rem;
  color: rgba(255,255,255,0.5);
}
.prediction-card {
  background: var(--surface);
  border: 1.5px solid var(--blue);
  border-radius: var(--radius);
  padding: 1.25rem 1.5rem;
  text-align: center;
  box-shadow: var(--shadow-xs);
}
.prediction-label {
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--text-muted);
  margin-bottom: 0.4rem;
}
.prediction-value {
  font-size: 1.25rem;
  font-weight: 800;
  color: var(--navy);
}
.login-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  box-shadow: var(--shadow-lg);
}
.login-card-header {
  background: var(--navy);
  padding: 2rem;
  text-align: center;
}
.login-card-header-title {
  font-size: 1.15rem;
  font-weight: 700;
  color: #ffffff;
  margin-bottom: 0.25rem;
}
.login-card-header-sub {
  font-size: 0.78rem;
  color: rgba(255,255,255,0.5);
}
.login-card-body {
  padding: 1.75rem;
}

/* ── Misc ────────────────────────────────────────────────────────────────── */
.stAlert { border-radius: var(--radius-sm) !important; }
.stDataFrame { border-radius: var(--radius) !important; box-shadow: var(--shadow-xs) !important; }
[data-testid="metric-container"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-top: 3px solid var(--blue) !important;
  border-radius: var(--radius) !important;
  padding: 1rem 1.25rem !important;
  box-shadow: var(--shadow-xs) !important;
}
[data-testid="stMetricLabel"] {
  font-size: 0.68rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.8px !important;
  color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.8rem !important;
  font-weight: 800 !important;
  color: var(--navy) !important;
  letter-spacing: -1px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ADHD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_adhd_model():
    model         = joblib.load("adhd_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, label_encoder

@st.cache_data
def load_questions():
    df = pd.read_excel("questions.xlsx")
    return df["question_text"].tolist(), df["scale_type"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "tasks":                  [],
        "task_id_counter":        1,
        "timer_running":          False,
        "timer_start":            None,
        "timer_duration":         25 * 60,
        "timer_mode":             "Work Session",
        "sessions_done":          0,
        "timer_paused_remaining": None,
        "mood_logs":              [],
        "selected_mood":          None,
        "voice_transcript":       "",
        "adhd_user":              None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()
init_db()  # ensure new DB columns exist


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MOODS = [
    {"symbol": "A", "label": "Great",   "score": 5, "color": "#15803d"},
    {"symbol": "B", "label": "Good",    "score": 4, "color": "#0d9488"},
    {"symbol": "C", "label": "Neutral", "score": 3, "color": "#1d4ed8"},
    {"symbol": "D", "label": "Low",     "score": 2, "color": "#b45309"},
    {"symbol": "E", "label": "Sad",     "score": 1, "color": "#b91c1c"},
]
MOOD_DISPLAY = {m["label"]: m for m in MOODS}

EMOTION_TO_MOOD = {
    "happy":    {"label": "Great",   "score": 5, "color": "#15803d"},
    "surprise": {"label": "Good",    "score": 4, "color": "#0d9488"},
    "neutral":  {"label": "Neutral", "score": 3, "color": "#1d4ed8"},
    "fear":     {"label": "Low",     "score": 2, "color": "#b45309"},
    "sad":      {"label": "Low",     "score": 2, "color": "#b45309"},
    "angry":    {"label": "Sad",     "score": 1, "color": "#b91c1c"},
    "disgust":  {"label": "Sad",     "score": 1, "color": "#b91c1c"},
}
EMOTION_COLORS = {
    "happy":    "#059669",
    "surprise": "#0891b2",
    "neutral":  "#2563eb",
    "fear":     "#d97706",
    "sad":      "#7c3aed",
    "angry":    "#dc2626",
    "disgust":  "#db2777",
}

SENTIMENT_TO_MOOD = [
    ( 0.50,  1.00, {"label": "Great",   "score": 5}),
    ( 0.10,  0.50, {"label": "Good",    "score": 4}),
    (-0.10,  0.10, {"label": "Neutral", "score": 3}),
    (-0.50, -0.10, {"label": "Low",     "score": 2}),
    (-1.00, -0.50, {"label": "Sad",     "score": 1}),
]

CHART_COLORS = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]

_vader = SentimentIntensityAnalyzer()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def format_seconds(secs):
    secs = max(0, int(secs))
    m, s = divmod(secs, 60)
    return f"{m:02d}:{s:02d}"


def page_header(eyebrow, title, subtitle, badge=None, badge_style=""):
    badge_html = f'<span class="meta-badge {badge_style}">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="page-header">
      <div class="page-header-eyebrow">{eyebrow}</div>
      <h1>{title}</h1>
      <p>{subtitle}</p>
      {"<div class='page-header-meta'>" + badge_html + "</div>" if badge else ""}
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label, value, sub="", accent="accent-blue"):
    st.markdown(f"""
    <div class="kpi-card {accent}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)


def section_label(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def analyze_sentiment(text):
    scores = _vader.polarity_scores(text)
    c = scores["compound"]
    for lo, hi, mood in SENTIMENT_TO_MOOD:
        if lo <= c <= hi:
            return scores, mood
    return scores, SENTIMENT_TO_MOOD[2][2]


def transcribe_audio(audio_bytes):
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)


def detect_emotion(image_bytes):
    from deepface import DeepFace
    img       = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)
    try:
        result = DeepFace.analyze(img_array, actions=["emotion"],
                                  enforce_detection=True, silent=True)
        return result[0]
    except Exception as e:
        if "face" in str(e).lower() or "detect" in str(e).lower():
            raise ValueError("No face detected. Ensure your face is clearly visible, "
                             "well-lit, and centred in the frame.")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
tasks_done  = sum(1 for t in st.session_state.tasks if t["done"])
tasks_total = len(st.session_state.tasks)

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div class="sidebar-brand-name">Wellness &amp; Focus</div>
      <div class="sidebar-brand-sub">Clinical Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Dashboard", "Focus Timer", "Daily Planner",
         "Mood Tracker", "Tips & Strategies", "ADHD Assessment"],
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <div class="sidebar-date">{datetime.date.today().strftime('%A, %B %d, %Y')}</div>
    <div class="sidebar-stats">
      <div class="sidebar-stat-row">
        <span class="sidebar-stat-label">Tasks completed</span>
        <span class="sidebar-stat-value">{tasks_done} / {tasks_total}</span>
      </div>
      <div class="sidebar-stat-row">
        <span class="sidebar-stat-label">Focus sessions</span>
        <span class="sidebar-stat-value">{st.session_state.sessions_done}</span>
      </div>
      <div class="sidebar-stat-row">
        <span class="sidebar-stat-label">Mood entries</span>
        <span class="sidebar-stat-value">{len(st.session_state.mood_logs)}</span>
      </div>
      <div class="sidebar-stat-row">
        <span class="sidebar-stat-label">ADHD session</span>
        <span class="sidebar-stat-value">{"Active" if st.session_state.adhd_user else "Signed out"}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def show_home():
    page_header("Overview", "Dashboard",
                "Your daily summary across focus, tasks, mood, and assessments.")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Focus Sessions", st.session_state.sessions_done,
                      f"{st.session_state.sessions_done * 25} min total", "accent-blue")
    with c2: kpi_card("Tasks Completed", tasks_done,
                      f"of {tasks_total} total", "accent-teal")
    with c3: kpi_card("Tasks Pending",   tasks_total - tasks_done,
                      "remaining today", "accent-amber")
    with c4: kpi_card("Mood Entries",    len(st.session_state.mood_logs),
                      "logged sessions", "accent-slate")

    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Platform Modules")

    modules = [
        ("Focus Timer",     "Pomodoro-based timer with configurable work and break intervals to maintain deep focus."),
        ("Daily Planner",   "Structured task management with priority levels, due dates, and completion tracking."),
        ("Mood Tracker",    "Multi-modal mood logging via camera expression analysis, voice sentiment, or manual input."),
        ("Tips & Strategies","Evidence-based cognitive, productivity, and well-being strategies curated for daily use."),
        ("ADHD Assessment", "Vanderbilt clinical questionnaire with machine-learning ADHD classification and PDF reports."),
    ]
    cols = st.columns(len(modules))
    for col, (title, desc) in zip(cols, modules):
        with col:
            st.markdown(f"""
            <div class="feature-card">
              <div class="feature-card-title">{title}</div>
              <div class="feature-card-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    if st.session_state.mood_logs or [t for t in st.session_state.tasks if not t["done"]]:
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)

        with r1:
            section_label("Latest Mood Entry")
            if st.session_state.mood_logs:
                last = st.session_state.mood_logs[-1]
                m    = MOOD_DISPLAY.get(last["mood"], {})
                st.markdown(f"""
                <div class="card">
                  <div style="font-size:1.1rem;font-weight:700;color:var(--navy);">
                    {last['mood']}
                  </div>
                  <div style="font-size:0.8rem;color:var(--text-muted);margin:0.2rem 0 0.75rem;">
                    {last['date']} &nbsp;·&nbsp; Energy {last['energy']}/10
                  </div>
                  {"<div style='font-size:0.85rem;color:var(--text-secondary);'>" + last['note'] + "</div>" if last['note'] else ""}
                </div>""", unsafe_allow_html=True)
            else:
                st.info("No mood entries yet.")

        with r2:
            section_label("Pending Tasks")
            pending = [t for t in st.session_state.tasks if not t["done"]][:5]
            if pending:
                for t in pending:
                    tc = {"High": "tag-high", "Medium": "tag-medium", "Low": "tag-low"}.get(t["priority"], "tag-medium")
                    st.markdown(f"""
                    <div class="task-row">
                      <span class="priority-tag {tc}">{t['priority']}</span>
                      <span style="font-size:0.875rem;color:var(--text-primary);">{t['text']}</span>
                      {"<span style='margin-left:auto;font-size:0.72rem;color:var(--text-muted);'>Due " + t['due'] + "</span>" if t['due'] else ""}
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("All tasks completed.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FOCUS TIMER
# ─────────────────────────────────────────────────────────────────────────────
def show_focus_timer():
    page_header("Productivity", "Focus Timer",
                "Use structured work intervals with scheduled breaks to sustain concentration.")

    # Mode buttons
    m1, m2, m3, _ = st.columns([1, 1, 1, 2])
    with m1:
        if st.button("Work Session  —  25 min", use_container_width=True):
            st.session_state.update(timer_mode="Work Session", timer_duration=25*60,
                                    timer_running=False, timer_paused_remaining=None)
    with m2:
        if st.button("Short Break  —  5 min", use_container_width=True):
            st.session_state.update(timer_mode="Short Break", timer_duration=5*60,
                                    timer_running=False, timer_paused_remaining=None)
    with m3:
        if st.button("Long Break  —  15 min", use_container_width=True):
            st.session_state.update(timer_mode="Long Break", timer_duration=15*60,
                                    timer_running=False, timer_paused_remaining=None)

    st.markdown("<br>", unsafe_allow_html=True)

    # Calculate remaining
    if st.session_state.timer_running and st.session_state.timer_start:
        elapsed   = time.time() - st.session_state.timer_start
        remaining = st.session_state.timer_duration - elapsed
        if remaining <= 0:
            remaining = 0
            st.session_state.timer_running = False
            if st.session_state.timer_mode == "Work Session":
                st.session_state.sessions_done += 1
            st.balloons()
    elif st.session_state.timer_paused_remaining is not None:
        remaining = st.session_state.timer_paused_remaining
    else:
        remaining = st.session_state.timer_duration

    progress = 1.0 - (remaining / st.session_state.timer_duration) if st.session_state.timer_duration else 0

    # Timer face
    mode_css  = {"Work Session": "mode-work", "Short Break": "mode-short", "Long Break": "mode-long"}
    ring_cls  = mode_css.get(st.session_state.timer_mode, "mode-work")
    _, tc, _ = st.columns([1, 1, 1])
    with tc:
        st.markdown(f"""
        <div class="timer-wrap">
          <div class="timer-ring {ring_cls}">
            <div class="timer-digits">{format_seconds(remaining)}</div>
            <div class="timer-mode-label {ring_cls}">{st.session_state.timer_mode}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.progress(progress)
    st.markdown("<br>", unsafe_allow_html=True)

    # Controls
    _, b1, b2, b3, _ = st.columns([1, 0.8, 0.8, 0.8, 1])
    with b1:
        if not st.session_state.timer_running:
            if st.button("Start", use_container_width=True):
                if st.session_state.timer_paused_remaining is not None:
                    st.session_state.timer_duration = int(st.session_state.timer_paused_remaining)
                    st.session_state.timer_paused_remaining = None
                st.session_state.timer_start   = time.time()
                st.session_state.timer_running = True
                st.rerun()
        else:
            if st.button("Pause", use_container_width=True):
                elapsed = time.time() - st.session_state.timer_start
                st.session_state.timer_paused_remaining = max(0, st.session_state.timer_duration - elapsed)
                st.session_state.timer_running = False
                st.rerun()
    with b2:
        if st.button("Reset", use_container_width=True):
            mode_dur = {"Work Session": 25*60, "Short Break": 5*60, "Long Break": 15*60}
            st.session_state.update(timer_running=False, timer_paused_remaining=None,
                                    timer_duration=mode_dur.get(st.session_state.timer_mode, 25*60))
            st.rerun()
    with b3:
        if st.session_state.timer_running:
            if st.button("Refresh", use_container_width=True):
                st.rerun()

    if st.session_state.timer_running and remaining > 0:
        time.sleep(1)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Session Statistics")
    s1, s2, s3 = st.columns(3)
    with s1: kpi_card("Sessions Today",  st.session_state.sessions_done, "", "accent-blue")
    with s2: kpi_card("Focus Minutes",   st.session_state.sessions_done * 25, "", "accent-teal")
    next_long = 4 - (st.session_state.sessions_done % 4) if st.session_state.sessions_done % 4 != 0 else 4
    with s3: kpi_card("Until Long Break", f"{next_long}", f"session{'s' if next_long != 1 else ''}", "accent-slate")

    with st.expander("Custom Duration"):
        custom_min = st.slider("Duration (minutes)", 1, 90, st.session_state.timer_duration // 60)
        if st.button("Apply"):
            st.session_state.update(timer_duration=custom_min*60, timer_running=False,
                                    timer_paused_remaining=None)
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DAILY PLANNER
# ─────────────────────────────────────────────────────────────────────────────
def show_daily_planner():
    page_header("Task Management", "Daily Planner",
                "Organise and prioritise your tasks with real-time completion tracking.")

    with st.expander("Add New Task", expanded=not bool(st.session_state.tasks)):
        with st.form("add_task_form", clear_on_submit=True):
            tc1, tc2, tc3 = st.columns([3, 1, 1])
            with tc1: task_text = st.text_input("Description", placeholder="Describe the task...")
            with tc2: priority  = st.selectbox("Priority", ["High", "Medium", "Low"])
            with tc3: due_date  = st.date_input("Due Date", value=None)
            if st.form_submit_button("Add Task", use_container_width=True) and task_text.strip():
                st.session_state.tasks.append({
                    "id":       st.session_state.task_id_counter,
                    "text":     task_text.strip(),
                    "priority": priority,
                    "due":      str(due_date) if due_date else "",
                    "done":     False,
                    "created":  str(datetime.date.today()),
                })
                st.session_state.task_id_counter += 1
                st.rerun()

    tasks = st.session_state.tasks
    if not tasks:
        st.info("No tasks yet. Add your first task above.")
        return

    f1, f2, _ = st.columns([1, 1, 2])
    with f1: filter_status   = st.selectbox("Status",   ["All", "Pending", "Completed"])
    with f2: filter_priority = st.selectbox("Priority", ["All", "High", "Medium", "Low"])

    filtered = tasks
    if filter_status   == "Pending":   filtered = [t for t in filtered if not t["done"]]
    elif filter_status == "Completed": filtered = [t for t in filtered if t["done"]]
    if filter_priority != "All":       filtered = [t for t in filtered if t["priority"] == filter_priority]

    pord     = {"High": 0, "Medium": 1, "Low": 2}
    filtered = sorted(filtered, key=lambda t: (t["done"], pord.get(t["priority"], 1)))

    done_count = sum(1 for t in tasks if t["done"])
    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(done_count / len(tasks) if tasks else 0)
    st.markdown(f"<p style='font-size:0.8rem;color:var(--text-muted);margin-top:0.3rem;'>"
                f"{done_count} of {len(tasks)} tasks completed</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    ids_to_delete = []
    for task in filtered:
        tc       = {"High": "tag-high", "Medium": "tag-medium", "Low": "tag-low"}.get(task["priority"], "tag-medium")
        done_cls = "done" if task["done"] else ""
        text_cls = "task-text-done" if task["done"] else ""
        cc, ct, cd = st.columns([0.4, 5.5, 0.5])
        with cc:
            checked = st.checkbox("", value=task["done"], key=f"chk_{task['id']}")
            if checked != task["done"]:
                for t in st.session_state.tasks:
                    if t["id"] == task["id"]:
                        t["done"] = checked
                st.rerun()
        with ct:
            st.markdown(f"""
            <div class="task-row {done_cls}">
              <span class="priority-tag {tc}">{task['priority']}</span>
              <span class="{text_cls}" style="font-size:0.875rem;">{task['text']}</span>
              {"<span style='margin-left:auto;font-size:0.72rem;color:var(--text-muted);'>Due " + task['due'] + "</span>" if task['due'] else ""}
            </div>""", unsafe_allow_html=True)
        with cd:
            if st.button("Remove", key=f"del_{task['id']}"):
                ids_to_delete.append(task["id"])

    if ids_to_delete:
        st.session_state.tasks = [t for t in st.session_state.tasks if t["id"] not in ids_to_delete]
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    ba1, ba2, _ = st.columns([1, 1, 3])
    with ba1:
        if st.button("Mark All Complete", use_container_width=True):
            for t in st.session_state.tasks: t["done"] = True
            st.rerun()
    with ba2:
        if st.button("Remove Completed", use_container_width=True):
            st.session_state.tasks = [t for t in st.session_state.tasks if not t["done"]]
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MOOD TRACKER
# ─────────────────────────────────────────────────────────────────────────────
def show_mood_tracker():
    page_header("Emotional Intelligence", "Mood Tracker",
                "Log your mood through expression detection, voice sentiment, or manual selection.")

    # ── Camera Detection ────────────────────────────────────────────────────
    with st.expander("Expression Analysis  —  Camera Detection", expanded=True):
        st.markdown("""
        <div class="voice-header">
          <div class="voice-header-title">Facial Expression Analysis</div>
          <div class="voice-header-sub">
            Capture a photo and the model will identify your dominant emotion
            and map it to a mood category.
          </div>
        </div>
        """, unsafe_allow_html=True)

        camera_photo = st.camera_input("Capture image for expression analysis", key="mood_camera")

        if camera_photo is not None:
            with st.spinner("Running expression analysis..."):
                try:
                    result      = detect_emotion(camera_photo.getvalue())
                    dominant    = result["dominant_emotion"]
                    emotions    = result["emotion"]
                    mapped      = EMOTION_TO_MOOD.get(dominant, {"label": "Neutral", "score": 3, "color": "#1d4ed8"})
                    confidence  = emotions[dominant]

                    section_label("Analysis Result")
                    dr1, dr2 = st.columns([1, 1.3])

                    with dr1:
                        st.markdown(f"""
                        <div class="detection-panel">
                          <div class="dominant-emotion">{dominant.title()}</div>
                          <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                          <hr style="margin:0.75rem 0;">
                          <div class="kpi-label" style="margin-bottom:0.6rem;">Emotion Breakdown</div>
                          {"".join([
                              f'<div class="bar-row">'
                              f'<div class="emotion-bar-label">{e}</div>'
                              f'<div class="bar-track"><div class="bar-fill" style="width:{min(v,100):.0f}%;background:{EMOTION_COLORS.get(e,"#1d4ed8")};"></div></div>'
                              f'<div class="bar-pct">{v:.1f}%</div>'
                              f'</div>'
                              for e, v in sorted(emotions.items(), key=lambda x: -x[1])
                          ])}
                        </div>""", unsafe_allow_html=True)

                    with dr2:
                        labels = list(emotions.keys())
                        values = list(emotions.values())
                        dom_color = EMOTION_COLORS.get(dominant, "#2563eb")
                        fig = go.Figure(go.Scatterpolar(
                            r=values + [values[0]],
                            theta=labels + [labels[0]],
                            fill="toself",
                            line=dict(color=dom_color, width=2.5),
                            fillcolor=f"rgba({int(dom_color[1:3],16)},{int(dom_color[3:5],16)},{int(dom_color[5:7],16)},0.18)",
                            marker=dict(size=6, color=dom_color),
                        ))
                        fig.update_layout(
                            polar=dict(
                                bgcolor="#f8fafc",
                                radialaxis=dict(visible=True, range=[0, 100],
                                               tickfont=dict(size=9, color="#94a3b8"),
                                               gridcolor="#e2e8f0"),
                                angularaxis=dict(tickfont=dict(size=10, color="#334155")),
                            ),
                            paper_bgcolor="white",
                            font=dict(family="Inter", size=11),
                            margin=dict(t=25, b=25, l=25, r=25),
                            height=290, showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown(f"""
                    <div class="suggested-mood-card">
                      <div>
                        <div class="suggested-mood-label">Detected Mood</div>
                        <div class="suggested-mood-value">{mapped['label']}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    if st.button(f"Accept  —  Log as '{mapped['label']}'",
                                 use_container_width=True, key="accept_cam_mood"):
                        st.session_state.selected_mood = {
                            "label": mapped["label"], "score": mapped["score"]
                        }
                        st.rerun()

                except Exception as e:
                    st.warning(str(e))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Voice / Text Sentiment ──────────────────────────────────────────────
    with st.expander("Sentiment Analysis  —  Voice or Text Input", expanded=False):
        st.markdown("""
        <div class="voice-header">
          <div class="voice-header-title">Voice & Text Sentiment Analysis</div>
          <div class="voice-header-sub">
            Speak naturally or type how you feel. VADER sentiment analysis will
            determine your mood from the language you use.
          </div>
        </div>
        """, unsafe_allow_html=True)

        audio = mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop  —  Analyse",
            just_once=True,
            key="voice_mood_recorder",
        )

        st.markdown('<div class="section-label">Or type your mood</div>', unsafe_allow_html=True)
        typed_text    = st.text_area("Describe how you feel",
                                     placeholder="e.g. I had a productive day and feel accomplished.",
                                     height=80, key="voice_typed_text",
                                     label_visibility="collapsed")
        analyse_btn   = st.button("Analyse Text", key="analyse_typed")

        if audio and audio.get("bytes"):
            with st.spinner("Transcribing recording..."):
                try:
                    st.session_state.voice_transcript = transcribe_audio(audio["bytes"])
                except sr.UnknownValueError:
                    st.warning("Could not interpret audio. Please speak clearly and try again.")
                except sr.RequestError as e:
                    st.error(f"Speech recognition service unavailable: {e}")
                except Exception as e:
                    st.error(f"Transcription error: {e}")

        if analyse_btn and typed_text.strip():
            st.session_state.voice_transcript = typed_text.strip()

        transcript = st.session_state.get("voice_transcript", "")
        if transcript:
            section_label("Transcript")
            edited = st.text_area("Edit if needed", value=transcript, height=70,
                                  key="voice_edit", label_visibility="collapsed")
            if edited != transcript:
                st.session_state.voice_transcript = edited
                transcript = edited

            scores, s_mood = analyze_sentiment(transcript)
            compound = scores["compound"]
            pos, neg, neu = scores["pos"], scores["neg"], scores["neu"]

            if compound >= 0.1:
                s_cls, s_desc, s_color = "sentiment-pos", "Positive", "#15803d"
            elif compound <= -0.1:
                s_cls, s_desc, s_color = "sentiment-neg", "Negative", "#b91c1c"
            else:
                s_cls, s_desc, s_color = "sentiment-neu", "Neutral",  "#1d4ed8"

            section_label("Sentiment Result")
            p1, p2 = st.columns([1, 1.4])

            with p1:
                st.markdown(f"""
                <div class="sentiment-panel">
                  <div class="kpi-label">Compound Score</div>
                  <div class="sentiment-score-large {s_cls}">{compound:+.2f}</div>
                  <div class="sentiment-descriptor" style="color:{s_color};">{s_desc}</div>
                  <hr style="margin:0.75rem 0;">
                  <div class="kpi-label" style="margin-bottom:0.6rem;">Component Breakdown</div>
                  <div class="bar-row">
                    <div class="bar-label">Positive</div>
                    <div class="bar-track"><div class="bar-fill" style="width:{pos*100:.0f}%;background:#15803d;"></div></div>
                    <div class="bar-pct">{pos*100:.0f}%</div>
                  </div>
                  <div class="bar-row">
                    <div class="bar-label">Neutral</div>
                    <div class="bar-track"><div class="bar-fill" style="width:{neu*100:.0f}%;background:#1d4ed8;"></div></div>
                    <div class="bar-pct">{neu*100:.0f}%</div>
                  </div>
                  <div class="bar-row">
                    <div class="bar-label">Negative</div>
                    <div class="bar-track"><div class="bar-fill" style="width:{neg*100:.0f}%;background:#b91c1c;"></div></div>
                    <div class="bar-pct">{neg*100:.0f}%</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            with p2:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=compound,
                    number={"valueformat": "+.2f",
                            "font": {"size": 26, "family": "Inter", "color": s_color}},
                    gauge={
                        "axis": {"range": [-1, 1], "tickwidth": 1,
                                 "tickcolor": "#94a3b8",
                                 "tickvals": [-1, -0.5, 0, 0.5, 1],
                                 "ticktext": ["-1.0", "-0.5", "0", "+0.5", "+1.0"],
                                 "tickfont": {"size": 9}},
                        "bar": {"color": s_color, "thickness": 0.22},
                        "bgcolor": "white", "borderwidth": 0,
                        "steps": [
                            {"range": [-1.0, -0.5], "color": "#fee2e2"},
                            {"range": [-0.5, -0.1], "color": "#fef3c7"},
                            {"range": [-0.1,  0.1], "color": "#eff6ff"},
                            {"range": [ 0.1,  0.5], "color": "#dcfce7"},
                            {"range": [ 0.5,  1.0], "color": "#bbf7d0"},
                        ],
                        "threshold": {"line": {"color": s_color, "width": 3},
                                      "thickness": 0.7, "value": compound},
                    },
                    title={"text": "Sentiment Gauge",
                           "font": {"size": 12, "family": "Inter", "color": "#475569"}},
                ))
                fig_g.update_layout(height=230, paper_bgcolor="white",
                                    font=dict(family="Inter"),
                                    margin=dict(t=40, b=10, l=20, r=20))
                st.plotly_chart(fig_g, use_container_width=True)

            st.markdown(f"""
            <div class="suggested-mood-card">
              <div>
                <div class="suggested-mood-label">Suggested Mood</div>
                <div class="suggested-mood-value">{s_mood['label']}</div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            va1, va2 = st.columns(2)
            with va1:
                if st.button(f"Accept  —  Log as '{s_mood['label']}'",
                             use_container_width=True, key="accept_voice_mood"):
                    st.session_state.selected_mood = s_mood
                    st.session_state.voice_transcript = ""
                    st.rerun()
            with va2:
                if st.button("Clear", use_container_width=True, key="clear_voice"):
                    st.session_state.voice_transcript = ""
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Manual Selection ────────────────────────────────────────────────────
    section_label("Manual Mood Selection")
    mood_cols = st.columns(len(MOODS))
    for i, mood in enumerate(MOODS):
        with mood_cols[i]:
            if st.button(mood["label"], key=f"mood_btn_{i}", use_container_width=True):
                st.session_state.selected_mood = mood

    selected = st.session_state.selected_mood
    if selected:
        st.success(f"Selected mood: {selected['label']}")
        with st.form("mood_form"):
            m1, m2 = st.columns(2)
            with m1: energy   = st.slider("Energy Level  (1 = low, 10 = high)", 1, 10, 5)
            with m2: log_date = st.date_input("Date", value=datetime.date.today())
            note = st.text_area("Notes", placeholder="Optional — describe your day or thoughts.",
                                height=80)
            if st.form_submit_button("Save Entry", use_container_width=True):
                st.session_state.mood_logs.append({
                    "date":   str(log_date),
                    "mood":   selected["label"],
                    "score":  selected["score"],
                    "energy": energy,
                    "note":   note.strip(),
                })
                st.session_state.selected_mood = None
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── History & Charts ────────────────────────────────────────────────────
    logs = st.session_state.mood_logs
    if not logs:
        st.info("No mood entries recorded yet.")
        return

    section_label("Trends")
    df = pd.DataFrame(logs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    ch1, ch2 = st.columns(2)

    MOOD_SCORE_COLORS = {5: "#059669", 4: "#0891b2", 3: "#2563eb", 2: "#d97706", 1: "#dc2626"}
    marker_colors = [MOOD_SCORE_COLORS.get(int(s), "#2563eb") for s in df["score"]]

    with ch1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["score"], mode="lines+markers",
            line=dict(color="#2563eb", width=2.5, shape="spline"),
            marker=dict(size=9, color=marker_colors,
                        line=dict(width=2, color="white")),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
            name="Mood"))
        fig.update_layout(
            title=dict(text="Mood Over Time", font=dict(size=13, family="Inter", color="#0c1f3f")),
            yaxis=dict(tickvals=[1,2,3,4,5],
                       ticktext=["Sad","Low","Neutral","Good","Great"],
                       range=[0.5, 5.5],
                       gridcolor="#e2e8f0", gridwidth=1,
                       tickfont=dict(size=11, color="#64748b")),
            xaxis=dict(gridcolor="#e2e8f0", gridwidth=1,
                       tickfont=dict(size=10, color="#64748b")),
            paper_bgcolor="white", plot_bgcolor="#fafbfc",
            font=dict(family="Inter", size=11),
            margin=dict(t=45, b=20, l=10, r=10), height=280)
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["date"], y=df["energy"], mode="lines+markers",
            line=dict(color="#0d9488", width=2.5, shape="spline"),
            marker=dict(size=9, color="#0d9488",
                        line=dict(width=2, color="white")),
            fill="tozeroy",
            fillcolor="rgba(13,148,136,0.10)",
            name="Energy"))
        fig2.update_layout(
            title=dict(text="Energy Level Over Time", font=dict(size=13, family="Inter", color="#0c1f3f")),
            yaxis=dict(range=[0, 11],
                       gridcolor="#e2e8f0", gridwidth=1,
                       tickfont=dict(size=11, color="#64748b")),
            xaxis=dict(gridcolor="#e2e8f0", gridwidth=1,
                       tickfont=dict(size=10, color="#64748b")),
            paper_bgcolor="white", plot_bgcolor="#fafbfc",
            font=dict(family="Inter", size=11),
            margin=dict(t=45, b=20, l=10, r=10), height=280)
        st.plotly_chart(fig2, use_container_width=True)

    dist  = df["mood"].value_counts()
    MOOD_COLORS_PIE = {"Great":"#059669","Good":"#0891b2","Neutral":"#2563eb","Low":"#d97706","Sad":"#dc2626"}
    pie_colors = [MOOD_COLORS_PIE.get(l, "#2563eb") for l in dist.index]
    fig3  = go.Figure(data=[go.Pie(
        labels=dist.index, values=dist.values, hole=0.48,
        marker=dict(colors=pie_colors, line=dict(color="white", width=2)),
        textinfo="label+percent",
        textfont=dict(family="Inter", size=12, color="white"),
        pull=[0.04]*len(dist))])
    fig3.update_layout(
        title=dict(text="Mood Distribution", font=dict(size=13, family="Inter", color="#0c1f3f")),
        paper_bgcolor="white", font=dict(family="Inter"),
        legend=dict(font=dict(size=11), bgcolor="white"),
        margin=dict(t=45, b=20, l=10, r=10), height=300)
    st.plotly_chart(fig3, use_container_width=True)

    sc1, sc2, sc3 = st.columns(3)
    with sc1: kpi_card("Average Mood",   f"{df['score'].mean():.1f} / 5",   "", "accent-blue")
    with sc2: kpi_card("Average Energy", f"{df['energy'].mean():.1f} / 10", "", "accent-teal")
    with sc3: kpi_card("Total Entries",  len(df),                            "", "accent-slate")

    section_label("Entry Log")
    disp = df[["date","mood","score","energy","note"]].copy()
    disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
    disp.columns = ["Date","Mood","Score","Energy","Notes"]
    st.dataframe(disp.sort_values("Date", ascending=False),
                 use_container_width=True, hide_index=True)

    if st.button("Clear All Entries", key="clear_mood_logs"):
        st.session_state.mood_logs = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TIPS
# ─────────────────────────────────────────────────────────────────────────────
TIPS = {
    "Focus & Productivity": [
        ("Pomodoro Technique",
         "Work in 25-minute focused sprints separated by 5-minute breaks. After four sessions, "
         "take a longer 15-20 minute break. This cycle prevents cognitive fatigue while maintaining output quality."),
        ("Single-Tasking",
         "Multitasking reduces cognitive efficiency by up to 40%. Commit to one task, reach a natural stopping point, "
         "then transition. Context-switching incurs a mental overhead of 20-30 minutes per switch."),
        ("Time Blocking",
         "Assign specific calendar windows to specific task categories. Reserve high-energy periods for demanding work "
         "and protect those blocks from meetings and interruptions."),
        ("Two-Minute Rule",
         "If a task requires less than two minutes, complete it immediately. Deferring small tasks creates "
         "cognitive overhead and inflates your mental to-do list."),
    ],
    "Cognitive Performance": [
        ("Brain Dumping",
         "Externalise all open mental loops onto paper before focused work. "
         "This frees working-memory capacity and reduces cognitive interference during deep work."),
        ("Chunking",
         "Decompose large projects into discrete, completable sub-tasks. Each completed unit "
         "triggers a dopamine response that reinforces continued progress."),
        ("Active Recall",
         "Test yourself on material rather than re-reading. Retrieval practice is significantly more effective "
         "for long-term retention than passive review."),
        ("Spaced Repetition",
         "Review material at exponentially increasing intervals (1, 3, 7, 14 days). "
         "This exploits the spacing effect for durable memory consolidation."),
    ],
    "Energy & Recovery": [
        ("Sleep Architecture",
         "Target 7-9 hours with consistent sleep and wake times. Deep sleep consolidates declarative memory, "
         "while REM sleep integrates procedural knowledge and emotional regulation."),
        ("Aerobic Exercise",
         "20-30 minutes of moderate aerobic activity elevates BDNF levels, improving attention, "
         "working memory, and executive function for 3-4 hours post-exercise."),
        ("Strategic Rest",
         "Use breaks for genuine neural recovery: walking, stretching, or eyes-closed rest. "
         "Social media activates the same attention networks as work and does not constitute rest."),
        ("Hydration",
         "Even 1-2% dehydration measurably impairs concentration and mood. "
         "Establish a consistent hydration routine rather than drinking reactively when thirsty."),
    ],
    "Mindset & Behaviour": [
        ("Implementation Intentions",
         "Specify the when, where, and how of planned actions. Research shows that concrete plans "
         "('I will run at 7am on Tuesday at the track') are 2-3x more likely to be executed."),
        ("Habit Stacking",
         "Anchor new behaviours to established routines. The format 'After [existing habit], I will [new habit]' "
         "exploits existing neural pathways to reduce adoption friction."),
        ("Growth Mindset",
         "Interpret challenges as indicators of growth rather than signs of incompetence. "
         "Difficulty is the mechanism through which skills develop."),
        ("Self-Compassion",
         "Treating yourself with kindness after failure leads to better performance outcomes than self-criticism. "
         "Self-criticism activates the threat-response system, which impairs learning."),
    ],
    "Distraction Management": [
        ("Environment Design",
         "Remove friction from desired behaviours and add friction to undesired ones. "
         "Physical and digital environments shape behaviour more reliably than willpower."),
        ("Batch Communication",
         "Designate fixed windows for email and messaging rather than responding reactively. "
         "Continuous notification monitoring fragments attention and chronically elevates cortisol."),
        ("Focus Rituals",
         "Establish a brief pre-work ritual (e.g., clearing desk, reviewing priorities, headphones on) "
         "to signal cognitive mode-switching to your brain."),
        ("The Ten-Minute Commitment",
         "When resistance to starting is high, commit to working for just ten minutes. "
         "Task engagement typically overrides initial resistance once started."),
    ],
}

def show_tips():
    page_header("Knowledge Base", "Tips & Strategies",
                "Evidence-based techniques for focus, cognitive performance, and well-being.")

    search = st.text_input("Search strategies", placeholder="e.g. sleep, focus, habits, memory...")

    for category, tip_list in TIPS.items():
        filtered = [t for t in tip_list
                    if not search or search.lower() in t[0].lower() or search.lower() in t[1].lower()]
        if not filtered:
            continue
        section_label(category)
        cols = st.columns(2)
        for i, (title, body) in enumerate(filtered):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="card" style="border-left:3px solid var(--blue);margin-bottom:0.75rem;">
                  <div style="font-size:0.875rem;font-weight:700;color:var(--navy);margin-bottom:0.4rem;">
                    {title}
                  </div>
                  <div style="font-size:0.82rem;color:var(--text-muted);line-height:1.65;">
                    {body}
                  </div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ADHD ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
def show_adhd_login():
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("""
        <div class="login-card">
          <div class="login-card-header">
            <div class="login-card-header-title">Vanderbilt ADHD Clinical System</div>
            <div class="login-card-header-sub">
              Department of Child &amp; Adolescent Psychiatry
            </div>
          </div>
          <div class="login-card-body">
        """, unsafe_allow_html=True)

        tab      = st.selectbox("", ["Login", "Register"], label_visibility="collapsed")
        username = st.text_input("Username", placeholder="Enter username", key="adhd_username")
        password = st.text_input("Password", type="password", placeholder="Enter password", key="adhd_password")

        if tab == "Register":
            role = st.selectbox("Role", ["user", "admin"], key="adhd_reg_role")
            if st.button("Create Account", use_container_width=True, key="adhd_reg_btn"):
                try:
                    register_user(username, password, role)
                    st.success("Account created. Please sign in.")
                except Exception as e:
                    st.error(f"Registration failed: {e}")

        if tab == "Login":
            if st.button("Sign In", use_container_width=True, key="adhd_login_btn"):
                try:
                    user = login_user(username, password)
                    if user:
                        st.session_state.adhd_user = user
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
                except Exception as e:
                    st.error(f"Login error: {e}")

        st.markdown("</div></div>", unsafe_allow_html=True)


def show_adhd_admin():
    st.markdown("""
    <div class="clinical-header">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div class="clinical-header-title">Vanderbilt ADHD Clinical System</div>
          <div class="clinical-header-sub">Analytics &amp; Administration Panel</div>
        </div>
        <span class="clinical-badge">Admin Access</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    admin_page = st.radio("View", ["Overview Dashboard", "Parent vs Teacher Comparison", "Raw Data"],
                          horizontal=True)

    try:
        conn = get_connection()
        df   = pd.read_sql("SELECT * FROM assessments", conn)
        conn.close()
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return

    if df.empty:
        st.warning("No assessment records in the database.")
        return

    if admin_page == "Overview Dashboard":
        section_label("Filters")
        r1, r2 = st.columns(2)
        with r1: role_f = st.selectbox("Respondent Role", ["All"] + df["role"].unique().tolist())
        with r2: sev_f  = st.selectbox("Severity",        ["All"] + df["severity"].unique().tolist())

        fdf = df.copy()
        if role_f != "All": fdf = fdf[fdf["role"]     == role_f]
        if sev_f  != "All": fdf = fdf[fdf["severity"] == sev_f]

        section_label("Key Metrics")
        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("Total Assessments",      len(fdf), "", "accent-blue")
        with c2: kpi_card("Avg Inattention Score",  f"{fdf['inatt_score'].mean():.2f}", "", "accent-teal")
        with c3: kpi_card("Avg Hyperactivity Score",f"{fdf['hyper_score'].mean():.2f}", "", "accent-amber")

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            pred_dist = fdf["prediction"].value_counts()
            fig1 = go.Figure(data=[go.Pie(
                labels=pred_dist.index, values=pred_dist.values,
                hole=0.48,
                marker=dict(colors=CHART_COLORS, line=dict(color="white", width=2)),
                textinfo="label+percent",
                textfont=dict(family="Inter", size=12),
                pull=[0.03]*len(pred_dist))])
            fig1.update_layout(
                title=dict(text="ADHD Type Distribution", font=dict(size=13, family="Inter", color="#0c1f3f")),
                paper_bgcolor="white", font=dict(family="Inter"),
                legend=dict(font=dict(size=11)),
                margin=dict(t=45, b=20, l=10, r=10))
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            sev   = fdf["severity"].value_counts()
            s_clr = {"Severe": "#dc2626", "Moderate": "#d97706", "Mild": "#16a34a"}
            fig2  = go.Figure(data=[go.Bar(
                x=sev.index, y=sev.values,
                marker=dict(
                    color=[s_clr.get(s, "#2563eb") for s in sev.index],
                    line=dict(color="white", width=1.5)
                ),
                text=sev.values, textposition="outside",
                textfont=dict(family="Inter", size=12, color="#334155"),
                width=0.5)])
            fig2.update_layout(
                title=dict(text="Severity Distribution", font=dict(size=13, family="Inter", color="#0c1f3f")),
                paper_bgcolor="white", plot_bgcolor="#fafbfc",
                font=dict(family="Inter"),
                yaxis=dict(gridcolor="#e2e8f0", gridwidth=1,
                           tickfont=dict(size=11, color="#64748b")),
                xaxis=dict(tickfont=dict(size=12, color="#334155")),
                margin=dict(t=45, b=20, l=10, r=10))
            st.plotly_chart(fig2, use_container_width=True)

    elif admin_page == "Parent vs Teacher Comparison":
        section_label("Radar Comparison")
        comparison = df.groupby("role")[["inatt_score","hyper_score","odd_score",
                                         "conduct_score","anxiety_score","performance_score"]].mean()
        fig = go.Figure()
        role_colors = {"Parent": "#2563eb", "Teacher": "#059669"}
        role_fills  = {"Parent": "rgba(37,99,235,0.15)", "Teacher": "rgba(5,150,105,0.15)"}
        for role_name in comparison.index:
            fig.add_trace(go.Scatterpolar(
                r=comparison.loc[role_name].values,
                theta=["Inattention","Hyperactivity","ODD","Conduct","Anxiety","Performance"],
                fill='toself', name=role_name,
                line=dict(color=role_colors.get(role_name, "#7c3aed"), width=2.5),
                fillcolor=role_fills.get(role_name, "rgba(124,58,237,0.15)"),
                marker=dict(size=7),
            ))
        fig.update_layout(
            paper_bgcolor="white", font=dict(family="Inter", size=11),
            polar=dict(bgcolor="#f8fafc",
                       radialaxis=dict(gridcolor="#e2e8f0", tickfont=dict(size=9, color="#94a3b8")),
                       angularaxis=dict(tickfont=dict(size=11, color="#334155"))),
            legend=dict(bgcolor="white", bordercolor="#e2e8f0", borderwidth=1,
                        font=dict(size=12)),
            margin=dict(t=40, b=40, l=40, r=40), height=420)
        st.plotly_chart(fig, use_container_width=True)

    elif admin_page == "Raw Data":
        section_label("Assessment Records")
        st.dataframe(df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Sign Out", key="admin_signout"):
        st.session_state.adhd_user = None
        st.rerun()


def show_adhd_user_assessment():
    st.markdown("""
    <div class="clinical-header">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div class="clinical-header-title">Vanderbilt ADHD Clinical Evaluation</div>
          <div class="clinical-header-sub">Department of Child &amp; Adolescent Psychiatry</div>
        </div>
        <span class="clinical-badge">Assessment Form</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        model, label_encoder = load_adhd_model()
        questions, _         = load_questions()
    except Exception as e:
        st.error(f"Failed to load assessment resources: {e}")
        return

    section_label("Patient Information")
    p1, p2, p3 = st.columns(3)
    with p1: patient_name   = st.text_input("Full Name",       placeholder="Patient full name")
    with p2: patient_age    = st.number_input("Age",           min_value=3, max_value=18)
    with p3: patient_gender = st.selectbox("Gender",           ["Male", "Female", "Other"])

    pa, pb = st.columns([1, 2])
    with pa: assessment_date = st.date_input("Assessment Date")
    with pb: role = st.selectbox("Respondent Role", ["Parent", "Teacher"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Mood Tracker (at time of assessment) ──────────────────────────────────
    _MOOD_MAP = {
        "😢 Very Low": 1,
        "😟 Low":      2,
        "😐 Neutral":  3,
        "🙂 Good":     4,
        "😄 Excellent": 5,
    }
    st.markdown("""
    <div style="background:white;border:1px solid #e2e8f0;border-left:5px solid #7c3aed;
                border-radius:11px;padding:1rem 1.4rem;box-shadow:0 2px 8px rgba(15,23,42,0.1);
                margin-bottom:1rem;">
      <div style="font-weight:600;color:#4c1d95;font-size:0.95rem;margin-bottom:0.3rem;">
        🎭 Current Mood Tracker
      </div>
      <div style="color:#64748b;font-size:0.8rem;">
        Rate the patient's current mood before starting the assessment —
        this will be correlated with the clinical scores.
      </div>
    </div>
    """, unsafe_allow_html=True)
    _mood_selection = st.radio(
        "Patient mood",
        list(_MOOD_MAP.keys()),
        index=2,
        horizontal=True,
        label_visibility="collapsed",
        key="adhd_mood_radio"
    )
    _mood_score = _MOOD_MAP[_mood_selection]

    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Clinical Questionnaire")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Inattention", "Hyperactivity", "ODD", "Conduct", "Anxiety", "Performance"
    ])

    opts_main = ["Never", "Occasionally", "Often", "Very Often"]
    opts_perf = ["Excellent", "Above Average", "Average", "Somewhat of a Problem", "Problematic"]
    responses = []

    def render_questions(tab, prefix, q_range, opts):
        with tab:
            for idx, i in enumerate(q_range):
                st.markdown(f"""
                <div class="question-row">
                  <span class="question-index">{idx+1}</span>
                  <span class="question-text">{questions[i]}</span>
                </div>""", unsafe_allow_html=True)
                ans = st.radio("", opts, key=f"{prefix}_{i}", horizontal=True,
                               label_visibility="collapsed")
                responses.append(opts.index(ans))
                st.divider()

    render_questions(tab1, "inatt",   range(0,  9),  opts_main)
    render_questions(tab2, "hyper",   range(9,  18), opts_main)
    render_questions(tab3, "odd",     range(18, 26), opts_main)
    render_questions(tab4, "conduct", range(26, 40), opts_main)
    render_questions(tab5, "anxiety", range(40, 47), opts_main)
    render_questions(tab6, "perf",    range(47, 55), opts_perf)

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        submit = st.button("Submit Assessment", use_container_width=True)

    if submit:
        r           = np.array(responses)
        inatt       = r[0:9].sum()
        hyper       = r[9:18].sum()
        odd         = r[18:26].sum()
        conduct     = r[26:40].sum()
        anxiety     = r[40:47].sum()
        performance = r[47:55].sum()

        pred        = model.predict(np.array([[inatt, hyper, odd, conduct, anxiety, performance]]))
        prediction  = label_encoder.inverse_transform(pred)[0]

        severity    = "Mild"
        if   inatt + hyper > 36: severity = "Severe"
        elif inatt + hyper > 18: severity = "Moderate"

        _assessed_at = _dt.now()
        # Fetch history BEFORE saving so we can compare
        _history = get_patient_history(st.session_state.adhd_user["id"], patient_name) if patient_name.strip() else []

        try:
            save_assessment((
                int(st.session_state.adhd_user["id"]), role,
                int(inatt), int(hyper), int(odd),
                int(conduct), int(anxiety), int(performance),
                str(prediction), str(severity),
                str(patient_name), int(patient_age), str(patient_gender),
                str(_mood_selection), int(_mood_score),
                _assessed_at
            ))
        except Exception as e:
            st.warning(f"Could not save to database: {e}")

        st.markdown("""
        <div class="result-banner">
          <div class="result-banner-title">Clinical Assessment Results</div>
          <div class="result-banner-sub">Assessment submitted and saved to the clinical database.</div>
        </div>""", unsafe_allow_html=True)

        section_label("Subscale Scores")
        c1, c2, c3 = st.columns(3)
        c1.metric("Inattention Score",   inatt)
        c2.metric("Hyperactivity Score", hyper)
        c3.metric("Total Core Score",    inatt + hyper)

        st.markdown("<br>", unsafe_allow_html=True)
        cs, cp = st.columns(2)

        with cs:
            sev_cls = {"Severe": "sev-severe", "Moderate": "sev-moderate", "Mild": "sev-mild"}[severity]
            st.markdown(f"""
            <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
                        padding:1.5rem;text-align:center;box-shadow:var(--shadow-xs);">
              <div class="kpi-label" style="margin-bottom:0.6rem;">Severity Level</div>
              <span class="severity-pill {sev_cls}">{severity}</span>
            </div>""", unsafe_allow_html=True)

        with cp:
            st.markdown(f"""
            <div class="prediction-card">
              <div class="prediction-label">ADHD Classification</div>
              <div class="prediction-value">{prediction}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure()
        scores_list = [inatt, hyper, odd, conduct, anxiety, performance]
        max_score   = max(scores_list) if max(scores_list) > 0 else 1
        fig.add_trace(go.Scatterpolar(
            r=scores_list,
            theta=["Inattention","Hyperactivity","ODD","Conduct","Anxiety","Performance"],
            fill='toself', name="Score Profile",
            line=dict(color="#2563eb", width=3),
            fillcolor="rgba(37,99,235,0.15)",
            marker=dict(size=8, color="#2563eb", line=dict(width=2, color="white")),
        ))
        fig.update_layout(
            title=dict(text="Subscale Score Profile", font=dict(size=14, family="Inter", color="#0c1f3f")),
            paper_bgcolor="white", font=dict(family="Inter", size=12),
            polar=dict(
                bgcolor="#f8fafc",
                radialaxis=dict(visible=True, gridcolor="#e2e8f0",
                               tickfont=dict(size=9, color="#94a3b8")),
                angularaxis=dict(tickfont=dict(size=11, color="#334155")),
            ),
            margin=dict(t=55, b=35, l=35, r=35), height=380)
        st.plotly_chart(fig, use_container_width=True)

        # ── Mood at assessment ────────────────────────────────────────────────
        _mood_clr = {1:"#ef5350",2:"#ffa726",3:"#42a5f5",4:"#66bb6a",5:"#26c6da"}.get(_mood_score,"#42a5f5")
        _mood_note = ("Low mood may amplify symptom perception — interpret scores with caution."
                      if _mood_score <= 2 else
                      "Neutral mood — scores reflect baseline behaviour." if _mood_score == 3 else
                      "Positive mood noted — scores reflect current state.")
        st.markdown(f"""
        <div style="background:white;border:1px solid #e2e8f0;border-left:5px solid {_mood_clr};
                    border-radius:11px;padding:1rem 1.4rem;margin-bottom:1rem;
                    box-shadow:0 2px 8px rgba(15,23,42,0.1);">
          <div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;
                      letter-spacing:1px;color:#64748b;margin-bottom:0.3rem;">
            Mood at Time of Assessment
          </div>
          <span style="font-size:1.2rem;font-weight:700;color:{_mood_clr};">{_mood_selection}</span>
          <span style="font-size:0.8rem;color:#64748b;margin-left:0.8rem;">{_mood_note}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Improvement Analysis ──────────────────────────────────────────────
        if _history and patient_name.strip():
            _prev = _history[-1]

            def _badge(label, pv, cv, lib=True):
                diff = cv - pv
                if diff == 0:
                    cls, arrow, msg = "improve-same", "→", "No change"
                elif (diff < 0) == lib:
                    cls, arrow, msg = "improve-up",   "▲ Improved", f"{abs(diff):+} pts"
                else:
                    cls, arrow, msg = "improve-down", "▼ Declined",  f"{abs(diff):+} pts"
                return (f'<span style="display:inline-flex;align-items:center;gap:0.4rem;'
                        f'padding:0.35rem 0.9rem;border-radius:50px;font-weight:700;font-size:0.82rem;'
                        f'margin:0.2rem;'
                        + ("background:#e8f5e9;color:#1b5e20;border:2px solid #81c784;" if cls=="improve-up" else
                           "background:#ffebee;color:#b71c1c;border:2px solid #ef9a9a;" if cls=="improve-down" else
                           "background:#e3f2fd;color:#0d47a1;border:2px solid #90caf9;")
                        + f'">{arrow} {label}: {pv} → {cv} ({msg})</span>')

            section_label("Progress & Improvement Analysis")
            prev_i, prev_h = _prev.get("inatt_score",0), _prev.get("hyper_score",0)
            prev_o, prev_c = _prev.get("odd_score",0),   _prev.get("conduct_score",0)
            prev_a, prev_p = _prev.get("anxiety_score",0), _prev.get("performance_score",0)

            st.markdown(
                '<div style="display:flex;flex-wrap:wrap;gap:0.3rem;margin-bottom:1rem;">'
                + _badge("Inattention",   prev_i, inatt)
                + _badge("Hyperactivity", prev_h, hyper)
                + _badge("ODD",           prev_o, odd)
                + _badge("Conduct",       prev_c, conduct)
                + _badge("Anxiety",       prev_a, anxiety)
                + _badge("Performance",   prev_p, performance)
                + '</div>', unsafe_allow_html=True)

            _ct, _pt = inatt + hyper, prev_i + prev_h
            if _ct < _pt:
                _ov = f"Overall Improved — core score dropped by {_pt - _ct} pts"
                _ov_style = "background:#e8f5e9;color:#1b5e20;border:2px solid #81c784;"
            elif _ct > _pt:
                _ov = f"Overall Declined — core score rose by {_ct - _pt} pts"
                _ov_style = "background:#ffebee;color:#b71c1c;border:2px solid #ef9a9a;"
            else:
                _ov = "Overall Stable — no change in core score"
                _ov_style = "background:#e3f2fd;color:#0d47a1;border:2px solid #90caf9;"
            st.markdown(
                f'<span style="display:inline-flex;padding:0.45rem 1.1rem;border-radius:50px;'
                f'font-weight:700;font-size:0.95rem;{_ov_style}">{_ov}</span>',
                unsafe_allow_html=True)

            _prev_sev = _prev.get("severity","")
            _prev_date = str(_prev.get("assessed_at",""))[:10]
            if _prev_sev and _prev_sev != severity:
                st.info(f"Severity changed from **{_prev_sev}** (assessed {_prev_date}) → **{severity}** today.")

            # Trend chart
            _all = list(_history) + [{"assessed_at": _assessed_at, "inatt_score": inatt,
                                       "hyper_score": hyper, "mood_score": _mood_score}]
            _dates  = [str(r.get("assessed_at",""))[:10] for r in _all]
            _i_vals = [r.get("inatt_score",0) for r in _all]
            _h_vals = [r.get("hyper_score",0) for r in _all]
            _m_vals = [r.get("mood_score",0)  for r in _all]

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=_dates, y=_i_vals, name="Inattention",
                mode="lines+markers", line=dict(color="#2563eb", width=2), marker=dict(size=8)))
            fig_trend.add_trace(go.Scatter(x=_dates, y=_h_vals, name="Hyperactivity",
                mode="lines+markers", line=dict(color="#d97706", width=2), marker=dict(size=8)))
            fig_trend.add_trace(go.Scatter(x=_dates, y=_m_vals, name="Mood Score",
                mode="lines+markers", line=dict(color="#7c3aed", width=2, dash="dot"),
                marker=dict(size=8), yaxis="y2"))
            fig_trend.update_layout(
                title=dict(text="Score Trend Over Time (with Mood)", font=dict(size=14, family="Inter")),
                paper_bgcolor="white", font=dict(family="Inter", size=12),
                yaxis=dict(title="ADHD Score", gridcolor="#e2e8f0"),
                yaxis2=dict(title="Mood (1–5)", overlaying="y", side="right",
                            range=[0,6], showgrid=False),
                legend=dict(bgcolor="white", bordercolor="#e2e8f0", borderwidth=1),
                margin=dict(t=50, b=30, l=40, r=40))
            st.plotly_chart(fig_trend, use_container_width=True)

        elif patient_name.strip():
            st.info("First assessment recorded for this patient. Future assessments will show improvement trends here.")

        section_label("Clinical Interpretation")
        interp = {
            "Combined Type":
                "**Combined Type ADHD** — Clinically significant symptoms in both the inattention and "
                "hyperactivity-impulsivity domains. A comprehensive intervention addressing both dimensions "
                "is recommended.",
            "Inattentive Type":
                "**Predominantly Inattentive Presentation** — Primary difficulties with sustained attention "
                "and organisation. Hyperactive-impulsive symptom levels are below threshold.",
            "Hyperactive Type":
                "**Predominantly Hyperactive-Impulsive Presentation** — Primary difficulties with activity "
                "regulation and impulse control. Inattentive symptom levels are below threshold.",
        }
        st.info(interp.get(prediction,
            "**Below Diagnostic Threshold** — Current symptom levels do not meet full diagnostic criteria. "
            "Monitoring and follow-up assessment is recommended."))

        st.markdown("<br>", unsafe_allow_html=True)
        buffer = BytesIO()
        doc    = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        elems  = [
            Paragraph("Vanderbilt ADHD Clinical Report", styles["Title"]),
            Spacer(1, 0.3 * inch),
            Paragraph(f"Patient Name: {patient_name}",        styles["Normal"]),
            Paragraph(f"Age: {patient_age}",                  styles["Normal"]),
            Paragraph(f"Gender: {patient_gender}",            styles["Normal"]),
            Paragraph(f"Assessment Date: {assessment_date}",  styles["Normal"]),
            Spacer(1, 0.3 * inch),
            Paragraph(f"Classification: {prediction}",        styles["Normal"]),
            Paragraph(f"Severity: {severity}",                styles["Normal"]),
            Spacer(1, 0.3 * inch),
        ]
        tbl_data = [
            ["Subscale", "Score"],
            ["Inattention",   str(inatt)],
            ["Hyperactivity", str(hyper)],
            ["ODD",           str(odd)],
            ["Conduct",       str(conduct)],
            ["Anxiety",       str(anxiety)],
            ["Performance",   str(performance)],
        ]
        tbl = Table(tbl_data)
        tbl.setStyle([
            ('BACKGROUND',    (0,0), (-1,0), rl_colors.HexColor("#0c1f3f")),
            ('TEXTCOLOR',     (0,0), (-1,0), rl_colors.white),
            ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID',          (0,0), (-1,-1), 1, rl_colors.HexColor("#e2e8f0")),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [rl_colors.white, rl_colors.HexColor("#f1f5f9")]),
        ])
        elems.append(tbl)
        doc.build(elems)
        buffer.seek(0)
        st.download_button("Download Clinical Report (PDF)", buffer,
                           "ADHD_Report.pdf", mime="application/pdf")

    st.markdown("<br>", unsafe_allow_html=True)
    co, _ = st.columns([1, 4])
    with co:
        if st.button("Sign Out", key="user_signout"):
            st.session_state.adhd_user = None
            st.rerun()


def show_adhd_page():
    page_header("Clinical Module", "ADHD Assessment",
                "Vanderbilt rating scale with machine-learning classification and PDF reporting.",
                badge="Clinical Access", badge_style="navy")

    if st.session_state.adhd_user is None:
        show_adhd_login()
    elif st.session_state.adhd_user["role"] == "admin":
        show_adhd_admin()
    else:
        show_adhd_user_assessment()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
if   page == "Dashboard":        show_home()
elif page == "Focus Timer":      show_focus_timer()
elif page == "Daily Planner":    show_daily_planner()
elif page == "Mood Tracker":     show_mood_tracker()
elif page == "Tips & Strategies":show_tips()
elif page == "ADHD Assessment":  show_adhd_page()
