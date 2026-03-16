"""
asrs_assessment.py
ASRS-based ADHD Assessment & Mood Tracker Module
Integrated with the Vanderbilt ADHD Clinical System
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
from io import BytesIO
import json
import os

# ─────────────────────────────────────────────
# DATABASE  (SQLite – always local, always present)
# ─────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "asrs_tracker.db")


def _get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_asrs_db():
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS asrs_sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            patient_name  TEXT    NOT NULL,
            session_date  TEXT    NOT NULL,
            inattention   REAL    DEFAULT 0,
            hyperactivity REAL    DEFAULT 0,
            impulsivity   REAL    DEFAULT 0,
            emotional_reg REAL    DEFAULT 0,
            focus_org     REAL    DEFAULT 0,
            total_score   REAL    DEFAULT 0,
            severity      TEXT    DEFAULT '',
            mood_detected TEXT    DEFAULT '',
            mood_score    REAL    DEFAULT 0,
            responses     TEXT    DEFAULT '{}',
            created_at    TEXT    DEFAULT (datetime('now','localtime'))
        );
        CREATE INDEX IF NOT EXISTS idx_asrs_user ON asrs_sessions(user_id, patient_name);
        CREATE INDEX IF NOT EXISTS idx_asrs_date ON asrs_sessions(session_date);
    """)
    conn.commit()
    conn.close()


def save_asrs_session(data: dict) -> int:
    conn = _get_db()
    cur = conn.execute("""
        INSERT INTO asrs_sessions
            (user_id, patient_name, session_date, inattention, hyperactivity,
             impulsivity, emotional_reg, focus_org, total_score, severity,
             mood_detected, mood_score, responses)
        VALUES
            (:user_id,:patient_name,:session_date,:inattention,:hyperactivity,
             :impulsivity,:emotional_reg,:focus_org,:total_score,:severity,
             :mood_detected,:mood_score,:responses)
    """, data)
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_asrs_history(user_id: int, patient_name: str, limit: int = 60):
    conn = _get_db()
    rows = conn.execute("""
        SELECT * FROM asrs_sessions
        WHERE user_id=? AND patient_name=?
        ORDER BY created_at DESC LIMIT ?
    """, (user_id, patient_name, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_last_session(user_id: int, patient_name: str):
    conn = _get_db()
    row = conn.execute("""
        SELECT * FROM asrs_sessions
        WHERE user_id=? AND patient_name=?
        ORDER BY created_at DESC LIMIT 1
    """, (user_id, patient_name)).fetchone()
    conn.close()
    return dict(row) if row else None


def count_consecutive_streak(history: list) -> int:
    """Count how many consecutive days ending today have a session."""
    if not history:
        return 0
    dates = sorted({r["session_date"] for r in history}, reverse=True)
    streak = 0
    check = date.today()
    for d in dates:
        if d == str(check):
            streak += 1
            check -= timedelta(days=1)
        else:
            break
    return streak


# ─────────────────────────────────────────────
# QUESTIONS  (ASRS v1.1 extended – 24 items)
# ─────────────────────────────────────────────
RESPONSE_OPTIONS = ["Never", "Rarely", "Sometimes", "Often", "Very Often"]
SCORE_MAP = {opt: i for i, opt in enumerate(RESPONSE_OPTIONS)}   # 0–4

ASRS_QUESTIONS = [
    # ── INATTENTION ──
    {"id": "IN1", "domain": "Inattention",
     "text": "How often do you have trouble wrapping up the final details of a project once the challenging parts are done?",
     "emoji": "📋"},
    {"id": "IN2", "domain": "Inattention",
     "text": "How often do you have difficulty getting things in order when you have to do a task that requires organization?",
     "emoji": "🗂️"},
    {"id": "IN3", "domain": "Inattention",
     "text": "How often do you have problems remembering appointments or obligations?",
     "emoji": "📅"},
    {"id": "IN4", "domain": "Inattention",
     "text": "When you have a task that requires a lot of thought, how often do you avoid or delay getting started?",
     "emoji": "⏳"},
    {"id": "IN5", "domain": "Inattention",
     "text": "How often do you make careless mistakes when working on a boring or difficult project?",
     "emoji": "✏️"},
    {"id": "IN6", "domain": "Inattention",
     "text": "How often do you have difficulty keeping attention on repetitive or boring work?",
     "emoji": "🔍"},
    # ── HYPERACTIVITY ──
    {"id": "HY1", "domain": "Hyperactivity",
     "text": "How often do you fidget or squirm when you have to sit for a long time?",
     "emoji": "🦶"},
    {"id": "HY2", "domain": "Hyperactivity",
     "text": "How often do you feel overly active and compelled to do things, like you were driven by a motor?",
     "emoji": "⚡"},
    {"id": "HY3", "domain": "Hyperactivity",
     "text": "How often do you feel restless or have difficulty relaxing in your downtime?",
     "emoji": "🌀"},
    {"id": "HY4", "domain": "Hyperactivity",
     "text": "How often do you talk too much in social situations?",
     "emoji": "💬"},
    {"id": "HY5", "domain": "Hyperactivity",
     "text": "How often do you find yourself moving around or tapping even when it's inappropriate?",
     "emoji": "🎵"},
    # ── IMPULSIVITY ──
    {"id": "IM1", "domain": "Impulsivity",
     "text": "How often do you blurt out answers before questions have been completed?",
     "emoji": "💥"},
    {"id": "IM2", "domain": "Impulsivity",
     "text": "How often do you have difficulty waiting your turn in situations where turn-taking is required?",
     "emoji": "⏱️"},
    {"id": "IM3", "domain": "Impulsivity",
     "text": "How often do you interrupt others when they are busy?",
     "emoji": "🚧"},
    {"id": "IM4", "domain": "Impulsivity",
     "text": "How often do you make decisions without thinking them through completely?",
     "emoji": "🎲"},
    # ── EMOTIONAL REGULATION ──
    {"id": "ER1", "domain": "Emotional Regulation",
     "text": "How often do you feel easily frustrated when things do not go as planned?",
     "emoji": "😤"},
    {"id": "ER2", "domain": "Emotional Regulation",
     "text": "How often do you have trouble controlling your temper in stressful situations?",
     "emoji": "🌋"},
    {"id": "ER3", "domain": "Emotional Regulation",
     "text": "How often do your emotions interfere with your ability to concentrate?",
     "emoji": "🎭"},
    {"id": "ER4", "domain": "Emotional Regulation",
     "text": "How often do you feel overwhelmed by your responsibilities or daily tasks?",
     "emoji": "🏋️"},
    # ── FOCUS & ORGANIZATION ──
    {"id": "FO1", "domain": "Focus",
     "text": "How often do you lose things necessary for tasks (wallet, keys, phone, papers)?",
     "emoji": "🔑"},
    {"id": "FO2", "domain": "Focus",
     "text": "How often are you easily distracted by external stimuli or unrelated thoughts?",
     "emoji": "🌪️"},
    {"id": "FO3", "domain": "Focus",
     "text": "How often do you find it hard to stay focused in direct conversations?",
     "emoji": "👂"},
    {"id": "FO4", "domain": "Focus",
     "text": "How often do you have difficulty sustaining attention during tasks or leisure activities?",
     "emoji": "🔭"},
    {"id": "FO5", "domain": "Focus",
     "text": "How often do you fail to follow through on instructions and fail to finish tasks?",
     "emoji": "📌"},
]

TOTAL_Qs = len(ASRS_QUESTIONS)   # 24
DOMAIN_ORDER = ["Inattention", "Hyperactivity", "Impulsivity", "Emotional Regulation", "Focus"]
DOMAIN_WEIGHTS = {
    "Inattention": 0.28,
    "Hyperactivity": 0.22,
    "Impulsivity": 0.18,
    "Emotional Regulation": 0.17,
    "Focus": 0.15,
}


# ─────────────────────────────────────────────
# MOOD DETECTION
# ─────────────────────────────────────────────
MOOD_STATES = {
    "calm": {
        "label": "Calm", "emoji": "😌",
        "color": "#2E7D32", "bg": "#E8F5E9", "border": "#66BB6A",
        "description": "Your responses suggest a calm, balanced state.",
        "tip": "Excellent baseline! Maintain your current routine.",
    },
    "focused": {
        "label": "Focused", "emoji": "🎯",
        "color": "#1565C0", "bg": "#E3F2FD", "border": "#42A5F5",
        "description": "Your response pattern suggests you are in a focused state.",
        "tip": "You're in a great headspace — channel this energy productively.",
    },
    "anxious": {
        "label": "Anxious", "emoji": "😰",
        "color": "#E65100", "bg": "#FFF3E0", "border": "#FFA726",
        "description": "Your responses indicate signs of anxiety or restlessness.",
        "tip": "Try box breathing: inhale 4s → hold 4s → exhale 4s → hold 4s.",
    },
    "frustrated": {
        "label": "Frustrated", "emoji": "😤",
        "color": "#B71C1C", "bg": "#FFEBEE", "border": "#EF5350",
        "description": "Patterns suggest some frustration or emotional dysregulation.",
        "tip": "It's okay to pause. A short walk can reset your emotional state.",
    },
    "overwhelmed": {
        "label": "Overwhelmed", "emoji": "😵",
        "color": "#4A148C", "bg": "#F3E5F5", "border": "#AB47BC",
        "description": "You appear to be experiencing high cognitive load right now.",
        "tip": "Break tasks into smaller steps. Focus on just one thing at a time.",
    },
}


def _domain_avgs(responses: dict) -> dict:
    """Return average score (0-4) per domain from current responses."""
    buckets = {d: [] for d in DOMAIN_ORDER}
    for q in ASRS_QUESTIONS:
        if q["id"] in responses:
            buckets[q["domain"]].append(SCORE_MAP.get(responses[q["id"]], 0))
    return {d: (np.mean(v) if v else 0.0) for d, v in buckets.items()}


def detect_mood(responses: dict) -> tuple:
    """Returns (mood_key, confidence 0-100)."""
    if not responses:
        return "calm", 50

    avgs = _domain_avgs(responses)
    hyper = avgs["Hyperactivity"]
    imp   = avgs["Impulsivity"]
    inatt = avgs["Inattention"]
    emo   = avgs["Emotional Regulation"]
    overall = np.mean(list(avgs.values()))

    if overall >= 3.2:
        mood, conf = "overwhelmed", min(100, int(overall / 4 * 100))
    elif emo >= 2.8 and (hyper >= 2.0 or inatt >= 2.5):
        mood, conf = "frustrated", min(100, int((emo + inatt) / 8 * 100))
    elif hyper >= 2.5 and imp >= 2.0:
        mood, conf = "anxious", min(100, int((hyper + imp) / 8 * 100))
    elif overall <= 1.0:
        mood, conf = "calm", min(100, int((4 - overall) / 4 * 100))
    else:
        mood, conf = "focused", min(100, int((4 - overall) / 4 * 80) + 10)

    return mood, conf


# ─────────────────────────────────────────────
# SEVERITY CALCULATION
# ─────────────────────────────────────────────
SEVERITY_LEVELS = [
    (25,  "Minimal",  "#2E7D32", "#E8F5E9"),
    (50,  "Mild",     "#558B2F", "#F9FBE7"),
    (72,  "Moderate", "#E65100", "#FFF3E0"),
    (101, "Severe",   "#B71C1C", "#FFEBEE"),
]


def calculate_severity(responses: dict) -> dict:
    avgs = _domain_avgs(responses)
    domain_pct = {d: round((avgs[d] / 4.0) * 100, 1) for d in DOMAIN_ORDER}
    total = round(sum(domain_pct[d] * DOMAIN_WEIGHTS[d] for d in DOMAIN_ORDER), 1)

    severity, color, bg = "Minimal", "#2E7D32", "#E8F5E9"
    for threshold, label, c, b in SEVERITY_LEVELS:
        if total < threshold:
            severity, color, bg = label, c, b
            break

    return {
        "total": total,
        "severity": severity,
        "color": color,
        "bg": bg,
        "domains": domain_pct,
        "inattention":   domain_pct["Inattention"],
        "hyperactivity": domain_pct["Hyperactivity"],
        "impulsivity":   domain_pct["Impulsivity"],
        "emotional_reg": domain_pct["Emotional Regulation"],
        "focus_org":     domain_pct["Focus"],
    }


# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
ASRS_CSS = """
<style>
/* ─── ASRS Page Variables ─── */
:root {
  --as-teal:   #00897B;
  --as-blue:   #1E88E5;
  --as-green:  #43A047;
  --as-bg:     #F0F7F4;
  --as-card:   #FFFFFF;
  --as-border: #CFE8DF;
  --as-radius: 14px;
  --as-shadow: 0 2px 18px rgba(0,137,123,0.10);
}

/* ─── Question Card ─── */
.asrs-question-card {
  background: var(--as-card);
  border: 1.5px solid var(--as-border);
  border-left: 5px solid var(--as-teal);
  border-radius: var(--as-radius);
  padding: 1.1rem 1.4rem;
  margin-bottom: 1rem;
  box-shadow: var(--as-shadow);
  transition: border-color 0.2s;
}
.asrs-question-card:hover { border-left-color: var(--as-blue); }
.asrs-question-card .q-domain {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--as-teal);
  margin-bottom: 0.3rem;
}
.asrs-question-card .q-text {
  font-size: 0.97rem;
  font-weight: 500;
  color: #1A3A3A;
  line-height: 1.55;
}

/* ─── Mood Indicator ─── */
.mood-card {
  border-radius: 14px;
  padding: 1.1rem 1.2rem;
  text-align: center;
  margin-bottom: 0.8rem;
  border: 2px solid transparent;
  transition: all 0.4s ease;
}
.mood-emoji { font-size: 2.6rem; line-height: 1.2; }
.mood-label { font-size: 1.05rem; font-weight: 700; margin: 0.2rem 0; }
.mood-desc  { font-size: 0.78rem; color: #555; line-height: 1.45; }
.mood-tip   {
  font-size: 0.75rem;
  margin-top: 0.5rem;
  padding: 0.4rem 0.7rem;
  border-radius: 8px;
  background: rgba(255,255,255,0.6);
  color: #333;
  font-style: italic;
}

/* ─── Progress Bar ─── */
.asrs-progress-wrap {
  background: #E0F2F1;
  border-radius: 20px;
  height: 10px;
  margin: 0.8rem 0 1.2rem;
  overflow: hidden;
}
.asrs-progress-fill {
  height: 100%;
  border-radius: 20px;
  background: linear-gradient(90deg, #00897B, #26C6DA);
  transition: width 0.4s ease;
}

/* ─── Severity Badge ─── */
.sev-badge {
  display: inline-block;
  padding: 0.4rem 1.3rem;
  border-radius: 50px;
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: 0.04em;
}

/* ─── Result Domain Bar ─── */
.domain-bar-wrap {
  margin-bottom: 0.6rem;
}
.domain-bar-label {
  display: flex;
  justify-content: space-between;
  font-size: 0.83rem;
  font-weight: 600;
  color: #2C4A3E;
  margin-bottom: 3px;
}
.domain-bar-bg {
  background: #E0F2F1;
  border-radius: 8px;
  height: 12px;
  overflow: hidden;
}
.domain-bar-fill {
  height: 100%;
  border-radius: 8px;
  transition: width 0.5s ease;
}

/* ─── History Card ─── */
.hist-card {
  background: #fff;
  border: 1.5px solid #CFE8DF;
  border-radius: 12px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.8rem;
  box-shadow: 0 1px 8px rgba(0,0,0,0.05);
}
.hist-card .hc-date { font-size: 0.75rem; color: #777; }
.hist-card .hc-score { font-size: 1.4rem; font-weight: 700; color: #00897B; }

/* ─── Comparison Delta ─── */
.delta-up   { color: #C62828; font-weight: 700; }
.delta-down { color: #2E7D32; font-weight: 700; }
.delta-same { color: #888;    font-weight: 600; }

/* ─── Section Header ─── */
.asrs-section-header {
  background: linear-gradient(135deg, #00897B 0%, #1E88E5 100%);
  color: white;
  border-radius: 12px;
  padding: 1.2rem 1.6rem;
  margin-bottom: 1.4rem;
}
.asrs-section-header h2 { color: white !important; margin: 0; font-size: 1.4rem; }
.asrs-section-header p  { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0; font-size: 0.9rem; }

/* ─── Encouraging Banner ─── */
.encourage-banner {
  background: linear-gradient(90deg, #E8F5E9, #E3F2FD);
  border-left: 4px solid #43A047;
  border-radius: 10px;
  padding: 0.8rem 1.1rem;
  font-size: 0.9rem;
  color: #1B5E20;
  margin-bottom: 1rem;
}

/* ─── Streak Badge ─── */
.streak-badge {
  background: linear-gradient(135deg, #FF8F00, #FFB300);
  color: white;
  border-radius: 50px;
  padding: 0.35rem 1rem;
  font-size: 0.85rem;
  font-weight: 700;
  display: inline-block;
}
</style>
"""


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _rkey(qid): return f"asrs_r_{qid}"

def _current_responses() -> dict:
    return {q["id"]: st.session_state[_rkey(q["id"])]
            for q in ASRS_QUESTIONS
            if _rkey(q["id"]) in st.session_state
            and st.session_state[_rkey(q["id"])] != "— Select —"}

def _answered_count() -> int:
    return len(_current_responses())

def _domain_color(domain: str) -> str:
    palette = {
        "Inattention":        "#1E88E5",
        "Hyperactivity":      "#E53935",
        "Impulsivity":        "#FB8C00",
        "Emotional Regulation": "#8E24AA",
        "Focus":              "#00897B",
    }
    return palette.get(domain, "#607D8B")


def _encouraging_msg(answered: int) -> str:
    msgs = {
        0:  "Take your time — there are no right or wrong answers. 💙",
        4:  "Good start! Keep going, you're doing great. 🌱",
        8:  "One third done! You're building a clearer picture. 🌟",
        12: "Halfway there — stay honest with yourself. 💪",
        16: "Almost done — just a few more questions! 🎯",
        20: "Final stretch! Your insights are nearly complete. ✨",
        24: "All done! Scroll down to see your results. 🎉",
    }
    for threshold in sorted(msgs.keys(), reverse=True):
        if answered >= threshold:
            return msgs[threshold]
    return msgs[0]


# ─────────────────────────────────────────────
# MOOD INDICATOR WIDGET
# ─────────────────────────────────────────────
def render_mood_indicator(responses: dict):
    mood_key, conf = detect_mood(responses)
    m = MOOD_STATES[mood_key]
    answered = len(responses)

    st.markdown(f"""
    <div class="mood-card" style="background:{m['bg']};border-color:{m['border']};">
      <div class="mood-emoji">{m['emoji']}</div>
      <div class="mood-label" style="color:{m['color']};">{m['label']}</div>
      <div class="mood-desc">{m['description']}</div>
      <div class="mood-tip">💡 {m['tip']}</div>
    </div>
    """, unsafe_allow_html=True)

    if answered > 0:
        st.caption(f"Based on {answered}/{TOTAL_Qs} responses · Confidence {conf}%")

    # Mini radar of answered domains
    if answered >= 4:
        avgs = _domain_avgs(responses)
        labels = list(avgs.keys())
        values = [round(v / 4 * 100, 1) for v in avgs.values()]

        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill='toself',
            fillcolor='rgba(0,137,123,0.15)',
            line=dict(color='#00897B', width=2),
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100],
                                tickfont=dict(size=8), gridcolor="#CFE8DF"),
                angularaxis=dict(tickfont=dict(size=9))
            ),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            height=220,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True, key="mood_radar")


# ─────────────────────────────────────────────
# ASSESSMENT PAGE
# ─────────────────────────────────────────────
def render_assessment_page(user_id: int):
    st.markdown("""
    <div class="asrs-section-header">
      <h2>🧠 ASRS ADHD Self-Assessment</h2>
      <p>Based on the WHO Adult ADHD Self-Report Scale (ASRS v1.1) — 24 validated questions</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Patient name ──────────────────────────────
    col_name, col_streak = st.columns([3, 1])
    with col_name:
        patient_name = st.text_input(
            "Your name (used to load your history)",
            value=st.session_state.get("asrs_patient", ""),
            placeholder="Enter your name…",
            key="asrs_name_input",
        )
        if patient_name:
            st.session_state["asrs_patient"] = patient_name

    if not patient_name:
        st.info("Enter your name above to begin the assessment.")
        return

    # ── Load last session info ────────────────────
    last = get_last_session(user_id, patient_name)
    history = get_asrs_history(user_id, patient_name)
    streak  = count_consecutive_streak(history)

    with col_streak:
        if streak > 0:
            st.markdown(f'<div class="streak-badge">🔥 {streak}-day streak</div>',
                        unsafe_allow_html=True)

    if last:
        st.markdown(f"""
        <div class="encourage-banner">
          👋 Welcome back, <b>{patient_name}</b>!
          Your last assessment was on <b>{last['session_date']}</b>
          with a score of <b>{last['total_score']}%</b> ({last['severity']}).
        </div>
        """, unsafe_allow_html=True)

    # ── Progress bar ──────────────────────────────
    answered = _answered_count()
    pct = int(answered / TOTAL_Qs * 100)
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#555;margin-top:0.5rem;">
      <span>{_encouraging_msg(answered)}</span>
      <span><b>{answered}/{TOTAL_Qs}</b> answered</span>
    </div>
    <div class="asrs-progress-wrap">
      <div class="asrs-progress-fill" style="width:{pct}%;"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column layout: questions | mood live ──
    q_col, mood_col = st.columns([2, 1], gap="large")

    with mood_col:
        st.markdown("#### 🎭 Live Mood Indicator")
        st.caption("Updates as you answer each question")
        render_mood_indicator(_current_responses())

    with q_col:
        # Group questions by domain for nicer display
        from itertools import groupby
        domain_grouped = {}
        for q in ASRS_QUESTIONS:
            domain_grouped.setdefault(q["domain"], []).append(q)

        for domain in DOMAIN_ORDER:
            questions = domain_grouped.get(domain, [])
            color = _domain_color(domain)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.6rem;margin:1.2rem 0 0.6rem;">
              <div style="width:4px;height:22px;background:{color};border-radius:2px;"></div>
              <span style="font-weight:700;font-size:1rem;color:{color};">{domain}</span>
            </div>
            """, unsafe_allow_html=True)

            for q in questions:
                st.markdown(f"""
                <div class="asrs-question-card" style="border-left-color:{color};">
                  <div class="q-domain">{q['emoji']} {q['domain']}</div>
                  <div class="q-text">{q['text']}</div>
                </div>
                """, unsafe_allow_html=True)

                current_val = st.session_state.get(_rkey(q["id"]), "— Select —")
                opts = ["— Select —"] + RESPONSE_OPTIONS
                idx = opts.index(current_val) if current_val in opts else 0

                st.selectbox(
                    label=f"Response for {q['id']}",
                    options=opts,
                    index=idx,
                    key=_rkey(q["id"]),
                    label_visibility="collapsed",
                )

    # ── Refresh mood after all questions rendered ──
    st.markdown("---")
    answered = _answered_count()
    responses = _current_responses()

    if answered < TOTAL_Qs:
        remaining = TOTAL_Qs - answered
        st.warning(f"Please answer all questions. {remaining} remaining.")
        return

    # ── Show Results ──────────────────────────────
    _render_results(user_id, patient_name, responses, last, history)


# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
def _domain_bar_html(label: str, pct: float, color: str) -> str:
    return f"""
    <div class="domain-bar-wrap">
      <div class="domain-bar-label">
        <span>{label}</span><span>{pct:.1f}%</span>
      </div>
      <div class="domain-bar-bg">
        <div class="domain-bar-fill" style="width:{pct}%;background:{color};"></div>
      </div>
    </div>"""


def _render_results(user_id, patient_name, responses, last, history):
    result = calculate_severity(responses)
    mood_key, conf = detect_mood(responses)
    m = MOOD_STATES[mood_key]

    st.markdown("""
    <div class="asrs-section-header">
      <h2>📊 Your Assessment Results</h2>
      <p>Based on your 24 responses — scroll down for detailed breakdown</p>
    </div>
    """, unsafe_allow_html=True)

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Overall Score", f"{result['total']}%")
    with c2:
        st.markdown(f"""
        <div style="text-align:center;">
          <div style="font-size:0.8rem;color:#555;">Severity</div>
          <div class="sev-badge" style="background:{result['bg']};color:{result['color']};
               border:2px solid {result['color']};margin-top:0.3rem;">
            {result['severity']}
          </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div style="text-align:center;">
          <div style="font-size:0.8rem;color:#555;">Current Mood</div>
          <div class="mood-card" style="background:{m['bg']};border-color:{m['border']};
               padding:0.5rem;margin-top:0.3rem;">
            <span style="font-size:1.5rem;">{m['emoji']}</span>
            <span style="font-weight:700;color:{m['color']};margin-left:0.4rem;">{m['label']}</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.metric("Questions Answered", f"{TOTAL_Qs}/{TOTAL_Qs}")

    st.markdown("<br>", unsafe_allow_html=True)

    res_col, chart_col = st.columns([1, 1], gap="large")

    with res_col:
        st.markdown("#### Domain Breakdown")
        domain_colors = {d: _domain_color(d) for d in DOMAIN_ORDER}
        bars_html = "".join(
            _domain_bar_html(d, result["domains"][d], domain_colors[d])
            for d in DOMAIN_ORDER
        )
        st.markdown(bars_html, unsafe_allow_html=True)

        # Severity description
        sev_desc = {
            "Minimal": "Your symptoms are minimal. Excellent self-management!",
            "Mild": "Mild symptoms detected. Some strategies may help.",
            "Moderate": "Moderate symptoms. Consider speaking with a professional.",
            "Severe": "Significant symptoms. Please consult a healthcare provider.",
        }
        st.info(f"**{result['severity']} ADHD Indicators** — {sev_desc.get(result['severity'], '')}")

    with chart_col:
        st.markdown("#### Symptom Radar")
        labels = DOMAIN_ORDER
        values = [result["domains"][d] for d in labels]
        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill='toself',
            fillcolor=f"rgba(0,137,123,0.2)",
            line=dict(color='#00897B', width=2.5),
            marker=dict(size=6, color='#00897B'),
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100],
                                tickfont=dict(size=9), gridcolor="#CFE8DF"),
                angularaxis=dict(tickfont=dict(size=10))
            ),
            showlegend=False,
            margin=dict(l=30, r=30, t=30, b=30),
            height=320,
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True, key="result_radar")

    # Comparison with previous session
    if last:
        st.markdown("---")
        st.markdown("#### 📅 Comparison with Previous Session")
        prev_score = last.get("total_score", 0)
        delta = result["total"] - prev_score
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        delta_class = "delta-up" if delta > 10 else ("delta-down" if delta < -10 else "delta-same")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Previous Score", f"{prev_score:.1f}%")
        with col_b:
            st.metric("Today's Score", f"{result['total']:.1f}%")
        with col_c:
            st.markdown(f"""
            <div style="text-align:center;">
              <div style="font-size:0.8rem;color:#555;">Change</div>
              <div class="{delta_class}" style="font-size:1.5rem;">{delta_str}</div>
            </div>""", unsafe_allow_html=True)

        # Domain-level comparison
        prev_domains = {
            "Inattention": last.get("inattention", 0),
            "Hyperactivity": last.get("hyperactivity", 0),
            "Impulsivity": last.get("impulsivity", 0),
            "Emotional Regulation": last.get("emotional_reg", 0),
            "Focus": last.get("focus_org", 0),
        }
        cmp_rows = []
        for d in DOMAIN_ORDER:
            cur = result["domains"][d]
            prev = prev_domains.get(d, 0)
            diff = cur - prev
            flag = ""
            if abs(diff) >= 20:
                flag = "⚠️ Significant change"
            elif diff > 0:
                flag = "↑ Increased"
            elif diff < 0:
                flag = "↓ Improved"
            cmp_rows.append({"Domain": d, "Previous": f"{prev:.1f}%",
                             "Today": f"{cur:.1f}%",
                             "Change": f"{diff:+.1f}%", "Note": flag})
        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

    # Save to DB
    save_data = {
        "user_id": user_id,
        "patient_name": patient_name,
        "session_date": str(date.today()),
        "inattention":   result["inattention"],
        "hyperactivity": result["hyperactivity"],
        "impulsivity":   result["impulsivity"],
        "emotional_reg": result["emotional_reg"],
        "focus_org":     result["focus_org"],
        "total_score":   result["total"],
        "severity":      result["severity"],
        "mood_detected": mood_key,
        "mood_score":    conf,
        "responses":     json.dumps(responses),
    }
    session_key = f"asrs_saved_{patient_name}_{date.today()}"
    if not st.session_state.get(session_key):
        save_asrs_session(save_data)
        st.session_state[session_key] = True
        st.success("✅ Assessment saved successfully!")

    # PDF download
    st.markdown("---")
    _pdf_section(patient_name, result, mood_key, m, last)

    # Reset button
    if st.button("🔄 Start New Assessment", use_container_width=True):
        for q in ASRS_QUESTIONS:
            k = _rkey(q["id"])
            if k in st.session_state:
                del st.session_state[k]
        if session_key in st.session_state:
            del st.session_state[session_key]
        st.rerun()


# ─────────────────────────────────────────────
# PDF EXPORT
# ─────────────────────────────────────────────
def _pdf_section(patient_name, result, mood_key, m, last):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, rightMargin=0.75*inch, leftMargin=0.75*inch,
                             topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('title', parent=styles['Title'],
                                  fontSize=18, spaceAfter=6,
                                  textColor=rl_colors.HexColor('#00897B'))
    sub_style = ParagraphStyle('sub', parent=styles['Normal'],
                                fontSize=10, textColor=rl_colors.HexColor('#555555'))
    head_style = ParagraphStyle('head', parent=styles['Heading2'],
                                 fontSize=12, textColor=rl_colors.HexColor('#1E88E5'),
                                 spaceBefore=14, spaceAfter=4)

    story.append(Paragraph("ASRS ADHD Self-Assessment Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}", sub_style))
    story.append(Spacer(1, 0.2*inch))

    info_data = [
        ["Patient Name", patient_name],
        ["Assessment Date", str(date.today())],
        ["Overall Score", f"{result['total']}%"],
        ["Severity", result['severity']],
        ["Mood Detected", m['label']],
    ]
    info_table = Table(info_data, colWidths=[2.2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), rl_colors.HexColor('#E0F2F1')),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#CFE8DF')),
        ('ROWBACKGROUNDS', (1, 0), (1, -1), [rl_colors.white]),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Domain Scores", head_style))
    domain_data = [["Domain", "Score", "Level"]] + [
        [d, f"{result['domains'][d]:.1f}%",
         "High" if result['domains'][d] >= 65 else ("Moderate" if result['domains'][d] >= 40 else "Low")]
        for d in DOMAIN_ORDER
    ]
    domain_table = Table(domain_data, colWidths=[2.8*inch, 1.5*inch, 1.5*inch])
    domain_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#00897B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [rl_colors.HexColor('#F0F7F4'), rl_colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#CFE8DF')),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(domain_table)

    if last:
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Comparison with Previous Session", head_style))
        prev_score = last.get("total_score", 0)
        delta = result["total"] - prev_score
        story.append(Paragraph(
            f"Previous score: {prev_score:.1f}% | Today: {result['total']:.1f}% | "
            f"Change: {delta:+.1f}%",
            styles['Normal']))

    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(
        "⚠️ This tool is for self-screening purposes only and does not constitute a medical diagnosis. "
        "Please consult a qualified healthcare professional for clinical evaluation.",
        ParagraphStyle('disclaimer', parent=styles['Normal'], fontSize=8,
                       textColor=rl_colors.HexColor('#888888'))))

    doc.build(story)
    buf.seek(0)

    st.download_button(
        label="⬇️ Download PDF Report",
        data=buf,
        file_name=f"ASRS_Report_{patient_name}_{date.today()}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ─────────────────────────────────────────────
# PROGRESS REPORT PAGE
# ─────────────────────────────────────────────
def render_progress_page(user_id: int):
    st.markdown("""
    <div class="asrs-section-header">
      <h2>📈 Progress Report</h2>
      <p>Track your ADHD score and mood trends over time</p>
    </div>
    """, unsafe_allow_html=True)

    patient_name = st.text_input(
        "Patient name", value=st.session_state.get("asrs_patient", ""),
        placeholder="Enter your name to load history…", key="prog_name"
    )
    if patient_name:
        st.session_state["asrs_patient"] = patient_name

    if not patient_name:
        st.info("Enter your name to view your progress report.")
        return

    history = get_asrs_history(user_id, patient_name, limit=60)
    if len(history) < 2:
        st.info("Complete at least 2 assessments to see your progress report.")
        return

    df = pd.DataFrame(history)
    df["session_date"] = pd.to_datetime(df["session_date"])
    df = df.sort_values("session_date")

    streak = count_consecutive_streak(history)
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Sessions", len(history))
    with col2: st.metric("Current Streak", f"🔥 {streak} days")
    with col3: st.metric("Best Score", f"{df['total_score'].min():.1f}%")
    with col4: st.metric("Latest Score", f"{df['total_score'].iloc[-1]:.1f}%")

    # ── Overall Score Trend ───────────────────────
    st.markdown("#### Overall ADHD Score Over Time")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df["session_date"], y=df["total_score"],
        mode='lines+markers',
        line=dict(color='#00897B', width=2.5),
        marker=dict(size=8, color='#00897B'),
        name='Score',
        fill='tozeroy',
        fillcolor='rgba(0,137,123,0.10)',
    ))
    for threshold, label, color, _ in SEVERITY_LEVELS:
        fig_trend.add_hline(y=threshold if threshold < 101 else 100,
                            line_dash="dot", line_color=color,
                            annotation_text=label, annotation_position="right")
    fig_trend.update_layout(
        xaxis_title="Date", yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100]),
        height=350, margin=dict(l=20, r=60, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,253,251,1)',
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Domain Trend ──────────────────────────────
    st.markdown("#### Domain Scores Over Time")
    domain_col_map = {
        "Inattention":        "inattention",
        "Hyperactivity":      "hyperactivity",
        "Impulsivity":        "impulsivity",
        "Emotional Regulation": "emotional_reg",
        "Focus":              "focus_org",
    }
    fig_domains = go.Figure()
    for domain, col in domain_col_map.items():
        if col in df.columns:
            fig_domains.add_trace(go.Scatter(
                x=df["session_date"], y=df[col],
                mode='lines+markers',
                name=domain,
                line=dict(color=_domain_color(domain), width=2),
                marker=dict(size=6),
            ))
    fig_domains.update_layout(
        xaxis_title="Date", yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100]),
        height=350, margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,253,251,1)',
    )
    st.plotly_chart(fig_domains, use_container_width=True)

    # ── Mood Pattern ─────────────────────────────
    st.markdown("#### Mood Pattern Over Time")
    mood_order = ["calm", "focused", "anxious", "frustrated", "overwhelmed"]
    mood_labels = {k: MOOD_STATES[k]["label"] for k in mood_order}
    mood_colors = {k: MOOD_STATES[k]["color"] for k in mood_order}

    df["mood_label"] = df["mood_detected"].map(
        lambda x: MOOD_STATES.get(x, {}).get("label", x))
    df["mood_color"] = df["mood_detected"].map(
        lambda x: MOOD_STATES.get(x, {}).get("color", "#888"))

    fig_mood = go.Figure()
    for mood in mood_order:
        mask = df["mood_detected"] == mood
        if mask.any():
            fig_mood.add_trace(go.Scatter(
                x=df.loc[mask, "session_date"],
                y=df.loc[mask, "mood_score"],
                mode='markers',
                name=MOOD_STATES[mood]["label"],
                marker=dict(size=14, color=MOOD_STATES[mood]["color"],
                            symbol='circle',
                            line=dict(width=1, color='white')),
                text=df.loc[mask, "mood_label"],
                hovertemplate="%{text}<br>Confidence: %{y}%<br>%{x}<extra></extra>",
            ))
    fig_mood.update_layout(
        xaxis_title="Date", yaxis_title="Confidence (%)",
        height=300, margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,253,251,1)',
    )
    st.plotly_chart(fig_mood, use_container_width=True)

    # ── Mood Distribution ─────────────────────────
    mood_counts = df["mood_detected"].value_counts().reset_index()
    mood_counts.columns = ["mood", "count"]
    mood_counts["label"] = mood_counts["mood"].map(
        lambda x: MOOD_STATES.get(x, {}).get("label", x))
    mood_counts["color"] = mood_counts["mood"].map(
        lambda x: MOOD_STATES.get(x, {}).get("color", "#888"))

    fig_pie = go.Figure(go.Pie(
        labels=mood_counts["label"], values=mood_counts["count"],
        marker=dict(colors=mood_counts["color"].tolist()),
        hole=0.4,
        textinfo='label+percent',
    ))
    fig_pie.update_layout(
        title="Mood Distribution", height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
    )

    # ── Calendar Heatmap ──────────────────────────
    st.markdown("#### Assessment Calendar (Last 60 Days)")
    all_dates = pd.date_range(end=date.today(), periods=60).date
    session_dates = set(df["session_date"].dt.date.tolist())
    cal_data = pd.DataFrame({
        "date": all_dates,
        "done": [1 if d in session_dates else 0 for d in all_dates],
        "score": [
            df.loc[df["session_date"].dt.date == d, "total_score"].values[0]
            if d in session_dates else None
            for d in all_dates
        ],
    })
    cal_data["week"] = [d.isocalendar()[1] for d in cal_data["date"]]
    cal_data["weekday"] = [d.weekday() for d in cal_data["date"]]
    cal_data["label"] = cal_data["date"].astype(str)

    fig_cal = go.Figure(go.Heatmap(
        x=cal_data["week"],
        y=cal_data["weekday"],
        z=cal_data["score"].fillna(-1),
        text=cal_data["label"],
        hovertemplate="Date: %{text}<br>Score: %{z:.1f}%<extra></extra>",
        colorscale=[
            [0, "#ECFDF5"], [0.001, "#ECFDF5"],
            [0.001, "#A7F3D0"], [0.4, "#6EE7B7"],
            [0.6, "#34D399"], [0.8, "#10B981"], [1.0, "#047857"],
        ],
        showscale=True,
        colorbar=dict(title="Score %", thickness=12),
        zmin=0, zmax=100,
    ))
    fig_cal.update_layout(
        yaxis=dict(
            tickvals=[0,1,2,3,4,5,6],
            ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            autorange="reversed",
        ),
        xaxis=dict(title="Week"),
        height=230,
        margin=dict(l=40, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # ── Week-over-week insight ─────────────────────
    if len(df) >= 2:
        latest  = df.iloc[-1]["total_score"]
        prev    = df.iloc[-2]["total_score"]
        delta   = latest - prev
        dir_str = "improved" if delta < 0 else ("worsened" if delta > 0 else "unchanged")
        emoji   = "✅" if delta < 0 else ("⚠️" if delta > 5 else "➡️")

        # Domain with biggest improvement
        domain_deltas = {}
        for d, col in domain_col_map.items():
            if col in df.columns and len(df) >= 2:
                domain_deltas[d] = df.iloc[-1][col] - df.iloc[-2][col]
        if domain_deltas:
            worst_d  = max(domain_deltas, key=domain_deltas.get)
            best_d   = min(domain_deltas, key=domain_deltas.get)

        st.markdown(f"""
        <div class="encourage-banner">
          {emoji} Your overall score <b>{dir_str}</b> by <b>{abs(delta):.1f}%</b>
          compared to your previous session.<br>
          🏆 Best improvement: <b>{best_d}</b> ({domain_deltas.get(best_d, 0):+.1f}%)
          &nbsp;|&nbsp;
          ⚠️ Needs attention: <b>{worst_d}</b> ({domain_deltas.get(worst_d, 0):+.1f}%)
        </div>
        """, unsafe_allow_html=True)

    # ── Raw History Table ─────────────────────────
    with st.expander("📄 Full Session History"):
        display_df = df[["session_date", "total_score", "severity",
                         "mood_detected", "inattention", "hyperactivity",
                         "impulsivity", "emotional_reg", "focus_org"]].copy()
        display_df.columns = ["Date", "Total %", "Severity", "Mood",
                              "Inattention", "Hyperactivity", "Impulsivity",
                              "Emotional Reg", "Focus"]
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# HISTORY PAGE
# ─────────────────────────────────────────────
def render_history_page(user_id: int):
    st.markdown("""
    <div class="asrs-section-header">
      <h2>🗂️ Assessment History</h2>
      <p>Review and compare your past ADHD assessments</p>
    </div>
    """, unsafe_allow_html=True)

    patient_name = st.text_input(
        "Patient name", value=st.session_state.get("asrs_patient", ""),
        placeholder="Enter your name…", key="hist_name"
    )
    if patient_name:
        st.session_state["asrs_patient"] = patient_name

    if not patient_name:
        st.info("Enter your name to view your assessment history.")
        return

    history = get_asrs_history(user_id, patient_name, limit=60)
    if not history:
        st.info("No assessments found. Complete your first assessment to see history here.")
        return

    streak = count_consecutive_streak(history)
    if streak > 0:
        st.markdown(f'<div class="streak-badge">🔥 {streak}-day streak! Keep it up!</div><br>',
                    unsafe_allow_html=True)

    # Side-by-side comparison: last 2 sessions
    if len(history) >= 2:
        st.markdown("#### Side-by-Side Comparison: Last Two Sessions")
        s1, s2 = history[0], history[1]  # newest first
        c1, c2 = st.columns(2)
        for col, sess, label in [(c1, s2, "Previous"), (c2, s1, "Latest")]:
            with col:
                m_info = MOOD_STATES.get(sess.get("mood_detected", "calm"), MOOD_STATES["calm"])
                st.markdown(f"""
                <div class="hist-card" style="border-top:4px solid #00897B;">
                  <div class="hc-date">📅 {label} — {sess['session_date']}</div>
                  <div class="hc-score">{sess['total_score']:.1f}%</div>
                  <div style="margin:0.3rem 0;">
                    <span class="sev-badge" style="font-size:0.85rem;
                      background:#E0F2F1;color:#00897B;border:1px solid #80CBC4;">
                      {sess['severity']}
                    </span>
                    &nbsp; {m_info['emoji']} {m_info['label']}
                  </div>
                  <hr style="border-color:#E0F2F1;margin:0.5rem 0;">
                  <div style="font-size:0.82rem;color:#555;">
                    Inattention: {sess['inattention']:.0f}% &nbsp;|&nbsp;
                    Hyperactivity: {sess['hyperactivity']:.0f}%<br>
                    Impulsivity: {sess['impulsivity']:.0f}% &nbsp;|&nbsp;
                    Emotional Reg: {sess['emotional_reg']:.0f}%<br>
                    Focus: {sess['focus_org']:.0f}%
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # Flag significant domain changes
        flags = []
        domain_col_map = {
            "Inattention": "inattention", "Hyperactivity": "hyperactivity",
            "Impulsivity": "impulsivity", "Emotional Regulation": "emotional_reg",
            "Focus": "focus_org",
        }
        for d, col in domain_col_map.items():
            delta = s1.get(col, 0) - s2.get(col, 0)
            if delta >= 20:
                flags.append(f"⚠️ **{d}** increased by **{delta:.0f}%** — worth monitoring.")
            elif delta <= -20:
                flags.append(f"✅ **{d}** improved by **{abs(delta):.0f}%** — great progress!")
        if flags:
            st.markdown("**Significant Changes Detected:**")
            for f in flags:
                st.markdown(f)

    # All past sessions
    st.markdown("#### All Past Sessions")
    for i, sess in enumerate(history):
        m_info = MOOD_STATES.get(sess.get("mood_detected", "calm"), MOOD_STATES["calm"])
        sev_color = {"Minimal": "#2E7D32", "Mild": "#558B2F",
                     "Moderate": "#E65100", "Severe": "#B71C1C"}.get(sess["severity"], "#607D8B")
        with st.expander(
            f"📅 {sess['session_date']} — Score: {sess['total_score']:.1f}% "
            f"| {sess['severity']} | {m_info['emoji']} {m_info['label']}"
        ):
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.metric("Inattention",    f"{sess['inattention']:.1f}%")
                st.metric("Hyperactivity",  f"{sess['hyperactivity']:.1f}%")
            with dc2:
                st.metric("Impulsivity",    f"{sess['impulsivity']:.1f}%")
                st.metric("Emotional Reg",  f"{sess['emotional_reg']:.1f}%")
            with dc3:
                st.metric("Focus",          f"{sess['focus_org']:.1f}%")
                st.metric("Total Score",    f"{sess['total_score']:.1f}%")
            st.caption(f"Recorded: {sess['created_at']} | Mood confidence: {sess.get('mood_score', 0):.0f}%")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────
def render_asrs_module():
    """Call this from app.py when the user selects the ASRS module."""
    # Ensure DB is ready
    init_asrs_db()

    # Inject CSS
    st.markdown(ASRS_CSS, unsafe_allow_html=True)

    user = st.session_state.get("user")
    if not user:
        st.warning("Please log in to use the ASRS Assessment module.")
        return

    user_id = user.get("id", 0)

    # Sub-navigation (sidebar)
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🧠 ASRS Module")
        asrs_page = st.radio(
            "Navigate",
            ["Take Assessment", "Progress Report", "History"],
            key="asrs_nav",
            label_visibility="collapsed",
        )

        # Quick stats
        patient = st.session_state.get("asrs_patient", "")
        if patient:
            history = get_asrs_history(user_id, patient, limit=60)
            if history:
                streak = count_consecutive_streak(history)
                st.markdown(f"**Patient:** {patient}")
                st.markdown(f"**Sessions:** {len(history)}")
                if streak:
                    st.markdown(f"**Streak:** 🔥 {streak} days")

    if asrs_page == "Take Assessment":
        render_assessment_page(user_id)
    elif asrs_page == "Progress Report":
        render_progress_page(user_id)
    elif asrs_page == "History":
        render_history_page(user_id)
