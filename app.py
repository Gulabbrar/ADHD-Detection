import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from datetime import datetime
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from database import get_connection, save_assessment, get_patient_history, init_db
from auth import login_user, register_user
from asrs_assessment import render_asrs_module, init_asrs_db
from eeg_dashboard import render_eeg_dashboard

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Vanderbilt ADHD Clinical System",
    page_icon="",
    layout="wide"
)

# =============================
# COMPREHENSIVE CSS STYLING
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ─── Variables ─── */
:root {
  --primary:       #1565c0;
  --primary-dark:  #0d47a1;
  --primary-light: #42a5f5;
  --bg:            #f0f4f8;
  --card:          #ffffff;
  --text:          #0a2540;
  --muted:         #546e7a;
  --border:        #dde3ec;
  --radius:        12px;
  --shadow:        0 2px 16px rgba(10,37,64,0.09);
}

/* ─── Base ─── */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.main { background-color: var(--bg) !important; }
.main .block-container { padding: 1.5rem 2.5rem 2rem; max-width: 1280px; }

/* ─── Headings ─── */
h1 { color: var(--primary) !important; font-weight: 700 !important; }
h2 { color: var(--primary) !important; font-weight: 600 !important; }
h3 { color: var(--text)    !important; font-weight: 600 !important; }

/* ─── App Header Banner ─── */
.app-header {
  background: linear-gradient(135deg, #0d47a1 0%, #1565c0 55%, #1976d2 100%);
  padding: 1.4rem 2rem;
  border-radius: var(--radius);
  margin-bottom: 1.75rem;
  display: flex;
  align-items: center;
  gap: 1.25rem;
  box-shadow: 0 4px 24px rgba(13,71,161,0.28);
}
.app-header-icon { font-size: 2.8rem; line-height: 1; }
.app-header h1  { color: white !important; font-size: 1.5rem !important; font-weight: 700 !important; margin: 0 !important; }
.app-header p   { color: rgba(255,255,255,0.8); font-size: 0.82rem; margin: 0.2rem 0 0; }
.header-badge {
  margin-left: auto;
  background: rgba(255,255,255,0.18);
  border: 1px solid rgba(255,255,255,0.35);
  color: white;
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.3rem 0.9rem;
  border-radius: 50px;
  white-space: nowrap;
}

/* ─── Login Card ─── */
.login-header {
  background: linear-gradient(135deg, #0d47a1, #1565c0);
  padding: 2rem;
  border-radius: 14px 14px 0 0;
  text-align: center;
}
.login-header h2 { color: white !important; font-size: 1.25rem !important; font-weight: 700 !important; margin: 0.5rem 0 0 !important; }
.login-header p  { color: rgba(255,255,255,0.8); font-size: 0.82rem; margin: 0.3rem 0 0; }
.login-icon { font-size: 2.8rem; }
.login-body {
  background: white;
  border: 1px solid var(--border);
  border-top: none;
  border-radius: 0 0 14px 14px;
  padding: 1.75rem;
  box-shadow: 0 8px 32px rgba(10,37,64,0.12);
}

/* ─── Section Header ─── */
.section-hdr {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.7rem 1.2rem;
  background: white;
  border-radius: var(--radius);
  border-left: 5px solid var(--primary);
  box-shadow: var(--shadow);
  margin-bottom: 1rem;
}
.section-hdr span { font-size: 0.95rem; font-weight: 600; color: var(--primary); }

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a2540 0%, #0d47a1 100%) !important;
  border-right: none !important;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span { color: rgba(255,255,255,0.9) !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: white !important; }
section[data-testid="stSidebar"] hr  { border-color: rgba(255,255,255,0.2) !important; }

/* ─── Metric Cards ─── */
[data-testid="metric-container"] {
  background: white !important;
  border: 1px solid var(--border) !important;
  border-top: 4px solid var(--primary) !important;
  border-radius: var(--radius) !important;
  padding: 1rem 1.25rem !important;
  box-shadow: var(--shadow) !important;
  transition: box-shadow 0.2s;
}
[data-testid="metric-container"]:hover { box-shadow: 0 6px 24px rgba(10,37,64,0.14) !important; }
[data-testid="stMetricLabel"] {
  font-size: 0.72rem !important; font-weight: 600 !important;
  color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.6px;
}
[data-testid="stMetricValue"] {
  font-size: 1.85rem !important; font-weight: 700 !important; color: var(--primary) !important;
}

/* ─── Primary Buttons ─── */
.stButton > button {
  background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.55rem 1.6rem !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.88rem !important;
  font-weight: 600 !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 10px rgba(21,101,192,0.28) !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 5px 18px rgba(21,101,192,0.38) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ─── Download Button ─── */
.stDownloadButton > button {
  background: linear-gradient(135deg, #00897b, #00695c) !important;
  color: white !important; border: none !important; border-radius: 8px !important;
  font-weight: 600 !important; box-shadow: 0 2px 10px rgba(0,105,92,0.28) !important;
}
.stDownloadButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 5px 16px rgba(0,105,92,0.36) !important; }

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px; background: #e2e8f0; padding: 4px; border-radius: 10px; margin-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 7px !important; font-weight: 500 !important; font-size: 0.82rem !important;
  padding: 0.45rem 0.9rem !important; color: #64748b !important;
  background: transparent !important; border: none !important;
}
.stTabs [aria-selected="true"] {
  background: white !important; color: var(--primary) !important;
  box-shadow: 0 1px 6px rgba(0,0,0,0.1) !important; font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: white; border-radius: 0 0 var(--radius) var(--radius);
  padding: 1.25rem 1rem; box-shadow: var(--shadow);
  border: 1px solid var(--border); border-top: none;
}

/* ─── Question Items ─── */
.question-item {
  background: #f8faff;
  border: 1px solid #dde3ec;
  border-left: 4px solid var(--primary-light);
  border-radius: 8px;
  padding: 0.85rem 1.1rem;
  margin-bottom: 0.25rem;
}
.q-num {
  display: inline-block; background: var(--primary); color: white;
  font-size: 0.68rem; font-weight: 700; padding: 0.15rem 0.5rem;
  border-radius: 50px; margin-right: 0.4rem; vertical-align: middle;
}
.q-text { font-size: 0.9rem; font-weight: 500; color: var(--text); line-height: 1.5; }

/* ─── Inputs ─── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
  border: 1.5px solid var(--border) !important; border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important; font-size: 0.88rem !important;
  padding: 0.5rem 0.75rem !important; transition: all 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
  border-color: var(--primary) !important; box-shadow: 0 0 0 3px rgba(21,101,192,0.12) !important;
}
.stSelectbox > div > div { border: 1.5px solid var(--border) !important; border-radius: 8px !important; }

/* ─── Severity Badges ─── */
.severity-badge {
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.55rem 1.3rem; border-radius: 50px; font-weight: 700; font-size: 0.95rem;
}
.severity-severe   { background: #ffebee; color: #b71c1c; border: 2px solid #ef9a9a; }
.severity-moderate { background: #fff3e0; color: #bf360c; border: 2px solid #ffb74d; }
.severity-mild     { background: #e8f5e9; color: #1b5e20; border: 2px solid #81c784; }

/* ─── Result Header ─── */
.result-header {
  background: linear-gradient(135deg, #0d47a1, #1976d2);
  padding: 1.4rem 2rem; border-radius: var(--radius); margin: 1.5rem 0 1.25rem;
  box-shadow: 0 4px 20px rgba(13,71,161,0.25);
}
.result-header h3 { color: white !important; font-size: 1.15rem !important; margin: 0 !important; }
.result-header p  { color: rgba(255,255,255,0.75); font-size: 0.82rem; margin: 0.25rem 0 0; }

/* ─── Prediction Card ─── */
.prediction-card {
  background: white; border: 2px solid var(--primary); border-radius: var(--radius);
  padding: 1.25rem 1.5rem; text-align: center; box-shadow: var(--shadow);
}
.prediction-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); }
.prediction-value { font-size: 1.35rem; font-weight: 700; color: var(--primary); margin-top: 0.3rem; }

/* ─── Misc ─── */
hr { border: none !important; border-top: 1px solid #edf2f7 !important; margin: 0.5rem 0 !important; }
.stDataFrame { border-radius: var(--radius) !important; box-shadow: var(--shadow) !important; }
.stAlert { border-radius: var(--radius) !important; }
.stRadio label { font-size: 0.85rem !important; font-family: 'Inter', sans-serif !important; }

/* ─── Mood Tracker ─── */
.mood-card {
  background: white;
  border: 1px solid var(--border);
  border-left: 5px solid #7e57c2;
  border-radius: var(--radius);
  padding: 1.1rem 1.4rem;
  box-shadow: var(--shadow);
  margin-bottom: 1rem;
}
.mood-card h4 { color: #4527a0 !important; font-size: 0.95rem !important; margin: 0 0 0.4rem !important; }
.mood-card p  { color: var(--muted); font-size: 0.8rem; margin: 0; }

/* ─── Improvement Badge ─── */
.improve-badge {
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.4rem 1rem; border-radius: 50px; font-weight: 700; font-size: 0.85rem;
}
.improve-up   { background: #e8f5e9; color: #1b5e20; border: 2px solid #81c784; }
.improve-down { background: #ffebee; color: #b71c1c; border: 2px solid #ef9a9a; }
.improve-same { background: #e3f2fd; color: #0d47a1; border: 2px solid #90caf9; }
</style>
""", unsafe_allow_html=True)

# =============================
# LOAD MODEL
# =============================
model = joblib.load("adhd_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Ensure new DB columns exist
init_db()
init_asrs_db()

# =============================
# SESSION INIT
# =============================
if "user" not in st.session_state:
    st.session_state.user = None

# =============================
# LOGIN / REGISTER
# =============================
if st.session_state.user is None:

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("""
        <div class="login-header">
          <div class="login-icon">🏥</div>
          <h2>Vanderbilt ADHD Clinical System</h2>
          <p>Department of Child &amp; Adolescent Psychiatry</p>
        </div>
        <div class="login-body">
        """, unsafe_allow_html=True)

        tab = st.selectbox("", ["Login", "Register"], label_visibility="collapsed")
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        if tab == "Register":
            role = st.selectbox("Register As", ["user", "admin"])
            if st.button("Create Account", use_container_width=True):
                register_user(username, password, role)
                st.success("Account created! Please sign in.")

        if tab == "Login":
            if st.button("Sign In", use_container_width=True):
                user = login_user(username, password)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# =============================
# ADMIN DASHBOARD
# =============================
if st.session_state.user["role"] == "admin":

    st.markdown("""
    <div class="app-header">
      <div class="app-header-icon">🏥</div>
      <div>
        <h1>Vanderbilt ADHD Clinical System</h1>
        <p>Clinical Analytics &amp; Administration Panel</p>
      </div>
      <div class="header-badge">⚙ ADMIN</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("## Admin Panel")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", [
        "Overview Dashboard",
        "Parent vs Teacher Comparison",
        "Mood & Improvement Analysis",
        "Raw Data",
        "Logout"
    ])

    if page == "Logout":
        st.session_state.user = None
        st.rerun()

    conn = get_connection()
    df = pd.read_sql("SELECT * FROM assessments", conn)
    conn.close()

    if df.empty:
        st.warning("No assessments found in the database.")
        st.stop()

    # ── Overview Dashboard ──
    if page == "Overview Dashboard":
        st.markdown('<div class="section-hdr"><span>📊 Clinical Analytics Overview</span></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            role_filter = st.selectbox("Filter by Respondent Role", ["All"] + df["role"].unique().tolist())
        with col2:
            severity_filter = st.selectbox("Filter by Severity Level", ["All"] + df["severity"].unique().tolist())

        filtered_df = df.copy()
        if role_filter != "All":
            filtered_df = filtered_df[filtered_df["role"] == role_filter]
        if severity_filter != "All":
            filtered_df = filtered_df[filtered_df["severity"] == severity_filter]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Assessments",      len(filtered_df))
        col2.metric("Avg Inattention Score",  round(filtered_df["inatt_score"].mean(), 2))
        col3.metric("Avg Hyperactivity Score", round(filtered_df["hyper_score"].mean(), 2))

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            fig1 = go.Figure(data=[go.Pie(
                labels=filtered_df["prediction"].value_counts().index,
                values=filtered_df["prediction"].value_counts().values,
                hole=0.42,
                marker=dict(colors=["#1565c0", "#42a5f5", "#90caf9", "#bbdefb"]),
            )])
            fig1.update_layout(
                title=dict(text="ADHD Type Distribution", font=dict(size=14, family="Inter")),
                paper_bgcolor="white", font=dict(family="Inter"),
                margin=dict(t=50, b=20, l=10, r=10),
                legend=dict(bgcolor="white", bordercolor="#dde3ec", borderwidth=1)
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            sev_counts = filtered_df["severity"].value_counts()
            bar_colors = {"Severe": "#c62828", "Moderate": "#ef6c00", "Mild": "#2e7d32"}
            fig2 = go.Figure(data=[go.Bar(
                x=sev_counts.index,
                y=sev_counts.values,
                marker_color=[bar_colors.get(s, "#1565c0") for s in sev_counts.index],
                text=sev_counts.values, textposition="outside"
            )])
            fig2.update_layout(
                title=dict(text="Severity Distribution", font=dict(size=14, family="Inter")),
                paper_bgcolor="white", font=dict(family="Inter"),
                margin=dict(t=50, b=20, l=10, r=10),
                yaxis=dict(gridcolor="#f0f4f8", showgrid=True)
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Parent vs Teacher ──
    if page == "Parent vs Teacher Comparison":
        st.markdown('<div class="section-hdr"><span>📈 Parent vs Teacher Score Comparison</span></div>', unsafe_allow_html=True)

        comparison = df.groupby("role")[[
            "inatt_score", "hyper_score", "odd_score",
            "conduct_score", "anxiety_score", "performance_score"
        ]].mean()

        fig = go.Figure()
        role_colors = {"Parent": "#1565c0", "Teacher": "#00897b"}
        for role_name in comparison.index:
            color = role_colors.get(role_name, "#42a5f5")
            fig.add_trace(go.Scatterpolar(
                r=comparison.loc[role_name].values,
                theta=["Inattention", "Hyperactivity", "ODD", "Conduct", "Anxiety", "Performance"],
                fill='toself',
                name=role_name,
                line=dict(color=color, width=2),
            ))
        fig.update_layout(
            paper_bgcolor="white",
            font=dict(family="Inter", size=12),
            polar=dict(bgcolor="#f8fafc"),
            legend=dict(bgcolor="white", bordercolor="#dde3ec", borderwidth=1),
            margin=dict(t=30, b=30, l=30, r=30)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Mood & Improvement Analysis ──
    if page == "Mood & Improvement Analysis":
        st.markdown('<div class="section-hdr"><span>🎭 Mood Correlation & Patient Progress</span></div>', unsafe_allow_html=True)

        if "mood_score" in df.columns and "inatt_score" in df.columns:
            mood_df = df.dropna(subset=["mood_score", "inatt_score", "hyper_score"])
            if not mood_df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    fig_m1 = go.Figure()
                    fig_m1.add_trace(go.Scatter(
                        x=mood_df["mood_score"], y=mood_df["inatt_score"],
                        mode="markers",
                        marker=dict(color="#1565c0", size=9, opacity=0.7),
                        name="Inattention"
                    ))
                    fig_m1.update_layout(
                        title=dict(text="Mood vs Inattention Score", font=dict(size=14, family="Inter")),
                        xaxis=dict(title="Mood Score (1=Very Low, 5=Excellent)", dtick=1),
                        yaxis=dict(title="Inattention Score"),
                        paper_bgcolor="white", font=dict(family="Inter"),
                        margin=dict(t=50, b=30, l=40, r=20)
                    )
                    st.plotly_chart(fig_m1, use_container_width=True)
                with col2:
                    fig_m2 = go.Figure()
                    fig_m2.add_trace(go.Scatter(
                        x=mood_df["mood_score"], y=mood_df["hyper_score"],
                        mode="markers",
                        marker=dict(color="#ef6c00", size=9, opacity=0.7),
                        name="Hyperactivity"
                    ))
                    fig_m2.update_layout(
                        title=dict(text="Mood vs Hyperactivity Score", font=dict(size=14, family="Inter")),
                        xaxis=dict(title="Mood Score (1=Very Low, 5=Excellent)", dtick=1),
                        yaxis=dict(title="Hyperactivity Score"),
                        paper_bgcolor="white", font=dict(family="Inter"),
                        margin=dict(t=50, b=30, l=40, r=20)
                    )
                    st.plotly_chart(fig_m2, use_container_width=True)

                avg_by_mood = mood_df.groupby("mood_score")[["inatt_score", "hyper_score"]].mean().reset_index()
                fig_avg = go.Figure()
                fig_avg.add_trace(go.Bar(
                    x=avg_by_mood["mood_score"], y=avg_by_mood["inatt_score"],
                    name="Avg Inattention", marker_color="#1565c0"
                ))
                fig_avg.add_trace(go.Bar(
                    x=avg_by_mood["mood_score"], y=avg_by_mood["hyper_score"],
                    name="Avg Hyperactivity", marker_color="#ef6c00"
                ))
                fig_avg.update_layout(
                    title=dict(text="Average Scores by Mood Level", font=dict(size=14, family="Inter")),
                    barmode="group",
                    xaxis=dict(title="Mood Score", dtick=1),
                    yaxis=dict(title="Average Score"),
                    paper_bgcolor="white", font=dict(family="Inter"),
                    margin=dict(t=50, b=30, l=40, r=20)
                )
                st.plotly_chart(fig_avg, use_container_width=True)
            else:
                st.info("No mood data yet. Mood tracking data will appear here after assessments are submitted.")
        else:
            st.info("Mood data columns not found. Run an assessment to populate mood data.")

        # Patient progress: patients with multiple assessments
        if "patient_name" in df.columns:
            patient_counts = df[df["patient_name"].notna() & (df["patient_name"] != "")]["patient_name"].value_counts()
            repeat_patients = patient_counts[patient_counts > 1].index.tolist()
            if repeat_patients:
                st.markdown('<div class="section-hdr"><span>📈 Repeat Patient Progress</span></div>', unsafe_allow_html=True)
                selected_patient = st.selectbox("Select Patient", repeat_patients)
                p_df = df[df["patient_name"] == selected_patient].sort_values("assessed_at")
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(
                    x=p_df["assessed_at"].astype(str).str[:10],
                    y=p_df["inatt_score"], name="Inattention",
                    mode="lines+markers", line=dict(color="#1565c0", width=2)
                ))
                fig_p.add_trace(go.Scatter(
                    x=p_df["assessed_at"].astype(str).str[:10],
                    y=p_df["hyper_score"], name="Hyperactivity",
                    mode="lines+markers", line=dict(color="#ef6c00", width=2)
                ))
                if "mood_score" in p_df.columns:
                    fig_p.add_trace(go.Scatter(
                        x=p_df["assessed_at"].astype(str).str[:10],
                        y=p_df["mood_score"], name="Mood",
                        mode="lines+markers", line=dict(color="#7e57c2", width=2, dash="dot"),
                        yaxis="y2"
                    ))
                fig_p.update_layout(
                    title=dict(text=f"Progress for {selected_patient}", font=dict(size=14, family="Inter")),
                    paper_bgcolor="white", font=dict(family="Inter"),
                    yaxis=dict(title="Score", gridcolor="#f0f4f8"),
                    yaxis2=dict(title="Mood (1-5)", overlaying="y", side="right", range=[0, 6], showgrid=False),
                    legend=dict(bgcolor="white", bordercolor="#dde3ec", borderwidth=1),
                    margin=dict(t=50, b=30, l=40, r=40)
                )
                st.plotly_chart(fig_p, use_container_width=True)

    # ── Raw Data ──
    if page == "Raw Data":
        st.markdown('<div class="section-hdr"><span>📋 Raw Assessment Records</span></div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

    st.stop()

# =============================
# USER NAVIGATION
# =============================
with st.sidebar:
    st.markdown("## 🏠 Navigation")
    st.markdown("---")
    _main_page = st.radio(
        "Select Module",
        ["🏥 Vanderbilt Assessment", "🧠 ASRS Self-Assessment", "📡 EEG Live Dashboard"],
        key="user_main_nav",
        label_visibility="collapsed",
    )

if "ASRS" in _main_page:
    render_asrs_module()
    st.stop()

if "EEG" in _main_page:
    render_eeg_dashboard()
    st.stop()

# =============================
# USER ASSESSMENT (Vanderbilt)
# =============================
st.markdown("""
<div class="app-header">
  <div class="app-header-icon">🏥</div>
  <div>
    <h1>Vanderbilt ADHD Clinical Evaluation System</h1>
    <p>Department of Child &amp; Adolescent Psychiatry</p>
  </div>
  <div class="header-badge">📋 Assessment Form</div>
</div>
""", unsafe_allow_html=True)

# ── Patient Information ──
st.markdown('<div class="section-hdr"><span>👤 Patient Information</span></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    patient_name = st.text_input("Patient Name", placeholder="Full name")
with col2:
    patient_age = st.number_input("Age", min_value=3, max_value=18)
with col3:
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

col_a, col_b = st.columns([1, 2])
with col_a:
    assessment_date = st.date_input("Assessment Date")
with col_b:
    role = st.selectbox("Respondent Role", ["Parent", "Teacher"])

st.markdown("<br>", unsafe_allow_html=True)

# ── Mood Tracker ──
MOOD_OPTIONS = {
    "😢 Very Low": 1,
    "😟 Low":      2,
    "😐 Neutral":  3,
    "🙂 Good":     4,
    "😄 Excellent": 5,
}

st.markdown("""
<div class="mood-card">
  <h4>🎭 Current Mood Tracker</h4>
  <p>Rate the patient's current mood before starting the assessment.
     This will be correlated with the clinical scores.</p>
</div>
""", unsafe_allow_html=True)

mood_selection = st.radio(
    "Patient's current mood",
    list(MOOD_OPTIONS.keys()),
    index=2,
    horizontal=True,
    label_visibility="collapsed"
)
mood_score = MOOD_OPTIONS[mood_selection]

st.markdown("<br>", unsafe_allow_html=True)

# =============================
# SECTION-WISE ADHD FORM
# =============================
questions_df = pd.read_excel("questions.xlsx")
questions   = questions_df["question_text"].tolist()
scale_types = questions_df["scale_type"].tolist()

responses = []

st.markdown('<div class="section-hdr"><span>📝 Clinical Assessment Questionnaire</span></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧠 Inattention",
    "⚡ Hyperactivity",
    "⚠ ODD",
    "🔴 Conduct",
    "😟 Anxiety",
    "📚 Performance"
])

options_main = ["Never", "Occasionally", "Often", "Very Often"]

# ── 1. Inattention ──
with tab1:
    st.markdown("**Inattention Subscale** — Rate the frequency of each behavior below.")
    for idx, i in enumerate(range(0, 9)):
        st.markdown(f"""
        <div class="question-item">
          <span class="q-num">Q{idx+1}</span>
          <span class="q-text">{questions[i]}</span>
        </div>
        """, unsafe_allow_html=True)
        ans = st.radio("", options_main, key=f"inatt_{i}", horizontal=True)
        responses.append(options_main.index(ans))
        st.divider()

# ── 2. Hyperactivity ──
with tab2:
    st.markdown("**Hyperactivity / Impulsivity Subscale** — Rate the frequency of each behavior below.")
    for idx, i in enumerate(range(9, 18)):
        st.markdown(f"""
        <div class="question-item">
          <span class="q-num">Q{idx+1}</span>
          <span class="q-text">{questions[i]}</span>
        </div>
        """, unsafe_allow_html=True)
        ans = st.radio("", options_main, key=f"hyper_{i}", horizontal=True)
        responses.append(options_main.index(ans))
        st.divider()

# ── 3. ODD ──
with tab3:
    st.markdown("**Oppositional Defiant Disorder Subscale** — Rate the frequency of each behavior below.")
    for idx, i in enumerate(range(18, 26)):
        st.markdown(f"""
        <div class="question-item">
          <span class="q-num">Q{idx+1}</span>
          <span class="q-text">{questions[i]}</span>
        </div>
        """, unsafe_allow_html=True)
        ans = st.radio("", options_main, key=f"odd_{i}", horizontal=True)
        responses.append(options_main.index(ans))
        st.divider()

# ── 4. Conduct ──
with tab4:
    st.markdown("**Conduct Disorder Subscale** — Rate the frequency of each behavior below.")
    for idx, i in enumerate(range(26, 40)):
        st.markdown(f"""
        <div class="question-item">
          <span class="q-num">Q{idx+1}</span>
          <span class="q-text">{questions[i]}</span>
        </div>
        """, unsafe_allow_html=True)
        ans = st.radio("", options_main, key=f"conduct_{i}", horizontal=True)
        responses.append(options_main.index(ans))
        st.divider()

# ── 5. Anxiety ──
with tab5:
    st.markdown("**Anxiety / Depression Subscale** — Rate the frequency of each behavior below.")
    for idx, i in enumerate(range(40, 47)):
        st.markdown(f"""
        <div class="question-item">
          <span class="q-num">Q{idx+1}</span>
          <span class="q-text">{questions[i]}</span>
        </div>
        """, unsafe_allow_html=True)
        ans = st.radio("", options_main, key=f"anxiety_{i}", horizontal=True)
        responses.append(options_main.index(ans))
        st.divider()

# ── 6. Performance ──
with tab6:
    options_perf = ["Excellent", "Above Average", "Average", "Somewhat of a Problem", "Problematic"]
    st.markdown("**Academic & Social Performance Subscale** — Rate the overall performance level.")
    for idx, i in enumerate(range(47, 55)):
        st.markdown(f"""
        <div class="question-item">
          <span class="q-num">Q{idx+1}</span>
          <span class="q-text">{questions[i]}</span>
        </div>
        """, unsafe_allow_html=True)
        ans = st.radio("", options_perf, key=f"performance_{i}", horizontal=True)
        responses.append(options_perf.index(ans))
        st.divider()

# =============================
# SUBMIT
# =============================
st.markdown("<br>", unsafe_allow_html=True)
col_btn, _ = st.columns([1, 3])
with col_btn:
    submit = st.button("Submit Assessment", use_container_width=True)

if submit:

    responses = np.array(responses)

    inatt       = responses[0:9].sum()
    hyper       = responses[9:18].sum()
    odd         = responses[18:26].sum()
    conduct     = responses[26:40].sum()
    anxiety     = responses[40:47].sum()
    performance = responses[47:55].sum()

    input_data = np.array([[inatt, hyper, odd, conduct, anxiety, performance]])

    pred       = model.predict(input_data)
    prediction = label_encoder.inverse_transform(pred)[0]

    severity = "Mild"
    if inatt + hyper > 36:
        severity = "Severe"
    elif inatt + hyper > 18:
        severity = "Moderate"

    assessed_at = datetime.now()

    # Fetch history BEFORE saving so we can compare
    history = get_patient_history(st.session_state.user["id"], patient_name)

    save_assessment((
        int(st.session_state.user["id"]),
        role,
        int(inatt), int(hyper), int(odd),
        int(conduct), int(anxiety), int(performance),
        str(prediction), str(severity),
        str(patient_name), int(patient_age), str(patient_gender),
        str(mood_selection), int(mood_score),
        assessed_at
    ))

    # ── Results Header ──
    st.markdown("""
    <div class="result-header">
      <h3>🏥 Clinical Assessment Results</h3>
      <p>Assessment completed — results saved to clinical database</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Score Metrics ──
    col1, col2, col3 = st.columns(3)
    col1.metric("Inattention Score",   inatt,        help="Sum of inattention subscale (max 24)")
    col2.metric("Hyperactivity Score", hyper,        help="Sum of hyperactivity subscale (max 27)")
    col3.metric("Total Core Score",    inatt + hyper, help="Combined inattention + hyperactivity")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Severity & Prediction ──
    col_sev, col_pred = st.columns(2)

    with col_sev:
        if severity == "Severe":
            badge_cls, badge_icon = "severity-severe", "🔴"
        elif severity == "Moderate":
            badge_cls, badge_icon = "severity-moderate", "🟠"
        else:
            badge_cls, badge_icon = "severity-mild", "🟢"

        st.markdown(f"""
        <div style="text-align:center; padding:1.5rem; background:white;
                    border:1px solid #dde3ec; border-radius:12px;
                    box-shadow:0 2px 16px rgba(10,37,64,0.09);">
          <div style="font-size:0.72rem; font-weight:600; text-transform:uppercase;
                      letter-spacing:1px; color:#546e7a; margin-bottom:0.6rem;">
            Severity Level
          </div>
          <span class="severity-badge {badge_cls}">{badge_icon} {severity}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_pred:
        st.markdown(f"""
        <div class="prediction-card">
          <div class="prediction-label">ADHD Classification</div>
          <div class="prediction-value">{prediction}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Radar Chart ──
    st.markdown("<br>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[inatt, hyper, odd, conduct, anxiety, performance],
        theta=["Inattention", "Hyperactivity", "ODD", "Conduct", "Anxiety", "Performance"],
        fill='toself',
        name="Score Profile",
        line=dict(color="#1565c0", width=2),
        fillcolor="rgba(21,101,192,0.15)"
    ))
    fig.update_layout(
        title=dict(text="Subscale Score Profile", font=dict(size=14, family="Inter")),
        paper_bgcolor="white",
        font=dict(family="Inter", size=12),
        polar=dict(bgcolor="#f8fafc"),
        margin=dict(t=50, b=30, l=30, r=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Mood Correlation ──
    mood_color_map = {1: "#ef5350", 2: "#ffa726", 3: "#42a5f5", 4: "#66bb6a", 5: "#26c6da"}
    mood_color = mood_color_map.get(mood_score, "#42a5f5")
    st.markdown(f"""
    <div style="background:white; border:1px solid #dde3ec; border-left:5px solid {mood_color};
                border-radius:12px; padding:1rem 1.4rem; margin-bottom:1rem;
                box-shadow:0 2px 16px rgba(10,37,64,0.09);">
      <div style="font-size:0.72rem; font-weight:600; text-transform:uppercase;
                  letter-spacing:1px; color:#546e7a; margin-bottom:0.4rem;">
        Mood at Time of Assessment
      </div>
      <span style="font-size:1.3rem; font-weight:700; color:{mood_color};">{mood_selection}</span>
      <span style="font-size:0.8rem; color:#546e7a; margin-left:0.8rem;">
        Score: {mood_score}/5 &nbsp;|&nbsp;
        {'Low mood may amplify symptom perception — interpret scores with caution.' if mood_score <= 2 else
         'Neutral mood — scores reflect baseline behaviour.' if mood_score == 3 else
         'Positive mood noted — scores reflect current state.'}
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── History & Improvement Analysis ──
    if history and patient_name.strip():
        prev = history[-1]  # Most recent prior assessment
        st.markdown('<div class="section-hdr"><span>📈 Progress & Improvement Analysis</span></div>', unsafe_allow_html=True)

        prev_inatt   = prev.get("inatt_score",   prev.get("inatt_score", 0))
        prev_hyper   = prev.get("hyper_score",   0)
        prev_odd     = prev.get("odd_score",     0)
        prev_conduct = prev.get("conduct_score", 0)
        prev_anxiety = prev.get("anxiety_score", 0)
        prev_perf    = prev.get("performance_score", 0)
        prev_total   = prev_inatt + prev_hyper
        curr_total   = inatt + hyper

        def delta_badge(label, prev_val, curr_val, lower_is_better=True):
            diff = curr_val - prev_val
            if diff == 0:
                cls, arrow = "improve-same", "→"
                msg = "No change"
            elif (diff < 0) == lower_is_better:
                cls, arrow = "improve-up", "▲ Improved"
                msg = f"{abs(diff):+} pts"
            else:
                cls, arrow = "improve-down", "▼ Declined"
                msg = f"{abs(diff):+} pts"
            return f'<span class="improve-badge {cls}">{arrow} {label}: {prev_val} → {curr_val} ({msg})</span>'

        st.markdown(f"""
        <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-bottom:1rem;">
          {delta_badge("Inattention",   prev_inatt,   inatt)}
          {delta_badge("Hyperactivity", prev_hyper,   hyper)}
          {delta_badge("ODD",           prev_odd,     odd)}
          {delta_badge("Conduct",       prev_conduct, conduct)}
          {delta_badge("Anxiety",       prev_anxiety, anxiety)}
          {delta_badge("Performance",   prev_perf,    performance)}
        </div>
        """, unsafe_allow_html=True)

        # Overall verdict
        if curr_total < prev_total:
            ov_cls, ov_msg = "improve-up",   f"Overall Improved — Core score dropped by {prev_total - curr_total} pts"
        elif curr_total > prev_total:
            ov_cls, ov_msg = "improve-down", f"Overall Declined — Core score rose by {curr_total - prev_total} pts"
        else:
            ov_cls, ov_msg = "improve-same", "Overall Stable — No change in core score"
        st.markdown(f'<span class="improve-badge {ov_cls}" style="font-size:1rem;">{ov_msg}</span><br><br>', unsafe_allow_html=True)

        prev_severity = prev.get("severity", "")
        prev_date     = str(prev.get("assessed_at", "Previous"))[:10]
        if prev_severity and prev_severity != severity:
            st.info(f"Severity changed from **{prev_severity}** (assessed {prev_date}) to **{severity}** today.")

        # Trend chart if more than one past record
        all_history = list(history) + [{
            "assessed_at": assessed_at, "inatt_score": inatt, "hyper_score": hyper,
            "odd_score": odd, "conduct_score": conduct, "anxiety_score": anxiety,
            "performance_score": performance, "mood_score": mood_score
        }]

        dates   = [str(r.get("assessed_at", ""))[:10] for r in all_history]
        inatt_t = [r.get("inatt_score", 0)   for r in all_history]
        hyper_t = [r.get("hyper_score", 0)   for r in all_history]
        mood_t  = [r.get("mood_score", 0)    for r in all_history]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=dates, y=inatt_t, name="Inattention",
            mode="lines+markers", line=dict(color="#1565c0", width=2),
            marker=dict(size=8)
        ))
        fig_trend.add_trace(go.Scatter(
            x=dates, y=hyper_t, name="Hyperactivity",
            mode="lines+markers", line=dict(color="#ef6c00", width=2),
            marker=dict(size=8)
        ))
        fig_trend.add_trace(go.Scatter(
            x=dates, y=mood_t, name="Mood Score",
            mode="lines+markers", line=dict(color="#7e57c2", width=2, dash="dot"),
            marker=dict(size=8), yaxis="y2"
        ))
        fig_trend.update_layout(
            title=dict(text="Score Trend Over Time (with Mood)", font=dict(size=14, family="Inter")),
            paper_bgcolor="white", font=dict(family="Inter", size=12),
            yaxis=dict(title="ADHD Score", gridcolor="#f0f4f8"),
            yaxis2=dict(title="Mood (1-5)", overlaying="y", side="right",
                        range=[0, 6], showgrid=False),
            legend=dict(bgcolor="white", bordercolor="#dde3ec", borderwidth=1),
            margin=dict(t=50, b=30, l=40, r=40)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    elif patient_name.strip():
        st.info("This is the first assessment recorded for this patient. Future assessments will show improvement trends here.")

    # ── Clinical Interpretation ──
    st.markdown('<div class="section-hdr"><span>🔬 Clinical Interpretation</span></div>', unsafe_allow_html=True)

    if prediction == "Combined Type":
        st.info("**Combined Type ADHD** — Significant symptoms in both inattention and hyperactivity-impulsivity domains. Comprehensive intervention addressing both dimensions is recommended.")
    elif prediction == "Inattentive Type":
        st.info("**Predominantly Inattentive Presentation** — Primary difficulties with sustained attention and organization. Hyperactive-impulsive symptoms are below threshold.")
    elif prediction == "Hyperactive Type":
        st.info("**Predominantly Hyperactive-Impulsive Presentation** — Primary difficulties with activity regulation and impulse control. Inattentive symptoms are below threshold.")
    else:
        st.info("**Below Diagnostic Threshold** — Current symptom levels do not meet full diagnostic criteria. Monitoring and follow-up assessment is recommended.")

    # ── PDF Report ──
    st.markdown("<br>", unsafe_allow_html=True)
    buffer = BytesIO()
    doc    = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Vanderbilt ADHD Clinical Report", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Patient Name: {patient_name}", styles["Normal"]))
    elements.append(Paragraph(f"Age: {patient_age}",            styles["Normal"]))
    elements.append(Paragraph(f"Gender: {patient_gender}",      styles["Normal"]))
    elements.append(Paragraph(f"Assessment Date: {assessment_date}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Mood at Assessment: {mood_selection}", styles["Normal"]))
    elements.append(Paragraph(f"Prediction: {prediction}", styles["Normal"]))
    elements.append(Paragraph(f"Severity: {severity}",     styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    table_data = [
        ["Subscale",      "Score"],
        ["Inattention",   str(inatt)],
        ["Hyperactivity", str(hyper)],
        ["ODD",           str(odd)],
        ["Conduct",       str(conduct)],
        ["Anxiety",       str(anxiety)],
        ["Performance",   str(performance)]
    ]
    table = Table(table_data)
    table.setStyle([
        ('BACKGROUND',   (0, 0), (-1, 0), colors.HexColor("#1565c0")),
        ('TEXTCOLOR',    (0, 0), (-1, 0), colors.white),
        ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID',         (0, 0), (-1, -1), 1, colors.HexColor("#dde3ec")),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")])
    ])
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)

    st.download_button(
        "📄 Download Clinical Report (PDF)",
        buffer,
        "ADHD_Report.pdf",
        mime="application/pdf"
    )

# ── Sidebar: Assessment History ──
with st.sidebar:
    st.markdown("## My History")
    st.markdown("---")
    history_name = st.text_input("Search patient history", placeholder="Enter patient name")
    if st.button("Load History", use_container_width=True) and history_name.strip():
        h = get_patient_history(st.session_state.user["id"], history_name.strip())
        if h:
            h_df = pd.DataFrame(h)
            display_cols = [c for c in ["assessed_at", "severity", "prediction", "mood",
                                         "inatt_score", "hyper_score", "mood_score"] if c in h_df.columns]
            st.dataframe(h_df[display_cols], use_container_width=True)
        else:
            st.warning("No records found for that patient.")
    st.markdown("---")
    if st.button("Sign Out", use_container_width=True):
        st.session_state.user = None
        st.rerun()


