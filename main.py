"""
main.py — ADHD Assessment Platform
Entry point: streamlit run main.py
"""
import streamlit as st
from database import init_db, get_patients, add_patient, authenticate

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ADHD Assessment Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DB Init ────────────────────────────────────────────────────────────────────
init_db()

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
.main { background-color: #f0f4f8 !important; }
.main .block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1300px !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0d47a1 100%) !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.9) !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: white !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }
section[data-testid="stSidebar"] .stSelectbox div { color: #0a1628 !important; }

/* ── Cards ──────────────────────────────────────────────────────────── */
.card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 10px rgba(15,23,42,0.07);
    margin-bottom: 12px;
}

/* ── Page Title ─────────────────────────────────────────────────────── */
.page-title {
    color: #0d47a1 !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
}

/* ── Metrics ────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-top: 4px solid #1565c0 !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.07) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.5px; color: #64748b !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important; font-weight: 700 !important; color: #0d47a1 !important;
}

/* ── Buttons ────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 2px 8px rgba(13,71,161,0.3) !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }
.stDownloadButton > button {
    background: linear-gradient(135deg, #2e7d32, #1b5e20) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
}

/* ── Tabs ───────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #e2e8f0; padding: 4px; border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important; font-weight: 500 !important;
    font-size: 0.82rem !important; padding: 0.4rem 0.9rem !important;
    color: #64748b !important; background: transparent !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: white !important; color: #0d47a1 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.1) !important; font-weight: 700 !important;
}

/* ── Question cards ──────────────────────────────────────────────────── */
.q-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-left: 4px solid #1565c0;
    border-radius: 8px; padding: 10px 14px; margin-bottom: 4px;
    font-size: 0.88rem; font-weight: 500; color: #0f172a;
}
.q-num {
    display: inline-block; background: #1565c0; color: white;
    font-size: 0.68rem; font-weight: 700; padding: 2px 7px;
    border-radius: 50px; margin-right: 6px;
}

/* ── DataFrames ─────────────────────────────────────────────────────── */
.stDataFrame { border-radius: 10px !important; box-shadow: 0 2px 8px rgba(15,23,42,0.07) !important; }

/* ── Login card ──────────────────────────────────────────────────────── */
.login-wrap { max-width: 420px; margin: 80px auto 0; }
.login-header {
    background: linear-gradient(135deg, #0d47a1, #1565c0);
    padding: 28px 24px; border-radius: 14px 14px 0 0; text-align: center;
}
.login-header h2 { color: white !important; margin: 8px 0 4px !important; font-size: 1.3rem !important; }
.login-header p  { color: rgba(255,255,255,0.75); font-size: 0.82rem; margin: 0; }
.login-body {
    background: white; border: 1px solid #e2e8f0; border-top: none;
    border-radius: 0 0 14px 14px; padding: 24px;
    box-shadow: 0 10px 40px rgba(13,71,161,0.15);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════
if "user" not in st.session_state:
    st.session_state.user = None
if "current_patient" not in st.session_state:
    st.session_state.current_patient = None

if st.session_state.user is None:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div class="login-wrap">
        <div class="login-header">
            <div style="font-size:2.5rem;">🧠</div>
            <h2>ADHD Assessment Platform</h2>
            <p>EEG · Questionnaire · Emotion · Cognitive Testing</p>
        </div>
        <div class="login-body">
        """, unsafe_allow_html=True)

        mode = st.selectbox("", ["Login", "Register"], label_visibility="collapsed")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        if mode == "Register":
            role = st.selectbox("Register As", ["clinician", "admin"])
            if st.button("Create Account", use_container_width=True):
                if username and password:
                    try:
                        from database import get_conn
                        with get_conn() as conn:
                            conn.execute(
                                "INSERT INTO users (username, password, role) VALUES (?,?,?)",
                                (username, password, role)
                            )
                        st.success("Account created! Please log in.")
                    except Exception:
                        st.error("Username already exists.")
                else:
                    st.warning("Please enter username and password.")

        if mode == "Login":
            if st.button("Sign In", use_container_width=True):
                user = authenticate(username, password)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 ADHD Platform")
    st.markdown(f"*Logged in as:* **{st.session_state.user.get('role','user')}**")
    st.markdown("---")

    # ── Patient Selector ─────────────────────────────────────────────────────
    patients = get_patients()
    st.markdown("### 👤 Active Patient")

    if patients:
        pt_opts  = {"— Select Patient —": None}
        pt_opts.update({f"{p['name']} (#{p['id']})": p for p in patients})
        sel_key  = st.selectbox("Patient", list(pt_opts.keys()),
                                label_visibility="collapsed")
        st.session_state.current_patient = pt_opts[sel_key]
        if st.session_state.current_patient:
            p = st.session_state.current_patient
            st.caption(f"Age: {p['age']} | Gender: {p['gender']}")
    else:
        st.caption("No patients yet.")

    # Quick-add patient
    with st.expander("➕ Quick Add Patient"):
        with st.form("quick_add"):
            qname   = st.text_input("Name *")
            qa, qg  = st.columns(2)
            qage    = qa.number_input("Age", 3, 100, 10)
            qgender = qg.selectbox("Gender", ["Male","Female","Other"])
            if st.form_submit_button("Add"):
                if qname.strip():
                    new_id = add_patient(qname.strip(), qage, qgender)
                    st.success(f"Added #{new_id}")
                    st.rerun()
                else:
                    st.error("Name required.")

    st.markdown("---")

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown("### Navigation")
    pages = [
        "🏠 Home Dashboard",
        "🧠 EEG Assessment",
        "📋 ADHD Questionnaire",
        "😊 Emotion Monitoring",
        "🎮 Activity Builder",
        "📈 Patient Progress",
        "🗂 Historical Data",
        "📄 ADHD Report",
    ]
    if st.session_state.user.get("role") == "admin":
        pages.append("⚙ Admin Panel")

    page = st.radio("Go to", pages, label_visibility="collapsed")

    st.markdown("---")
    if st.button("Logout", use_container_width=True):
        st.session_state.user            = None
        st.session_state.current_patient = None
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTING
# ══════════════════════════════════════════════════════════════════════════════
if   "Home"        in page:
    from modules.home          import render_home;          render_home()
elif "EEG"         in page:
    from modules.eeg           import render_eeg;           render_eeg()
elif "Questionnaire" in page:
    from modules.questionnaire import render_questionnaire; render_questionnaire()
elif "Emotion"     in page:
    from modules.emotion       import render_emotion;       render_emotion()
elif "Activity"    in page:
    from modules.activity      import render_activity;      render_activity()
elif "Progress"    in page:
    from modules.progress      import render_progress;      render_progress()
elif "Historical"  in page:
    from modules.history       import render_history;       render_history()
elif "Report"      in page:
    from modules.report        import render_report;        render_report()
elif "Admin"       in page:
    from modules.admin         import render_admin;         render_admin()
