"""
eeg_dashboard.py
================
Streamlit rendering module for the Live EEG Brainwave Dashboard.

Called from app.py when the user selects "EEG Live Dashboard" in the
sidebar. Automatically starts the serial reader in a background thread —
no separate terminal required.
"""

import os
import time
import threading
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from serial_reader import run as _serial_run

# ─── Configuration ─────────────────────────────────────────────────────────
CSV_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_data.csv")
REFRESH_SECONDS = 3      # Auto-refresh interval (seconds)
MAX_CHART_ROWS  = 100    # How many recent rows to show in charts
# ───────────────────────────────────────────────────────────────────────────


def _ensure_serial_thread():
    """
    Start the serial reader in a daemon background thread the first time
    this function is called in a session. Subsequent calls are no-ops.
    """
    if st.session_state.get("_serial_thread_started"):
        return
    t = threading.Thread(target=_serial_run, daemon=True, name="serial_reader")
    t.start()
    st.session_state["_serial_thread_started"] = True


def _load_data() -> pd.DataFrame:
    """
    Read live_data.csv and return the last MAX_CHART_ROWS rows.
    Returns an empty DataFrame if the file is missing or unreadable.
    """
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
        return df.tail(MAX_CHART_ROWS).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _line_chart(
    df: pd.DataFrame,
    cols: list,
    title: str,
    colors: list,
    height: int = 240,
) -> go.Figure:
    """Build a Plotly line chart for one or more columns vs timestamp."""
    fig = go.Figure()
    for col, color in zip(cols, colors):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["timestamp"],
                y=df[col],
                name=col,
                mode="lines",
                line=dict(color=color, width=2),
            ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, family="Inter", color="#0a2540")),
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        font=dict(family="Inter", size=11),
        xaxis=dict(showgrid=False, title="Time", tickfont=dict(size=9)),
        yaxis=dict(gridcolor="#e2e8f0", title="Value"),
        legend=dict(
            bgcolor="white",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(t=40, b=30, l=45, r=15),
        height=height,
    )
    return fig


def render_eeg_dashboard():
    """Entry point called by app.py."""

    # ── Start serial reader thread (once per session) ────────────────────────
    _ensure_serial_thread()

    # ── Page header ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
      <div class="app-header-icon"></div>
      <div>
        <h1>Live EEG Brainwave Dashboard</h1>
        <p>Real-time data from COM6 · Auto-refreshes every 3 seconds</p>
      </div>
      <div class="header-badge">⚡ LIVE</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Status bar ──────────────────────────────────────────────────────────
    status_col, btn_col = st.columns([4, 1])
    with status_col:
        thread_ok = st.session_state.get("_serial_thread_started", False)
        if os.path.exists(CSV_PATH):
            size_kb = os.path.getsize(CSV_PATH) / 1024
            thread_status = "Serial reader: running" if thread_ok else "Serial reader: starting…"
            st.caption(
                f"Source: `live_data.csv`  ·  "
                f"Size: {size_kb:.1f} KB  ·  "
                f"Port: COM6 @ 9600 baud  ·  {thread_status}"
            )
        else:
            st.info("Connecting to COM6… Please wait a moment and the data will appear.")
    with btn_col:
        if st.button("Refresh Now", use_container_width=True):
            st.rerun()

    # ── Load data ────────────────────────────────────────────────────────────
    df = _load_data()

    if df.empty:
        st.info(
            "Connecting to COM6 and waiting for EEG data…  \n"
            "This usually takes a few seconds. The page will refresh automatically."
        )
        time.sleep(REFRESH_SECONDS)
        st.rerun()
        return

    # ── Latest reading metrics ────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr"><span>📡 Latest Reading</span></div>',
        unsafe_allow_html=True,
    )

    latest = df.iloc[-1]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Signal Quality", f"{int(latest.get('quality', 0))} %")
    m2.metric("Attention",      int(latest.get("attention", 0)))
    m3.metric("Meditation",     int(latest.get("meditation", 0)))
    m4.metric("Delta",          int(latest.get("delta", 0)))
    m5.metric(
        "Timestamp",
        str(latest.get("timestamp", ""))[-8:],   # show HH:MM:SS only
    )

    # ── Signal quality colour indicator ─────────────────────────────────────
    quality = int(latest.get("quality", 0))
    if quality > 60:
        q_label, q_class = "Good Signal", "severity-mild"
    elif quality > 30:
        q_label, q_class = "Fair Signal", "severity-moderate"
    else:
        q_label, q_class = "Poor Signal", "severity-severe"

    st.markdown(
        f'<span class="severity-badge {q_class}">{q_label} — Quality: {quality}%</span>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Recent data table ─────────────────────────────────────────────────
    with st.expander("Latest 20 Rows (newest first)", expanded=False):
        display_cols = [
            "timestamp", "quality", "attention", "meditation",
            "delta", "theta", "lowAlpha", "highAlpha",
            "lowBeta", "highBeta", "lowGamma", "midGamma",
        ]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[available].tail(20).iloc[::-1].reset_index(drop=True),
            use_container_width=True,
        )

    # ── Brainwave charts ─────────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr"><span>📊 Live Brainwave Charts</span></div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.plotly_chart(
            _line_chart(df, ["attention"], "Attention", ["#1565c0"]),
            use_container_width=True,
            key="eeg_attention",
        )
        st.plotly_chart(
            _line_chart(df, ["lowAlpha", "highAlpha"], "Alpha Waves", ["#00897b", "#26c6da"]),
            use_container_width=True,
            key="eeg_alpha",
        )
        st.plotly_chart(
            _line_chart(df, ["lowGamma", "midGamma"], "Gamma Waves", ["#7b1fa2", "#ab47bc"]),
            use_container_width=True,
            key="eeg_gamma",
        )
        st.plotly_chart(
            _line_chart(df, ["delta"], "Delta Waves", ["#0d47a1"]),
            use_container_width=True,
            key="eeg_delta",
        )

    with col_right:
        st.plotly_chart(
            _line_chart(df, ["meditation"], "Meditation", ["#00897b"]),
            use_container_width=True,
            key="eeg_meditation",
        )
        st.plotly_chart(
            _line_chart(df, ["lowBeta", "highBeta"], "Beta Waves", ["#e65100", "#ff8f00"]),
            use_container_width=True,
            key="eeg_beta",
        )
        st.plotly_chart(
            _line_chart(df, ["theta"], "Theta Waves", ["#4527a0"]),
            use_container_width=True,
            key="eeg_theta",
        )

    # ── Auto-refresh footer ──────────────────────────────────────────────
    st.markdown("---")
    st.caption(f"Auto-refreshing every {REFRESH_SECONDS} seconds …")
    time.sleep(REFRESH_SECONDS)
    st.rerun()
