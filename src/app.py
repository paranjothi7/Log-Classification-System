"""
app.py — Streamlit dashboard for the Intelligent SOC Log Classification System.
"""
from __future__ import annotations
import os, sys, io, time
from pathlib import Path

# path setup
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()


# Category colour palette (all 9 real labels + Unknown)
_CAT_COLORS = {
    "HTTP Status":         "#38bdf8",
    "Security Alert":      "#ef4444",
    "System Notification": "#22c55e",
    "Error":               "#a78bfa",
    "Resource Usage":      "#f97316",
    "Critical Error":      "#dc2626",
    "User Action":         "#06b6d4",
    "Workflow Error":      "#eab308",
    "Deprecation Warning": "#94a3b8",
    "Unknown":             "#475569",
}

from processors.enhanced_processor import EnhancedProcessor
from processors.high_performance_processor import HighPerformanceProcessor
from database.connection import test_connection
from database.service import LogService

# Streamlit config
st.set_page_config(
    page_title="SOC Log Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1,h2,h3 { font-family: 'IBM Plex Mono', monospace; }

.metric-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: #e2e8f0;
}
.metric-card .label { font-size: .75rem; color: #94a3b8; letter-spacing:.08em; text-transform:uppercase; }
.metric-card .value { font-size: 2rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }

.badge-security { background:#ef4444; color:white; padding:2px 10px; border-radius:20px; font-size:.75rem; }
.badge-resource  { background:#f97316; color:white; padding:2px 10px; border-radius:20px; font-size:.75rem; }
.badge-workflow  { background:#eab308; color:white; padding:2px 10px; border-radius:20px; font-size:.75rem; }
.badge-unknown   { background:#64748b; color:white; padding:2px 10px; border-radius:20px; font-size:.75rem; }

.sev-critical { color:#ef4444; font-weight:700; }
.sev-high     { color:#f97316; font-weight:700; }
.sev-medium   { color:#eab308; }
.sev-low      { color:#22c55e; }
.sev-info     { color:#94a3b8; }

[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# Singletons

@st.cache_resource
def get_processor():
    return EnhancedProcessor()

@st.cache_resource
def get_hp_processor():
    return HighPerformanceProcessor()

# Sidebar

PAGES = [
    "Dashboard",
    "Log Classification",
    "Analytics",
    "Log History",
    "Single Log Test",
    "System Status",
]

with st.sidebar:
    st.markdown("## ️ SOC Log Classifier")
    st.markdown("---")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    db_ok = test_connection()
    st.markdown(
        f"**DB:** {'🟢 Connected' if db_ok else '🔴 Disconnected'}",
        unsafe_allow_html=True,
    )

# PAGE: Dashboard

if page == PAGES[0]:
    st.title("Security Operations Dashboard")
    st.caption("Real-time log classification & incident monitoring")

    stats = LogService.summary_stats() if db_ok else {
        "total_logs": 0, "avg_confidence": 0.0, "critical_alerts": 0, "security_alerts": 0
    }

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, color in [
        (c1, "Total Logs",       stats["total_logs"],      "#38bdf8"),
        (c2, "Security Alerts",  stats["security_alerts"],  "#ef4444"),
        (c3, "Critical Errors",  stats["critical_alerts"],  "#f97316"),
        (c4, "Avg Confidence",   f"{stats['avg_confidence']:.1%}", "#22c55e"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value" style="color:{color}">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("Category Trend (7 days)")
        if db_ok:
            df_trend = LogService.hourly_trend(days=7)
            if not df_trend.empty:
                fig = px.area(
                    df_trend, x="hour", y="count", color="category",
                    color_discrete_map=_CAT_COLORS,
                    template="plotly_dark",
                )
                fig.update_layout(
                    plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                    legend_title="Category", margin=dict(l=0,r=0,t=0,b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trend data yet. Process some logs first.")
        else:
            st.warning("Database unavailable — showing demo chart.")
            _demo = pd.DataFrame({
                "hour":     ["00:00","04:00","08:00","12:00","16:00","20:00"]*4,
                "count":    [4,2,15,28,22,10, 3,1,8,12,9,5, 1,0,4,9,7,3, 20,15,40,70,60,30],
                "category": (["Security Alert"]*6 + ["Resource Usage"]*6 +
                             ["Workflow Error"]*6 + ["HTTP Status"]*6),
            })
            fig = px.area(_demo, x="hour", y="count", color="category",
                          template="plotly_dark")
            fig.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                              margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Category Distribution")
        if db_ok:
            df_dist = LogService.category_distribution(days=7)
        else:
            df_dist = pd.DataFrame({
                "category": ["HTTP Status","Security Alert","System Notification","Error",
                             "Resource Usage","Critical Error","User Action","Workflow Error","Deprecation Warning"],
                "count":    [420, 154, 148, 74, 74, 67, 60, 2, 1],
            })
        if not df_dist.empty:
            fig2 = px.pie(
                df_dist, names="category", values="count", hole=0.5,
                color_discrete_map=_CAT_COLORS,
                template="plotly_dark",
            )
            fig2.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                showlegend=True, margin=dict(l=0,r=0,t=0,b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

# PAGE: Log Classification (CSV upload)

elif page == PAGES[1]:
    st.title("Batch Log Classification")
    st.caption("Upload a CSV file with a `message` column (and optional `source` column)")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"**{len(df)} rows detected.**")
        st.dataframe(df.head(5), use_container_width=True)

        if "message" not in df.columns:
            st.error("CSV must contain a `message` column.")
        else:
            if st.button("Classify All Logs", type="primary"):
                proc = get_hp_processor()
                messages = df["message"].astype(str).tolist()
                progress = st.progress(0, text="Classifying …")
                container = st.empty()

                def _progress(done, total):
                    progress.progress(done / total, text=f"{done}/{total} logs processed")

                batch = proc.process_batch(messages, progress_cb=_progress)
                progress.empty()

                results_df = pd.DataFrame([
                    {
                        "message":    messages[i],
                        "category":   r.category,
                        "severity":   r.severity,
                        "confidence": f"{r.confidence:.1%}",
                        "method":     r.method,
                        "time_ms":    r.processing_time_ms,
                    }
                    for i, r in enumerate(batch.results)
                ])

                st.success(
                    f" Done — {batch.success}/{batch.total} classified "
                    f"in {batch.duration_sec:.1f}s ({batch.throughput_rpm} logs/min)"
                )
                st.dataframe(results_df, use_container_width=True)

                csv_out = results_df.to_csv(index=False).encode()
                st.download_button(
                    " Download Results",
                    data=csv_out,
                    file_name="classified_logs.csv",
                    mime="text/csv",
                )

                # Save to DB
                if db_ok:
                    rows = [
                        {
                            "raw_message":           messages[i],
                            "source":                df.get("source", pd.Series(["upload"]*len(messages))).iloc[i],
                            "category":              r.category,
                            "severity":              r.severity,
                            "confidence":            r.confidence,
                            "classification_method": r.method,
                            "processing_time_ms":    r.processing_time_ms,
                        }
                        for i, r in enumerate(batch.results)
                    ]
                    saved = LogService.bulk_save(rows)
                    st.info(f" {saved} entries saved to database.")

# PAGE: Analytics

elif page == PAGES[2]:
    st.title("Analytics & Trends")
    days = st.slider("Time window (days)", 1, 30, 7)

    if not db_ok:
        st.warning("Database unavailable — charts will be empty.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Category Distribution")
        df_cat = LogService.category_distribution(days) if db_ok else pd.DataFrame()
        if not df_cat.empty:
            fig = px.bar(
                df_cat, x="category", y="count",
                color="category",
                color_discrete_map=_CAT_COLORS,
                template="plotly_dark",
            )
            fig.update_layout(
                showlegend=False, plot_bgcolor="#0f172a",
                paper_bgcolor="#0f172a", margin=dict(l=0,r=0,t=0,b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for the selected period.")

    with col2:
        st.subheader("Hourly Log Volume")
        df_hour = LogService.hourly_trend(days) if db_ok else pd.DataFrame()
        if not df_hour.empty:
            fig2 = px.line(
                df_hour, x="hour", y="count", color="category",
                template="plotly_dark",
            )
            fig2.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                margin=dict(l=0,r=0,t=0,b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data for the selected period.")

# PAGE: Log History

elif page == PAGES[3]:
    st.title("Log History")

    col_a, col_b, col_c = st.columns(3)
    cat_filter = col_a.selectbox(
        "Category", ["All","HTTP Status","Security Alert","System Notification",
                     "Error","Resource Usage","Critical Error","User Action",
                     "Workflow Error","Deprecation Warning","Unknown"]
    )
    sev_filter = col_b.selectbox(
        "Severity", ["All", "Critical", "High", "Medium", "Low", "Info"]
    )
    limit = col_c.slider("Show last N logs", 10, 500, 100)

    if db_ok:
        logs = LogService.get_recent(
            limit=limit,
            category=None if cat_filter == "All" else cat_filter,
            severity=None if sev_filter == "All" else sev_filter,
        )
        if logs:
            df_hist = pd.DataFrame(logs)
            st.dataframe(df_hist, use_container_width=True)
            csv = df_hist.to_csv(index=False).encode()
            st.download_button(" Export CSV", csv, "log_history.csv", "text/csv")
        else:
            st.info("No logs found for the selected filters.")
    else:
        st.error("Database connection unavailable.")

# PAGE: Single Log Test

elif page == PAGES[4]:
    st.title("Single Log Test")
    st.caption("Test a single log message through the classification pipeline")

    EXAMPLES = [
        "Multiple login failures detected from IP 192.168.1.105",
        "Memory usage exceeded 97% on prod-worker-02",
        "Escalation workflow failed for ticket TKT-9021",
        "User admin logged in from 10.0.0.1",
        "SQL injection attempt detected in search parameter",
        "Disk quota exceeded on /data partition (98% full)",
    ]
    example = st.selectbox("Or pick an example:", ["(type your own)"] + EXAMPLES)
    default_msg = "" if example == "(type your own)" else example
    message = st.text_area("Log message:", value=default_msg, height=100)

    if st.button("Classify", type="primary") and message.strip():
        proc = get_processor()
        with st.spinner("Classifying …"):
            result = proc.process(message.strip())

        color = _CAT_COLORS.get(result.category, "#64748b")

        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(
            f'<div class="metric-card"><div class="label">Category</div>'
            f'<div class="value" style="color:{color};font-size:1.2rem">{result.category}</div></div>',
            unsafe_allow_html=True,
        )
        col2.markdown(
            f'<div class="metric-card"><div class="label">Severity</div>'
            f'<div class="value" style="font-size:1.2rem">{result.severity}</div></div>',
            unsafe_allow_html=True,
        )
        col3.markdown(
            f'<div class="metric-card"><div class="label">Confidence</div>'
            f'<div class="value" style="font-size:1.2rem">{result.confidence:.1%}</div></div>',
            unsafe_allow_html=True,
        )
        col4.markdown(
            f'<div class="metric-card"><div class="label">Method / Time</div>'
            f'<div class="value" style="font-size:1.2rem">{result.method} / {result.processing_time_ms:.0f}ms</div></div>',
            unsafe_allow_html=True,
        )

        if result.reasoning:
            st.info(f"**Reasoning:** {result.reasoning}")

        conf_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result.confidence * 100,
            title={"text": "Confidence %"},
            gauge={
                "axis":    {"range": [0, 100]},
                "bar":     {"color": color},
                "steps": [
                    {"range":[0,50],  "color":"#1e293b"},
                    {"range":[50,75], "color":"#334155"},
                    {"range":[75,100],"color":"#475569"},
                ],
            },
        ))
        conf_fig.update_layout(
            height=250, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font_color="#e2e8f0",
        )
        st.plotly_chart(conf_fig, use_container_width=True)

# PAGE: System Status

elif page == PAGES[5]:
    st.title("System Status")
    st.caption("Health checks and configuration overview")

    checks = {
        "PostgreSQL Database":   test_connection(),
        "GROQ API key set":      bool(os.getenv("GROQ_API_KEY")),
        "JIRA configured":       bool(os.getenv("JIRA_API_TOKEN")),
    }

    for label, ok in checks.items():
        icon = "✅" if ok else "❌"
        st.markdown(f"{icon} **{label}**")

    st.markdown("---")
    st.subheader("Environment")
    st.code(
        f"DATABASE_URL  = {os.getenv('DATABASE_URL', '(not set)')}\n"
        f"GROQ_MODEL    = {os.getenv('GROQ_MODEL', 'llama3-70b-8192')}\n"
        f"BERT_PATH     = {os.getenv('BERT_MODEL_PATH', 'models/bert_log_classifier')}\n"
        f"MAX_WORKERS   = {os.getenv('MAX_WORKERS', '4')}\n"
        f"BATCH_SIZE    = {os.getenv('BATCH_SIZE', '100')}",
        language="ini",
    )

    if db_ok:
        st.markdown("---")
        st.subheader("Database Stats")
        stats = LogService.summary_stats()
        st.json(stats)