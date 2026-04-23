"""
CryptoSentinel — AI-Powered Cryptocurrency Fraud Detection on Web3
==================================================================
Single-file Streamlit app.  Run with:    streamlit run app.py

Features
--------
- Trains 3 fraud-detection models inline at startup (cached, ~3s first run).
- Real-time transaction simulator with working Generate/Analyze/AI buttons.
- Graph-based anomaly detection via Isolation Forest on wallet network.
- Local AI-insights generator (no external API needed).
- Optional Gemini integration if GEMINI_API_KEY env var is set.
- Polished dark dashboard with metrics, heatmap, network graph, time series.

Author: refactored and hardened for reliability.
"""

import os
import io
import time
import random
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import joblib

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
#  🔑 OPTIONAL — PASTE YOUR API KEYS HERE
#  ────────────────────────────────────────────────────────────────────────
#  The app works fine without either key (it falls back to a local
#  analyser). If you want live LLM commentary in the AI Insights tab,
#  paste ONE of these keys between the quotes below.
#
#  Get a free Groq key:  https://console.groq.com/keys
#  Get a free Gemini key: https://aistudio.google.com/app/apikey
# ═══════════════════════════════════════════════════════════════════════════

GROQ_API_KEY   = ""   # <── paste your "gsk_..." key here, keeping the quotes
GEMINI_API_KEY = ""   # <── optional Gemini fallback key

# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CryptoSentinel — AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Custom CSS — polished dark theme with neon accents
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-0: #0a0e1a;
    --bg-1: #111827;
    --bg-2: #1a2332;
    --border: #2a3548;
    --accent: #00d9ff;
    --accent-glow: rgba(0, 217, 255, 0.3);
    --good: #22c55e;
    --warn: #f59e0b;
    --bad: #ef4444;
    --text-1: #f1f5f9;
    --text-2: #94a3b8;
    --text-3: #64748b;
}

html, body, [class*="css"], .stApp {
    background: var(--bg-0) !important;
    color: var(--text-1) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* Hero title */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent) 0%, #a855f7 50%, #22c55e 100%);
}
.hero h1 {
    color: var(--text-1) !important;
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    margin: 0 0 6px 0 !important;
    font-family: 'Inter', sans-serif !important;
}
.hero .subtitle {
    color: var(--text-2);
    font-size: 0.98rem;
    margin: 0;
}
.hero .badge {
    display: inline-block;
    background: rgba(0,217,255,0.12);
    color: var(--accent);
    border: 1px solid rgba(0,217,255,0.3);
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
    text-transform: uppercase;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    transition: all 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: var(--accent);
    box-shadow: 0 0 24px rgba(0,217,255,0.15);
}
[data-testid="stMetricLabel"] {
    color: var(--text-2) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    color: var(--text-1) !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-1) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--text-1) !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 20px !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: var(--bg-2) !important;
    color: var(--text-1) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: 0 0 16px var(--accent-glow) !important;
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: var(--bg-1);
    border-radius: 10px;
    padding: 6px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-2) !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-2) !important;
    color: var(--accent) !important;
}

/* Alert cards */
.alert-card {
    background: linear-gradient(135deg, rgba(239,68,68,0.08) 0%, rgba(239,68,68,0.04) 100%);
    border-left: 3px solid var(--bad);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    color: var(--text-1);
}
.alert-card.warn {
    background: linear-gradient(135deg, rgba(245,158,11,0.08) 0%, rgba(245,158,11,0.04) 100%);
    border-left-color: var(--warn);
}
.alert-card.info {
    background: linear-gradient(135deg, rgba(0,217,255,0.08) 0%, rgba(0,217,255,0.04) 100%);
    border-left-color: var(--accent);
}
.alert-card .tag {
    display: inline-block;
    background: rgba(239,68,68,0.2);
    color: var(--bad);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    margin-right: 8px;
    letter-spacing: 0.05em;
}
.alert-card.warn .tag { background: rgba(245,158,11,0.2); color: var(--warn); }
.alert-card.info .tag { background: rgba(0,217,255,0.2); color: var(--accent); }

.blocked-pill {
    display: inline-block;
    background: var(--bg-2);
    border: 1px solid var(--bad);
    color: var(--bad);
    padding: 4px 10px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    margin: 3px 3px;
}

.section-title {
    color: var(--text-1);
    font-size: 1.1rem;
    font-weight: 700;
    margin: 18px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::before {
    content: '';
    width: 3px; height: 16px;
    background: var(--accent);
    border-radius: 2px;
}

.insights-box {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 26px;
    line-height: 1.7;
    font-size: 0.95rem;
    color: var(--text-1);
}
.insights-box h3, .insights-box h4 {
    color: var(--accent) !important;
    margin-top: 16px !important;
    font-weight: 700 !important;
}
.insights-box ul { margin: 10px 0; padding-left: 20px; }
.insights-box li { margin-bottom: 6px; }
.insights-box strong { color: var(--accent); }

[data-testid="stDataFrame"] {
    background: var(--bg-1);
    border-radius: 10px;
    border: 1px solid var(--border);
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

/* Status indicator in sidebar */
.model-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.25);
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 0.83rem;
    color: var(--good);
    font-weight: 500;
}
.model-status::before {
    content: '●';
    color: var(--good);
    font-size: 1.2rem;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Model training (cached — runs once per server start)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_all_models():
    """Train Fake/Sybil/Bot detectors on the compact schemas the app produces."""
    raw = pd.read_csv("transaction_dataset.csv")
    raw = raw.drop(columns=[c for c in ["Unnamed: 0", "Index", "Address"] if c in raw.columns])
    y = raw["FLAG"]
    np.random.seed(42)

    # Shared base features
    freq = raw["Sent tnx"] + raw["Received Tnx"]
    variance = raw[["avg val sent", "avg val received"]].std(axis=1).fillna(0)

    # --- Fake identity: 7 numeric features ---
    f_df = pd.DataFrame({
        "Transaction Frequency": freq,
        "Average Sent Amount": raw["avg val sent"],
        "Average Received Amount": raw["avg val received"],
        "Unique Sent Addresses": raw["Unique Sent To Addresses"],
        "Unique Received Addresses": raw["Unique Received From Addresses"],
        "Transaction Time Consistency": raw["Time Diff between first and last (Mins)"],
        "Time Diff between Transactions (Minutes)": raw["Avg min between sent tnx"].fillna(0),
    })
    f_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=150))
    ])
    f_X_tr, f_X_te, f_y_tr, f_y_te = train_test_split(f_df, y, test_size=0.2, stratify=y, random_state=42)
    f_pipe.fit(f_X_tr, f_y_tr)
    f_metrics = {
        "accuracy": accuracy_score(f_y_te, f_pipe.predict(f_X_te)),
        "f1":       f1_score(f_y_te, f_pipe.predict(f_X_te)),
        "auc":      roc_auc_score(f_y_te, f_pipe.predict_proba(f_X_te)[:, 1]),
    }

    # --- Sybil: 13 features (10 numeric + 3 categorical) ---
    s_df = f_df.copy()
    s_df["Account Age"] = np.random.randint(1, 100, len(raw))
    s_df["Total Sent Transactions"] = raw["Sent tnx"]
    s_df["Total Received Transactions"] = raw["Received Tnx"]
    s_df["Device Fingerprint"] = np.random.choice(["DeviceA", "DeviceB", "DeviceC"], len(raw))
    s_df["IP Address"] = np.random.choice(["192.168.0.1", "192.168.0.2"], len(raw))
    s_df["Geolocation"] = np.random.choice(["US", "EU", "Asia"], len(raw))
    num_s = [c for c in s_df.columns if c not in ("Device Fingerprint", "IP Address", "Geolocation")]
    cat_s = ["Device Fingerprint", "IP Address", "Geolocation"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_s),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_s)
    ])
    s_pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=150))])
    s_X_tr, s_X_te, s_y_tr, s_y_te = train_test_split(s_df, y, test_size=0.2, stratify=y, random_state=42)
    s_pipe.fit(s_X_tr, s_y_tr)
    s_metrics = {
        "accuracy": accuracy_score(s_y_te, s_pipe.predict(s_X_te)),
        "f1":       f1_score(s_y_te, s_pipe.predict(s_X_te)),
        "auc":      roc_auc_score(s_y_te, s_pipe.predict_proba(s_X_te)[:, 1]),
    }

    # --- Bot: 4 numeric features ---
    b_df = pd.DataFrame({
        "Transaction Time Diff": raw["Time Diff between first and last (Mins)"],
        "Transaction Amount Variance": variance,
        "Unique Sent Addresses": raw["Unique Sent To Addresses"],
        "Bot Activity Indicator": freq * variance * raw["Unique Sent To Addresses"],
    })
    b_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=150))
    ])
    b_X_tr, b_X_te, b_y_tr, b_y_te = train_test_split(b_df, y, test_size=0.2, stratify=y, random_state=42)
    b_pipe.fit(b_X_tr, b_y_tr)
    b_metrics = {
        "accuracy": accuracy_score(b_y_te, b_pipe.predict(b_X_te)),
        "f1":       f1_score(b_y_te, b_pipe.predict(b_X_te)),
        "auc":      roc_auc_score(b_y_te, b_pipe.predict_proba(b_X_te)[:, 1]),
    }

    return {
        "fake":  (f_pipe, f_metrics),
        "sybil": (s_pipe, s_metrics),
        "bot":   (b_pipe, b_metrics),
        "trained_on": len(raw),
    }


# ─────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "transaction_history": [],
        "alerts": [],
        "blocked_accounts": set(),
        "anomalies_detected": set(),
        "graph_img": None,
        "insights_html": "",
        "last_action": "",
        "detection_counts": {"fake": 0, "sybil": 0, "bot": 0, "graph": 0},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────
# Transaction simulation + feature engineering
# ─────────────────────────────────────────────────────────────────────
KNOWN_GOOD = ["0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
              "0x1aD91ee08f21bE3dE0BA2ba6918E714dA6B45836"]

def random_address():
    return "0x" + "".join(random.choices("0123456789abcdef", k=40))

def simulate_transaction(force_suspicious=False):
    # Mix of known-good and random addresses; occasionally force suspicious patterns
    if random.random() < 0.2:
        addr = random.choice(KNOWN_GOOD)
    else:
        addr = random_address()

    if force_suspicious or random.random() < 0.3:
        # Suspicious profile
        sent = random.randint(8, 15)
        recv = random.randint(8, 15)
        avg_sent = random.uniform(0.001, 0.1)
        avg_recv = random.uniform(0.001, 0.1)
        time_diff = random.uniform(1, 10)
    else:
        sent = random.randint(1, 7)
        recv = random.randint(1, 7)
        avg_sent = random.uniform(0.1, 5.0)
        avg_recv = random.uniform(0.1, 5.0)
        time_diff = random.uniform(10, 120)

    return {
        "Sent tnx": sent,
        "Received Tnx": recv,
        "avg val sent": avg_sent,
        "avg val received": avg_recv,
        "Unique Sent To Addresses": random.randint(1, 20),
        "Unique Received From Addresses": random.randint(1, 20),
        "Time Diff between first and last (Mins)": time_diff,
        "Timestamp": datetime.now(),
        "Account Age": random.randint(1, 100),
        "Device Fingerprint": random.choice(["DeviceA", "DeviceB", "DeviceC"]),
        "IP Address": random.choice(["192.168.0.1", "192.168.0.2", "10.0.0.1"]),
        "Geolocation": random.choice(["US", "EU", "Asia", "LATAM"]),
        "Address": addr,
    }


def fe_fake(tx, prev=None):
    td = (tx["Timestamp"] - prev["Timestamp"]).total_seconds() / 60 if prev else 0
    return pd.DataFrame([{
        "Transaction Frequency": tx["Sent tnx"] + tx["Received Tnx"],
        "Average Sent Amount": tx["avg val sent"],
        "Average Received Amount": tx["avg val received"],
        "Unique Sent Addresses": tx["Unique Sent To Addresses"],
        "Unique Received Addresses": tx["Unique Received From Addresses"],
        "Transaction Time Consistency": tx["Time Diff between first and last (Mins)"],
        "Time Diff between Transactions (Minutes)": td,
    }])


def fe_sybil(tx, prev=None):
    td = (tx["Timestamp"] - prev["Timestamp"]).total_seconds() / 60 if prev else 0
    return pd.DataFrame([{
        "Transaction Frequency": tx["Sent tnx"] + tx["Received Tnx"],
        "Average Sent Amount": tx["avg val sent"],
        "Average Received Amount": tx["avg val received"],
        "Unique Sent Addresses": tx["Unique Sent To Addresses"],
        "Unique Received Addresses": tx["Unique Received From Addresses"],
        "Transaction Time Consistency": tx["Time Diff between first and last (Mins)"],
        "Time Diff between Transactions (Minutes)": td,
        "Account Age": tx["Account Age"],
        "Total Sent Transactions": tx["Sent tnx"],
        "Total Received Transactions": tx["Received Tnx"],
        "Device Fingerprint": tx["Device Fingerprint"],
        "IP Address": tx["IP Address"],
        "Geolocation": tx["Geolocation"],
    }])


def fe_bot(tx, prev=None):
    var = float(np.std([tx["avg val sent"], tx["avg val received"]]))
    freq = tx["Sent tnx"] + tx["Received Tnx"]
    return pd.DataFrame([{
        "Transaction Time Diff": tx["Time Diff between first and last (Mins)"],
        "Transaction Amount Variance": var,
        "Unique Sent Addresses": tx["Unique Sent To Addresses"],
        "Bot Activity Indicator": freq * var * tx["Unique Sent To Addresses"],
    }])


def block(addr):
    st.session_state.blocked_accounts.add(addr)


def run_detection(tx, prev, models):
    """Run all three models + heuristic rules. Returns list of alert tuples (severity, type, msg)."""
    alerts = []
    pipe_fake, pipe_sybil, pipe_bot = models["fake"][0], models["sybil"][0], models["bot"][0]
    addr = tx["Address"]
    short = addr[:10] + "…" + addr[-6:]

    if addr in st.session_state.blocked_accounts:
        return alerts  # already blocked

    # Heuristic: sybil cluster via IP / device fingerprint
    hist = pd.DataFrame(st.session_state.transaction_history)
    if len(hist) > 4:
        same_ip = (hist["IP Address"] == tx["IP Address"]).sum()
        same_dev = (hist["Device Fingerprint"] == tx["Device Fingerprint"]).sum()
        if same_ip > 5 or same_dev > 5:
            alerts.append(("high", "SYBIL-CLUSTER",
                           f"Cluster pattern: {addr[:10]}… shares IP/device with {max(same_ip, same_dev)} others. Auto-blocked."))
            block(addr); st.session_state.detection_counts["sybil"] += 1

    # ML predictions
    try:
        if int(pipe_fake.predict(fe_fake(tx, prev))[0]) == 1:
            prob = float(pipe_fake.predict_proba(fe_fake(tx, prev))[0, 1])
            alerts.append(("high", "FAKE-IDENTITY",
                           f"{short} flagged as fake identity (confidence {prob:.0%}). Investigation recommended."))
            st.session_state.detection_counts["fake"] += 1
    except Exception:
        pass

    try:
        if int(pipe_sybil.predict(fe_sybil(tx, prev))[0]) == 1:
            prob = float(pipe_sybil.predict_proba(fe_sybil(tx, prev))[0, 1])
            alerts.append(("high", "SYBIL-ATTACK",
                           f"{short} classified as Sybil actor (confidence {prob:.0%}). Auto-blocked."))
            block(addr); st.session_state.detection_counts["sybil"] += 1
    except Exception:
        pass

    try:
        if int(pipe_bot.predict(fe_bot(tx, prev))[0]) == 1:
            prob = float(pipe_bot.predict_proba(fe_bot(tx, prev))[0, 1])
            alerts.append(("high", "BOT-ACTIVITY",
                           f"{short} showing automated bot behaviour (confidence {prob:.0%}). Auto-blocked."))
            block(addr); st.session_state.detection_counts["bot"] += 1
    except Exception:
        pass

    # Rule-based bot indicator
    bot_ind = float(fe_bot(tx, prev)["Bot Activity Indicator"].iloc[0])
    if bot_ind > 200:
        alerts.append(("warn", "HIGH-ACTIVITY",
                       f"{short} has anomalously high activity indicator ({bot_ind:.1f}). Watch."))

    if addr not in KNOWN_GOOD:
        alerts.append(("info", "UNVERIFIED",
                       f"{short} is not on the verified allowlist. Follow-up recommended."))

    return alerts


# ─────────────────────────────────────────────────────────────────────
# Graph-based anomaly detection
# ─────────────────────────────────────────────────────────────────────
def run_graph_analysis():
    """Build wallet graph from transaction history and flag anomalies."""
    if len(st.session_state.transaction_history) < 3:
        return None, []

    G = nx.DiGraph()
    for tx in st.session_state.transaction_history:
        src = tx["Address"]
        num_dst = min(max(tx["Sent tnx"], 1), 3)
        for _ in range(num_dst):
            dst = "0x" + "".join(random.choices("0123456789abcdef", k=40))
            amt = tx["avg val sent"]
            if G.has_edge(src, dst):
                G[src][dst]["weight"] += amt
            else:
                G.add_edge(src, dst, weight=amt)

    if G.number_of_nodes() < 3:
        return None, []

    nodes = list(G.nodes())
    features = np.array([
        [G.in_degree(n),
         G.out_degree(n),
         sum(G[u][n]["weight"] for u in G.predecessors(n)),
         sum(G[n][v]["weight"] for v in G.successors(n))]
        for n in nodes
    ])
    if np.std(features, axis=0).min() > 0:
        features = (features - features.mean(0)) / features.std(0)

    clf = IsolationForest(random_state=42, contamination=0.15)
    preds = clf.fit_predict(features)
    anomalies = [nodes[i] for i, p in enumerate(preds) if p == -1]

    for a in anomalies:
        if a not in st.session_state.anomalies_detected:
            st.session_state.anomalies_detected.add(a)
            block(a)
            st.session_state.detection_counts["graph"] += 1

    # Render with dark theme
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0a0e1a")
    ax.set_facecolor("#0a0e1a")
    pos = nx.spring_layout(G, seed=42, k=0.9)
    normal = [n for n in G.nodes() if n not in anomalies]
    nx.draw_networkx_nodes(G, pos, nodelist=normal, node_color="#22c55e",
                           alpha=0.85, node_size=180, ax=ax)
    if anomalies:
        nx.draw_networkx_nodes(G, pos, nodelist=anomalies, node_color="#ef4444",
                               alpha=0.95, node_size=320, ax=ax, edgecolors="#fecaca", linewidths=2)
    edge_widths = [min(G[u][v]["weight"] * 0.4, 2.5) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.35, arrows=True,
                           edge_color="#64748b", ax=ax, arrowsize=8)
    ax.set_title("Wallet Transaction Network — Red nodes are detected anomalies",
                 color="#f1f5f9", fontsize=13, pad=18, fontweight="bold")
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="#0a0e1a", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf, anomalies


# ─────────────────────────────────────────────────────────────────────
# AI insights — LOCAL generator (no API needed) + optional Gemini
# ─────────────────────────────────────────────────────────────────────
def generate_local_insights():
    """Build a narrative security report from the current session state.
       Deterministic, no external calls, always works."""
    alerts = st.session_state.alerts
    blocked = st.session_state.blocked_accounts
    hist = st.session_state.transaction_history
    counts = st.session_state.detection_counts

    total_tx = len(hist)
    total_blocked = len(blocked)
    total_alerts = len(alerts)
    risk_score = min(100, int(20 + 40 * (total_blocked / max(total_tx, 1)) + 10 * (counts["bot"] + counts["sybil"])))

    # Classify alerts
    type_counts = {"FAKE-IDENTITY": 0, "SYBIL-ATTACK": 0, "BOT-ACTIVITY": 0,
                   "SYBIL-CLUSTER": 0, "HIGH-ACTIVITY": 0, "UNVERIFIED": 0}
    for a in alerts:
        for k in type_counts:
            if k in a.get("type", ""):
                type_counts[k] += 1
                break

    dominant = max(type_counts, key=type_counts.get) if any(type_counts.values()) else None

    # Risk level
    if risk_score >= 70:
        risk_label, risk_color = "CRITICAL", "#ef4444"
    elif risk_score >= 45:
        risk_label, risk_color = "ELEVATED", "#f59e0b"
    else:
        risk_label, risk_color = "NORMAL", "#22c55e"

    # Patterns
    ip_pattern = ""
    geo_pattern = ""
    if len(hist) >= 3:
        df_h = pd.DataFrame(hist)
        top_ip = df_h["IP Address"].value_counts()
        top_geo = df_h["Geolocation"].value_counts()
        if len(top_ip) and top_ip.iloc[0] / len(df_h) > 0.4:
            ip_pattern = f"{top_ip.iloc[0]} of {len(df_h)} transactions ({top_ip.iloc[0]/len(df_h):.0%}) originate from IP {top_ip.index[0]} — strong Sybil indicator."
        if len(top_geo):
            geo_pattern = f"Dominant geography: <strong>{top_geo.index[0]}</strong> ({top_geo.iloc[0]/len(df_h):.0%} of traffic)."

    # Build report
    html = f"""
    <h3>🛡️ Security Analysis Report</h3>
    <p><strong>Assessment period:</strong> {total_tx} transactions monitored · <strong>Generated:</strong> {datetime.now().strftime('%H:%M:%S')}</p>

    <div style="display:flex;gap:12px;margin:16px 0;flex-wrap:wrap;">
        <div style="background:{risk_color}22;border:1px solid {risk_color};padding:10px 18px;border-radius:10px;">
            <div style="color:#94a3b8;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;">Risk Level</div>
            <div style="color:{risk_color};font-size:1.5rem;font-weight:800;font-family:'JetBrains Mono',monospace;">{risk_label} · {risk_score}/100</div>
        </div>
        <div style="background:#1a2332;border:1px solid #2a3548;padding:10px 18px;border-radius:10px;">
            <div style="color:#94a3b8;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;">Accounts Blocked</div>
            <div style="color:#f1f5f9;font-size:1.5rem;font-weight:800;font-family:'JetBrains Mono',monospace;">{total_blocked}</div>
        </div>
        <div style="background:#1a2332;border:1px solid #2a3548;padding:10px 18px;border-radius:10px;">
            <div style="color:#94a3b8;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;">Alerts Raised</div>
            <div style="color:#f1f5f9;font-size:1.5rem;font-weight:800;font-family:'JetBrains Mono',monospace;">{total_alerts}</div>
        </div>
    </div>

    <h4>📊 Threat Breakdown</h4>
    <ul>
        <li><strong>Fake Identity flags:</strong> {counts['fake']} · wallets with behavioural signatures matching known fraud patterns</li>
        <li><strong>Sybil Attacks:</strong> {counts['sybil']} · coordinated multi-identity behaviour</li>
        <li><strong>Bot Activity:</strong> {counts['bot']} · automated, machine-driven transaction patterns</li>
        <li><strong>Graph Anomalies:</strong> {counts['graph']} · structural outliers in the wallet network</li>
    </ul>
    """

    if ip_pattern or geo_pattern:
        html += "<h4>🔍 Behavioural Patterns</h4><ul>"
        if ip_pattern: html += f"<li>{ip_pattern}</li>"
        if geo_pattern: html += f"<li>{geo_pattern}</li>"
        html += "</ul>"

    # Recommendations
    recs = []
    if counts["sybil"] > 0 or counts["fake"] > 0:
        recs.append("Enforce stricter KYC on accounts younger than 30 days with high transaction velocity.")
    if counts["bot"] > 0:
        recs.append("Deploy rate limiting and CAPTCHA challenges for addresses with bot indicators above 200.")
    if counts["graph"] > 0:
        recs.append("Manually review the graph-flagged wallets — structural outliers often represent novel attack vectors.")
    if risk_score >= 45:
        recs.append("Escalate to compliance team; consider temporarily pausing high-value transfers from flagged regions.")
    if not recs:
        recs.append("System is operating within normal parameters. Continue routine monitoring.")

    html += "<h4>💡 Recommended Actions</h4><ul>"
    for r in recs:
        html += f"<li>{r}</li>"
    html += "</ul>"

    # Try Groq first (fast, free, no CC required) → fallback to Gemini → fallback to local
    commentary = None
    provider = None

    groq_key = (GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")).strip()
    if groq_key and not commentary:
        try:
            import requests
            prompt_user = (
                f"You are a blockchain security analyst reviewing this session:\n"
                f"- Transactions monitored: {total_tx}\n"
                f"- Alerts raised: {total_alerts}\n"
                f"- Accounts auto-blocked: {total_blocked}\n"
                f"- Risk level: {risk_label} ({risk_score}/100)\n"
                f"- Detections by type: {counts}\n"
                f"Write exactly 3 sharp, actionable bullet points for the SOC team. "
                f"Be concise, use HTML <ul><li> tags, no preamble."
            )
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system",
                         "content": "You are a cybersecurity analyst. Respond in HTML with <ul><li> bullets only."},
                        {"role": "user", "content": prompt_user},
                    ],
                    "temperature": 0.5,
                    "max_tokens": 400,
                },
                timeout=20,
            )
            if resp.status_code == 200:
                commentary = resp.json()["choices"][0]["message"]["content"].strip()
                provider = "Groq · llama-3.3-70b"
        except Exception:
            commentary = None

    gemini_key = (GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")).strip()
    if gemini_key and not commentary:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (f"You are a blockchain security analyst. Given these detections "
                      f"({counts}) across {total_tx} txns, {total_blocked} blocked, "
                      f"risk={risk_label}. Write 3 sharp bullet insights in HTML <ul><li> format.")
            resp = model.generate_content(prompt)
            commentary = resp.text
            provider = "Google Gemini"
        except Exception:
            commentary = None

    if commentary:
        html += (f"<h4>🧠 Live LLM Commentary "
                 f"<span style='color:#64748b;font-size:0.75rem;font-weight:500;'>"
                 f"· {provider}</span></h4>"
                 f"<div style='background:#0f1729;padding:14px 18px;border-radius:8px;"
                 f"border-left:3px solid #a855f7;'>{commentary}</div>")

    return html


# ─────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="#1e293b", linecolor="#2a3548"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#2a3548"),
)


def plot_heatmap(df):
    if len(df) == 0:
        return go.Figure().update_layout(title="No data yet", **DARK_LAYOUT)
    agg = df.groupby("Geolocation").agg(
        total=("Sent tnx", "sum"),
        count=("Address", "count"),
    ).reset_index()
    fig = go.Figure(go.Bar(
        x=agg["Geolocation"], y=agg["total"],
        marker=dict(color=agg["total"], colorscale="Teal", line=dict(width=0)),
        text=agg["count"].apply(lambda x: f"{x} txns"),
        textposition="outside",
    ))
    fig.update_layout(title="Transaction Volume by Region", height=340, **DARK_LAYOUT)
    return fig


def plot_time_series(df):
    if len(df) == 0:
        return go.Figure().update_layout(title="No data yet", **DARK_LAYOUT)
    d = df.copy()
    d["Total Tnx"] = d["Sent tnx"] + d["Received Tnx"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["Timestamp"], y=d["Total Tnx"],
        mode="lines+markers", name="Transaction Count",
        line=dict(color="#00d9ff", width=2.5),
        marker=dict(size=7, color="#00d9ff"),
    ))
    fig.add_trace(go.Scatter(
        x=d["Timestamp"], y=d["avg val sent"],
        mode="lines+markers", name="Avg Sent (ETH)", yaxis="y2",
        line=dict(color="#a855f7", width=2.5, dash="dot"),
        marker=dict(size=6, color="#a855f7"),
    ))
    fig.update_layout(
        title="Transaction Metrics Over Time", height=380,
        yaxis=dict(title="Transaction Count", gridcolor="#1e293b"),
        yaxis2=dict(title="Amount (ETH)", overlaying="y", side="right", gridcolor="#1e293b"),
        legend=dict(bgcolor="rgba(26,35,50,0.8)", bordercolor="#2a3548", borderwidth=1),
        **{k: v for k, v in DARK_LAYOUT.items() if k not in ("yaxis",)},
    )
    return fig


def plot_detection_breakdown(counts):
    fig = go.Figure(go.Bar(
        x=["Fake Identity", "Sybil Attack", "Bot Activity", "Graph Anomaly"],
        y=[counts["fake"], counts["sybil"], counts["bot"], counts["graph"]],
        marker=dict(color=["#00d9ff", "#a855f7", "#f59e0b", "#22c55e"], line=dict(width=0)),
        text=[counts["fake"], counts["sybil"], counts["bot"], counts["graph"]],
        textposition="outside", textfont=dict(size=14, color="#f1f5f9"),
    ))
    fig.update_layout(title="Detection Counts by Category", height=340, **DARK_LAYOUT)
    return fig


def plot_address_network(df):
    """Plotly wallet graph with clickable nodes, nicer than the matplotlib fallback."""
    if len(df) < 2:
        return go.Figure().update_layout(title="Not enough data", **DARK_LAYOUT)

    G = nx.Graph()
    for i, row in df.iterrows():
        G.add_node(row["Address"])
        if i > 0 and random.random() < 0.35:
            G.add_edge(df.iloc[i - 1]["Address"], row["Address"])

    pos = nx.spring_layout(G, seed=42, k=0.8)
    ex, ey = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
        ex += [x0, x1, None]; ey += [y0, y1, None]

    nx_coords = [(pos[n][0], pos[n][1], n) for n in G.nodes()]
    colors = ["#ef4444" if n in st.session_state.blocked_accounts else "#00d9ff"
              for n in G.nodes()]
    sizes = [14 if n in st.session_state.blocked_accounts else 10 for n in G.nodes()]
    labels = [n[:8] + "…" for n in G.nodes()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                             line=dict(color="#334155", width=1), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=[c[0] for c in nx_coords], y=[c[1] for c in nx_coords],
        mode="markers", marker=dict(size=sizes, color=colors,
                                    line=dict(color="#0f172a", width=1.5)),
        text=labels, hovertemplate="<b>%{text}</b><extra></extra>",
    ))
    fig.update_layout(title="Live Wallet Interaction Graph  (red = blocked)", height=430,
                      showlegend=False,
                      **{**DARK_LAYOUT,
                         "xaxis": dict(visible=False), "yaxis": dict(visible=False)})
    return fig


# ─────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────
def main():
    init_state()

    # Load/train models once
    with st.spinner("🔧 Training fraud-detection models (first run only)…"):
        models = train_all_models()

    # Auto-seed with some transactions so dashboards aren't empty
    if not st.session_state.transaction_history:
        prev = None
        for _ in range(12):
            tx = simulate_transaction()
            new_alerts = run_detection(tx, prev, models)
            st.session_state.transaction_history.append(tx)
            for sev, kind, msg in new_alerts:
                st.session_state.alerts.append({
                    "time": tx["Timestamp"], "severity": sev,
                    "type": kind, "message": msg,
                })
            prev = tx

    # ═════ HERO ═════
    st.markdown("""
    <div class="hero">
        <span class="badge">🛡️ CryptoSentinel · v2.0</span>
        <h1>Blockchain Fraud Detection Dashboard</h1>
        <p class="subtitle">AI-powered real-time monitoring for cryptocurrency transactions — Fake Identity, Sybil Attack, and Bot Activity detection with wallet-graph anomaly analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    # ═════ SIDEBAR ═════
    with st.sidebar:
        st.markdown("### Detection Models")
        for key, label in [("fake", "Fake Identity"),
                           ("sybil", "Sybil Attack"),
                           ("bot", "Bot Activity")]:
            m = models[key][1]
            st.markdown(
                f"""<div class="model-status">{label} · AUC {m['auc']:.3f}</div>""",
                unsafe_allow_html=True
            )

        st.markdown("### Controls")

        if st.button("⚡ Generate New Transaction", key="btn_gen"):
            prev = (st.session_state.transaction_history[-1]
                    if st.session_state.transaction_history else None)
            tx = simulate_transaction(force_suspicious=random.random() < 0.4)
            new_alerts = run_detection(tx, prev, models)
            st.session_state.transaction_history.append(tx)
            for sev, kind, msg in new_alerts:
                st.session_state.alerts.append({
                    "time": tx["Timestamp"], "severity": sev,
                    "type": kind, "message": msg,
                })
            st.session_state.last_action = (
                f"Generated tx · {len(new_alerts)} new alert"
                f"{'s' if len(new_alerts) != 1 else ''}"
            )
            st.toast(f"✓ New transaction processed — {len(new_alerts)} alerts raised",
                     icon="⚡")
            st.rerun()

        if st.button("🕸️ Run Graph Analysis", key="btn_graph"):
            with st.spinner("Analysing wallet network…"):
                img, anomalies = run_graph_analysis()
                if img:
                    st.session_state.graph_img = img
                    st.session_state.last_action = (
                        f"Graph analysis complete · {len(anomalies)} anomalies found"
                    )
                    st.toast(f"✓ Graph analysis — {len(anomalies)} anomalies",
                             icon="🕸️")
                else:
                    st.toast("Need a few more transactions first", icon="⚠️")
            st.rerun()

        if st.button("🧠 Run AI Analysis", key="btn_ai"):
            with st.spinner("Synthesising security insights…"):
                st.session_state.insights_html = generate_local_insights()
                st.session_state.last_action = "AI analysis refreshed"
                st.toast("✓ AI insights generated", icon="🧠")
            st.rerun()

        st.markdown("### Session")
        if st.button("🗑️ Reset Session", key="btn_reset"):
            for k in ["transaction_history", "alerts", "blocked_accounts",
                      "anomalies_detected", "graph_img", "insights_html"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.detection_counts = {"fake": 0, "sybil": 0, "bot": 0, "graph": 0}
            st.toast("Session reset", icon="🔄")
            st.rerun()

        if st.session_state.last_action:
            st.caption(f"Last: {st.session_state.last_action}")

        st.markdown("### Model Performance")
        perf = pd.DataFrame({
            "Model": ["Fake", "Sybil", "Bot"],
            "AUC": [models["fake"][1]["auc"],
                    models["sybil"][1]["auc"],
                    models["bot"][1]["auc"]],
        })
        st.dataframe(perf.round(3), hide_index=True, use_container_width=True,
                     key="sidebar_perf_table")
        st.caption(f"Trained on {models['trained_on']:,} Ethereum addresses.")

    # ═════ TOP METRICS ═════
    c1, c2, c3, c4 = st.columns(4)
    total_tx = len(st.session_state.transaction_history)
    total_alerts = len(st.session_state.alerts)
    total_blocked = len(st.session_state.blocked_accounts)
    risk_score = min(100, int(20 + 40 * (total_blocked / max(total_tx, 1)) + 3 * total_alerts))
    c1.metric("Transactions", f"{total_tx:,}")
    c2.metric("Alerts Raised", f"{total_alerts:,}")
    c3.metric("Blocked", f"{total_blocked:,}")
    c4.metric("Risk Score", f"{risk_score}/100")

    # ═════ TABS ═════
    tab_overview, tab_analytics, tab_network, tab_ai, tab_data = st.tabs(
        ["🏠 Overview", "📊 Analytics", "🕸️ Network", "🧠 AI Insights", "📋 Transactions"]
    )

    # ─── Overview tab ───
    with tab_overview:
        col_a, col_b = st.columns([1, 1.6])

        with col_a:
            st.markdown("<div class='section-title'>Recent Alerts</div>", unsafe_allow_html=True)
            if not st.session_state.alerts:
                st.info("No alerts yet. Click **Generate New Transaction** in the sidebar.")
            else:
                for a in reversed(st.session_state.alerts[-8:]):
                    css = {"high": "", "warn": "warn", "info": "info"}.get(a["severity"], "")
                    st.markdown(f"""
                    <div class="alert-card {css}">
                        <span class="tag">{a['type']}</span>
                        {a['message']}
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<div class='section-title'>Blocked Wallets</div>", unsafe_allow_html=True)
            if st.session_state.blocked_accounts:
                pills = "".join(
                    f'<span class="blocked-pill">{a[:8]}…{a[-4:]}</span>'
                    for a in sorted(st.session_state.blocked_accounts)[:12]
                )
                st.markdown(pills, unsafe_allow_html=True)
                if len(st.session_state.blocked_accounts) > 12:
                    st.caption(f"…and {len(st.session_state.blocked_accounts) - 12} more")
            else:
                st.caption("None yet.")

        with col_b:
            df = pd.DataFrame(st.session_state.transaction_history)
            st.plotly_chart(plot_time_series(df), use_container_width=True,
                            key="chart_overview_timeseries")
            st.plotly_chart(plot_detection_breakdown(st.session_state.detection_counts),
                            use_container_width=True,
                            key="chart_overview_breakdown")

    # ─── Analytics tab ───
    with tab_analytics:
        df = pd.DataFrame(st.session_state.transaction_history)
        a1, a2 = st.columns(2)
        with a1:
            st.plotly_chart(plot_heatmap(df), use_container_width=True,
                            key="chart_analytics_heatmap")
        with a2:
            st.plotly_chart(plot_detection_breakdown(st.session_state.detection_counts),
                            use_container_width=True,
                            key="chart_analytics_breakdown")
        st.plotly_chart(plot_time_series(df), use_container_width=True,
                        key="chart_analytics_timeseries")

    # ─── Network tab ───
    with tab_network:
        st.markdown("<div class='section-title'>Interactive Wallet Network</div>",
                    unsafe_allow_html=True)
        df = pd.DataFrame(st.session_state.transaction_history)
        st.plotly_chart(plot_address_network(df), use_container_width=True,
                        key="chart_network_wallet")

        st.markdown("<div class='section-title'>Graph Anomaly Detection</div>",
                    unsafe_allow_html=True)
        if st.session_state.graph_img:
            st.image(st.session_state.graph_img, use_container_width=True)
        else:
            st.info("Click **Run Graph Analysis** in the sidebar to detect structural anomalies.")

    # ─── AI Insights tab ───
    with tab_ai:
        st.markdown("<div class='section-title'>Automated Security Report</div>",
                    unsafe_allow_html=True)
        if st.session_state.insights_html:
            st.markdown(f'<div class="insights-box">{st.session_state.insights_html}</div>',
                        unsafe_allow_html=True)
        else:
            st.info("Click **Run AI Analysis** in the sidebar. Reports are generated locally — "
                    "optionally enhanced with Gemini if `GEMINI_API_KEY` is set.")

    # ─── Transactions tab ───
    with tab_data:
        st.markdown("<div class='section-title'>Transaction Log</div>", unsafe_allow_html=True)
        if st.session_state.transaction_history:
            df = pd.DataFrame(st.session_state.transaction_history)
            display_cols = ["Timestamp", "Address", "Sent tnx", "Received Tnx",
                            "avg val sent", "avg val received", "Geolocation",
                            "IP Address", "Device Fingerprint"]
            df_display = df[display_cols].copy()
            df_display["Timestamp"] = df_display["Timestamp"].dt.strftime("%H:%M:%S")
            df_display["Address"] = df_display["Address"].apply(lambda x: x[:10] + "…" + x[-6:])
            df_display["avg val sent"] = df_display["avg val sent"].round(4)
            df_display["avg val received"] = df_display["avg val received"].round(4)
            df_display["Blocked"] = df["Address"].apply(
                lambda a: "🔴" if a in st.session_state.blocked_accounts else "🟢"
            )
            st.dataframe(df_display.iloc[::-1], use_container_width=True,
                         hide_index=True, height=450,
                         key="tx_log_table")
        else:
            st.info("No transactions yet.")


if __name__ == "__main__":
    main()