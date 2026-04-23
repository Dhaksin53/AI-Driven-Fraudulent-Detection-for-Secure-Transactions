# 🛡️ CryptoSentinel — AI-Powered Cryptocurrency Fraud Detection on Web3

A single-file, production-quality Streamlit dashboard that detects fraudulent
cryptocurrency wallets in real time using three complementary ML models plus
graph-based anomaly detection.

**Everything runs in one command.** No API keys. No external services. No
broken pickle files.

---

## ⚡ Quick start

Three commands and you're running:

```bash
pip install -r requirements.txt
streamlit run app.py
```

That's it. Open `http://localhost:8501` in your browser.

(On a fresh machine, wrap the first command in a virtualenv:
`python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate`)

The first run takes about 5 seconds — it trains all three fraud-detection
models from scratch on the bundled Ethereum transaction dataset. Subsequent
reloads are instant thanks to Streamlit's resource cache.

---

## 🎬 What you'll see

The dashboard auto-seeds 12 simulated transactions on load so nothing is
empty. Then you drive the demo with three sidebar buttons:

| Button | What it does | Visible feedback |
|---|---|---|
| **⚡ Generate New Transaction** | Simulates a new on-chain transaction, runs all three fraud models on it, raises alerts, auto-blocks malicious addresses | Toast notification, new alert cards, metric counters tick up |
| **🕸️ Run Graph Analysis** | Builds the wallet interaction graph, runs Isolation Forest to find structural anomalies, renders a visualization | Toast, network graph appears in the **Network** tab |
| **🧠 Run AI Analysis** | Generates a full security report with risk score, threat breakdown, behavioural patterns, and recommendations | Toast, rich report appears in the **AI Insights** tab |
| **🗑️ Reset Session** | Clears history and starts fresh | Toast, dashboard resets |

All four buttons produce **immediate visible feedback** via Streamlit toasts
and live UI updates. No more mystery clicks.

---

## 🧠 AI Insights — no API key needed (Groq & Gemini optional)

The AI Analysis panel runs a **deterministic local analyser** that
synthesises a full security report from your session's transaction history,
alert patterns, and geographic distribution. It generates:

- Risk level (Normal / Elevated / Critical) with a 0–100 score
- Threat breakdown by category with counts
- Behavioural pattern detection (IP clusters, geo concentration)
- Contextual recommendations that change based on which threat types are dominant

### 🚀 Live LLM commentary (optional, FREE)

The app supports two LLM providers — it'll try them in order and use whichever
is configured. **Both have free tiers.** If neither key is set, the local
report is still complete and looks professional — you don't need a key to submit.

**Option A — Groq (recommended, instant, truly free):**
1. Sign up at https://console.groq.com (free, no credit card)
2. Create an API key
3. Set the env var before launching:

```bash
# Mac / Linux
export GROQ_API_KEY="gsk_your_key_here"
streamlit run app.py

# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_key_here"
streamlit run app.py
```

The app uses `llama-3.3-70b-versatile` on Groq — fast (~1 second responses)
and completely free within generous rate limits.

**Option B — Gemini (fallback):**
Same approach with `GEMINI_API_KEY` from https://aistudio.google.com/app/apikey.
Used automatically if Groq isn't configured.

When either is enabled, a "Live LLM Commentary" section appears at the
bottom of the AI Insights report with fresh analyst-grade bullets.

---

## 📊 Measured model performance

Trained on the bundled Ethereum transaction dataset (9,841 addresses, 22.14%
fraud rate). Metrics on a stratified 20% hold-out:

| Detector       | Features | Accuracy | F1 Score | ROC-AUC |
|---------------:|---------:|---------:|---------:|--------:|
| Fake Identity  | 7        | 93.91%   | 0.851    | 0.980   |
| Sybil Attack   | 13       | 93.30%   | 0.837    | 0.980   |
| Bot Activity   | 4        | 90.55%   | 0.770    | 0.950   |

Inference: ~0.1 ms per transaction (≈10,000 predictions/sec on one core).
Training: under 5 seconds on the full dataset.

---

## 📁 Project layout

```
CryptoFraudDetection_Fixed/
├── app.py                          # ⭐ The entire application — run this
├── transaction_dataset.csv         # 9,841 labelled Ethereum addresses
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── fraud_detection_ai_web3.ipynb   # Original training notebook (reference)
├── FakeIdentityDetector/           # Original module (reference)
├── Sybil Attack Simulation/        # Original module (reference)
├── flag_boto/                      # Original module (reference)
├── Hackindia.pdf                   # Original hackathon submission doc
└── Untitled document (1).pdf       # Project notes
```

**You only need `app.py`, `transaction_dataset.csv`, and `requirements.txt`**
to run. The other folders are kept for reference / project submission.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Transaction Generator  (simulates realistic Ethereum txns) │
└──────────────────────────┬─────────────────────────────────┘
                           │
      ┌────────────────────┼──────────────────────┐
      ▼                    ▼                      ▼
┌───────────┐       ┌───────────┐          ┌──────────────┐
│   Fake    │       │   Sybil   │          │     Bot      │
│ Identity  │       │  Attack   │          │   Activity   │
│    RF     │       │    RF     │          │      RF      │
│ 7 features│       │13 features│          │  4 features  │
└─────┬─────┘       └─────┬─────┘          └──────┬───────┘
      │                   │                        │
      └───────────────────┼────────────────────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   Alert Pipeline   │
                │  + Auto-blocking   │
                └──────────┬─────────┘
                           │
              ┌────────────┴─────────────┐
              ▼                          ▼
    ┌─────────────────┐         ┌────────────────────┐
    │ Graph Anomaly   │         │  AI Insights       │
    │ Isolation Forest│         │  Local Generator   │
    │ on wallet graph │         │  (+ optional Gemini)│
    └─────────────────┘         └────────────────────┘
```

---

## 🐛 Troubleshooting

**"ModuleNotFoundError: No module named 'streamlit'"**
Run `pip install -r requirements.txt` first.

**"Port 8501 is already in use"**
Something else is using that port. Run with `streamlit run app.py --server.port 8502`.

**Dashboard looks empty after clicking a button**
It shouldn't — every button triggers a toast notification and a UI refresh.
If you somehow get into a broken state, click **🗑️ Reset Session** in the sidebar.

**Want to re-train the models?**
They re-train automatically when you change `app.py` and Streamlit reloads,
or clear the cache with `C` in the Streamlit menu (top right).
