# PCA-SPC Industrial Process Monitor

> Multivariate Statistical Process Control for manufacturing —
> built by a Process Engineer, for Process Engineers.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-All%20rights%20reserved-lightgrey)](#license)

---

## 🎬 Live Demo

🔗 **[process-monitor.streamlit.app](https://process-monitor-6qzenpxqzttrvce3ie3qoz.streamlit.app/)**

A demo dataset is available in this repository (`demo_industrial_process.csv`).
Load it directly into the app to explore the full workflow without any setup.

<!-- ![Demo](docs/demo.gif) -->

---

## The problem this solves

Modern industrial machines monitor variables individually with fixed thresholds.
But process anomalies are often **multivariate** — no single variable exceeds
its limit, yet the combination signals a real problem.

| Traditional (univariate SPC) | This tool (multivariate PCA-SPC) |
|---|---|
| Each variable monitored independently | All variables monitored together |
| Misses correlated anomalies | Detects subtle combined shifts |
| Binary alarm: in / out of tolerance | Severity score with statistical confidence |
| No root cause information | Contribution plots identify the cause |
| No memory of past process state | Persistent model calibrated on your data |
| No explanation for the technician | AI-generated diagnosis in plain language |

Applicable to any continuous or batch manufacturing process:
injection molding, extrusion, chemical reactors, semiconductor fabrication,
pharmaceutical manufacturing, food processing, and more.

---

## Workflow

```
① Process Context
   Describe your process, equipment, and analysis objective
   The AI uses this to contextualise every response

② Dataset
   Load CSV/Excel from your data acquisition system or machine USB
   Descriptive statistics · missing value handling · optional train/test split

③ PC Selection
   Scree plot · Cumulative variance · RMSECV cross-validation
   Choose the number of principal components

④ Calibration (Phase I)
   Build PCA-SPC reference model on stable production data
   T² UCL and Q UCL from your actual data distribution
   Iterative outlier cleaning · contribution plots on residual flags

⑤ Loadings & Score Plots
   All PC pairs selectable
   Loading biplot (variable correlations)
   Score plot with correct bivariate confidence ellipse

⑥ Monitoring (Phase II)
   Upload multiple files (different time periods or batches)
   Combined T² and Q control charts with file-source shading
   Click any anomalous observation → contribution plots → AI explanation

⑦ Summary & Root Cause
   Top anomalies ranked by severity
   Variable exceedance frequency across all anomalies
   AI-generated structured root cause analysis and action plan
```

---

## Key methods

| Step | Method |
|---|---|
| Preprocessing | Autoscaling (μ=0, σ=1) |
| PC selection | Kaiser criterion, cumulative variance, RMSECV |
| T² control limit | Hotelling T²: `T²_lim = A(N-1)/(N-A) · F(A, N-A, 1-α)` |
| Q control limit | Jackson–Mudholkar approximation |
| Score plot ellipse | Bivariate F limit (k=2) — correct for 2D projection |
| Contribution plots | Signed contributions per variable with UCL/LCL (±1.96σ) |
| Phase I cleaning | Iterative outlier removal until convergence |
| AI explanations | Google Gemini 2.5 Flash (free) with automatic fallback chain |

---

## Demo dataset

`demo_industrial_process.csv` — 1200 production cycles, 15 process variables
(temperatures, pressures, times, positions, speeds) and 2 quality variables.

Contains three realistic anomaly scenarios:

| Period | Event | Variables involved |
|---|---|---|
| Cycles 0–699 | Stable reference period (train) | — |
| Cycles 700–749 | Gradual thermal drift | Temperature Zone 1–3, Fill Time |
| Cycles 800–809 | Pressure spike | Injection Pressure, Holding Pressure |
| Cycles 950–1199 | Material change (viscosity shift) | Back Pressure, Screw Speed, Fill Time |

Suggested setup for demo:
- Y Variables (sidebar): `Part_Weight_g`, `Cushion_mm`
- Split: Temporal, 60% train (cycles 0–699)
- Objective: Statistical Process Control

---

## Features

- **Adaptive workflow** — guided steps adapt to your objective
  (SPC monitoring / Diagnostics / Exploratory analysis)
- **Multi-file monitoring** — load multiple exports and analyse them together
  with time-source shading on charts
- **AI in every section** — context-aware responses that know your process,
  model state, and detected anomalies
- **AI chat popup** — always-accessible chat with full project context
- **Language detection** — AI responds in the language you use to describe
  your process (Italian, English, German, French, ...)
- **Model persistence** — save and reload your session as a `.pkl` file
- **Multi-model AI fallback** — Gemini 2.5 Flash → Gemini 2.5 Flash-Lite → Claude Haiku

---

## Run locally

```bash
git clone https://github.com/MarDan93/process-monitor
cd process-monitor
pip install -r requirements.txt
streamlit run process_monitor_app.py
```

**AI setup (optional — free):**
Get a key at [aistudio.google.com](https://aistudio.google.com), then:

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "AIza..."
ANTHROPIC_API_KEY = "sk-ant-..."   # optional paid fallback
```

---

## Repository structure

```
process-monitor/
├── process_monitor_app.py                   # Streamlit web application
├── demo_industrial_process.csv              # Demo dataset (synthetic, realistic)
├── notebooks/
│   └── PCA_SPC_Industrial_Template.ipynb   # Full analysis notebook (Colab / Jupyter)
├── requirements.txt
└── README.md
```

---

## Status

Active development. Core analytical pipeline and SPC logic are stable.
UI and AI features are continuously improved.

---

## Roadmap

- [x] PCA component selection (Scree, cumulative variance, RMSECV)
- [x] Phase I iterative cleaning
- [x] T² and Q control charts with contribution plots
- [x] Adaptive workflow (SPC / Diagnostics / Exploratory)
- [x] Multi-file monitoring with time-source indicators
- [x] AI explanations with language detection and model fallback
- [x] Model persistence (save/load session)
- [ ] PLS model for quality variable prediction (Y from X)
- [ ] Operator View — simplified interface for production floor use
- [ ] Direct connectivity (OPC-UA / machine APIs)
- [ ] PDF report export

---

## Author

**Daniele Marangon** — Process Engineer | Lean Six Sigma Green Belt

6+ years of experience in chemical, plastics, and manufacturing industries.
Applying machine learning to industrial process monitoring and quality control.

[![GitHub](https://img.shields.io/badge/GitHub-MarDan93-black?logo=github)](https://github.com/MarDan93)

---

## License

© 2025 Daniele Marangon — All rights reserved.

The source code is publicly visible for portfolio and demonstration purposes.
Reproduction, redistribution, or commercial use requires explicit written permission.
