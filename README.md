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

A demo dataset is available — see the **Demo dataset** section below for download instructions and app configuration.

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

The `demo/` folder contains the **Multi-stage Continuous-Flow Manufacturing Process** dataset
— real process data from a multi-stage continuous production line, ideal for demonstrating
PCA-SPC monitoring on industrial sensor data.

**Source:** Kaggle —
[Multi-stage continuous-flow manufacturing process](https://www.kaggle.com/datasets/supergus/multistage-continuousflow-manufacturing-process)
**License:** CC0 Public Domain — free to use and redistribute without restrictions.

| Property | Value |
|---|---|
| Observations | ~14,000 production cycles |
| Variables | Ambient conditions, raw material properties, zone temperatures, motor amperages, material pressures |
| Quality outputs (Y) | Stage 1 and Stage 2 output measurements |
| Missing values | None |

### ⬇️ Try the demo

1. Download the dataset from Kaggle (free account required):
   [→ kaggle.com/datasets/supergus/multistage-continuousflow-manufacturing-process](https://www.kaggle.com/datasets/supergus/multistage-continuousflow-manufacturing-process)

   *Or use the file in this repository's `demo/` folder if available.*

2. Open the app:
   **[→ process-monitor.streamlit.app](https://process-monitor-6qzenpxqzttrvce3ie3qoz.streamlit.app/)**

3. Configure the sidebar:
   ```
   Columns to exclude:   time_stamp

   Y Variables:          Stage1.Output.Measurement0
                         Stage1.Output.Measurement1
                         Stage2.Output.Measurement0
                         Stage2.Output.Measurement1
   ```

4. Go to **⚙️ Process Context** → select **Statistical Process Control** → Save

5. Go to **📂 Dataset** → upload the CSV → apply Temporal split (70%)

6. Follow the guided workflow: PC Selection → Calibration → Monitoring → Summary

> **Note:** Several temperature columns have near-zero variance (measurement noise only)
> and are automatically removed by the app during preprocessing. This is expected and normal.

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

## Dataset requirements & current limitations

### What works well
- **Continuous processes** — injection molding, extrusion, reactors, assembly lines
- **Numerical sensor data** — temperatures, pressures, flows, positions, speeds
- **Stationary processes** — stable operating conditions during the calibration period
- **5 to ~50 variables** — beyond that, RMSECV slows down and contribution plots lose interpretability
- **200+ observations** in the train set (minimum ~5–10× number of variables)

### Current limitations

| Limitation | Detail |
|---|---|
| **No batch data support** | Each row is treated as an independent observation. Batch processes (with start-up, steady-state, and end phases) require Multiway PCA preprocessing before use. |
| **No temporal structure** | The app does not model autocorrelation or time dependencies. If observations are strongly correlated in time, UCL estimates may be slightly off. |
| **Numerical variables only** | Categorical variables (material type, operator, recipe, shift) are ignored. If they drive process variability, contribution plots may point to the wrong variables. |
| **Discretised variables** | Variables with few distinct values (fixed setpoints, rounded readings) can create stripe patterns in score plots — not anomalies, just data structure. |
| **Static model** | The Phase I model does not update automatically. If the process drifts permanently (tool wear, supplier change), the model must be recalibrated manually. |
| **No slow drift detection** | Gradual process changes over weeks or months are not automatically flagged. The user must decide when to recalibrate. |

> **In short:** this tool works best on **continuous, stationary, numerical process data**
> where the goal is to detect deviations from a known stable reference period.
> Batch processes, time-series forecasting, and categorical process variables
> are out of scope in the current version.



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
