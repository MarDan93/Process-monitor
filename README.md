# Process-monitor
# PCA-SPC Industrial Process Monitor

> Multivariate Statistical Process Control for injection molding and manufacturing — built by a Process Engineer, for Process Engineers.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This project implements a full **PCA-based Statistical Process Control (SPC)** pipeline designed for real-world industrial use. It goes beyond univariate control charts by detecting multivariate anomalies that single-variable monitoring would miss — identifying **which process variables are responsible** through contribution plots, and generating **plain-language explanations** for production technicians via AI.

The system is built around the workflow used in advanced manufacturing environments (pharmaceutical, semiconductor, plastics) and aims to make it accessible to production teams without dedicated data science resources.

---

## Why PCA-SPC?

Modern injection molding machines already monitor individual variables and trigger alarms when they exceed fixed tolerances. However, this approach has critical limitations:

| Traditional (univariate) | This system (multivariate) |
|---|---|
| Each variable monitored independently | All variables monitored together |
| Misses correlated anomalies | Detects subtle combined shifts |
| Binary alarm: in / out of tolerance | Severity score: how far, how fast |
| No root cause information | Contribution plots identify cause |
| No memory of past events | Intervention log with history |
| No explanation for technician | AI-generated diagnosis in plain language |

---

## Workflow

```
Raw data (USB export from machine panel)
        ↓
① Dataset Explorer
  Descriptive statistics · Missing values · Correlation heatmap · AI summary

        ↓
② Principal Component Selection
  Scree plot · Cumulative variance · RMSECV cross-validation

        ↓
③ Phase I — Model Calibration
  Iterative outlier cleaning · PCA-SPC fit
  T² (Hotelling) + Q (SPE) control limits
  Contribution plots on training outliers

        ↓
④ Loadings & Score Plots
  All PC pairs selectable · Loading biplot · Score plot with T² ellipse

        ↓
⑤ Phase II — Production Monitoring
  Upload new data → T² and Q charts
  → Anomaly table sorted by severity
  → Contribution plots per variable with UCL/LCL
  → AI explanation for technician
  → Intervention log
```

---

## Repository Structure

```
process-monitor/
├── process_monitor_app.py                  # Streamlit web application
├── notebooks/
│   └── PCA_SPC_Industrial_Template.ipynb  # Full analysis notebook (Jupyter / Colab)
├── requirements.txt                        # Python dependencies
└── README.md
```

---

## Key Methods

| Step | Method |
|---|---|
| Preprocessing | Autoscaling (mean = 0, std = 1) |
| PC selection | Kaiser criterion, cumulative variance, RMSECV |
| T² control limit | Hotelling T² — F distribution |
| Q control limit | Jackson–Mudholkar approximation |
| Contribution plots | Signed contributions per variable |
| Phase I cleaning | Iterative outlier removal until convergence |
| AI explanations | Google Gemini 2.5 Flash API (free tier) |

---

## Web Application

A **Streamlit app** is included, providing a zero-code interface for production engineers and shift supervisors.

### Live demo
🔗 [process-monitor.streamlit.app](https://process-monitor-6qzenpxqzttrvce3ie3qoz.streamlit.app/)

### Run locally
```bash
pip install -r requirements.txt
streamlit run process_monitor_app.py
```

### AI explanations setup (optional — free)
The app uses **Google Gemini 2.5 Flash** for plain-language interpretations of results.
Get a free API key at [aistudio.google.com](https://aistudio.google.com), then:

- **Streamlit Cloud:** Settings → Secrets → add `GEMINI_API_KEY = "your-key"`
- **Local:** create `.streamlit/secrets.toml` with the same line

---

## Notebook

The Jupyter notebook (`PCA_SPC_Industrial_Template.ipynb`) implements the same pipeline with full transparency — every step is commented and explained. Designed to run on **Google Colab** or locally.

Built as a **generic template**: change only the configuration block (Section 1) to apply it to any industrial dataset.

Tested on:
- Injection molding process data (proprietary, anonymised)

---

## Status

This project is under active development. The analytical pipeline and core SPC logic are functional. The web application interface is continuously being refined.

---

## Roadmap

- [x] PCA component selection (Scree, cumulative variance, RMSECV)
- [x] Phase I iterative cleaning
- [x] T² and Q control charts with contribution plots
- [x] Streamlit web application with AI-powered explanations
- [ ] PLS model for quality variable prediction (Y from X)
- [ ] Adaptive monitoring with process drift detection
- [ ] Multi-product / multi-mold model management
- [ ] Direct machine connectivity (Euromap 63 / OPC-UA)
- [ ] PDF report export

---

## Author

**Daniele Marangon** — Process Engineer | Lean Six Sigma Green Belt

6+ years of experience in chemical, plastics, and manufacturing industries.
Currently exploring the application of machine learning to industrial process monitoring and quality control.

[![GitHub](https://img.shields.io/badge/GitHub-MarDan93-black?logo=github)](https://github.com/MarDan93)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/YOUR-PROFILE)

---

## License

MIT License — free to use, modify, and distribute with attribution.
