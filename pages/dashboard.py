import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

dash.register_page(__name__, path="/", name="Dashboard", order=0)

# ── Layout ────────────────────────────────────────────────────
layout = html.Div([
    html.Div([
        html.P("Dashboard", className="page-title"),
        html.P("Panoramica del processo e stato del modello.", className="page-subtitle"),
    ]),
    html.Div(id="dash-content"),
])


@callback(
    Output("dash-content", "children"),
    Input("ctx-store",   "data"),
    Input("model-store", "data"),
    Input("mon-store",   "data"),
    Input("data-store",  "data"),
)
def render_dashboard(ctx, model_data, mon, data):
    ctx        = ctx        or {}
    model_data = model_data or {}
    mon        = mon        or {}
    data       = data       or {}

    model_b64 = model_data.get("model_b64")
    mon_files = mon.get("mon_files", [])
    df_X_ok   = bool(data.get("df_X_json"))

    # ── Empty state ───────────────────────────────────────────
    if not model_b64 and not df_X_ok:
        return dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("🏭", style={"fontSize": "48px", "marginBottom": "12px"}),
                    html.H4("Benvenuto in Process Monitor",
                            style={"fontWeight": 700, "color": "#1E293B"}),
                    html.P(
                        "Strumento PCA-SPC per il monitoraggio multivariato di processi industriali. "
                        "Inizia dalla sezione Setup per caricare i tuoi dati.",
                        style={"color": "#64748B", "fontSize": "14px", "maxWidth": "480px",
                               "margin": "0 auto 20px"},
                    ),
                    dcc.Link(
                        dbc.Button("🚀 Vai al Setup", color="primary", size="lg"),
                        href="/setup",
                    ),
                ], style={"textAlign": "center", "padding": "60px 20px"}),
            ], width=12),
        ])

    children = []

    # ── Process context banner ────────────────────────────────
    if ctx.get("process_context"):
        wf = _get_workflow(ctx.get("analysis_objective", ""))
        wf_styles = {
            "spc":         ("#EFF6FF", "#2563EB", "🔵 SPC"),
            "diagnostic":  ("#FFF7ED", "#D97706", "🟠 Diagnostics"),
            "exploratory": ("#F0FDF4", "#16A34A", "🟢 Exploratory"),
        }
        bg, col, lbl = wf_styles.get(wf, wf_styles["spc"])
        children.append(
            html.Div(
                [html.Strong(f"{lbl} — "), ctx["process_context"][:120]],
                className="wf-banner",
                style={"background": bg, "borderLeftColor": col, "color": col},
            )
        )

    # ── KPI row ───────────────────────────────────────────────
    kpi_cards = []
    if model_b64:
        from core.store_utils import b64_to_model
        try:
            mdl = b64_to_model(model_b64)
            nf  = int(((mdl["T2"] > mdl["T2_UCL"]) | (mdl["Q"] > mdl["Q_UCL"])).sum())
            kpi_cards += [
                _kpi("k PCs",            str(mdl["k"])),
                _kpi("UCL T²",           f"{mdl['T2_UCL']:.3f}"),
                _kpi("UCL Q",            f"{mdl['Q_UCL']:.3f}"),
                _kpi("Flag calibrazione",f"{nf} ({nf/len(mdl['T2'])*100:.1f}%)"),
            ]
        except Exception:
            pass

    if data.get("feature_names"):
        fn = data["feature_names"]
        kpi_cards.insert(0, _kpi("Variabili X", str(len(fn))))

    if kpi_cards:
        children.append(
            dbc.Row([dbc.Col(c, md=2, sm=4, xs=6) for c in kpi_cards],
                    className="mb-3 g-2")
        )

    # ── Monitoring status ─────────────────────────────────────
    if mon_files and model_b64:
        from core.store_utils import b64_to_model, mon_dict_to_mon
        from components.charts import chart_line_multi, COLORS
        try:
            mdl = b64_to_model(model_b64)
            all_t2 = np.concatenate([mon_dict_to_mon(mf)["T2"]      for mf in mon_files])
            all_q  = np.concatenate([mon_dict_to_mon(mf)["Q"]       for mf in mon_files])
            all_t2f= np.concatenate([mon_dict_to_mon(mf)["T2_flag"] for mf in mon_files])
            all_qf = np.concatenate([mon_dict_to_mon(mf)["Q_flag"]  for mf in mon_files])
            boundaries = [0] + list(np.cumsum([mf["n_rows"] for mf in mon_files]))
            names = [mf["name"] for mf in mon_files]
            pct = (all_t2f | all_qf).mean() * 100

            if pct < 5:
                status_badge = html.Span("🟢 Processo in controllo", className="badge badge-ok",
                                         style={"fontSize": "15px", "padding": "8px 16px"})
            elif pct < 15:
                status_badge = html.Span("🟡 Attenzione", className="badge badge-warn",
                                         style={"fontSize": "15px", "padding": "8px 16px"})
            else:
                status_badge = html.Span("🔴 Anomalie rilevate", className="badge badge-alarm",
                                         style={"fontSize": "15px", "padding": "8px 16px"})

            children.append(
                html.Div([
                    html.Div([
                        html.H6("Stato monitoraggio",
                                style={"fontSize": "12px", "fontWeight": 600,
                                       "textTransform": "uppercase", "letterSpacing": "0.05em",
                                       "color": "#94A3B8", "marginBottom": "8px"}),
                        status_badge,
                        html.Span(
                            f"  {int((all_t2f|all_qf).sum())} anomalie su "
                            f"{len(all_t2)} cicli ({pct:.1f}%)",
                            style={"fontSize": "13px", "color": "#64748B", "marginLeft": "10px"},
                        ),
                    ], style={"marginBottom": "16px"}),

                    dbc.Row([
                        dbc.Col(dcc.Graph(
                            figure=chart_line_multi(
                                all_t2, mdl["T2_UCL"], "Hotelling T²",
                                COLORS["primary"], all_t2f, names, boundaries,
                            ),
                            config={"toImageButtonOptions": {"format": "png", "scale": 2},
                                    "displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d"]},
                        ), md=6),
                        dbc.Col(dcc.Graph(
                            figure=chart_line_multi(
                                all_q, mdl["Q_UCL"], "Q (SPE)",
                                COLORS["success"], all_qf, names, boundaries,
                            ),
                            config={"toImageButtonOptions": {"format": "png", "scale": 2},
                                    "displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d"]},
                        ), md=6),
                    ], className="g-2"),
                ], className="card mb-3")
            )
        except Exception as e:
            children.append(html.Div(f"Errore grafici: {e}", className="alert alert-warn"))

    elif model_b64:
        children.append(
            html.Div([
                html.Span("ℹ️  Modello calibrato — carica dati di monitoraggio per vedere "
                          "il controllo in tempo reale."),
                "  ",
                dcc.Link(dbc.Button("Vai al Monitoring →", color="primary", size="sm"),
                         href="/monitoring"),
            ], className="alert alert-info")
        )

    # ── Action next-step ─────────────────────────────────────
    if not ctx.get("context_saved") or not df_X_ok:
        children.append(
            html.Div([
                html.Strong("Passo successivo: "),
                "Configura il processo e carica i dati nella sezione ",
                dcc.Link("Setup →", href="/setup"),
            ], className="alert alert-info mt-2")
        )
    elif not model_b64:
        children.append(
            html.Div([
                html.Strong("Passo successivo: "),
                "Calibra il modello PCA-SPC nella sezione ",
                dcc.Link("Training →", href="/training"),
            ], className="alert alert-info mt-2")
        )
    elif not mon_files:
        children.append(
            html.Div([
                html.Strong("Passo successivo: "),
                "Carica i dati di monitoraggio nella sezione ",
                dcc.Link("Monitoring →", href="/monitoring"),
            ], className="alert alert-info mt-2")
        )

    return children


def _kpi(label, value):
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value"),
    ], className="metric-card")


def _get_workflow(obj):
    o = obj.lower()
    if "diagnostic" in o or "diagnos" in o:
        return "diagnostic"
    if "explorer" in o or "explorat" in o or "esplo" in o:
        return "exploratory"
    return "spc"
