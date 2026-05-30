import dash
from dash import html, dcc, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

dash.register_page(__name__, path="/analysis", name="Analysis", order=5)

layout = html.Div([
    html.Div([
        html.P("Root Cause Analysis", className="page-title"),
        html.P("Analisi delle anomalie, variabili ricorrenti e validazione Y.",
               className="page-subtitle"),
    ]),
    html.Div(id="analysis-content"),
])


@callback(
    Output("analysis-content", "children"),
    Input("model-store", "data"),
    Input("mon-store",   "data"),
    Input("ctx-store",   "data"),
    Input("data-store",  "data"),
    Input("split-store", "data"),
    Input("url-store",   "data"),
)
def render_analysis(model_data, mon, ctx, data, split, pathname):
    if pathname and pathname != "/analysis":
        return no_update
    if not model_data or not model_data.get("model_b64"):
        return html.Div([
            html.Div("⬆ Calibra prima il modello nella sezione Training.",
                     className="alert alert-info"),
            dcc.Link(dbc.Button("Vai al Training", color="primary", size="sm"), href="/training"),
        ])

    from core.store_utils import b64_to_model, mon_dict_to_mon, json_to_df
    mdl   = b64_to_model(model_data["model_b64"])
    fn    = mdl["feature_names"]
    obj   = (ctx or {}).get("analysis_objective", "").lower()
    is_diag = "diagnostic" in obj or "explorat" in obj
    mon_files = (mon or {}).get("mon_files", [])

    # Determine data source
    if is_diag:
        all_t2   = mdl["T2"]
        all_q    = mdl["Q"]
        all_t2f  = mdl["T2"] > mdl["T2_UCL"]
        all_qf   = mdl["Q"]  > mdl["Q_UCL"]
        all_Xns  = mdl["X_scaled"]
        all_En   = mdl["E"]
        all_Tn   = mdl["scores"]
        source   = "Phase I (dataset completo)"
    elif mon_files:
        all_t2   = np.concatenate([mon_dict_to_mon(mf)["T2"]      for mf in mon_files])
        all_q    = np.concatenate([mon_dict_to_mon(mf)["Q"]       for mf in mon_files])
        all_t2f  = np.concatenate([mon_dict_to_mon(mf)["T2_flag"] for mf in mon_files])
        all_qf   = np.concatenate([mon_dict_to_mon(mf)["Q_flag"]  for mf in mon_files])
        all_Xns  = np.concatenate([mon_dict_to_mon(mf)["Xn_s"]    for mf in mon_files])
        all_En   = np.concatenate([mon_dict_to_mon(mf)["En"]       for mf in mon_files])
        all_Tn   = np.concatenate([mon_dict_to_mon(mf)["Tn"]       for mf in mon_files])
        source   = "Phase II monitoring"
    else:
        return html.Div([
            html.Div(
                "⬆ Carica dati di monitoring nella sezione Monitoring, "
                "oppure usa un obiettivo Diagnostics per analizzare i dati Phase I.",
                className="alert alert-info",
            ),
            dcc.Link(dbc.Button("Vai al Monitoring", color="primary", size="sm"),
                     href="/monitoring"),
        ])

    n_test  = len(all_t2)
    n_t2    = int(all_t2f.sum())
    n_q     = int(all_qf.sum())
    n_any   = int((all_t2f | all_qf).sum())
    pct     = n_any / n_test * 100
    stato   = "🟢 STABILE" if pct < 5 else ("🟡 WARNING" if pct < 15 else "🔴 ANOMALIE")
    kind    = "ok" if pct < 5 else ("warn" if pct < 15 else "alarm")

    flagged_idx = np.where(all_t2f | all_qf)[0]

    # ── KPI row ───────────────────────────────────────────────
    kpi_row = dbc.Row([
        dbc.Col(html.Div([html.Div("k PCs",        className="metric-label"),
                          html.Div(str(mdl["k"]),  className="metric-value",
                                   style={"fontSize": "20px"})], className="metric-card"), md=2),
        dbc.Col(html.Div([html.Div("UCL T²",       className="metric-label"),
                          html.Div(f"{mdl['T2_UCL']:.3f}", className="metric-value",
                                   style={"fontSize": "20px"})], className="metric-card"), md=2),
        dbc.Col(html.Div([html.Div("UCL Q",        className="metric-label"),
                          html.Div(f"{mdl['Q_UCL']:.3f}", className="metric-value",
                                   style={"fontSize": "20px"})], className="metric-card"), md=2),
        dbc.Col(html.Div([html.Div("Cicli totali", className="metric-label"),
                          html.Div(str(n_test),    className="metric-value",
                                   style={"fontSize": "20px"})], className="metric-card"), md=2),
        dbc.Col(html.Div([html.Div("Anomalie",     className="metric-label"),
                          html.Div(f"{n_any} ({pct:.1f}%)", className="metric-value",
                                   style={"fontSize": "20px"})], className="metric-card"), md=2),
        dbc.Col(html.Div([html.Div("Stato",        className="metric-label"),
                          html.Div(stato,           className="metric-value",
                                   style={"fontSize": "15px"})], className="metric-card"), md=2),
    ], className="g-2 mb-3")

    if len(flagged_idx) == 0:
        return html.Div([
            kpi_row,
            html.Div("✅ Nessuna anomalia rilevata — processo in controllo.",
                     className="alert alert-ok", style={"fontSize": "14px"}),
        ])

    # ── Top anomalies ─────────────────────────────────────────
    from dash import dash_table
    df_top = pd.DataFrame({
        "Ciclo":         flagged_idx,
        "T²":            all_t2[flagged_idx].round(3),
        "T²/UCL":        (all_t2[flagged_idx] / mdl["T2_UCL"]).round(2),
        "Q":             all_q[flagged_idx].round(3),
        "Q/UCL":         (all_q[flagged_idx] / mdl["Q_UCL"]).round(2),
        "Severity×UCL":  np.maximum(
            all_t2[flagged_idx] / mdl["T2_UCL"],
            all_q[flagged_idx]  / mdl["Q_UCL"],
        ).round(2),
    }).sort_values("Severity×UCL", ascending=False).head(10).reset_index(drop=True)

    # ── Variable frequency ────────────────────────────────────
    P         = mdl["loadings"]
    lam       = mdl["eigenvalues"]
    t2_ucl_v  = mdl["T2contrib_UCL"];  t2_lcl_v = mdl["T2contrib_LCL"]
    q_ucl_v   = mdl["Qcontrib_UCL"];   q_lcl_v  = mdl["Qcontrib_LCL"]
    t2_cnt    = np.zeros(len(fn))
    q_cnt     = np.zeros(len(fn))
    for obs in flagged_idx:
        c_t2 = all_Xns[obs] * (P @ (all_Tn[obs] / lam))
        c_q  = all_En[obs]
        for i in range(len(fn)):
            if c_t2[i] > t2_ucl_v[i] or c_t2[i] < t2_lcl_v[i]:
                t2_cnt[i] += 1
            if c_q[i] > q_ucl_v[i] or c_q[i] < q_lcl_v[i]:
                q_cnt[i] += 1

    df_vars = pd.DataFrame({
        "#":        range(1, len(fn) + 1),
        "Variabile": fn,
        "T² exceed": t2_cnt.astype(int),
        "Q exceed":  q_cnt.astype(int),
        "Totale":    (t2_cnt + q_cnt).astype(int),
    }).sort_values("Totale", ascending=False)

    df_vt = df_vars[df_vars["Totale"] > 0].head(10)
    from components.charts import chart_variable_frequency

    top_t2v = df_vars.nlargest(5, "T² exceed")["Variabile"].tolist()
    top_qv  = df_vars.nlargest(5, "Q exceed")["Variabile"].tolist()
    top_ov  = df_vars.nlargest(5, "Totale")["Variabile"].tolist()

    # ── Y Validation ──────────────────────────────────────────
    y_section = html.Div()
    y_names   = (data or {}).get("y_names", [])
    if y_names and data.get("df_Y_json"):
        try:
            df_Y = json_to_df(data["df_Y_json"])
            y_section = _y_validation(
                df_Y, y_names, all_t2f | all_qf, is_diag, data, split, n_test
            )
        except Exception:
            pass

    return html.Div([
        kpi_row,

        # Top anomalies card
        html.Div([
            html.Div(f"Top anomalie — {source}", className="card-title"),
            dash_table.DataTable(
                data=df_top.to_dict("records"),
                columns=[{"name": c, "id": c} for c in df_top.columns],
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": "12px", "padding": "7px 12px",
                            "fontFamily": "Inter, sans-serif"},
                style_header={"fontWeight": 600, "fontSize": "11px",
                              "textTransform": "uppercase", "color": "#64748B"},
                sort_action="native",
            ),
        ], className="card mb-3"),

        # Variable frequency card
        html.Div([
            html.Div("Variabili ricorrenti nelle anomalie", className="card-title"),
            dbc.Row([
                dbc.Col(dcc.Graph(
                    figure=chart_variable_frequency(df_vt),
                    config={"toImageButtonOptions": {"format": "png", "scale": 2},
                            "displayModeBar": True},
                ), md=8),
                dbc.Col(
                    dash_table.DataTable(
                        data=df_vt.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in df_vt.columns],
                        style_cell={"fontSize": "11px", "padding": "5px 8px",
                                    "fontFamily": "Inter, sans-serif"},
                        style_header={"fontWeight": 600, "fontSize": "10px",
                                      "color": "#64748B"},
                    ),
                    md=4,
                ),
            ], className="g-2"),
        ], className="card mb-3"),

        # Y Validation
        y_section,
    ])


def _y_validation(df_Y, y_names, pca_flags, is_diag, data, split, n_test):
    from core.store_utils import json_to_df
    from dash import dash_table

    items = []
    for y_col in y_names:
        if y_col not in df_Y.columns:
            continue

        # Align Y with the analysis observations
        if is_diag:
            df_Y_aligned = df_Y.reset_index(drop=True)
        else:
            df_X_full = json_to_df(data["df_X_json"]) if data.get("df_X_json") else None
            if split and split.get("df_test_builtin_json") and df_X_full is not None:
                test_size  = len(json_to_df(split["df_test_builtin_json"]))
                train_size = len(df_X_full) - test_size
                df_Y_aligned = df_Y.iloc[train_size:].reset_index(drop=True)
            else:
                continue

        if len(df_Y_aligned) != n_test:
            continue

        y_vals   = df_Y_aligned[y_col].values
        unique_v = np.unique(y_vals[~np.isnan(y_vals)])
        is_bin   = len(unique_v) <= 2

        if is_bin:
            y_bin   = (y_vals > 0).astype(int)
            pca_bin = pca_flags.astype(int)
            tp = int(((pca_bin == 1) & (y_bin == 1)).sum())
            fp = int(((pca_bin == 1) & (y_bin == 0)).sum())
            fn_ = int(((pca_bin == 0) & (y_bin == 1)).sum())
            tn = int(((pca_bin == 0) & (y_bin == 0)).sum())
            n_y1    = int(y_bin.sum())
            n_flagged = int(pca_bin.sum())
            prec  = tp / (tp + fp)  if (tp + fp)  > 0 else 0.0
            rec   = tp / (tp + fn_) if (tp + fn_) > 0 else 0.0
            f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            if rec > 0.7 and prec > 0.5:
                interp_kind = "ok"
                interp_text = (f"✅ Forte accordo — il modello rileva {rec:.0%} degli "
                               f"eventi {y_col} con precisione {prec:.0%}.")
            elif rec > 0.4:
                interp_kind = "warn"
                interp_text = (f"⚠ Accordo parziale — il modello rileva {rec:.0%} degli eventi. "
                               f"{fn_} eventi non rilevati.")
            else:
                interp_kind = "warn"
                interp_text = (f"⚠ Basso overlap — solo {rec:.0%} degli eventi {y_col} rilevati. "
                               f"Le variabili X potrebbero non spiegare completamente questo outcome.")

            df_cm = pd.DataFrame({
                "": ["PCA flag=1", "PCA flag=0"],
                "Y=1": [tp, fn_], "Y=0": [fp, tn],
            })
            items.append(html.Div([
                html.Div(f"Variabile Y: {y_col}", className="card-title"),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Div("Y=1 eventi",  className="metric-label"),
                        html.Div(f"{n_y1} ({n_y1/n_test*100:.1f}%)",
                                 className="metric-value", style={"fontSize": "18px"}),
                    ], className="metric-card"), md=3),
                    dbc.Col(html.Div([
                        html.Div("PCA flagged", className="metric-label"),
                        html.Div(f"{n_flagged} ({n_flagged/n_test*100:.1f}%)",
                                 className="metric-value", style={"fontSize": "18px"}),
                    ], className="metric-card"), md=3),
                    dbc.Col(html.Div([
                        html.Div("Precision",   className="metric-label"),
                        html.Div(f"{prec:.1%}", className="metric-value",
                                 style={"fontSize": "18px"}),
                    ], className="metric-card"), md=2),
                    dbc.Col(html.Div([
                        html.Div("Recall",      className="metric-label"),
                        html.Div(f"{rec:.1%}",  className="metric-value",
                                 style={"fontSize": "18px"}),
                    ], className="metric-card"), md=2),
                    dbc.Col(html.Div([
                        html.Div("F1 Score",    className="metric-label"),
                        html.Div(f"{f1:.2f}",   className="metric-value",
                                 style={"fontSize": "18px"}),
                    ], className="metric-card"), md=2),
                ], className="g-2 mb-2"),
                dbc.Row([
                    dbc.Col(
                        dash_table.DataTable(
                            data=df_cm.to_dict("records"),
                            columns=[{"name": c, "id": c} for c in df_cm.columns],
                            style_cell={"fontSize": "12px", "padding": "6px 12px",
                                        "fontFamily": "Inter, sans-serif"},
                            style_header={"fontWeight": 600},
                        ),
                        md=5,
                    ),
                    dbc.Col(
                        html.Div(interp_text, className=f"alert alert-{interp_kind}",
                                 style={"fontSize": "13px"}),
                        md=7,
                    ),
                ], className="g-2"),
            ], className="card mb-3"))

        else:
            # Continuous Y
            y_anom   = y_vals[pca_flags]
            y_normal = y_vals[~pca_flags]
            if len(y_anom) > 0 and len(y_normal) > 0:
                diff_pct = abs(y_anom.mean() - y_normal.mean()) / (abs(y_normal.mean()) + 1e-9) * 100
                kind_v   = "ok" if diff_pct > 10 else "warn"
                items.append(html.Div([
                    html.Div(f"Variabile Y (continua): {y_col}", className="card-title"),
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.Div("Media — cicli normali",  className="metric-label"),
                            html.Div(f"{y_normal.mean():.3f}", className="metric-value",
                                     style={"fontSize": "20px"}),
                        ], className="metric-card"), md=4),
                        dbc.Col(html.Div([
                            html.Div("Media — cicli anomali",  className="metric-label"),
                            html.Div(f"{y_anom.mean():.3f}",   className="metric-value",
                                     style={"fontSize": "20px"}),
                        ], className="metric-card"), md=4),
                        dbc.Col(html.Div([
                            html.Div("Differenza",             className="metric-label"),
                            html.Div(f"{diff_pct:.1f}%",       className="metric-value",
                                     style={"fontSize": "20px"}),
                        ], className="metric-card"), md=4),
                    ], className="g-2 mb-2"),
                    html.Div(
                        (f"✅ I cicli anomali mostrano una differenza significativa "
                         f"in {y_col} ({diff_pct:.1f}%). Il modello cattura variazioni "
                         f"che impattano questa metrica di qualità.")
                        if diff_pct > 10 else
                        (f"⚠ I cicli anomali mostrano poca differenza in {y_col} ({diff_pct:.1f}%). "
                         f"Le variazioni rilevate potrebbero non impattare direttamente "
                         f"questa variabile di qualità."),
                        className=f"alert alert-{kind_v}",
                        style={"fontSize": "13px"},
                    ),
                ], className="card mb-3"))

    if not items:
        return html.Div()

    return html.Div([
        html.Hr(className="section-sep"),
        html.Div([
            html.P("Validazione Y", className="page-title",
                   style={"fontSize": "16px", "marginBottom": "4px"}),
            html.P(
                "Confronta i flag PCA-SPC con le variabili di output/qualità definite.",
                style={"fontSize": "13px", "color": "#64748B", "marginBottom": "16px"},
            ),
            *items,
        ]),
    ])
