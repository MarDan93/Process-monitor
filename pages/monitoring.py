import dash
from dash import html, dcc, Input, Output, State, callback, no_update, ALL
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

dash.register_page(__name__, path="/monitoring", name="Monitoring", order=4)

CHART_PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2", "#BE185D"]

layout = html.Div([
    html.Div([
        html.P("Monitoring — Phase II", className="page-title"),
        html.P("Applica il modello PCA-SPC a nuovi dati per rilevare anomalie.",
               className="page-subtitle"),
    ]),
    html.Div(id="mon-gate"),
    html.Div(id="mon-content"),
    dcc.Store(id="selected-cycle-store", data=None),
])


@callback(
    Output("mon-gate",    "children"),
    Output("mon-content", "children"),
    Input("model-store",  "data"),
    Input("mon-store",    "data"),
    Input("ctx-store",    "data"),
    Input("url-store",    "data"),
)
def render_monitoring(model_data, mon, ctx, pathname):
    if pathname and pathname != "/monitoring":
        return no_update, no_update
    if not model_data or not model_data.get("model_b64"):
        return html.Div([
            html.Div("⬆ Calibra prima il modello nella sezione Training.",
                     className="alert alert-info"),
            dcc.Link(dbc.Button("Vai al Training", color="primary", size="sm"), href="/training"),
        ]), html.Div()

    from core.store_utils import b64_to_model, mon_dict_to_mon
    mdl      = b64_to_model(model_data["model_b64"])
    fn       = mdl["feature_names"]
    obj      = (ctx or {}).get("analysis_objective", "").lower()
    is_diag  = "diagnostic" in obj or "explorat" in obj
    mon_files = (mon or {}).get("mon_files", [])

    upload_section = html.Div([
        html.Div([
            html.Div(
                "Confronto dataset aggiuntivi" if is_diag
                else "Carica dati di monitoring",
                className="card-title",
            ),
            html.P(
                "Carica uno o più file CSV/Excel da sottoporre al modello calibrato. "
                "Il numero di colonne deve corrispondere alle variabili X del modello.",
                style={"fontSize": "13px", "color": "#64748B", "marginBottom": "12px"},
            ) if not is_diag else html.Div(
                "Workflow Diagnostics — le anomalie Phase I sono già disponibili nella sezione "
                "Root Cause. Puoi caricare file aggiuntivi per confronto.",
                className="alert alert-info", style={"fontSize": "13px"},
            ),
            dcc.Upload(
                id="upload-monitoring",
                children=html.Div([
                    "📂 Carica file CSV/Excel  ",
                    html.Small("(più file supportati)", style={"color": "#94A3B8"}),
                ], style={"fontSize": "13px"}),
                style={
                    "border": "2px dashed #CBD5E1", "borderRadius": "10px",
                    "padding": "20px", "textAlign": "center", "cursor": "pointer",
                    "background": "#F8FAFC",
                },
                multiple=True,
            ),
            html.Div(id="mon-upload-msg", className="mt-2"),
        ], className="card"),
    ])

    # Loaded files list
    files_list = html.Div()
    if mon_files:
        file_items = []
        for fi, mf in enumerate(mon_files):
            color = CHART_PALETTE[fi % len(CHART_PALETTE)]
            file_items.append(
                dbc.Row([
                    dbc.Col(
                        html.Span(
                            f"● {mf['name']} — {mf['n_rows']} cicli",
                            className="file-tag",
                            style={"background": f"{color}18", "color": color,
                                   "border": f"1px solid {color}44"},
                        ),
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    dbc.Col(
                        dbc.Button("✕", id={"type": "rm-mon-file", "index": fi},
                                   color="light", size="sm", n_clicks=0),
                        width="auto",
                    ),
                ], className="mb-1 g-1")
            )
        files_list = html.Div([
            html.Div("File caricati", style={"fontSize": "12px", "fontWeight": 600,
                                              "color": "#64748B", "marginBottom": "8px"}),
            *file_items,
        ], className="card mb-2")

    # Charts + analysis section
    charts_section = html.Div()
    if mon_files:
        try:
            all_t2  = np.concatenate([mon_dict_to_mon(mf)["T2"]      for mf in mon_files])
            all_q   = np.concatenate([mon_dict_to_mon(mf)["Q"]       for mf in mon_files])
            all_t2f = np.concatenate([mon_dict_to_mon(mf)["T2_flag"] for mf in mon_files])
            all_qf  = np.concatenate([mon_dict_to_mon(mf)["Q_flag"]  for mf in mon_files])
            boundaries = [0] + list(np.cumsum([mf["n_rows"] for mf in mon_files]))
            names = [mf["name"] for mf in mon_files]

            n_test  = len(all_t2)
            n_t2    = int(all_t2f.sum())
            n_q     = int(all_qf.sum())
            n_any   = int((all_t2f | all_qf).sum())
            pct     = n_any / n_test * 100

            stato = "🟢 STABILE" if pct < 5 else ("🟡 WARNING" if pct < 15 else "🔴 ANOMALIE")
            kind  = "ok" if pct < 5 else ("warn" if pct < 15 else "alarm")

            from components.charts import chart_line_multi, chart_contribution, COLORS

            # Per-file summary table
            rows_pf = []
            for mf in mon_files:
                m  = mon_dict_to_mon(mf)
                nf = len(m["T2"])
                nt2 = int(m["T2_flag"].sum())
                nq  = int(m["Q_flag"].sum())
                na  = int((m["T2_flag"] | m["Q_flag"]).sum())
                rows_pf.append({
                    "File":    mf["name"],
                    "Cicli":   nf,
                    "T² anom": f"{nt2} ({nt2/nf*100:.1f}%)",
                    "Q anom":  f"{nq}  ({nq/nf*100:.1f}%)",
                    "Totali":  f"{na}  ({na/nf*100:.1f}%)",
                    "Stato":   "🟢" if na/nf < 0.05 else ("🟡" if na/nf < 0.15 else "🔴"),
                })
            from dash import dash_table

            # Top anomaly table
            flagged_idx = np.where(all_t2f | all_qf)[0]
            df_anom = pd.DataFrame({
                "Ciclo":       flagged_idx,
                "T²":          all_t2[flagged_idx].round(3),
                "T²/UCL":      (all_t2[flagged_idx] / mdl["T2_UCL"]).round(2),
                "Q":           all_q[flagged_idx].round(3),
                "Q/UCL":       (all_q[flagged_idx] / mdl["Q_UCL"]).round(2),
                "Severity×UCL": np.maximum(
                    all_t2[flagged_idx] / mdl["T2_UCL"],
                    all_q[flagged_idx]  / mdl["Q_UCL"],
                ).round(2),
            }).sort_values("Severity×UCL", ascending=False).reset_index(drop=True)

            charts_section = html.Div([
                # KPI row
                dbc.Row([
                    dbc.Col(html.Div([html.Div("Cicli totali",   className="metric-label"),
                                     html.Div(str(n_test),       className="metric-value",
                                              style={"fontSize": "20px"})], className="metric-card"), md=3),
                    dbc.Col(html.Div([html.Div("T² anomalie",    className="metric-label"),
                                     html.Div(f"{n_t2} ({n_t2/n_test*100:.1f}%)",
                                              className="metric-value", style={"fontSize": "20px"})],
                                    className="metric-card"), md=3),
                    dbc.Col(html.Div([html.Div("Q anomalie",     className="metric-label"),
                                     html.Div(f"{n_q} ({n_q/n_test*100:.1f}%)",
                                              className="metric-value", style={"fontSize": "20px"})],
                                    className="metric-card"), md=3),
                    dbc.Col(html.Div([html.Div("Stato",          className="metric-label"),
                                     html.Div(stato,             className="metric-value",
                                              style={"fontSize": "16px"})], className="metric-card"), md=3),
                ], className="g-2 mb-3"),

                # Per-file table
                html.Div([
                    dash_table.DataTable(
                        data=rows_pf,
                        columns=[{"name": c, "id": c} for c in rows_pf[0].keys()],
                        style_table={"overflowX": "auto"},
                        style_cell={"fontSize": "12px", "padding": "7px 12px",
                                    "fontFamily": "Inter, sans-serif"},
                        style_header={"fontWeight": 600, "fontSize": "11px",
                                      "textTransform": "uppercase", "color": "#64748B"},
                    ),
                ], className="card mb-3"),

                # Control charts
                html.Div([
                    html.Div("Control Charts", className="card-title"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(
                            figure=chart_line_multi(
                                all_t2, mdl["T2_UCL"], "Phase II — Hotelling T²",
                                COLORS["primary"], all_t2f, names, boundaries,
                            ),
                            config={"toImageButtonOptions": {"format": "png", "scale": 2},
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["lasso2d"]},
                        ), md=6),
                        dbc.Col(dcc.Graph(
                            figure=chart_line_multi(
                                all_q, mdl["Q_UCL"], "Phase II — Q (SPE)",
                                COLORS["success"], all_qf, names, boundaries,
                            ),
                            config={"toImageButtonOptions": {"format": "png", "scale": 2},
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["lasso2d"]},
                        ), md=6),
                    ], className="g-2"),
                ], className="card mb-3"),

                # Anomaly table + contribution analysis
                html.Div([
                    html.Div("Analisi anomalie", className="card-title"),
                    html.Div(
                        f"{len(flagged_idx)} cicli fuori controllo — seleziona una riga per analizzare.",
                        style={"fontSize": "13px", "color": "#64748B", "marginBottom": "8px"},
                    ),
                    html.Small(
                        "Severity×UCL: max(T²/UCL, Q/UCL). "
                        "1.0 = appena fuori limite | 2.0 = doppio limite | 5.0+ = anomalia grave.",
                        style={"color": "#94A3B8"},
                    ),
                    dash_table.DataTable(
                        id="p2-anomaly-table",
                        data=df_anom.head(50).to_dict("records"),
                        columns=[{"name": c, "id": c} for c in df_anom.columns],
                        style_table={"overflowX": "auto", "marginTop": "8px"},
                        style_cell={"fontSize": "12px", "padding": "7px 12px",
                                    "fontFamily": "Inter, sans-serif"},
                        style_header={"fontWeight": 600, "fontSize": "11px",
                                      "textTransform": "uppercase", "color": "#64748B"},
                        row_selectable="single",
                        selected_rows=[0],
                        style_data_conditional=[{
                            "if": {"state": "selected"},
                            "backgroundColor": "#EFF6FF",
                            "border": "1px solid #2563EB",
                        }],
                        page_size=10,
                        sort_action="native",
                    ),
                    html.Div(id="p2-contrib-area", className="mt-3"),
                    html.Div(id="p2-log-form", className="mt-3"),
                ], className="card"),
            ])
        except Exception as e:
            charts_section = html.Div(f"Errore: {e}", className="alert alert-alarm")

    return html.Div(), html.Div([upload_section, files_list, charts_section])


# ── File upload ───────────────────────────────────────────────
@callback(
    Output("mon-store",      "data", allow_duplicate=True),
    Output("mon-upload-msg", "children"),
    Input("upload-monitoring", "contents"),
    State("upload-monitoring", "filename"),
    State("model-store",       "data"),
    State("mon-store",         "data"),
    prevent_initial_call=True,
)
def handle_mon_upload(contents_list, filenames, model_data, mon):
    if not contents_list or not model_data or not model_data.get("model_b64"):
        return no_update, no_update
    from core.store_utils import b64_to_model, parse_upload, mon_file_to_dict
    from core.pca_spc import monitor_new

    mdl = b64_to_model(model_data["model_b64"])
    fn  = mdl["feature_names"]
    mon = dict(mon or {"mon_files": [], "anomaly_log": []})
    loaded_names = {mf["name"] for mf in mon["mon_files"]}
    msgs = []

    for contents, fname in zip(contents_list, filenames):
        if fname in loaded_names:
            msgs.append(html.Span(f"ℹ {fname} già caricato  ",
                                  style={"fontSize": "12px", "color": "#64748B"}))
            continue
        try:
            df_raw = parse_upload(contents, fname)
            df_raw.columns = df_raw.columns.astype(str)
            miss_c = [c for c in fn if c not in df_raw.columns]
            if miss_c:
                msgs.append(html.Span(
                    f"❌ {fname}: colonne mancanti {miss_c[:3]}  ",
                    style={"fontSize": "12px", "color": "#DC2626"},
                ))
                continue
            df_new = df_raw[fn].fillna(df_raw[fn].mean())
            mon_res = monitor_new(mdl, df_new.values)
            mon["mon_files"].append(mon_file_to_dict(fname, len(df_new), mon_res))
            msgs.append(html.Span(
                f"✅ {fname} ({len(df_new)} cicli)  ",
                style={"fontSize": "12px", "color": "#16A34A"},
            ))
        except Exception as e:
            msgs.append(html.Span(f"❌ {fname}: {e}  ",
                                  style={"fontSize": "12px", "color": "#DC2626"}))

    return mon, html.Div(msgs)


# ── Remove monitoring file ────────────────────────────────────
@callback(
    Output("mon-store", "data", allow_duplicate=True),
    Input({"type": "rm-mon-file", "index": ALL}, "n_clicks"),
    State("mon-store", "data"),
    prevent_initial_call=True,
)
def remove_mon_file(n_clicks_list, mon):
    from dash import ctx as dash_ctx
    triggered = dash_ctx.triggered_id
    if not triggered or not any(n_clicks_list):
        return no_update
    idx = triggered["index"]
    mon = dict(mon or {"mon_files": [], "anomaly_log": []})
    if 0 <= idx < len(mon["mon_files"]):
        mon["mon_files"] = [mf for i, mf in enumerate(mon["mon_files"]) if i != idx]
    return mon


# ── Contribution analysis for selected anomaly ───────────────
@callback(
    Output("p2-contrib-area", "children"),
    Output("p2-log-form",     "children"),
    Input("p2-anomaly-table", "selected_rows"),
    State("p2-anomaly-table", "data"),
    State("model-store",      "data"),
    State("mon-store",        "data"),
    prevent_initial_call=True,
)
def show_p2_contrib(selected_rows, table_data, model_data, mon):
    if not selected_rows or not model_data or not model_data.get("model_b64"):
        return html.Div(), html.Div()
    from core.store_utils import b64_to_model, mon_dict_to_mon
    from components.charts import chart_contribution, COLORS
    from dash import dash_table

    row   = table_data[selected_rows[0]]
    obs   = int(row["Ciclo"])
    mdl   = b64_to_model(model_data["model_b64"])
    fn    = mdl["feature_names"]
    lam   = mdl["eigenvalues"]
    P     = mdl["loadings"]

    # Reconstruct arrays
    mon_files = (mon or {}).get("mon_files", [])
    all_Xns = np.concatenate([mon_dict_to_mon(mf)["Xn_s"] for mf in mon_files])
    all_En  = np.concatenate([mon_dict_to_mon(mf)["En"]   for mf in mon_files])
    all_Tn  = np.concatenate([mon_dict_to_mon(mf)["Tn"]   for mf in mon_files])
    all_t2  = np.concatenate([mon_dict_to_mon(mf)["T2"]   for mf in mon_files])
    all_q   = np.concatenate([mon_dict_to_mon(mf)["Q"]    for mf in mon_files])

    c_t2 = all_Xns[obs] * (P @ (all_Tn[obs] / lam))
    c_q  = all_En[obs]

    t2_ucl_v = mdl["T2contrib_UCL"]; t2_lcl_v = mdl["T2contrib_LCL"]
    q_ucl_v  = mdl["Qcontrib_UCL"];  q_lcl_v  = mdl["Qcontrib_LCL"]

    fig_t2, exc_t2 = chart_contribution(c_t2, t2_ucl_v, t2_lcl_v, fn, f"T² — Ciclo {obs}")
    fig_q,  exc_q  = chart_contribution(c_q,  q_ucl_v,  q_lcl_v,  fn, f"Q — Ciclo {obs}")

    t2_ratio = float(all_t2[obs] / mdl["T2_UCL"])
    q_ratio  = float(all_q[obs]  / mdl["Q_UCL"])
    severity = max(t2_ratio, q_ratio)
    kind     = "alarm" if severity >= 1.5 else "warn"

    # Find which file this cycle belongs to
    boundaries = [0] + list(np.cumsum([mf["n_rows"] for mf in mon_files]))
    file_of_obs = "sconosciuto"
    for fi, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if s <= obs < e:
            file_of_obs = mon_files[fi]["name"]
            break

    status_msg = html.Div([
        html.Strong(f"Ciclo {obs} — {'🔴 ANOMALIA' if severity >= 1.5 else '⚠ WARNING'}"),
        html.Span(f"   T²={all_t2[obs]:.3f} ({t2_ratio:.2f}×UCL)   "
                  f"Q={all_q[obs]:.3f} ({q_ratio:.2f}×UCL)   "
                  f"File: {file_of_obs}",
                  style={"fontSize": "12px", "color": "#475569"}),
    ], className=f"alert alert-{kind}", style={"fontSize": "13px"})

    def _exceed_tbl(exceedances):
        if not exceedances:
            return html.Small("Nessuna variabile fuori limite.", style={"color": "#64748B"})
        df_ex = __import__("pandas").DataFrame(
            exceedances, columns=["#", "Variabile", "Valore", "LCL", "UCL"]
        )
        return dash_table.DataTable(
            data=df_ex.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df_ex.columns],
            style_cell={"fontSize": "11px", "padding": "5px 8px",
                        "fontFamily": "Inter, sans-serif"},
            style_header={"fontWeight": 600},
        )

    contrib_section = html.Div([
        status_msg,
        dbc.Row([
            dbc.Col([
                html.Div("Contributi T²", style={"fontSize": "12px", "fontWeight": 600,
                                                  "color": "#64748B", "marginBottom": "6px"}),
                dcc.Graph(figure=fig_t2,
                          config={"toImageButtonOptions": {"format": "png", "scale": 2},
                                  "displayModeBar": True}),
                html.Div("Variabili fuori limite:", style={"fontSize": "12px", "marginTop": "6px"}),
                _exceed_tbl(exc_t2),
            ], md=6),
            dbc.Col([
                html.Div("Contributi Q", style={"fontSize": "12px", "fontWeight": 600,
                                                 "color": "#64748B", "marginBottom": "6px"}),
                dcc.Graph(figure=fig_q,
                          config={"toImageButtonOptions": {"format": "png", "scale": 2},
                                  "displayModeBar": True}),
                html.Div("Variabili fuori limite:", style={"fontSize": "12px", "marginTop": "6px"}),
                _exceed_tbl(exc_q),
            ], md=6),
        ], className="g-2"),
    ])

    # Intervention log form
    log_form = html.Div([
        html.Hr(className="section-sep"),
        html.Div("📝 Registra intervento", style={"fontSize": "13px", "fontWeight": 600,
                                                    "marginBottom": "8px"}),
        dbc.Row([
            dbc.Col(dbc.Textarea(
                id="log-action-text",
                placeholder="Descrivi l'azione correttiva presa...",
                style={"height": "70px", "fontSize": "13px"},
            ), md=9),
            dbc.Col(dbc.Button("💾 Salva", id="btn-save-log", color="primary",
                               size="sm", style={"marginTop": "4px"}), md=3),
        ], className="g-2"),
        dcc.Store(id="log-cycle-store", data=obs),
        html.Div(id="log-save-msg", className="mt-1"),

        # Show existing log
        html.Div(id="log-display"),
    ])

    return contrib_section, log_form


# ── Save intervention log ─────────────────────────────────────
@callback(
    Output("mon-store",    "data", allow_duplicate=True),
    Output("log-save-msg", "children"),
    Input("btn-save-log",  "n_clicks"),
    State("log-action-text", "value"),
    State("log-cycle-store", "data"),
    State("mon-store",     "data"),
    State("model-store",   "data"),
    prevent_initial_call=True,
)
def save_log(n, action, cycle, mon, model_data):
    if not action or not action.strip():
        return no_update, html.Span("⚠ Inserisci una descrizione.", style={"fontSize": "12px"})
    if not model_data or not model_data.get("model_b64"):
        return no_update, no_update
    from core.store_utils import b64_to_model, mon_dict_to_mon

    mdl  = b64_to_model(model_data["model_b64"])
    mon  = dict(mon or {"mon_files": [], "anomaly_log": []})
    mon_files = mon.get("mon_files", [])

    obs  = int(cycle or 0)
    all_t2 = np.concatenate([mon_dict_to_mon(mf)["T2"] for mf in mon_files]) if mon_files else np.array([])
    all_q  = np.concatenate([mon_dict_to_mon(mf)["Q"]  for mf in mon_files]) if mon_files else np.array([])

    t2_val = float(all_t2[obs]) if obs < len(all_t2) else 0.0
    q_val  = float(all_q[obs])  if obs < len(all_q)  else 0.0
    sev    = max(t2_val / mdl["T2_UCL"], q_val / mdl["Q_UCL"]) if obs < len(all_t2) else 0.0

    mon["anomaly_log"].append({
        "Ciclo":     obs,
        "T²":        round(t2_val, 3),
        "Q":         round(q_val, 3),
        "Severity":  f"{sev:.2f}×UCL",
        "Azione":    action.strip(),
    })

    return mon, html.Span("✅ Salvato.", style={"fontSize": "12px", "color": "#16A34A"})


# ── Show intervention log ─────────────────────────────────────
@callback(
    Output("log-display", "children"),
    Input("mon-store",    "data"),
)
def display_log(mon):
    alog = (mon or {}).get("anomaly_log", [])
    if not alog:
        return html.Div()
    from dash import dash_table
    df = pd.DataFrame(alog)
    return html.Div([
        html.Div("📋 Log interventi", style={"fontSize": "12px", "fontWeight": 600,
                                              "color": "#64748B", "marginTop": "12px",
                                              "marginBottom": "6px"}),
        dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df.columns],
            style_cell={"fontSize": "12px", "padding": "6px 10px",
                        "fontFamily": "Inter, sans-serif"},
            style_header={"fontWeight": 600},
        ),
    ])
