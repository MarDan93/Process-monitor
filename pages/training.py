import dash
from dash import html, dcc, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

dash.register_page(__name__, path="/training", name="Training", order=2)

layout = html.Div([
    html.Div([
        html.P("Model Training", className="page-title"),
        html.P("Selezione componenti principali e calibrazione Phase I.", className="page-subtitle"),
    ]),
    html.Div(id="training-gate"),
    html.Div(id="training-content"),
])


@callback(
    Output("training-gate",    "children"),
    Output("training-content", "children"),
    Input("data-store",  "data"),
    Input("model-store", "data"),
    Input("url-store",   "data"),
)
def render_training(data, model_data, pathname):
    if pathname and pathname != "/training":
        return no_update, no_update
    if not data or not data.get("df_X_json"):
        return html.Div([
            html.Div(
                "⬆ Carica prima i dati nella sezione Setup.",
                className="alert alert-info",
            ),
            dcc.Link(dbc.Button("Vai al Setup", color="primary", size="sm"),
                     href="/setup"),
        ]), html.Div()

    return html.Div(), _training_layout(model_data)


def _training_layout(model_data):
    model_b64 = (model_data or {}).get("model_b64")

    return html.Div([
        # ── Step 1 — PC Selection ─────────────────────────────
        dbc.Accordion([
            dbc.AccordionItem(
                title="1 — Selezione componenti principali",
                item_id="acc-pc",
                children=[
                    dbc.RadioItems(
                        id="pc-criterion",
                        options=[
                            {"label": "Scree plot",          "value": "scree"},
                            {"label": "Varianza cumulata",   "value": "cumvar"},
                            {"label": "RMSECV",              "value": "rmsecv"},
                        ],
                        value="scree",
                        inline=True,
                        style={"fontSize": "13px"},
                        className="mb-3",
                    ),
                    dcc.Loading(html.Div(id="pc-chart-area"), type="circle"),
                    html.Hr(className="section-sep"),
                    dbc.Row([
                        dbc.Col(html.Div(id="pc-metrics"), md=6),
                        dbc.Col([
                            html.Label("Numero di PC per il modello",
                                       style={"fontSize": "13px", "fontWeight": 500}),
                            dbc.Input(id="pc-k-input", type="number",
                                      min=2, max=20, step=1, value=4,
                                      style={"width": "120px", "fontSize": "13px"}),
                        ], md=3),
                        dbc.Col([
                            dbc.Button("▶ Calcola RMSECV", id="btn-rmsecv",
                                       color="light", size="sm",
                                       className="mt-4"),
                        ], md=3),
                    ], className="g-3"),
                    html.Div(id="pc-k-selected-msg", className="mt-2"),
                ],
            ),
        ], start_collapsed=False, always_open=True),

        html.Div(style={"height": "12px"}),

        # ── Step 2 — Calibration ──────────────────────────────
        dbc.Accordion([
            dbc.AccordionItem(
                title="2 — Calibrazione Phase I",
                item_id="acc-cal",
                children=[
                    # Cleaning option
                    dbc.Accordion([
                        dbc.AccordionItem(
                            title="🧹 Pulizia iterativa dati (opzionale)",
                            item_id="acc-clean",
                            children=[
                                html.P(
                                    "Rimuove iterativamente i cicli anomali prima della calibrazione "
                                    "per ottenere un modello di riferimento più pulito.",
                                    style={"fontSize": "13px", "color": "#64748B"},
                                ),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Switch(id="toggle-clean", label="Abilita pulizia",
                                                   value=False),
                                    ], md=4),
                                    dbc.Col([
                                        html.Label("Confidenza pulizia (α)",
                                                   style={"fontSize": "12px"}),
                                        dcc.Slider(
                                            id="clean-alpha", min=0.95, max=0.999,
                                            step=0.001, value=0.99,
                                            marks={0.95: "0.95", 0.99: "0.99", 0.999: "0.999"},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ], md=5),
                                    dbc.Col([
                                        dbc.Button("▶ Esegui pulizia", id="btn-clean",
                                                   color="light", size="sm", className="mt-4"),
                                    ], md=3),
                                ], className="g-2"),
                                dcc.Loading(html.Div(id="clean-result"), type="dot"),
                            ],
                        ),
                    ], start_collapsed=True),

                    html.Div(style={"height": "12px"}),

                    dbc.Button(
                        "🔧 Calibra modello PCA-SPC",
                        id="btn-fit",
                        color="primary",
                        style={"width": "100%", "fontWeight": 600},
                    ),
                    dcc.Loading(
                        html.Div(id="cal-result"),
                        type="circle",
                        className="mt-3",
                    ),
                ],
            ),
        ], start_collapsed=False, always_open=True),

        # ── Hidden stores for clean mask + rmsecv ────────────
        dcc.Store(id="clean-mask-store",  data=None),
        dcc.Store(id="rmsecv-store",      data=None),
    ])


# ── PC chart rendering ────────────────────────────────────────
@callback(
    Output("pc-chart-area",        "children"),
    Output("pc-metrics",           "children"),
    Output("pc-k-selected-msg",    "children"),
    Input("pc-criterion",          "value"),
    Input("rmsecv-store",          "data"),
    Input("pc-k-input",            "value"),
    State("data-store",            "data"),
    State("split-store",           "data"),
    State("cfg-store",             "data"),
)
def render_pc_chart(criterion, rmsecv_data, k_input, data, split, cfg):
    if not data or not data.get("df_X_json"):
        return html.Div(), html.Div(), html.Div()
    from core.store_utils import json_to_df
    from components.charts import COLORS
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import plotly.graph_objects as go

    df_X = json_to_df(data["df_X_json"])
    if split and split.get("df_train_json"):
        df_for_pca = json_to_df(split["df_train_json"])
    else:
        df_for_pca = df_X

    X_raw   = df_for_pca.fillna(df_for_pca.mean()).values
    sc_tmp  = StandardScaler()
    Xs_tmp  = sc_tmp.fit_transform(X_raw)
    n_obs, n_vars = Xs_tmp.shape
    max_k   = min(20, n_vars - 1, n_obs - 2)

    pca_tmp = PCA(n_components=max_k, svd_solver="full", random_state=42)
    pca_tmp.fit(Xs_tmp)
    eigs    = pca_tmp.explained_variance_
    evr_all = pca_tmp.explained_variance_ratio_ * 100
    cum     = np.cumsum(evr_all)
    ks      = list(range(1, max_k + 1))

    k_kaiser = max(2, min(int(np.sum(eigs > 1)), max_k))
    k_90     = int(np.argmax(cum >= 90.0)) + 1
    k_95     = int(np.argmax(cum >= 95.0)) + 1

    # Build chart based on criterion
    if criterion == "scree":
        fig = go.Figure()
        fig.add_scatter(x=ks, y=evr_all.tolist(), mode="lines+markers",
                        line=dict(color=COLORS["primary"], width=2),
                        marker=dict(size=6), name="Varianza per PC")
        fig.add_hline(y=float(100 / n_vars), line_dash="dot", line_color=COLORS["muted"],
                      annotation_text="Soglia Kaiser", annotation_position="right",
                      annotation_font_size=10)
        fig.add_vline(x=k_kaiser, line_dash="dash", line_color=COLORS["danger"],
                      annotation_text=f"k={k_kaiser}", annotation_position="top right")
        fig.update_layout(xaxis_title="PC", yaxis_title="Varianza spiegata (%)",
                          height=280, margin=dict(l=10, r=80, t=20, b=30),
                          plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["white"],
                          showlegend=False, font=dict(family="Inter"),
                          xaxis=dict(gridcolor=COLORS["border"]),
                          yaxis=dict(gridcolor=COLORS["border"]))
        chart_div = dcc.Graph(figure=fig, config={"displayModeBar": False})
        hint = html.Div(f"💡 Kaiser suggerisce k = {k_kaiser}",
                        className="alert alert-info mt-1",
                        style={"fontSize": "13px"})

    elif criterion == "cumvar":
        fig = go.Figure()
        fig.add_scatter(x=ks, y=cum.tolist(), mode="lines+markers",
                        line=dict(color=COLORS["primary"], width=2),
                        marker=dict(size=6), name="Varianza cumulata")
        for pct, c, lbl in [(90, COLORS["warning"], "90%"), (95, COLORS["danger"], "95%")]:
            fig.add_hline(y=pct, line_dash="dash", line_color=c,
                          annotation_text=lbl, annotation_position="right")
        fig.update_layout(xaxis_title="N° PC", yaxis_title="Varianza cumulata (%)",
                          yaxis=dict(range=[0, 105], gridcolor=COLORS["border"]),
                          xaxis=dict(gridcolor=COLORS["border"]),
                          height=280, margin=dict(l=10, r=80, t=20, b=30),
                          plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["white"],
                          showlegend=False, font=dict(family="Inter"))
        chart_div = dcc.Graph(figure=fig, config={"displayModeBar": False})
        hint = html.Div(
            f"💡 90% varianza → k={k_90}  |  95% varianza → k={k_95}",
            className="alert alert-info mt-1", style={"fontSize": "13px"},
        )

    else:  # rmsecv
        if rmsecv_data:
            bk   = rmsecv_data["best_k"]
            rcv  = rmsecv_data["rmsecv"]
            fig  = go.Figure()
            fig.add_scatter(
                x=ks[:len(rcv)], y=rcv, mode="lines+markers",
                line=dict(color=COLORS["success"], width=2),
                marker=dict(
                    size=[10 if i + 1 == bk else 5 for i in range(len(rcv))],
                    color=[COLORS["danger"] if i + 1 == bk else COLORS["success"]
                           for i in range(len(rcv))],
                ),
            )
            fig.add_vline(x=bk, line_dash="dash", line_color=COLORS["danger"],
                          annotation_text=f"min k={bk}", annotation_position="top right")
            fig.update_layout(xaxis_title="N° PC", yaxis_title="RMSECV",
                              height=280, margin=dict(l=10, r=80, t=20, b=30),
                              plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["white"],
                              showlegend=False, font=dict(family="Inter"),
                              xaxis=dict(gridcolor=COLORS["border"]),
                              yaxis=dict(gridcolor=COLORS["border"]))
            chart_div = dcc.Graph(figure=fig, config={"displayModeBar": False})
            hint = html.Div(f"💡 RMSECV minimo → k = {bk}",
                            className="alert alert-info mt-1", style={"fontSize": "13px"})
        else:
            chart_div = html.Div(
                "Clicca 'Calcola RMSECV' per eseguire la cross-validation.",
                className="alert alert-info", style={"fontSize": "13px"},
            )
            hint = html.Div()

    # Metrics row
    rmsecv_k = rmsecv_data["best_k"] if rmsecv_data else None
    metrics = dbc.Row([
        dbc.Col(html.Div([
            html.Div("Kaiser", className="metric-label"),
            html.Div(f"k = {k_kaiser}", className="metric-value",
                     style={"fontSize": "18px"}),
        ], className="metric-card"), md=4),
        dbc.Col(html.Div([
            html.Div("90% var", className="metric-label"),
            html.Div(f"k = {k_90}", className="metric-value",
                     style={"fontSize": "18px"}),
        ], className="metric-card"), md=4),
        dbc.Col(html.Div([
            html.Div("RMSECV", className="metric-label"),
            html.Div(f"k = {rmsecv_k}" if rmsecv_k else "—",
                     className="metric-value", style={"fontSize": "18px"}),
        ], className="metric-card"), md=4),
    ], className="g-2")

    # k confirmation
    k_val = k_input or k_kaiser
    k_val = max(2, min(int(k_val), max_k))
    k_cum = float(cum[k_val - 1]) if k_val - 1 < len(cum) else 0.0
    k_msg = html.Div(
        f"✅ k = {k_val} PC selezionati — {k_cum:.1f}% di varianza spiegata",
        className="alert alert-ok", style={"fontSize": "13px"},
    )

    return [chart_div, hint], metrics, k_msg


# ── RMSECV computation ────────────────────────────────────────
@callback(
    Output("rmsecv-store", "data"),
    Input("btn-rmsecv",    "n_clicks"),
    State("data-store",    "data"),
    State("split-store",   "data"),
    prevent_initial_call=True,
)
def compute_rmsecv(n, data, split):
    if not data or not data.get("df_X_json"):
        return no_update
    from core.store_utils import json_to_df
    from core.pca_spc import compute_rmsecv as _rmsecv
    from sklearn.preprocessing import StandardScaler

    df_X = json_to_df(data["df_X_json"])
    if split and split.get("df_train_json"):
        df_for = json_to_df(split["df_train_json"])
    else:
        df_for = df_X

    X_raw = df_for.fillna(df_for.mean()).values
    Xs    = StandardScaler().fit_transform(X_raw)
    n_obs, n_vars = Xs.shape
    max_k = min(20, n_vars - 1, n_obs - 2)
    bk, _, rcv = _rmsecv(Xs, max_k)
    return {"best_k": int(bk), "rmsecv": rcv.tolist()}


# ── Iterative cleaning ────────────────────────────────────────
@callback(
    Output("clean-mask-store", "data"),
    Output("clean-result",     "children"),
    Input("btn-clean",         "n_clicks"),
    State("toggle-clean",      "value"),
    State("clean-alpha",       "value"),
    State("pc-k-input",        "value"),
    State("data-store",        "data"),
    State("split-store",       "data"),
    prevent_initial_call=True,
)
def run_cleaning(n, enabled, alpha_clean, k_input, data, split):
    if not enabled or not data or not data.get("df_X_json"):
        return None, html.Div()
    from core.store_utils import json_to_df
    from core.pca_spc import iterative_cleaning
    from sklearn.preprocessing import StandardScaler
    from dash import dash_table

    df_X = json_to_df(data["df_X_json"])
    if split and split.get("df_train_json"):
        df_for = json_to_df(split["df_train_json"])
    else:
        df_for = df_X

    X_raw   = df_for.fillna(df_for.mean()).values
    n_obs, n_vars = X_raw.shape
    k_clean = max(2, min(int(k_input or 4), n_vars - 1, n_obs - 2))

    mask, log_df = iterative_cleaning(X_raw, k_clean, alpha_clean or 0.99)
    n_rem  = int((~mask).sum())
    pct    = n_rem / len(X_raw) * 100

    tbl = dash_table.DataTable(
        data=log_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in log_df.columns],
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "12px", "padding": "6px 10px",
                    "fontFamily": "Inter, sans-serif"},
        style_header={"fontWeight": 600},
    )

    kind = "warn" if pct > 20 else "ok"
    msg  = html.Div([
        html.Div(
            f"{'⚠' if pct>20 else '✅'} Rimossi {n_rem} cicli ({pct:.1f}%) — "
            f"Rimanenti: {mask.sum()}",
            className=f"alert alert-{kind}", style={"fontSize": "13px"},
        ),
        html.Div(tbl, className="mt-2"),
    ])

    return mask.tolist(), msg


# ── Calibrate model ───────────────────────────────────────────
@callback(
    Output("model-store",  "data", allow_duplicate=True),
    Output("mon-store",    "data", allow_duplicate=True),
    Output("cal-result",   "children"),
    Input("btn-fit",       "n_clicks"),
    State("pc-k-input",    "value"),
    State("cfg-store",     "data"),
    State("data-store",    "data"),
    State("split-store",   "data"),
    State("clean-mask-store", "data"),
    State("model-store",   "data"),
    State("mon-store",     "data"),
    prevent_initial_call=True,
)
def calibrate(n, k_input, cfg, data, split, clean_mask, existing_model, existing_mon):
    if not data or not data.get("df_X_json"):
        return no_update, no_update, html.Div("⚠ Carica prima i dati.", className="alert alert-warn")
    from core.store_utils import json_to_df, model_to_b64, mon_file_to_dict
    from core.pca_spc import fit_pca_spc, monitor_new

    df_X = json_to_df(data["df_X_json"])
    fn   = data.get("feature_names", list(df_X.columns))
    alpha = (cfg or {}).get("alpha", 0.95)
    k_use = max(2, min(int(k_input or 4), len(fn) - 1, len(df_X) - 2))

    if split and split.get("df_train_json"):
        df_cal = json_to_df(split["df_train_json"])
    else:
        df_cal = df_X

    df_cal_filled = df_cal[fn].fillna(df_cal[fn].mean()) if set(fn).issubset(df_cal.columns) \
                    else df_cal.fillna(df_cal.mean())
    X_raw = df_cal_filled.values

    if clean_mask:
        mask  = np.array(clean_mask, dtype=bool)
        if len(mask) == len(X_raw):
            X_raw = X_raw[mask]

    try:
        mdl = fit_pca_spc(X_raw, k_use, alpha)
        mdl["feature_names"] = fn
    except Exception as e:
        return no_update, no_update, html.Div(f"❌ Errore calibrazione: {e}",
                                              className="alert alert-alarm")

    new_model_store = dict(existing_model or {})
    new_model_store.update({
        "model_b64": model_to_b64(mdl),
        "k_chosen":  k_use,
    })

    # Auto-load test set if split was applied
    new_mon = dict(existing_mon or {"mon_files": [], "anomaly_log": []})
    if split and split.get("df_test_builtin_json"):
        df_te  = json_to_df(split["df_test_builtin_json"])
        df_te_f = df_te[fn].fillna(df_te[fn].mean()) if set(fn).issubset(df_te.columns) \
                  else df_te.fillna(df_te.mean())
        mon_res = monitor_new(mdl, df_te_f.values)
        new_mon["mon_files"] = [mon_file_to_dict("Built-in test set", len(df_te), mon_res)]

    # ── Result display ────────────────────────────────────────
    nf   = int(((mdl["T2"] > mdl["T2_UCL"]) | (mdl["Q"] > mdl["Q_UCL"])).sum())
    pct  = nf / len(mdl["T2"]) * 100
    evr_cum = float(np.cumsum(mdl["evr"])[k_use - 1])

    from components.charts import chart_line_multi, COLORS
    fT2 = mdl["T2"] > mdl["T2_UCL"]
    fQ  = mdl["Q"]  > mdl["Q_UCL"]

    result = html.Div([
        dbc.Row([
            dbc.Col(html.Div([html.Div("k PCs",        className="metric-label"),
                              html.Div(str(k_use),     className="metric-value",
                                       style={"fontSize": "20px"})], className="metric-card"), md=3),
            dbc.Col(html.Div([html.Div("UCL T²",       className="metric-label"),
                              html.Div(f"{mdl['T2_UCL']:.3f}", className="metric-value",
                                       style={"fontSize": "20px"})], className="metric-card"), md=3),
            dbc.Col(html.Div([html.Div("UCL Q",        className="metric-label"),
                              html.Div(f"{mdl['Q_UCL']:.3f}",  className="metric-value",
                                       style={"fontSize": "20px"})], className="metric-card"), md=3),
            dbc.Col(html.Div([html.Div("Flag Phase I", className="metric-label"),
                              html.Div(f"{nf} ({pct:.1f}%)", className="metric-value",
                                       style={"fontSize": "20px"})], className="metric-card"), md=3),
        ], className="g-2 mb-3"),

        html.Div(
            f"✅ Modello calibrato — {k_use} PC, {evr_cum:.1f}% varianza spiegata, α={alpha}",
            className="alert alert-ok", style={"fontSize": "13px"},
        ),

        dbc.Row([
            dbc.Col(dcc.Graph(
                figure=chart_line_multi(mdl["T2"], mdl["T2_UCL"],
                                        "Phase I — Hotelling T²",
                                        COLORS["primary"], fT2),
                config={"toImageButtonOptions": {"format": "png", "scale": 2},
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            ), md=6),
            dbc.Col(dcc.Graph(
                figure=chart_line_multi(mdl["Q"], mdl["Q_UCL"],
                                        "Phase I — Q (SPE)",
                                        COLORS["success"], fQ),
                config={"toImageButtonOptions": {"format": "png", "scale": 2},
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            ), md=6),
        ], className="g-2"),

        # Phase I anomaly table
        html.Div(id="p1-anomaly-section",
                 children=_phase1_anomalies(mdl, fn)),
    ])

    return new_model_store, new_mon, result


def _phase1_anomalies(mdl, fn):
    from dash import dash_table
    from components.charts import COLORS

    flagged = np.where((mdl["T2"] > mdl["T2_UCL"]) | (mdl["Q"] > mdl["Q_UCL"]))[0]
    if len(flagged) == 0:
        return html.Div("✅ Nessuna anomalia nel set di calibrazione — modello pulito.",
                        className="alert alert-ok mt-2", style={"fontSize": "13px"})

    df_anom = pd.DataFrame({
        "Ciclo":      flagged,
        "T²":         mdl["T2"][flagged].round(3),
        "T²/UCL":     (mdl["T2"][flagged] / mdl["T2_UCL"]).round(2),
        "Q":          mdl["Q"][flagged].round(3),
        "Q/UCL":      (mdl["Q"][flagged] / mdl["Q_UCL"]).round(2),
        "Severity×UCL": np.maximum(
            mdl["T2"][flagged] / mdl["T2_UCL"],
            mdl["Q"][flagged]  / mdl["Q_UCL"],
        ).round(2),
    }).sort_values("Severity×UCL", ascending=False).head(20).reset_index(drop=True)

    tbl = dash_table.DataTable(
        id="p1-anomaly-table",
        data=df_anom.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_anom.columns],
        style_table={"overflowX": "auto"},
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
    )
    return html.Div([
        html.Hr(className="section-sep"),
        html.Div(f"{len(flagged)} cicli anomali in Phase I — clicca una riga per analizzare.",
                 style={"fontSize": "13px", "color": "#64748B", "marginBottom": "8px"}),
        tbl,
        html.Div(id="p1-contrib-section", className="mt-3"),
    ])
