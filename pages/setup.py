import dash
from dash import html, dcc, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import io

dash.register_page(__name__, path="/setup", name="Setup", order=1)


def _step(n, label):
    return html.Div([
        html.Div(str(n), className="step-circle", id=f"step-circle-{n}"),
        html.Span(label, style={"fontSize": "12px", "fontWeight": 500}),
    ], className="step-item", id=f"step-item-{n}")

def _conn():
    return html.Div(className="step-connector")


_OBJECTIVES = [
    "Statistical Process Control — costruisci un modello e monitora la produzione",
    "Diagnostics — analizza dati storici per trovare anomalie e cause radice",
    "Exploratory analysis — comprendi la struttura del processo e le correlazioni",
    "Altro (descrivi sotto)",
]

layout = html.Div([
    html.Div([
        html.P("Setup", className="page-title"),
        html.P("Contesto del processo, caricamento dati e configurazione variabili.",
               className="page-subtitle"),
    ]),

    # ── Step indicator ────────────────────────────────────────
    html.Div([
        _step(1, "Contesto"), _conn(),
        _step(2, "Dati"),     _conn(),
        _step(3, "Variabili"),
    ], className="step-indicator", id="setup-step-indicator"),

    html.Div(id="setup-feedback"),

    # ── Step 1 — Context ──────────────────────────────────────
    html.Div([
        html.Div([
            html.Div("1 — Contesto del processo", className="card-title"),
            html.P(
                "L'AI utilizza questa descrizione per contestualizzare ogni risposta.",
                style={"fontSize": "13px", "color": "#64748B", "marginBottom": "12px"},
            ),
            dbc.Textarea(
                id="ctx-textarea",
                placeholder="Descrivi il processo industriale: tipo di impianto, variabili misurate, "
                            "criticità operative, obiettivi di qualità...",
                style={"height": "140px", "fontSize": "13px"},
                className="mb-3",
            ),
            html.Label("Obiettivo dell'analisi",
                       style={"fontSize": "13px", "fontWeight": 500, "marginBottom": "6px"}),
            dcc.Dropdown(
                id="ctx-objective",
                options=[{"label": o, "value": o} for o in _OBJECTIVES],
                value=_OBJECTIVES[0],
                clearable=False,
                style={"fontSize": "13px"},
            ),
            html.Div(id="ctx-objective-extra", className="mt-2"),
            dbc.Row([
                dbc.Col(dbc.Button("💾 Salva contesto", id="btn-save-ctx",
                                   color="primary", className="mt-3"), width="auto"),
                dbc.Col(dbc.Button("🗑 Cancella", id="btn-clear-ctx",
                                   color="light", className="mt-3"), width="auto"),
            ]),
            html.Div(id="ctx-save-msg", className="mt-2"),
        ], className="card"),
    ]),

    # ── Step 2 — Data ─────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div("2 — Caricamento dati", className="card-title"),
            html.P(
                "Carica un file CSV o Excel. Ogni riga = un'osservazione (ciclo/campione).",
                style={"fontSize": "13px", "color": "#64748B", "marginBottom": "12px"},
            ),
            dbc.Row([
                dbc.Col(
                    dcc.Upload(
                        id="upload-main",
                        children=html.Div([
                            "📂 Trascina qui un file o ",
                            html.A("seleziona dal computer", style={"color": "#2563EB"}),
                        ], style={"fontSize": "13px"}),
                        style={
                            "border": "2px dashed #CBD5E1",
                            "borderRadius": "10px",
                            "padding": "28px 20px",
                            "textAlign": "center",
                            "cursor": "pointer",
                            "background": "#F8FAFC",
                            "transition": "border-color 0.15s",
                        },
                        multiple=False,
                    ),
                    md=8,
                ),
                dbc.Col([
                    html.P("oppure", style={"fontSize": "12px", "color": "#94A3B8",
                                           "textAlign": "center", "margin": "12px 0 8px"}),
                    dbc.Button(
                        "⚗ Carica dataset demo",
                        id="btn-load-demo",
                        color="light",
                        style={"width": "100%", "fontSize": "12px"},
                    ),
                    html.P(
                        "Impianto trattamento acque — 527 cicli, 29 variabili",
                        style={"fontSize": "11px", "color": "#94A3B8",
                               "textAlign": "center", "marginTop": "6px"},
                    ),
                ], md=4),
            ]),
            html.Div(id="upload-status", className="mt-2"),
        ], className="card"),
    ]),

    # ── Dataset preview (shown after upload) ─────────────────
    html.Div(id="dataset-preview"),

    # ── Step 3 — Variable config ──────────────────────────────
    html.Div([
        html.Div([
            html.Div("3 — Configurazione variabili", className="card-title"),
            dbc.Row([
                dbc.Col([
                    html.Label("Variabili Y (output/qualità) — una per riga",
                               style={"fontSize": "13px", "fontWeight": 500}),
                    dbc.Textarea(
                        id="cfg-y-cols",
                        placeholder="RD-DBO-G\nRD-SS-G",
                        style={"height": "90px", "fontSize": "13px"},
                    ),
                ], md=5),
                dbc.Col([
                    html.Label("Colonne da escludere — una per riga",
                               style={"fontSize": "13px", "fontWeight": 500}),
                    dbc.Textarea(
                        id="cfg-excl-cols",
                        placeholder="Data\nShift",
                        style={"height": "90px", "fontSize": "13px"},
                    ),
                ], md=5),
                dbc.Col([
                    html.Label("Livello di confidenza UCL (α)",
                               style={"fontSize": "13px", "fontWeight": 500}),
                    dcc.Slider(
                        id="cfg-alpha",
                        min=0.90, max=0.99, step=0.01, value=0.95,
                        marks={v: f"{v:.2f}" for v in [0.90, 0.95, 0.99]},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], md=2),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(dbc.Button("💾 Salva configurazione", id="btn-save-cfg",
                                   color="primary", className="mt-3"), width="auto"),
            ]),
            html.Div(id="cfg-save-msg", className="mt-2"),
        ], className="card"),
    ]),

    # ── Train / test split ────────────────────────────────────
    html.Div(id="split-section"),
])


# ── Restore stored values on page load ───────────────────────
@callback(
    Output("ctx-textarea",    "value"),
    Output("ctx-objective",   "value"),
    Output("cfg-y-cols",      "value"),
    Output("cfg-excl-cols",   "value"),
    Output("cfg-alpha",       "value"),
    Input("ctx-store",        "data"),
    Input("cfg-store",        "data"),
)
def restore_form(ctx, cfg):
    ctx = ctx or {}
    cfg = cfg or {}
    return (
        ctx.get("process_context", ""),
        ctx.get("analysis_objective", _OBJECTIVES[0]),
        "\n".join(cfg.get("y_cols", [])),
        "\n".join(cfg.get("excl_cols", [])),
        cfg.get("alpha", 0.95),
    )


# ── Show extra textarea for "Altro" objective ─────────────────
@callback(
    Output("ctx-objective-extra", "children"),
    Input("ctx-objective", "value"),
)
def show_extra(val):
    if val and val.startswith("Altro"):
        return dbc.Textarea(
            id="ctx-objective-extra-text",
            placeholder="Descrivi il tuo obiettivo...",
            style={"height": "60px", "fontSize": "13px"},
        )
    return html.Div(id="ctx-objective-extra-text")


# ── Save context ──────────────────────────────────────────────
@callback(
    Output("ctx-store",    "data", allow_duplicate=True),
    Output("ctx-save-msg", "children"),
    Input("btn-save-ctx",  "n_clicks"),
    State("ctx-textarea",  "value"),
    State("ctx-objective", "value"),
    State("ctx-objective-extra-text", "value"),
    State("ctx-store",     "data"),
    prevent_initial_call=True,
)
def save_context(n, text, obj, obj_extra, existing):
    if not text or not text.strip():
        return no_update, html.Span("⚠ Inserisci una descrizione del processo.",
                                    style={"fontSize": "12px", "color": "#D97706"})
    final_obj = (obj_extra or "").strip() if obj and obj.startswith("Altro") else (obj or "")
    store = dict(existing or {})
    store.update({
        "process_context":    text.strip(),
        "analysis_objective": final_obj,
        "context_saved":      True,
    })
    return store, html.Span("✅ Contesto salvato.", style={"fontSize": "12px", "color": "#16A34A"})


# ── Clear context ─────────────────────────────────────────────
@callback(
    Output("ctx-store",    "data", allow_duplicate=True),
    Output("ctx-textarea", "value", allow_duplicate=True),
    Output("ctx-save-msg", "children", allow_duplicate=True),
    Input("btn-clear-ctx", "n_clicks"),
    prevent_initial_call=True,
)
def clear_context(n):
    return (
        {"process_context": "", "analysis_objective": "", "context_saved": False},
        "",
        "",
    )


# ── File upload or demo ───────────────────────────────────────
@callback(
    Output("data-store",    "data", allow_duplicate=True),
    Output("upload-status", "children"),
    Input("upload-main",    "contents"),
    Input("btn-load-demo",  "n_clicks"),
    State("upload-main",    "filename"),
    State("cfg-store",      "data"),
    State("data-store",     "data"),
    prevent_initial_call=True,
)
def handle_upload(contents, demo_clicks, filename, cfg, existing_data):
    from dash import ctx as dash_ctx
    from core.store_utils import df_to_json, parse_upload
    import os

    triggered = dash_ctx.triggered_id
    cfg = cfg or {}
    y_cols   = cfg.get("y_cols", [])
    excl_cols = cfg.get("excl_cols", [])

    try:
        if triggered == "btn-load-demo":
            demo_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "demo", "water_treatment_demo.csv"
            )
            df_raw = pd.read_csv(demo_path)
            fname = "water_treatment_demo.csv"
        elif contents:
            df_raw = parse_upload(contents, filename)
            fname = filename
        else:
            return no_update, no_update

        df_raw.columns = df_raw.columns.astype(str)

        # Apply exclusions
        drop_c = [c for c in excl_cols if c in df_raw.columns]
        if drop_c:
            df_raw.drop(columns=drop_c, inplace=True)

        y_valid = [c for c in y_cols if c in df_raw.columns]
        df_num  = df_raw.select_dtypes(include=[np.number]).copy()
        const_c = df_num.columns[df_num.std() == 0].tolist()
        if const_c:
            df_num.drop(columns=const_c, inplace=True)

        x_cols = [c for c in df_num.columns if c not in y_valid]
        df_X   = df_num[x_cols].copy()
        df_Y   = df_num[y_valid].copy() if y_valid else pd.DataFrame()
        miss   = int(df_X.isnull().sum().sum())

        store = dict(existing_data or {})
        store.update({
            "df_X_json":    df_to_json(df_X),
            "df_Y_json":    df_to_json(df_Y) if not df_Y.empty else "",
            "feature_names": x_cols,
            "y_names":       y_valid,
            "n_rows":        len(df_X),
            "n_vars":        len(x_cols),
        })

        msg = html.Div([
            html.Span(
                f"✅ Caricato: {fname} — {len(df_X)} cicli, {len(x_cols)} variabili X",
                style={"fontSize": "13px", "color": "#16A34A"},
            ),
            html.Span(
                f"  ⚠ {miss} valori mancanti (imputati con la media)" if miss else "",
                style={"fontSize": "12px", "color": "#D97706", "marginLeft": "8px"},
            ),
        ])
        return store, msg

    except Exception as e:
        return no_update, html.Span(f"❌ Errore: {e}",
                                    style={"fontSize": "13px", "color": "#DC2626"})


# ── Dataset preview ───────────────────────────────────────────
@callback(
    Output("dataset-preview", "children"),
    Input("data-store", "data"),
)
def show_preview(data):
    if not data or not data.get("df_X_json"):
        return html.Div()
    from core.store_utils import json_to_df
    try:
        df_X = json_to_df(data["df_X_json"])
        fn   = data.get("feature_names", [])
        yn   = data.get("y_names", [])

        # Stats
        df_Xf = df_X.fillna(df_X.mean())
        desc  = df_Xf.describe().T.round(3)
        desc["cv%"] = (desc["std"] / desc["mean"].abs() * 100).round(1)
        desc_records = desc.reset_index().rename(columns={"index": "Variable"}).to_dict("records")
        desc_cols = [{"name": c, "id": c} for c in desc.reset_index().rename(
            columns={"index": "Variable"}).columns]

        from dash import dash_table
        tbl = dash_table.DataTable(
            data=desc_records[:30],
            columns=desc_cols,
            style_table={"overflowX": "auto", "fontSize": "12px"},
            style_cell={"padding": "6px 10px", "fontFamily": "Inter, sans-serif"},
            style_header={"fontWeight": 600, "fontSize": "11px",
                          "textTransform": "uppercase", "color": "#64748B"},
            page_size=15,
        )

        vars_x = html.Div([
            html.Strong(f"Variabili X ({len(fn)}): "),
            html.Span(", ".join(fn[:20]) + ("..." if len(fn) > 20 else ""),
                      style={"fontSize": "13px", "color": "#64748B"}),
        ])
        vars_y = html.Div([
            html.Strong(f"Variabili Y ({len(yn)}): "),
            html.Span(", ".join(yn) if yn else "—",
                      style={"fontSize": "13px", "color": "#64748B"}),
        ], style={"marginTop": "4px"}) if yn else html.Div()

        return html.Div([
            html.Div([
                html.Div("Anteprima dataset", className="card-title"),
                vars_x, vars_y,
                html.Hr(className="section-sep"),
                html.Div("Statistiche descrittive",
                         style={"fontSize": "12px", "fontWeight": 600, "color": "#64748B",
                                "marginBottom": "8px"}),
                tbl,
            ], className="card"),
        ])
    except Exception as e:
        return html.Div(f"Errore anteprima: {e}", className="alert alert-warn")


# ── Save config ───────────────────────────────────────────────
@callback(
    Output("cfg-store",    "data", allow_duplicate=True),
    Output("cfg-save-msg", "children"),
    Input("btn-save-cfg",  "n_clicks"),
    State("cfg-y-cols",    "value"),
    State("cfg-excl-cols", "value"),
    State("cfg-alpha",     "value"),
    prevent_initial_call=True,
)
def save_config(n, y_raw, excl_raw, alpha):
    y_cols   = [c.strip() for c in (y_raw   or "").split("\n") if c.strip()]
    excl_cols = [c.strip() for c in (excl_raw or "").split("\n") if c.strip()]
    return (
        {"y_cols": y_cols, "excl_cols": excl_cols, "alpha": alpha or 0.95},
        html.Span("✅ Configurazione salvata.", style={"fontSize": "12px", "color": "#16A34A"}),
    )


# ── Train/test split section ──────────────────────────────────
@callback(
    Output("split-section", "children"),
    Input("data-store", "data"),
    Input("ctx-store",  "data"),
)
def show_split(data, ctx):
    if not data or not data.get("df_X_json"):
        return html.Div()
    ctx = ctx or {}
    obj = ctx.get("analysis_objective", "").lower()
    is_spc = "diagnostic" not in obj and "explorat" not in obj

    if not is_spc:
        return html.Div([
            html.Div([
                html.Div("Workflow Diagnostics / Exploratory",
                         className="card-title"),
                html.Div(
                    "Per questo obiettivo il dataset completo è usato per la calibrazione. "
                    "Nessun split train/test necessario.",
                    className="alert alert-info",
                ),
            ], className="card"),
        ])

    from core.store_utils import json_to_df
    try:
        df_X  = json_to_df(data["df_X_json"])
        n_tot = len(df_X)
    except Exception:
        return html.Div()

    return html.Div([
        html.Div([
            html.Div("Train / Test Split", className="card-title"),
            html.P(
                "Dividi il dataset in un set di calibrazione (Phase I) "
                "e un set di test (Phase II).",
                style={"fontSize": "13px", "color": "#64748B", "marginBottom": "12px"},
            ),
            dbc.Row([
                dbc.Col([
                    html.Label("Metodo", style={"fontSize": "13px", "fontWeight": 500}),
                    dbc.RadioItems(
                        id="split-method",
                        options=[{"label": "Temporale (primei N%)", "value": "Temporal"},
                                 {"label": "Casuale",               "value": "Random"}],
                        value="Temporal",
                        inline=True,
                        style={"fontSize": "13px"},
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("Dimensione train (%)",
                               style={"fontSize": "13px", "fontWeight": 500}),
                    dcc.Slider(
                        id="split-ratio",
                        min=5, max=95, step=5, value=70,
                        marks={v: f"{v}%" for v in [20, 50, 70, 90]},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], md=5),
                dbc.Col([
                    html.Div(id="split-info", style={"fontSize": "13px",
                                                      "color": "#64748B",
                                                      "paddingTop": "28px"}),
                ], md=3),
            ], className="g-3"),
            html.Div(id="split-preview-chart"),
            dbc.Button("▶ Applica split", id="btn-apply-split",
                       color="primary", size="sm", className="mt-3"),
            html.Div(id="split-msg", className="mt-2"),
        ], className="card"),
    ])


@callback(
    Output("split-info",          "children"),
    Output("split-preview-chart", "children"),
    Input("split-ratio",  "value"),
    Input("split-method", "value"),
    State("data-store",   "data"),
)
def update_split_preview(ratio, method, data):
    if not data or not data.get("df_X_json"):
        return "", html.Div()
    from core.store_utils import json_to_df
    from components.charts import COLORS
    import plotly.graph_objects as go

    df_X  = json_to_df(data["df_X_json"])
    n_tot = len(df_X)
    n_tr  = int(n_tot * (ratio / 100))
    n_te  = n_tot - n_tr
    info  = f"Train: {n_tr} ({ratio}%)  |  Test: {n_te} ({100-ratio}%)"

    fn = data.get("feature_names", [])
    if method == "Temporal" and fn:
        fc   = fn[0]
        vals = df_X[fc].fillna(df_X[fc].mean()).tolist()
        fig  = go.Figure()
        fig.add_vrect(x0=0, x1=n_tr, fillcolor=COLORS["primary"], opacity=0.08,
                      line_width=0, annotation_text=f"Train ({ratio}%)",
                      annotation_position="top left", annotation_font_size=9)
        fig.add_vrect(x0=n_tr, x1=n_tot, fillcolor=COLORS["danger"], opacity=0.08,
                      line_width=0, annotation_text=f"Test ({100-ratio}%)",
                      annotation_position="top right", annotation_font_size=9)
        fig.add_vline(x=n_tr, line_dash="dash", line_color=COLORS["danger"], line_width=1.5)
        fig.add_scatter(x=list(range(n_tot)), y=vals, mode="lines",
                        line=dict(color=COLORS["neutral"], width=1), showlegend=False)
        fig.update_layout(
            height=180, margin=dict(l=10, r=10, t=20, b=30),
            plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["white"],
            xaxis=dict(gridcolor=COLORS["border"]), yaxis=dict(gridcolor=COLORS["border"],
                       title=fc[:20]),
            font=dict(family="Inter"),
        )
        chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, className="mt-2")
    else:
        chart = html.Div()

    return info, chart


@callback(
    Output("split-store", "data", allow_duplicate=True),
    Output("split-msg",   "children"),
    Input("btn-apply-split", "n_clicks"),
    State("split-ratio",  "value"),
    State("split-method", "value"),
    State("data-store",   "data"),
    prevent_initial_call=True,
)
def apply_split(n, ratio, method, data):
    if not data or not data.get("df_X_json"):
        return no_update, html.Span("⚠ Carica prima i dati.", style={"fontSize": "12px"})
    from core.store_utils import json_to_df, df_to_json

    df_X = json_to_df(data["df_X_json"]).fillna(
        json_to_df(data["df_X_json"]).mean()
    )
    n_tot = len(df_X)
    n_tr  = int(n_tot * (ratio / 100))

    if method == "Temporal":
        df_tr = df_X.iloc[:n_tr].reset_index(drop=True)
        df_te = df_X.iloc[n_tr:].reset_index(drop=True)
    else:
        rng  = np.random.default_rng(42)
        idx  = rng.permutation(n_tot)
        df_tr = df_X.iloc[idx[:n_tr]].reset_index(drop=True)
        df_te = df_X.iloc[idx[n_tr:]].reset_index(drop=True)

    return (
        {"df_train_json": df_to_json(df_tr),
         "df_test_builtin_json": df_to_json(df_te),
         "split_applied": True},
        html.Span(f"✅ Split applicato — Train: {len(df_tr)} | Test: {len(df_te)}",
                  style={"fontSize": "12px", "color": "#16A34A"}),
    )
