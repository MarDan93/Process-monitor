import dash
from dash import html, dcc, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
from itertools import combinations
import numpy as np

dash.register_page(__name__, path="/structure", name="Structure", order=3)

layout = html.Div([
    html.Div([
        html.P("Loadings & Score Plots", className="page-title"),
        html.P("Struttura del modello: correlazioni tra variabili e posizione delle osservazioni.",
               className="page-subtitle"),
    ]),
    html.Div(id="structure-content"),
])


@callback(
    Output("structure-content", "children"),
    Input("model-store", "data"),
    Input("cfg-store",   "data"),
    Input("url-store",   "data"),
)
def render_structure(model_data, cfg, pathname):
    if pathname and pathname != "/structure":
        return no_update
    if not model_data or not model_data.get("model_b64"):
        return html.Div([
            html.Div("⬆ Calibra prima il modello nella sezione Training.",
                     className="alert alert-info"),
            dcc.Link(dbc.Button("Vai al Training", color="primary", size="sm"), href="/training"),
        ])

    from core.store_utils import b64_to_model
    mdl   = b64_to_model(model_data["model_b64"])
    fn    = mdl["feature_names"]
    P     = mdl["loadings"]
    k_m   = mdl["k"]
    evr_m = mdl["evr"]
    lam_m = mdl["eigenvalues"]
    T_m   = mdl["scores"]
    alpha = (cfg or {}).get("alpha", 0.95)

    all_pairs   = list(combinations(range(k_m), 2))
    pair_options = [
        {"label": f"PC{a+1} ({evr_m[a]:.1f}%)  vs  PC{b+1} ({evr_m[b]:.1f}%)",
         "value": f"{a},{b}"}
        for a, b in all_pairs
    ]

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Tipo di grafico", style={"fontSize": "13px", "fontWeight": 500}),
                dbc.RadioItems(
                    id="ls-tipo",
                    options=[
                        {"label": "Loading plot (struttura variabili)", "value": "loading"},
                        {"label": "Score plot (osservazioni)",          "value": "score"},
                    ],
                    value="loading",
                    inline=False,
                    style={"fontSize": "13px"},
                ),
            ], md=4),
            dbc.Col([
                html.Label("Coppia di PC", style={"fontSize": "13px", "fontWeight": 500}),
                dcc.Dropdown(
                    id="ls-pair",
                    options=pair_options,
                    value=pair_options[0]["value"] if pair_options else None,
                    clearable=False,
                    style={"fontSize": "13px"},
                ),
            ], md=5),
            dbc.Col([
                dbc.Button("📋 Indice variabili", id="btn-var-index",
                           color="light", size="sm", className="mt-4"),
            ], md=3),
        ], className="g-3 mb-3"),

        dbc.Collapse(
            html.Div(id="var-index-table"),
            id="collapse-var-index",
            is_open=False,
        ),

        dcc.Loading(html.Div(id="ls-chart-area"), type="circle"),
        html.Div(id="ls-table-area", className="mt-3"),
    ])


@callback(
    Output("collapse-var-index", "is_open"),
    Input("btn-var-index",       "n_clicks"),
    State("collapse-var-index",  "is_open"),
    prevent_initial_call=True,
)
def toggle_var_index(n, is_open):
    return not is_open


@callback(
    Output("var-index-table", "children"),
    Input("model-store", "data"),
)
def render_var_index(model_data):
    if not model_data or not model_data.get("model_b64"):
        return html.Div()
    from core.store_utils import b64_to_model
    from dash import dash_table
    mdl = b64_to_model(model_data["model_b64"])
    fn  = mdl["feature_names"]
    df  = __import__("pandas").DataFrame({"#": range(1, len(fn) + 1), "Variabile": fn})
    return html.Div([
        dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df.columns],
            style_table={"maxHeight": "220px", "overflowY": "auto"},
            style_cell={"fontSize": "12px", "padding": "5px 10px",
                        "fontFamily": "Inter, sans-serif"},
            style_header={"fontWeight": 600},
        ),
    ], className="card mb-3")


@callback(
    Output("ls-chart-area", "children"),
    Output("ls-table-area", "children"),
    Input("ls-tipo",     "value"),
    Input("ls-pair",     "value"),
    State("model-store", "data"),
    State("cfg-store",   "data"),
)
def render_ls_chart(tipo, pair_str, model_data, cfg):
    if not model_data or not model_data.get("model_b64") or not pair_str:
        return html.Div(), html.Div()
    from core.store_utils import b64_to_model
    from components.charts import chart_loading, chart_score

    mdl   = b64_to_model(model_data["model_b64"])
    fn    = mdl["feature_names"]
    P     = mdl["loadings"]
    evr_m = mdl["evr"]
    lam_m = mdl["eigenvalues"]
    T_m   = mdl["scores"]
    alpha = (cfg or {}).get("alpha", 0.95)

    pc_i, pc_j = [int(x) for x in pair_str.split(",")]
    n_tr = mdl.get("n_train", len(T_m))

    import pandas as pd
    from dash import dash_table

    if tipo == "loading":
        fig = chart_loading(P, fn, evr_m, pc_i, pc_j)
        chart = dcc.Graph(
            figure=fig,
            config={"toImageButtonOptions": {"format": "png", "scale": 2}, "displayModeBar": True},
        )
        caption = html.Div(
            "Numero = indice variabile (vedi tabella indice). "
            "Hover per nome completo. "
            "⚫ Vicini = correlati  |  Opposti = negativamente correlati.",
            style={"fontSize": "12px", "color": "#64748B", "marginTop": "4px"},
        )
        # Loadings table
        df_load = pd.DataFrame({
            "#":           range(1, P.shape[0] + 1),
            "Variabile":   fn,
            f"PC{pc_i+1}": P[:, pc_i].round(4),
            f"PC{pc_j+1}": P[:, pc_j].round(4),
            "Distanza":    np.sqrt(P[:, pc_i]**2 + P[:, pc_j]**2).round(4),
        }).sort_values("Distanza", ascending=False)
        tbl = dash_table.DataTable(
            data=df_load.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df_load.columns],
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": "12px", "padding": "6px 10px",
                        "fontFamily": "Inter, sans-serif"},
            style_header={"fontWeight": 600, "fontSize": "11px", "color": "#64748B"},
            page_size=15,
            sort_action="native",
        )
        table_section = html.Div([
            html.Div("Loading table — variabili ordinate per distanza dall'origine",
                     style={"fontSize": "12px", "fontWeight": 600, "color": "#64748B",
                            "marginBottom": "8px"}),
            tbl,
        ], className="card")

    else:  # score
        flag_m = (mdl["T2"] > mdl["T2_UCL"]) | (mdl["Q"] > mdl["Q_UCL"])
        fig = chart_score(T_m, lam_m, mdl["T2_UCL"], evr_m, pc_i, pc_j,
                          flag_m, alpha=alpha, n_train=n_tr)
        chart = dcc.Graph(
            figure=fig,
            config={"toImageButtonOptions": {"format": "png", "scale": 2}, "displayModeBar": True},
        )
        caption = html.Div(
            f"⚫ Dentro ellisse {int(alpha*100)}%  |  "
            f"🔴 Fuori ellisse  |  "
            f"Ellisse basata sulla distribuzione F (k=2, n={n_tr}, α={alpha})",
            style={"fontSize": "12px", "color": "#64748B", "marginTop": "4px"},
        )
        table_section = html.Div()

    return [chart, caption], table_section
