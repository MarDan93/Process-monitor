import numpy as np
import plotly.graph_objects as go
from scipy.stats import f

COLORS = dict(
    primary="#2563EB",
    success="#16A34A",
    warning="#D97706",
    danger="#DC2626",
    neutral="#374151",
    muted="#6B7280",
    surface="#F8FAFC",
    border="#E2E8F0",
    white="#FFFFFF",
)
CHART_PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2", "#BE185D"]

_LAYOUT_BASE = dict(
    font=dict(family="Inter, sans-serif"),
    plot_bgcolor=COLORS["surface"],
    paper_bgcolor=COLORS["white"],
    xaxis=dict(gridcolor=COLORS["border"], gridwidth=1),
    yaxis=dict(gridcolor=COLORS["border"], gridwidth=1),
    margin=dict(l=10, r=80, t=44, b=36),
    legend=dict(orientation="h", y=-0.28, font=dict(size=11)),
    hovermode="x unified",
)


def _apply_base(fig, **overrides):
    layout = {**_LAYOUT_BASE, **overrides}
    fig.update_layout(**layout)
    return fig


def chart_line_multi(values, ucl, title, color, flags=None,
                     file_labels=None, file_boundaries=None):
    fig = go.Figure()
    if file_boundaries and len(file_boundaries) > 1:
        for fi, (s, e) in enumerate(zip(file_boundaries[:-1], file_boundaries[1:])):
            fc = CHART_PALETTE[fi % len(CHART_PALETTE)]
            fig.add_vrect(
                x0=s, x1=e - 1, fillcolor=fc, opacity=0.06, line_width=0,
                annotation_text=file_labels[fi] if file_labels else f"File {fi + 1}",
                annotation_position="top left", annotation_font_size=9,
            )
    fig.add_scatter(
        x=np.arange(len(values)).tolist(), y=values.tolist(),
        mode="lines", line=dict(color=color, width=1.5),
        hovertemplate="Cycle %{x}<br>%{y:.3f}<extra></extra>", name="Value",
    )
    fig.add_hline(
        y=ucl, line_dash="dash", line_color=COLORS["danger"], line_width=1.5,
        annotation_text=f"UCL={ucl:.3f}", annotation_position="right",
        annotation_font_color=COLORS["danger"], annotation_font_size=11,
    )
    if flags is not None and flags.any():
        fi = np.where(flags)[0]
        fig.add_scatter(
            x=fi.tolist(), y=values[fi].tolist(), mode="markers",
            marker=dict(size=7, color=COLORS["danger"], line=dict(color="white", width=1)),
            name="Anomaly",
            hovertemplate="Cycle %{x}<br>%{y:.3f}<extra></extra>",
        )
    return _apply_base(
        fig,
        title=dict(text=title, font=dict(family="Inter", size=13, color=COLORS["neutral"])),
        xaxis_title="Cycle", height=280,
        margin=dict(l=10, r=80, t=40, b=30),
    )


def chart_contribution(contrib, ucl_v, lcl_v, fn, title):
    p = len(contrib)
    xax = list(range(1, p + 1))
    labels = fn if fn else [f"Var {i + 1}" for i in range(p)]
    colors = [
        COLORS["danger"] if (contrib[i] > ucl_v[i] or contrib[i] < lcl_v[i])
        else COLORS["primary"]
        for i in range(p)
    ]
    fig = go.Figure()
    fig.add_bar(
        x=xax, y=contrib.tolist(), marker_color=colors, marker_line_width=0,
        customdata=labels,
        hovertemplate="[%{x}] %{customdata}<br>%{y:.4f}<extra></extra>",
        name="Contribution",
    )
    fig.add_scatter(
        x=xax, y=ucl_v.tolist(), mode="lines",
        line=dict(color=COLORS["danger"], dash="dot", width=1.5),
        name="UCL±", hoverinfo="skip",
    )
    fig.add_scatter(
        x=xax, y=lcl_v.tolist(), mode="lines",
        line=dict(color=COLORS["danger"], dash="dot", width=1.5),
        showlegend=False, hoverinfo="skip",
    )
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.8)
    exceed = [
        (i + 1, labels[i], round(float(contrib[i]), 4),
         round(float(lcl_v[i]), 4), round(float(ucl_v[i]), 4))
        for i in range(p)
        if contrib[i] > ucl_v[i] or contrib[i] < lcl_v[i]
    ]
    _apply_base(
        fig,
        title=dict(text=title, font=dict(family="Inter", size=12, color=COLORS["neutral"])),
        xaxis=dict(title="Variable index", tickvals=xax, ticktext=[str(x) for x in xax],
                   gridcolor=COLORS["border"]),
        yaxis=dict(title="Contribution (signed)", gridcolor=COLORS["border"]),
        height=300, margin=dict(l=10, r=10, t=40, b=30),
        legend=dict(orientation="h", y=-0.32, font=dict(size=11)),
    )
    return fig, exceed


def chart_score(T, lam, T2_UCL_global, evr, pc_i, pc_j,
                flagged=None, alpha=0.95, n_train=None):
    ang = np.linspace(0, 2 * np.pi, 400)
    n = n_train if n_train else max(T.shape[0], 10)
    T2_UCL_2d = (2 * (n - 1) / (n - 2)) * f.ppf(alpha, 2, n - 2)
    a = np.sqrt(lam[pc_i] * T2_UCL_2d)
    b = np.sqrt(lam[pc_j] * T2_UCL_2d)
    outside = (T[:, pc_i] / a) ** 2 + (T[:, pc_j] / b) ** 2 > 1
    inside = ~outside
    fig = go.Figure()
    if inside.any():
        fig.add_scatter(
            x=T[inside, pc_i].tolist(), y=T[inside, pc_j].tolist(),
            mode="markers",
            marker=dict(size=5, color=COLORS["neutral"], opacity=0.6),
            name=f"Inside {int(alpha * 100)}% ellipse",
            hovertemplate=f"PC{pc_i + 1}:%{{x:.3f}}<br>PC{pc_j + 1}:%{{y:.3f}}<extra></extra>",
        )
    if outside.any():
        fig.add_scatter(
            x=T[outside, pc_i].tolist(), y=T[outside, pc_j].tolist(),
            mode="markers",
            marker=dict(size=8, color=COLORS["danger"], line=dict(color="white", width=1)),
            name=f"Outside {int(alpha * 100)}% ellipse",
            hovertemplate=f"PC{pc_i + 1}:%{{x:.3f}}<br>PC{pc_j + 1}:%{{y:.3f}}<extra></extra>",
        )
    fig.add_scatter(
        x=(a * np.cos(ang)).tolist(), y=(b * np.sin(ang)).tolist(),
        mode="lines", line=dict(color=COLORS["danger"], dash="dash", width=1.8),
        name=f"{int(alpha * 100)}% confidence", hoverinfo="skip",
    )
    fig.add_hline(y=0, line_color=COLORS["border"], line_width=0.8)
    fig.add_vline(x=0, line_color=COLORS["border"], line_width=0.8)
    title_text = (
        f"Score plot  PC{pc_i + 1} ({evr[pc_i]:.1f}%)  vs  PC{pc_j + 1} ({evr[pc_j]:.1f}%)"
        f"  —  {int(outside.sum())}/{len(T)} outside {int(alpha * 100)}% ellipse"
    )
    return _apply_base(
        fig,
        title=dict(text=title_text, font=dict(family="Inter", size=12, color=COLORS["neutral"])),
        xaxis_title=f"PC{pc_i + 1} ({evr[pc_i]:.1f}%)",
        yaxis_title=f"PC{pc_j + 1} ({evr[pc_j]:.1f}%)",
        height=420, margin=dict(l=10, r=10, t=55, b=30),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        hovermode="closest",
    )


def chart_loading(P, fn, evr, pc_i, pc_j):
    p = P.shape[0]
    labels = fn if fn else [f"Var {i + 1}" for i in range(p)]
    fig = go.Figure()
    fig.add_scatter(
        x=P[:, pc_i].tolist(), y=P[:, pc_j].tolist(),
        mode="markers+text",
        marker=dict(size=9, color=COLORS["primary"], line=dict(color="white", width=1.5)),
        text=[str(i + 1) for i in range(p)],
        textposition="top center",
        textfont=dict(size=9, color=COLORS["muted"]),
        customdata=labels,
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            f"PC{pc_i + 1}: %{{x:.4f}}<br>"
            f"PC{pc_j + 1}: %{{y:.4f}}<extra></extra>"
        ),
        showlegend=False,
    )
    fig.add_hline(y=0, line_color=COLORS["border"], line_width=1)
    fig.add_vline(x=0, line_color=COLORS["border"], line_width=1)
    return _apply_base(
        fig,
        title=dict(
            text=f"Loading plot  PC{pc_i + 1} ({evr[pc_i]:.1f}%)  vs  PC{pc_j + 1} ({evr[pc_j]:.1f}%)",
            font=dict(family="Inter", size=12, color=COLORS["neutral"]),
        ),
        xaxis_title=f"PC{pc_i + 1} loading",
        yaxis_title=f"PC{pc_j + 1} loading",
        height=480, margin=dict(l=10, r=10, t=50, b=30),
        hovermode="closest",
    )


def chart_variable_frequency(df_vt):
    fig = go.Figure()
    fig.add_bar(
        x=df_vt["Variable"].tolist(), y=df_vt["T² exceed"].tolist(),
        name="T² exceedances", marker_color=COLORS["primary"], marker_line_width=0,
    )
    fig.add_bar(
        x=df_vt["Variable"].tolist(), y=df_vt["Q exceed"].tolist(),
        name="Q exceedances", marker_color=COLORS["danger"], marker_line_width=0,
    )
    return _apply_base(
        fig,
        barmode="stack",
        title=dict(text="Variable exceedance frequency",
                   font=dict(family="Inter", size=12)),
        xaxis=dict(title="Variable", tickangle=-40, gridcolor=COLORS["border"]),
        yaxis=dict(title="N° anomalous cycles", gridcolor=COLORS["border"]),
        height=300, margin=dict(l=10, r=10, t=40, b=80),
        legend=dict(orientation="h", y=-0.45, font=dict(size=11)),
    )
