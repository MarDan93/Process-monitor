import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f, norm
import io

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Process Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #1a1a2e;
        padding: 16px 20px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px;
        font-weight: 500;
        color: #1a1a2e;
    }
    .metric-label {
        font-size: 12px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .status-ok    { border-left-color: #2ecc71; }
    .status-warn  { border-left-color: #f39c12; }
    .status-alarm { border-left-color: #e74c3c; }

    .explain-box {
        background: #1a1a2e;
        color: #e0e0e0;
        padding: 20px 24px;
        border-radius: 4px;
        font-size: 14px;
        line-height: 1.7;
        margin-top: 16px;
    }
    .explain-box strong { color: #f39c12; }

    .var-pill {
        display: inline-block;
        background: #e74c3c22;
        color: #c0392b;
        border: 1px solid #e74c3c44;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin: 2px;
        font-family: 'IBM Plex Mono', monospace;
    }

    .stProgress > div > div { background-color: #1a1a2e; }

    .section-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #999;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────

def fit_pca_spc(X_raw, k, alpha=0.95):
    """Fit PCA-SPC model. Returns results dict."""
    n, p = X_raw.shape
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_raw)

    pca = PCA(n_components=k, svd_solver='full', random_state=42)
    T   = pca.fit_transform(Xs)
    P   = pca.components_.T
    lam = pca.explained_variance_

    T2     = np.sum((T**2) / lam, axis=1)
    T2_UCL = (k*(n-1)/(n-k)) * f.ppf(alpha, k, n-k)

    X_hat = pca.inverse_transform(T)
    E     = Xs - X_hat
    Q     = np.sum(E**2, axis=1)

    pca_full = PCA(n_components=min(p, n-1), svd_solver='full', random_state=42)
    pca_full.fit(Xs)
    re = pca_full.explained_variance_[k:]
    t1, t2, t3 = re.sum(), (re**2).sum(), (re**3).sum()
    h0 = 1 - (2*t1*t3)/(3*t2**2)
    z  = norm.ppf(alpha)
    Q_UCL = t1 * ((z*np.sqrt(2*t2*h0**2)/t1)+1+(t2*h0*(h0-1)/t1**2))**(1/h0)

    return {
        'scaler': scaler, 'pca': pca, 'k': k,
        'scores': T, 'loadings': P, 'eigenvalues': lam,
        'T2': T2, 'Q': Q, 'T2_UCL': T2_UCL, 'Q_UCL': Q_UCL,
        'X_scaled': Xs, 'E': E,
        'evr': pca.explained_variance_ratio_ * 100,
        'feature_names': []
    }


def monitor_new(results, X_new_raw):
    """Apply model to new data."""
    sc  = results['scaler']
    pca = results['pca']
    lam = results['eigenvalues']
    T2_UCL = results['T2_UCL']
    Q_UCL  = results['Q_UCL']

    Xn_s = sc.transform(X_new_raw)
    Tn   = pca.transform(Xn_s)
    T2n  = np.sum((Tn**2)/lam, axis=1)
    En   = Xn_s - pca.inverse_transform(Tn)
    Qn   = np.sum(En**2, axis=1)

    return {
        'T2': T2n, 'Q': Qn,
        'T2_flag': T2n > T2_UCL,
        'Q_flag':  Qn  > Q_UCL,
        'Xn_s': Xn_s, 'Tn': Tn, 'En': En
    }


def contributions(results, mon, obs_idx):
    """Compute signed contributions for one observation."""
    P   = results['loadings']
    lam = results['eigenvalues']
    t   = mon['Tn'][obs_idx]
    xs  = mon['Xn_s'][obs_idx]
    e   = mon['En'][obs_idx]
    w   = t / lam
    c_T2 = xs * (P @ w)
    c_Q  = e
    return c_T2, c_Q


def severity_label(t2, q, T2_UCL, Q_UCL):
    ratio = max(t2/T2_UCL, q/Q_UCL)
    if ratio < 1.0:   return "✅ IN CONTROLLO", "ok"
    elif ratio < 1.5: return "⚠️ ATTENZIONE",   "warn"
    else:             return "🔴 ANOMALIA",      "alarm"


def explain_obs(t2, q, T2_UCL, Q_UCL, top_vars_t2, top_vars_q, feature_names):
    """Generate plain-Italian explanation for a flagged observation."""
    ratio_t2 = t2 / T2_UCL
    ratio_q  = q  / Q_UCL

    lines = []
    if ratio_t2 > 1:
        lines.append(
            f"Il <strong>T²</strong> è {ratio_t2:.1f}× il limite — "
            "il ciclo si discosta dalla baseline nel piano delle componenti principali."
        )
    if ratio_q > 1:
        lines.append(
            f"Il <strong>Q (SPE)</strong> è {ratio_q:.1f}× il limite — "
            "sono presenti variazioni non catturate dal modello."
        )

    if top_vars_t2:
        vnames = ", ".join(
            f"<span class='var-pill'>{feature_names[i] if feature_names else f'Var {i+1}'}</span>"
            for i in top_vars_t2
        )
        lines.append(f"Variabili che contribuiscono di più al T²: {vnames}")

    if top_vars_q:
        vnames = ", ".join(
            f"<span class='var-pill'>{feature_names[i] if feature_names else f'Var {i+1}'}</span>"
            for i in top_vars_q
        )
        lines.append(f"Variabili che contribuiscono di più al Q: {vnames}")

    if not lines:
        return "<em>Ciclo in controllo — nessuna anomalia rilevata.</em>"

    return "<br><br>".join(lines)


def plotly_control_chart(values, ucl, title, color, flags=None):
    """Interactive control chart."""
    n = len(values)
    idx = np.arange(n)

    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scatter(
        x=idx, y=values,
        mode='lines+markers',
        line=dict(color=color, width=1.2),
        marker=dict(size=4, color=color),
        name=title,
        hovertemplate='Ciclo %{x}<br>Valore: %{y:.3f}<extra></extra>'
    ))

    # UCL line
    fig.add_hline(y=ucl, line_dash='dash', line_color='#e74c3c',
                  line_width=1.5, annotation_text=f'UCL = {ucl:.2f}',
                  annotation_position='right')

    # Flagged points
    if flags is not None and flags.any():
        fi = np.where(flags)[0]
        fig.add_trace(go.Scatter(
            x=fi, y=values[fi],
            mode='markers',
            marker=dict(size=10, color='#e74c3c', symbol='x', line=dict(width=2)),
            name='Anomalia',
            hovertemplate='Ciclo %{x}<br>Valore: %{y:.3f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(family='IBM Plex Mono', size=13)),
        xaxis_title='Ciclo',
        yaxis_title=title.split('—')[-1].strip(),
        height=280,
        margin=dict(l=10, r=10, t=40, b=30),
        legend=dict(orientation='h', y=-0.2),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(family='IBM Plex Sans'),
        hovermode='x unified'
    )
    return fig


def plotly_contribution(contrib, ucl_v, lcl_v, feature_names, title):
    """Contribution bar chart with per-variable UCL/LCL."""
    p = len(contrib)
    xax = list(range(1, p+1))
    colors = [
        '#e74c3c' if (contrib[i] > ucl_v[i] or contrib[i] < lcl_v[i])
        else '#2c3e50'
        for i in range(p)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xax, y=contrib,
        marker_color=colors,
        name='Contributo',
        hovertemplate='Var %{x}<br>%{y:.4f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=xax, y=ucl_v,
        mode='lines', line=dict(color='#e74c3c', dash='dash', width=1.5),
        name='UCL'
    ))
    fig.add_trace(go.Scatter(
        x=xax, y=lcl_v,
        mode='lines', line=dict(color='#e74c3c', dash='dash', width=1.5),
        name='LCL'
    ))
    fig.add_hline(y=0, line_color='black', line_width=0.8)

    fig.update_layout(
        title=dict(text=title, font=dict(family='IBM Plex Mono', size=12)),
        xaxis_title='Indice variabile',
        yaxis_title='Contributo (signed)',
        height=280,
        margin=dict(l=10, r=10, t=40, b=30),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(family='IBM Plex Sans'),
        showlegend=True,
        legend=dict(orientation='h', y=-0.25)
    )
    return fig


# ── Session state ─────────────────────────────────────────────
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'df_train' not in st.session_state:
    st.session_state.df_train = None


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏭 Process Monitor")
    st.markdown("**PCA-SPC · Injection Molding**")
    st.divider()

    st.markdown('<div class="section-title">Configurazione</div>', unsafe_allow_html=True)

    alpha = st.slider("Confidenza UCL", 0.90, 0.99, 0.95, 0.01,
                      help="Livello di confidenza per i limiti T² e Q")

    y_cols_input = st.text_area(
        "Variabili Y (qualità) — una per riga",
        placeholder="Quota cuscino\nTempo ciclo\nQuota minimo cuscino",
        help="Queste variabili vengono escluse dal modello PCA e usate solo per validazione"
    )
    y_cols = [c.strip() for c in y_cols_input.split('\n') if c.strip()]

    exclude_input = st.text_area(
        "Colonne da escludere — una per riga",
        placeholder="1\ntimestamp",
        help="Colonne indice, timestamp o ID da rimuovere"
    )
    exclude_cols = [c.strip() for c in exclude_input.split('\n') if c.strip()]

    st.divider()
    st.markdown('<div class="section-title">Selezione PCs</div>', unsafe_allow_html=True)
    k_mode = st.radio("Metodo selezione k", ["Automatico (Kaiser)", "Manuale"], label_visibility="collapsed")
    k_manual = st.number_input("Numero PCs", 1, 20, 3, disabled=(k_mode == "Automatico (Kaiser)"))


# ── Main tabs ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📥 Phase I — Calibrazione", "📊 Phase II — Monitoraggio", "📋 Variabili"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — PHASE I
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Carica dati di calibrazione")
    st.caption("Cicli raccolti in condizioni stabili dopo setup stampo / manutenzione")

    uploaded_train = st.file_uploader(
        "Trascina qui il file CSV dalla USB (Phase I)",
        type=['csv', 'xlsx', 'xls'],
        key='train_upload'
    )

    if uploaded_train:
        # Load
        if uploaded_train.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_train)
        else:
            df_raw = pd.read_excel(uploaded_train)

        df_raw.columns = df_raw.columns.astype(str)

        # Exclude columns
        drop_cols = [c for c in exclude_cols if c in df_raw.columns]
        if drop_cols:
            df_raw.drop(columns=drop_cols, inplace=True)

        # Separate X and Y
        y_valid   = [c for c in y_cols if c in df_raw.columns]
        df_num    = df_raw.select_dtypes(include=[np.number])
        df_num.fillna(df_num.mean(), inplace=True)
        const     = df_num.columns[df_num.std() == 0].tolist()
        if const:
            df_num.drop(columns=const, inplace=True)

        x_cols    = [c for c in df_num.columns if c not in y_valid]
        df_X      = df_num[x_cols].reset_index(drop=True)
        df_Y      = df_num[y_valid].reset_index(drop=True) if y_valid else pd.DataFrame()

        st.success(f"✅ File caricato: **{len(df_X)} cicli · {len(x_cols)} variabili X · {len(y_valid)} variabili Y**")

        col1, col2, col3 = st.columns(3)
        col1.metric("Cicli", len(df_X))
        col2.metric("Variabili X", len(x_cols))
        col3.metric("Variabili Y", len(y_valid) if y_valid else "—")

        # Preview
        with st.expander("Anteprima dati"):
            st.dataframe(df_X.head(10), use_container_width=True)

        # Select k
        X_raw = df_X.values
        sc_tmp = StandardScaler()
        Xs_tmp = sc_tmp.fit_transform(X_raw)
        pca_tmp = PCA(svd_solver='full', random_state=42)
        pca_tmp.fit(Xs_tmp)
        eigs = pca_tmp.explained_variance_

        if k_mode == "Automatico (Kaiser)":
            k = max(2, int(np.sum(eigs > 1)))
            k = min(k, X_raw.shape[1]-1, X_raw.shape[0]-2)
        else:
            k = int(k_manual)

        st.info(f"**Componenti principali selezionate: k = {k}**  "
                f"({'Kaiser automatico' if k_mode == 'Automatico (Kaiser)' else 'impostato manualmente'})")

        # Scree plot
        evr = pca_tmp.explained_variance_ratio_[:min(15, len(eigs))] * 100
        cum = np.cumsum(evr)
        fig_scree = go.Figure()
        fig_scree.add_bar(x=list(range(1, len(evr)+1)), y=evr,
                          name='Varianza per PC', marker_color='#2c3e50')
        fig_scree.add_scatter(x=list(range(1, len(evr)+1)), y=cum,
                              mode='lines+markers', name='Cumulativa %',
                              line=dict(color='#e74c3c', width=2),
                              yaxis='y2')
        fig_scree.add_vline(x=k, line_dash='dash', line_color='#f39c12',
                            annotation_text=f'k={k}')
        fig_scree.update_layout(
            title=dict(text='Scree Plot', font=dict(family='IBM Plex Mono', size=13)),
            xaxis_title='PC', yaxis_title='Varianza spiegata %',
            yaxis2=dict(title='Cumulativa %', overlaying='y', side='right', range=[0,105]),
            height=280, margin=dict(l=10,r=10,t=40,b=30),
            plot_bgcolor='#fafafa', paper_bgcolor='white',
            font=dict(family='IBM Plex Sans'),
            legend=dict(orientation='h', y=-0.25)
        )
        st.plotly_chart(fig_scree, use_container_width=True)

        # Fit model button
        if st.button("🔧 Costruisci modello Phase I", type="primary", use_container_width=True):
            with st.spinner("Fitting PCA-SPC..."):
                model = fit_pca_spc(X_raw, k, alpha)
                model['feature_names'] = x_cols

                # Contribution limits
                T  = model['scores']
                E  = model['E']
                P  = model['loadings']
                lam = model['eigenvalues']
                Xs = model['X_scaled']
                z  = norm.ppf(0.975)
                mu_e = E.mean(0); sd_e = E.std(0, ddof=1)
                model['Qcontrib_LCL'] = mu_e - z*sd_e
                model['Qcontrib_UCL'] = mu_e + z*sd_e
                W = T/lam; cT2 = Xs*(W@P.T)
                mu_c = cT2.mean(0); sd_c = cT2.std(0, ddof=1)
                model['T2contrib_LCL'] = mu_c - z*sd_c
                model['T2contrib_UCL'] = mu_c + z*sd_c

                st.session_state.model        = model
                st.session_state.feature_names = x_cols
                st.session_state.df_train     = df_X

            st.success("✅ Modello costruito e salvato.")

            c1, c2, c3 = st.columns(3)
            c1.metric("T² UCL", f"{model['T2_UCL']:.3f}")
            c2.metric("Q UCL",  f"{model['Q_UCL']:.3f}")
            flagged = ((model['T2'] > model['T2_UCL']) | (model['Q'] > model['Q_UCL'])).sum()
            pct = flagged/len(X_raw)*100
            status = "🟢 Buono" if pct < 5 else "🟡 Accettabile" if pct < 15 else "🔴 Attenzione"
            c3.metric("Flag Phase I", f"{flagged} ({pct:.1f}%)", delta=status, delta_color="off")

            # Phase I charts
            fig_t2 = plotly_control_chart(
                model['T2'], model['T2_UCL'],
                'Phase I — Hotelling T²', '#2c3e50',
                model['T2'] > model['T2_UCL']
            )
            fig_q = plotly_control_chart(
                model['Q'], model['Q_UCL'],
                'Phase I — Q (SPE)', '#16a085',
                model['Q'] > model['Q_UCL']
            )
            st.plotly_chart(fig_t2, use_container_width=True)
            st.plotly_chart(fig_q,  use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — PHASE II
# ══════════════════════════════════════════════════════════════
with tab2:
    if st.session_state.model is None:
        st.warning("⚠️ Prima costruisci il modello in **Phase I — Calibrazione**.")
    else:
        model   = st.session_state.model
        fn      = st.session_state.feature_names

        st.markdown("### Carica nuovi dati dalla USB")
        st.caption("Cicli di produzione da monitorare rispetto alla baseline")

        uploaded_test = st.file_uploader(
            "Trascina qui il file CSV dalla USB (Phase II)",
            type=['csv', 'xlsx', 'xls'],
            key='test_upload'
        )

        if uploaded_test:
            if uploaded_test.name.endswith('.csv'):
                df_test_raw = pd.read_csv(uploaded_test)
            else:
                df_test_raw = pd.read_excel(uploaded_test)

            df_test_raw.columns = df_test_raw.columns.astype(str)

            # Align columns
            missing = [c for c in fn if c not in df_test_raw.columns]
            if missing:
                st.error(f"Colonne mancanti nel file test: {missing}")
            else:
                df_test_X = df_test_raw[fn].copy()
                df_test_X.fillna(df_test_X.mean(), inplace=True)

                mon = monitor_new(model, df_test_X.values)

                n_test    = len(df_test_X)
                n_t2_flag = mon['T2_flag'].sum()
                n_q_flag  = mon['Q_flag'].sum()
                n_any     = (mon['T2_flag'] | mon['Q_flag']).sum()
                pct_any   = n_any / n_test * 100

                st.success(f"✅ {n_test} cicli analizzati")

                # KPI row
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Cicli totali", n_test)
                c2.metric("T² anomali", f"{n_t2_flag} ({n_t2_flag/n_test*100:.1f}%)",
                          delta="⚠️" if n_t2_flag > 0 else "✅", delta_color="off")
                c3.metric("Q anomali", f"{n_q_flag} ({n_q_flag/n_test*100:.1f}%)",
                          delta="⚠️" if n_q_flag > 0 else "✅", delta_color="off")

                if pct_any < 5:
                    stato = "🟢 STABILE"
                elif pct_any < 15:
                    stato = "🟡 ATTENZIONE"
                else:
                    stato = "🔴 ANOMALIE"
                c4.metric("Stato processo", stato)

                # Control charts
                fig_t2 = plotly_control_chart(
                    mon['T2'], model['T2_UCL'],
                    'Phase II — Hotelling T²', '#2980b9', mon['T2_flag']
                )
                fig_q = plotly_control_chart(
                    mon['Q'], model['Q_UCL'],
                    'Phase II — Q (SPE)', '#27ae60', mon['Q_flag']
                )
                st.plotly_chart(fig_t2, use_container_width=True)
                st.plotly_chart(fig_q,  use_container_width=True)

                # Flagged observations detail
                flagged_idx = np.where(mon['T2_flag'] | mon['Q_flag'])[0]

                if len(flagged_idx) == 0:
                    st.success("✅ Nessuna anomalia rilevata — processo in controllo.")
                else:
                    st.markdown(f"### 🔍 Analisi anomalie — {len(flagged_idx)} cicli fuori controllo")

                    # Select observation to inspect
                    obs_choice = st.selectbox(
                        "Seleziona ciclo da analizzare",
                        options=flagged_idx,
                        format_func=lambda x: (
                            f"Ciclo {x} — "
                            f"T²: {mon['T2'][x]:.2f} ({mon['T2'][x]/model['T2_UCL']:.1f}×UCL) | "
                            f"Q: {mon['Q'][x]:.2f} ({mon['Q'][x]/model['Q_UCL']:.1f}×UCL)"
                        )
                    )

                    t2_obs = mon['T2'][obs_choice]
                    q_obs  = mon['Q'][obs_choice]
                    label, status_cls = severity_label(t2_obs, q_obs, model['T2_UCL'], model['Q_UCL'])

                    # Status card
                    st.markdown(
                        f"<div class='metric-card status-{status_cls}'>"
                        f"<div class='metric-label'>Ciclo {obs_choice}</div>"
                        f"<div class='metric-value'>{label}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    # Contribution plots
                    c_t2, c_q = contributions(model, mon, obs_choice)

                    top_t2 = np.argsort(np.abs(c_t2))[::-1][:3].tolist()
                    top_q  = np.argsort(np.abs(c_q))[::-1][:3].tolist()

                    col_l, col_r = st.columns(2)
                    with col_l:
                        fig_ct2 = plotly_contribution(
                            c_t2,
                            model['T2contrib_UCL'],
                            model['T2contrib_LCL'],
                            fn,
                            f'T² Contribution — Ciclo {obs_choice}'
                        )
                        st.plotly_chart(fig_ct2, use_container_width=True)

                    with col_r:
                        fig_cq = plotly_contribution(
                            c_q,
                            model['Qcontrib_UCL'],
                            model['Qcontrib_LCL'],
                            fn,
                            f'Q Contribution — Ciclo {obs_choice}'
                        )
                        st.plotly_chart(fig_cq, use_container_width=True)

                    # Plain-Italian explanation
                    spiegazione = explain_obs(
                        t2_obs, q_obs,
                        model['T2_UCL'], model['Q_UCL'],
                        top_t2, top_q, fn
                    )
                    st.markdown(
                        f"<div class='explain-box'>"
                        f"<strong>Spiegazione per il tecnico</strong><br><br>"
                        f"{spiegazione}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    # Log soluzione
                    st.divider()
                    st.markdown("#### 📝 Registra intervento")
                    with st.form("log_form"):
                        azione = st.text_area("Descrivi l'intervento effettuato", height=80,
                                              placeholder="Es: aumentata contropressione da 80 a 95 bar")
                        submitted = st.form_submit_button("💾 Salva nel log", use_container_width=True)
                        if submitted and azione:
                            if 'anomaly_log' not in st.session_state:
                                st.session_state.anomaly_log = []
                            st.session_state.anomaly_log.append({
                                'Ciclo':     obs_choice,
                                'T²':        round(t2_obs, 3),
                                'Q':         round(q_obs, 3),
                                'Intervento': azione,
                            })
                            st.success("✅ Intervento registrato.")

                    # Show log
                    if 'anomaly_log' in st.session_state and st.session_state.anomaly_log:
                        st.markdown("#### 📋 Log interventi")
                        st.dataframe(
                            pd.DataFrame(st.session_state.anomaly_log),
                            use_container_width=True
                        )


# ══════════════════════════════════════════════════════════════
# TAB 3 — VARIABILI
# ══════════════════════════════════════════════════════════════
with tab3:
    if st.session_state.model is None:
        st.warning("⚠️ Carica prima i dati in Phase I.")
    else:
        model = st.session_state.model
        fn    = st.session_state.feature_names
        P     = model['loadings']
        k     = model['k']
        evr   = model['evr']

        st.markdown("### Indice variabili e loadings")

        # Variable table
        var_df = pd.DataFrame({
            'Indice': range(1, len(fn)+1),
            'Nome variabile': fn,
        })
        st.dataframe(var_df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Loading plot per componente")

        pc_sel = st.selectbox(
            "Seleziona PC",
            options=range(k),
            format_func=lambda i: f"PC{i+1} — {evr[i]:.2f}% varianza"
        )

        loadings_i = P[:, pc_sel]
        colors_bar = [
            '#e74c3c' if abs(l) >= 0.2 else '#2c3e50'
            for l in loadings_i
        ]

        fig_load = go.Figure(go.Bar(
            x=list(range(1, len(fn)+1)),
            y=loadings_i,
            marker_color=colors_bar,
            hovertemplate='%{customdata}<br>Loading: %{y:.4f}<extra></extra>',
            customdata=fn
        ))
        fig_load.add_hline(y=0.2,  line_dash='dot', line_color='gray', line_width=1)
        fig_load.add_hline(y=-0.2, line_dash='dot', line_color='gray', line_width=1)
        fig_load.add_hline(y=0,    line_color='black', line_width=0.8)

        fig_load.update_layout(
            title=dict(
                text=f'PC{pc_sel+1} Loadings ({evr[pc_sel]:.2f}% varianza) — rosso = |loading| ≥ 0.2',
                font=dict(family='IBM Plex Mono', size=13)
            ),
            xaxis_title='Indice variabile',
            yaxis_title='Loading',
            height=320,
            margin=dict(l=10, r=10, t=50, b=30),
            plot_bgcolor='#fafafa',
            paper_bgcolor='white',
            font=dict(family='IBM Plex Sans')
        )
        st.plotly_chart(fig_load, use_container_width=True)

        # Top variables
        top5 = np.argsort(np.abs(loadings_i))[::-1][:5]
        st.markdown("**Top 5 variabili per questa PC:**")
        top_df = pd.DataFrame({
            'Indice': top5 + 1,
            'Variabile': [fn[i] for i in top5],
            'Loading': [round(loadings_i[i], 4) for i in top5],
            'Direzione': ['↑ positivo' if loadings_i[i] > 0 else '↓ negativo' for i in top5]
        })
        st.dataframe(top_df, use_container_width=True, hide_index=True)
