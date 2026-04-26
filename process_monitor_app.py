import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f, norm
import anthropic

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Process Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    .section-header {
        background: #1a1a2e; color: white;
        padding: 12px 20px; border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 14px; letter-spacing: 0.05em;
        margin: 24px 0 16px 0;
    }
    .step-number {
        background: #e74c3c; color: white;
        border-radius: 50%; width: 24px; height: 24px;
        display: inline-flex; align-items: center;
        justify-content: center; font-size: 12px;
        font-weight: 600; margin-right: 10px;
    }
    .llm-box {
        background: #f0f4ff; border: 1px solid #c5d0f5;
        border-left: 4px solid #3b5bdb;
        padding: 16px 20px; border-radius: 4px;
        font-size: 14px; line-height: 1.7;
        margin-top: 12px;
    }
    .status-ok    { color: #2ecc71; font-weight: 600; }
    .status-warn  { color: #f39c12; font-weight: 600; }
    .status-alarm { color: #e74c3c; font-weight: 600; }
    .var-pill {
        display: inline-block; background: #e74c3c22; color: #c0392b;
        border: 1px solid #e74c3c44; padding: 2px 8px;
        border-radius: 20px; font-size: 12px; margin: 2px;
        font-family: 'IBM Plex Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ═════════════════════════════════════════════════════════════

def fit_pca_spc(X_raw, k, alpha=0.95):
    n, p   = X_raw.shape
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X_raw)
    pca    = PCA(n_components=k, svd_solver='full', random_state=42)
    T      = pca.fit_transform(Xs)
    P      = pca.components_.T
    lam    = pca.explained_variance_
    T2     = np.sum((T**2) / lam, axis=1)
    T2_UCL = (k*(n-1)/(n-k)) * f.ppf(alpha, k, n-k)
    X_hat  = pca.inverse_transform(T)
    E      = Xs - X_hat
    Q      = np.sum(E**2, axis=1)
    pca_f  = PCA(n_components=min(p, n-1), svd_solver='full', random_state=42)
    pca_f.fit(Xs)
    re     = pca_f.explained_variance_[k:]
    t1, t2, t3 = re.sum(), (re**2).sum(), (re**3).sum()
    h0     = 1 - (2*t1*t3) / (3*t2**2)
    z      = norm.ppf(alpha)
    Q_UCL  = t1 * ((z*np.sqrt(2*t2*h0**2)/t1) + 1 + (t2*h0*(h0-1)/t1**2))**(1/h0)
    return dict(
        scaler=scaler, pca=pca, k=k,
        scores=T, loadings=P, eigenvalues=lam,
        T2=T2, Q=Q, T2_UCL=T2_UCL, Q_UCL=Q_UCL,
        X_scaled=Xs, E=E,
        evr=pca.explained_variance_ratio_ * 100,
        feature_names=[]
    )


def compute_contrib_limits(model, Xs_train):
    P   = model['loadings']; lam = model['eigenvalues']
    pca = model['pca']
    T   = pca.transform(Xs_train)
    E   = Xs_train - pca.inverse_transform(T)
    z   = norm.ppf(0.975)
    mu_e = E.mean(0); sd_e = E.std(0, ddof=1)
    model['Qcontrib_LCL'] = mu_e - z*sd_e
    model['Qcontrib_UCL'] = mu_e + z*sd_e
    W = T/lam; cT2 = Xs_train*(W@P.T)
    mu_c = cT2.mean(0); sd_c = cT2.std(0, ddof=1)
    model['T2contrib_LCL'] = mu_c - z*sd_c
    model['T2contrib_UCL'] = mu_c + z*sd_c
    return model


def monitor_new(model, X_new):
    sc  = model['scaler']; pca = model['pca']; lam = model['eigenvalues']
    Xns = sc.transform(X_new); Tn = pca.transform(Xns)
    T2n = np.sum((Tn**2)/lam, axis=1)
    En  = Xns - pca.inverse_transform(Tn); Qn = np.sum(En**2, axis=1)
    return dict(T2=T2n, Q=Qn,
                T2_flag=T2n > model['T2_UCL'],
                Q_flag=Qn   > model['Q_UCL'],
                Xn_s=Xns, Tn=Tn, En=En)


def compute_rmsecv(X_s, max_k, G=10):
    n, p   = X_s.shape
    max_k  = min(max_k, p, n-1)
    idx    = np.arange(n)
    sg     = [idx[g::G] for g in range(G)]
    vg     = [np.array([j]) for j in range(p)]
    press  = np.zeros(max_k); rmsecv = np.zeros(max_k)
    for k in range(1, max_k+1):
        PRESS = 0.0; COUNT = 0
        for ti in sg:
            tri = np.setdiff1d(idx, ti)
            Xtr = X_s[tri]; Xte = X_s[ti]
            pc  = PCA(n_components=min(k,p), svd_solver='full', random_state=42)
            pc.fit(Xtr); P = pc.components_.T
            for mc in vg:
                oc  = np.setdiff1d(np.arange(p), mc)
                kk  = min(k, P[oc,:].shape[0]-1)
                if kk < 1: continue
                Th, *_ = np.linalg.lstsq(P[oc,:kk], Xte[:,oc].T, rcond=None)
                Xh  = (Th.T) @ P[:,:kk].T
                r   = Xte[:,mc] - Xh[:,mc]
                PRESS += np.sum(r**2); COUNT += r.size
        press[k-1]  = PRESS
        rmsecv[k-1] = np.sqrt(PRESS/COUNT) if COUNT > 0 else np.inf
    return int(np.argmin(rmsecv))+1, press, rmsecv


def iterative_cleaning(X_raw, k_clean, alpha_clean, max_iter=10):
    mask = np.ones(len(X_raw), dtype=bool); log = []
    for it in range(1, max_iter+1):
        X_it  = X_raw[mask]; n_it = len(X_it)
        sc_it = StandardScaler(); Xs_it = sc_it.fit_transform(X_it)
        k_it  = min(k_clean, Xs_it.shape[1]-1, n_it-2)
        res   = fit_pca_spc(X_it, k_it, alpha_clean)
        flag  = (res['T2'] > res['T2_UCL']) | (res['Q'] > res['Q_UCL'])
        n_rem = int(flag.sum())
        log.append(dict(Iterazione=it, Cicli_prima=n_it,
                        Rimossi=n_rem, Cicli_dopo=n_it-n_rem))
        if n_rem == 0: break
        idx_c = np.where(mask)[0]; mask[idx_c[flag]] = False
    return mask, pd.DataFrame(log)


def llm_describe_dataset(df_X, df_Y, feature_names, y_names):
    """Call Claude API to describe the dataset in plain Italian."""
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
        if not api_key:
            return None, "API key non configurata nei Secrets di Streamlit."

        desc = df_X.describe().round(3).to_string()
        missing = df_X.isnull().sum()
        missing_info = missing[missing > 0].to_string() if missing.any() else "Nessun valore mancante"
        corr_top = ""
        if len(feature_names) > 1:
            corr = df_X.corr().abs()
            np.fill_diagonal(corr.values, 0)
            top = corr.unstack().nlargest(5)
            corr_top = "\n".join(f"  {a} ↔ {b}: {v:.2f}" for (a,b),v in top.items())

        prompt = f"""Sei un esperto di processi industriali e analisi dati per stampaggio a iniezione.
Analizza questo dataset di processo e fornisci una descrizione sintetica in italiano per un process engineer.

DATASET:
- Osservazioni: {len(df_X)}
- Variabili di processo (X): {len(feature_names)} → {', '.join(feature_names[:10])}{'...' if len(feature_names)>10 else ''}
- Variabili di qualità (Y): {len(y_names)} → {', '.join(y_names) if y_names else 'nessuna'}

STATISTICHE DESCRITTIVE:
{desc}

VALORI MANCANTI:
{missing_info}

TOP 5 CORRELAZIONI TRA VARIABILI X:
{corr_top}

Fornisci una descrizione strutturata con:
1. Panoramica generale del dataset (2-3 righe)
2. Variabili con maggiore variabilità (coefficiente di variazione alto)
3. Eventuali valori anomali o da verificare (min/max sospetti)
4. Correlazioni significative da tenere in considerazione per la PCA
5. Raccomandazioni prima di procedere con l'analisi

Sii conciso e pratico. Usa linguaggio tecnico ma comprensibile."""

        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text, None
    except Exception as e:
        return None, str(e)


def llm_explain_anomaly(t2, q, T2_UCL, Q_UCL, top_vars_t2, top_vars_q, fn):
    """Call Claude API to explain an anomaly in plain Italian for a technician."""
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
        if not api_key:
            return None, "API key non configurata."

        vars_t2 = [fn[i] for i in top_vars_t2] if fn else [f"Var {i+1}" for i in top_vars_t2]
        vars_q  = [fn[i] for i in top_vars_q]  if fn else [f"Var {i+1}" for i in top_vars_q]

        prompt = f"""Sei un esperto di stampaggio a iniezione e controllo statistico di processo.
Spiega questa anomalia a un capoturno/tecnico di produzione in modo chiaro e pratico.

DATI ANOMALIA:
- T² (Hotelling): {t2:.3f} — limite UCL: {T2_UCL:.3f} — rapporto: {t2/T2_UCL:.2f}×
- Q (SPE): {q:.3f} — limite UCL: {Q_UCL:.3f} — rapporto: {q/Q_UCL:.2f}×
- Variabili che contribuiscono al T²: {', '.join(vars_t2)}
- Variabili che contribuiscono al Q: {', '.join(vars_q)}

Fornisci:
1. Spiegazione semplice dell'anomalia (cosa sta succedendo nel processo)
2. Possibili cause fisiche basate sulle variabili indicate
3. Azioni correttive consigliate (cosa controllare sulla macchina)

Massimo 150 parole. Linguaggio diretto, niente statistica, parla di processo fisico."""

        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text, None
    except Exception as e:
        return None, str(e)


# ── Chart helpers ─────────────────────────────────────────────

def chart_line(values, ucl, title, color, flags=None, key=None):
    fig = go.Figure()
    fig.add_scatter(x=np.arange(len(values)), y=values,
                    mode='lines+markers',
                    line=dict(color=color, width=1.2),
                    marker=dict(size=4, color=color),
                    hovertemplate='Ciclo %{x}<br>%{y:.3f}<extra></extra>',
                    name='Valore')
    fig.add_hline(y=ucl, line_dash='dash', line_color='#e74c3c',
                  line_width=1.5,
                  annotation_text=f'UCL={ucl:.3f}',
                  annotation_position='right')
    if flags is not None and flags.any():
        fi = np.where(flags)[0]
        fig.add_scatter(x=fi, y=values[fi], mode='markers',
                        marker=dict(size=10, color='#e74c3c',
                                    symbol='x', line=dict(width=2)),
                        name='Anomalia',
                        hovertemplate='Ciclo %{x}<br>%{y:.3f}<extra></extra>')
    fig.update_layout(
        title=dict(text=title, font=dict(family='IBM Plex Mono', size=13)),
        xaxis_title='Ciclo', height=300,
        margin=dict(l=10, r=10, t=40, b=30),
        plot_bgcolor='#fafafa', paper_bgcolor='white',
        font=dict(family='IBM Plex Sans'),
        legend=dict(orientation='h', y=-0.25),
        hovermode='x unified'
    )
    return fig


def chart_contribution(contrib, ucl_v, lcl_v, fn, title):
    p      = len(contrib)
    xax    = list(range(1, p+1))
    colors = ['#e74c3c' if (contrib[i] > ucl_v[i] or contrib[i] < lcl_v[i])
              else '#2c3e50' for i in range(p)]
    fig = go.Figure()
    fig.add_bar(x=xax, y=contrib.tolist(),
                marker_color=colors,
                customdata=fn if fn else [f'Var {i+1}' for i in range(p)],
                hovertemplate='%{customdata}<br>Contributo: %{y:.4f}<extra></extra>',
                name='Contributo')
    fig.add_scatter(x=xax, y=ucl_v.tolist(), mode='lines',
                    line=dict(color='#e74c3c', dash='dash', width=1.5),
                    name='UCL', hoverinfo='skip')
    fig.add_scatter(x=xax, y=lcl_v.tolist(), mode='lines',
                    line=dict(color='#e74c3c', dash='dash', width=1.5),
                    name='LCL', hoverinfo='skip')
    fig.add_hline(y=0, line_color='black', line_width=0.8)
    fig.update_layout(
        title=dict(text=title, font=dict(family='IBM Plex Mono', size=12)),
        xaxis=dict(title='Variabile (indice)',
                   tickvals=xax, ticktext=[str(x) for x in xax]),
        yaxis_title='Contributo (signed)',
        height=320, margin=dict(l=10, r=10, t=40, b=30),
        plot_bgcolor='#fafafa', paper_bgcolor='white',
        font=dict(family='IBM Plex Sans'),
        legend=dict(orientation='h', y=-0.3)
    )
    exceed = [(i+1,
               fn[i] if fn else f'Var {i+1}',
               round(float(contrib[i]), 4),
               round(float(lcl_v[i]), 4),
               round(float(ucl_v[i]), 4))
              for i in range(p)
              if contrib[i] > ucl_v[i] or contrib[i] < lcl_v[i]]
    return fig, exceed


def chart_scores(T, lam, T2_UCL, evr, pc_i, pc_j, flagged):
    ang = np.linspace(0, 2*np.pi, 400)
    a   = np.sqrt(T2_UCL * lam[pc_i])
    b   = np.sqrt(T2_UCL * lam[pc_j])
    ok  = ~flagged
    fig = go.Figure()
    fig.add_scatter(x=T[ok, pc_i].tolist(), y=T[ok, pc_j].tolist(),
                    mode='markers',
                    marker=dict(size=5, color='#2c3e50', opacity=0.6),
                    name='In controllo',
                    hovertemplate=f'PC{pc_i+1}: %{{x:.3f}}<br>PC{pc_j+1}: %{{y:.3f}}<extra></extra>')
    if flagged.any():
        fig.add_scatter(x=T[flagged, pc_i].tolist(), y=T[flagged, pc_j].tolist(),
                        mode='markers',
                        marker=dict(size=9, color='#e74c3c',
                                    symbol='x', line=dict(width=2)),
                        name='Anomalia',
                        hovertemplate=f'PC{pc_i+1}: %{{x:.3f}}<br>PC{pc_j+1}: %{{y:.3f}}<extra></extra>')
    fig.add_scatter(x=(a*np.cos(ang)).tolist(), y=(b*np.sin(ang)).tolist(),
                    mode='lines',
                    line=dict(color='#7f8c8d', dash='dash', width=1.2),
                    name='UCL T² ellisse', hoverinfo='skip')
    fig.add_hline(y=0, line_color='lightgray', line_width=0.8)
    fig.add_vline(x=0, line_color='lightgray', line_width=0.8)
    fig.update_layout(
        title=dict(text=f'Score plot  PC{pc_i+1} ({evr[pc_i]:.1f}%)  vs  PC{pc_j+1} ({evr[pc_j]:.1f}%)',
                   font=dict(family='IBM Plex Mono', size=12)),
        xaxis_title=f'PC{pc_i+1} ({evr[pc_i]:.1f}%)',
        yaxis_title=f'PC{pc_j+1} ({evr[pc_j]:.1f}%)',
        height=360, margin=dict(l=10, r=10, t=40, b=30),
        plot_bgcolor='#fafafa', paper_bgcolor='white',
        font=dict(family='IBM Plex Sans'),
        legend=dict(orientation='h', y=-0.25)
    )
    return fig


def chart_loadings(P, fn, pc_i, evr):
    p   = P.shape[0]
    xax = list(range(1, p+1))
    lv  = P[:, pc_i].tolist()
    fig = go.Figure()
    fig.add_scatter(x=xax, y=lv, mode='lines+markers',
                    line=dict(color='#2980b9', width=1.5),
                    marker=dict(size=6, color='#2980b9'),
                    customdata=fn if fn else [f'Var {i+1}' for i in range(p)],
                    hovertemplate='%{customdata}<br>Loading: %{y:.4f}<extra></extra>',
                    name='Loading')
    fig.add_hline(y=0, line_color='black', line_width=0.8)
    fig.update_layout(
        title=dict(text=f'PC{pc_i+1} Loadings  ({evr[pc_i]:.1f}% varianza)',
                   font=dict(family='IBM Plex Mono', size=12)),
        xaxis=dict(title='Variabile (indice)',
                   tickvals=xax, ticktext=[str(x) for x in xax]),
        yaxis_title='Loading',
        height=280, margin=dict(l=10, r=10, t=40, b=30),
        plot_bgcolor='#fafafa', paper_bgcolor='white',
        font=dict(family='IBM Plex Sans'), showlegend=False
    )
    return fig


# ═════════════════════════════════════════════════════════════
#  SESSION STATE
# ═════════════════════════════════════════════════════════════
defaults = dict(model=None, feature_names=[], df_train=None,
                df_X_full=None, df_Y_full=None, y_names=[],
                anomaly_log=[], mon=None, df_test_X=None,
                clean_mask=None)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🏭 Process Monitor")
    st.markdown("**PCA-SPC · Injection Molding**")
    st.divider()
    st.markdown("**Configurazione**")
    alpha = st.slider("Confidenza UCL", 0.90, 0.99, 0.95, 0.01)
    y_cols_input = st.text_area(
        "Variabili Y (qualità) — una per riga",
        placeholder="Quota cuscino\nTempo ciclo\nQuota minimo cuscino"
    )
    y_cols = [c.strip() for c in y_cols_input.split('\n') if c.strip()]
    exclude_input = st.text_area(
        "Colonne da escludere — una per riga",
        placeholder="1\ntimestamp"
    )
    exclude_cols = [c.strip() for c in exclude_input.split('\n') if c.strip()]
    st.divider()
    st.markdown("**Navigazione rapida**")
    st.markdown("""
- [📂 1 · Dataset](#dataset)
- [📐 2 · Selezione PC](#selezione-pc)
- [🔧 3 · Calibrazione](#calibrazione)
- [📊 4 · Loadings & Scores](#loadings-scores)
- [🔍 5 · Monitoraggio](#monitoraggio)
""")


# ═════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═════════════════════════════════════════════════════════════
st.markdown("# 🏭 Process Monitor")
st.markdown("Analisi multivariata PCA-SPC per il monitoraggio di processo")
st.divider()


# ════════════════════════════════════════════════════════════
#  SEZIONE 1 — DATASET
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-header"><span class="step-number">1</span>Dataset</div>',
            unsafe_allow_html=True)
st.markdown("Carica il file esportato dalla USB della pressa.")

uploaded = st.file_uploader(
    "Trascina qui il file CSV o Excel",
    type=['csv', 'xlsx', 'xls'], key='upload_main'
)

if uploaded:
    # Load
    if uploaded.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded)
    df_raw.columns = df_raw.columns.astype(str)

    # Drop excluded
    drop_c = [c for c in exclude_cols if c in df_raw.columns]
    if drop_c:
        df_raw.drop(columns=drop_c, inplace=True)

    # Separate X / Y
    y_valid  = [c for c in y_cols if c in df_raw.columns]
    df_num   = df_raw.select_dtypes(include=[np.number]).copy()
    const_c  = df_num.columns[df_num.std() == 0].tolist()
    if const_c:
        df_num.drop(columns=const_c, inplace=True)
    x_cols   = [c for c in df_num.columns if c not in y_valid]
    df_X     = df_num[x_cols].copy()
    df_Y     = df_num[y_valid].copy() if y_valid else pd.DataFrame()

    # Store in session
    st.session_state.df_X_full    = df_X
    st.session_state.df_Y_full    = df_Y
    st.session_state.feature_names = x_cols
    st.session_state.y_names      = y_valid

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cicli totali", len(df_X))
    c2.metric("Variabili X", len(x_cols))
    c3.metric("Variabili Y", len(y_valid) if y_valid else "—")
    miss_tot = df_X.isnull().sum().sum()
    c4.metric("Valori mancanti", int(miss_tot),
              delta="⚠️ da gestire" if miss_tot > 0 else "✅ nessuno",
              delta_color="off")

    # Indice variabili
    with st.expander("📋 Indice variabili"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Variabili X (processo)**")
            st.dataframe(
                pd.DataFrame({'Indice': range(1, len(x_cols)+1), 'Variabile': x_cols}),
                use_container_width=True, hide_index=True
            )
        with col_b:
            if y_valid:
                st.markdown("**Variabili Y (qualità)**")
                st.dataframe(
                    pd.DataFrame({'Variabile': y_valid}),
                    use_container_width=True, hide_index=True
                )

    # Statistiche descrittive
    with st.expander("📊 Statistiche descrittive", expanded=True):
        desc = df_X.describe().T.round(3)
        desc['cv%'] = (desc['std'] / desc['mean'].abs() * 100).round(1)
        st.dataframe(desc, use_container_width=True)

    # Valori mancanti
    if miss_tot > 0:
        with st.expander("⚠️ Valori mancanti"):
            miss_df = df_X.isnull().sum()
            miss_df = miss_df[miss_df > 0].reset_index()
            miss_df.columns = ['Variabile', 'Valori mancanti']
            miss_df['%'] = (miss_df['Valori mancanti'] / len(df_X) * 100).round(1)
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
        df_X.fillna(df_X.mean(), inplace=True)
        st.info("ℹ️ Valori mancanti sostituiti con la media della colonna.")

    # LLM description
    st.markdown("**Descrizione automatica del dataset**")
    if st.button("🤖 Genera descrizione con AI", key='llm_describe'):
        with st.spinner("Analisi in corso..."):
            testo, errore = llm_describe_dataset(df_X, df_Y, x_cols, y_valid)
        if errore:
            st.error(f"Errore API: {errore}")
        else:
            st.markdown(
                f"<div class='llm-box'>{testo.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True
            )
    st.caption("Richiede ANTHROPIC_API_KEY nei Secrets di Streamlit Cloud.")


# ════════════════════════════════════════════════════════════
#  SEZIONE 2 — SELEZIONE PC
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-header"><span class="step-number">2</span>Selezione numero di componenti principali</div>',
            unsafe_allow_html=True)

if st.session_state.df_X_full is None:
    st.info("⬆️ Carica prima il dataset nella sezione 1.")
else:
    df_X  = st.session_state.df_X_full.copy()
    df_X.fillna(df_X.mean(), inplace=True)
    X_raw = df_X.values
    sc_tmp = StandardScaler()
    Xs_tmp = sc_tmp.fit_transform(X_raw)
    n_obs, n_vars = Xs_tmp.shape
    max_k = min(20, n_vars-1, n_obs-2)

    pca_tmp  = PCA(n_components=max_k, svd_solver='full', random_state=42)
    pca_tmp.fit(Xs_tmp)
    eigs     = pca_tmp.explained_variance_
    evr_all  = pca_tmp.explained_variance_ratio_ * 100
    cum      = np.cumsum(evr_all)
    ks       = list(range(1, max_k+1))
    k_kaiser = max(2, min(int(np.sum(eigs > 1)), max_k))
    k_90     = int(np.argmax(cum >= 90.0)) + 1
    k_95     = int(np.argmax(cum >= 95.0)) + 1

    # Scelta grafico
    grafico = st.radio(
        "Seleziona il criterio da visualizzare",
        ["1 · Scree plot (Kaiser)", "2 · Varianza cumulativa", "3 · RMSECV"],
        horizontal=True, key='pc_chart_radio'
    )

    if grafico == "1 · Scree plot (Kaiser)":
        fig_pc = go.Figure()
        fig_pc.add_scatter(x=ks, y=evr_all.tolist(),
                           mode='lines+markers',
                           line=dict(color='#2c3e50', width=2),
                           marker=dict(size=6),
                           hovertemplate='PC%{x}<br>Varianza: %{y:.2f}%<extra></extra>')
        fig_pc.add_hline(y=float(100/n_vars), line_dash='dot', line_color='#7f8c8d',
                         annotation_text=f'Soglia Kaiser ({100/n_vars:.1f}%)',
                         annotation_position='right')
        fig_pc.add_vline(x=k_kaiser, line_dash='dash', line_color='#e74c3c',
                         annotation_text=f'Kaiser k={k_kaiser}',
                         annotation_position='top right')
        fig_pc.update_layout(xaxis_title='PC', yaxis_title='Varianza spiegata (%)',
                             height=320, margin=dict(l=10,r=10,t=20,b=30),
                             plot_bgcolor='#fafafa', paper_bgcolor='white',
                             font=dict(family='IBM Plex Sans'), showlegend=False)
        st.plotly_chart(fig_pc, use_container_width=True, key='scree_chart')
        st.info(f"💡 Criterio Kaiser: **k = {k_kaiser}** PC")

    elif grafico == "2 · Varianza cumulativa":
        fig_pc = go.Figure()
        fig_pc.add_scatter(x=ks, y=cum.tolist(),
                           mode='lines+markers',
                           line=dict(color='#2980b9', width=2),
                           marker=dict(size=6),
                           hovertemplate='PC%{x}<br>Cumulativa: %{y:.1f}%<extra></extra>')
        fig_pc.add_hline(y=90, line_dash='dash', line_color='#f39c12',
                         annotation_text='90%', annotation_position='right')
        fig_pc.add_hline(y=95, line_dash='dash', line_color='#e74c3c',
                         annotation_text='95%', annotation_position='right')
        fig_pc.add_vline(x=k_90, line_dash='dot', line_color='#f39c12',
                         annotation_text=f'k={k_90}', annotation_position='top left')
        fig_pc.add_vline(x=k_95, line_dash='dot', line_color='#e74c3c',
                         annotation_text=f'k={k_95}', annotation_position='top right')
        fig_pc.update_layout(xaxis_title='Numero PC',
                             yaxis_title='Varianza cumulativa (%)',
                             yaxis=dict(range=[0,105]),
                             height=320, margin=dict(l=10,r=10,t=20,b=30),
                             plot_bgcolor='#fafafa', paper_bgcolor='white',
                             font=dict(family='IBM Plex Sans'), showlegend=False)
        st.plotly_chart(fig_pc, use_container_width=True, key='cumvar_chart')
        st.info(f"💡 90% varianza → **k = {k_90}**  |  95% varianza → **k = {k_95}**")

    else:  # RMSECV
        with st.spinner("Calcolo RMSECV in corso... (può richiedere qualche minuto)"):
            best_k_cv, _, rmsecv_cv = compute_rmsecv(Xs_tmp, max_k)
        fig_pc = go.Figure()
        fig_pc.add_scatter(x=ks, y=rmsecv_cv.tolist(),
                           mode='lines+markers',
                           line=dict(color='#16a085', width=2),
                           marker=dict(
                               size=[12 if i+1==best_k_cv else 6 for i in range(max_k)],
                               color=['#e74c3c' if i+1==best_k_cv else '#16a085'
                                      for i in range(max_k)]),
                           hovertemplate='PC%{x}<br>RMSECV: %{y:.4f}<extra></extra>')
        fig_pc.add_vline(x=best_k_cv, line_dash='dash', line_color='#e74c3c',
                         annotation_text=f'Minimo k={best_k_cv}',
                         annotation_position='top right')
        fig_pc.update_layout(xaxis_title='Numero PC', yaxis_title='RMSECV',
                             height=320, margin=dict(l=10,r=10,t=20,b=30),
                             plot_bgcolor='#fafafa', paper_bgcolor='white',
                             font=dict(family='IBM Plex Sans'), showlegend=False)
        st.plotly_chart(fig_pc, use_container_width=True, key='rmsecv_chart')
        st.info(f"💡 RMSECV minimo → **k = {best_k_cv}** PC")

    # Riepilogo e scelta
    st.markdown("**Riepilogo suggerimenti:**")
    rs1, rs2, rs3 = st.columns(3)
    rs1.metric("Kaiser",  f"k = {k_kaiser}")
    rs2.metric("90% var", f"k = {k_90}")
    rs3.metric("RMSECV",  "calcola →" if grafico != "3 · RMSECV" else f"k = {best_k_cv if grafico=='3 · RMSECV' else '—'}")

    k_chosen = st.number_input(
        "Numero di PC da usare nel modello",
        min_value=2, max_value=max_k,
        value=k_kaiser, step=1, key='k_chosen'
    )
    st.session_state['k_chosen_val'] = int(k_chosen)
    st.success(f"✅ k = **{k_chosen}** PC — varianza spiegata: **{cum[k_chosen-1]:.1f}%**")


# ════════════════════════════════════════════════════════════
#  SEZIONE 3 — CALIBRAZIONE (PHASE I)
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-header"><span class="step-number">3</span>Calibrazione — Phase I</div>',
            unsafe_allow_html=True)

if st.session_state.df_X_full is None:
    st.info("⬆️ Carica prima il dataset.")
elif 'k_chosen_val' not in st.session_state:
    st.info("⬆️ Scegli il numero di PC nella sezione 2.")
else:
    df_X   = st.session_state.df_X_full.copy()
    df_X.fillna(df_X.mean(), inplace=True)
    x_cols = st.session_state.feature_names
    X_raw  = df_X.values
    k_use  = st.session_state.k_chosen_val

    # Pulizia
    st.markdown("**🧹 Pulizia dati (opzionale)**")
    st.caption("Rimuove cicli anomali dal set di calibrazione prima di fittare il modello.")
    use_clean = st.toggle("Attiva pulizia iterativa degli outlier", value=False)

    clean_mask = np.ones(len(X_raw), dtype=bool)

    if use_clean:
        alpha_clean = st.slider("Confidenza pulizia", 0.95, 0.999, 0.99, 0.001,
                                format="%.3f",
                                help="0.99 = rimuove solo cicli chiaramente anomali")
        if st.button("▶ Esegui pulizia", key='btn_clean'):
            sc_k = StandardScaler()
            Xs_k = sc_k.fit_transform(X_raw)
            eig_k = PCA(svd_solver='full').fit(Xs_k).explained_variance_
            k_cl  = max(2, min(int(np.sum(eig_k > 1)), X_raw.shape[1]-1))
            with st.spinner("Pulizia in corso..."):
                clean_mask, log_df = iterative_cleaning(X_raw, k_cl, alpha_clean)
            n_rem = int((~clean_mask).sum())
            pct   = n_rem / len(X_raw) * 100
            st.success(f"✅ Rimossi **{n_rem}** cicli ({pct:.1f}%) · Rimasti: **{clean_mask.sum()}**")
            st.dataframe(log_df, use_container_width=True, hide_index=True)
            if pct > 20:
                st.warning("⚠️ >20% rimossi — il periodo di calibrazione potrebbe non essere stato stabile.")
            st.session_state.clean_mask = clean_mask
    else:
        st.session_state.clean_mask = clean_mask

    # Fit
    st.markdown("**🔧 Costruisci modello**")
    if st.button("Costruisci modello Phase I", type="primary",
                 use_container_width=True, key='btn_fit'):
        mask   = st.session_state.clean_mask
        if mask is None:
            mask = np.ones(len(X_raw), dtype=bool)
        X_clean = X_raw[mask]
        with st.spinner("Fitting PCA-SPC..."):
            model = fit_pca_spc(X_clean, k_use, alpha)
            model['feature_names'] = x_cols
            model = compute_contrib_limits(model, model['X_scaled'])
            st.session_state.model        = model
            st.session_state.df_train     = pd.DataFrame(X_clean, columns=x_cols)

        st.success("✅ Modello costruito.")
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("T² UCL", f"{model['T2_UCL']:.3f}")
        cm2.metric("Q UCL",  f"{model['Q_UCL']:.3f}")
        n_flag = int(((model['T2'] > model['T2_UCL']) |
                      (model['Q']  > model['Q_UCL'])).sum())
        cm3.metric("Flag residui", f"{n_flag} ({n_flag/len(X_clean)*100:.1f}%)")

        fT2 = model['T2'] > model['T2_UCL']
        fQ  = model['Q']  > model['Q_UCL']
        st.plotly_chart(chart_line(model['T2'], model['T2_UCL'],
                                   'Phase I — Hotelling T²', '#2c3e50', fT2),
                        use_container_width=True, key='p1_t2')
        st.plotly_chart(chart_line(model['Q'], model['Q_UCL'],
                                   'Phase I — Q (SPE)', '#16a085', fQ),
                        use_container_width=True, key='p1_q')


# ════════════════════════════════════════════════════════════
#  SEZIONE 4 — LOADINGS & SCORES
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-header"><span class="step-number">4</span>Loadings & Score plots</div>',
            unsafe_allow_html=True)

if st.session_state.model is None:
    st.info("⬆️ Costruisci prima il modello nella sezione 3.")
else:
    model  = st.session_state.model
    fn     = model['feature_names']
    P      = model['loadings']
    k_m    = model['k']
    evr_m  = model['evr']
    lam_m  = model['eigenvalues']
    T_m    = model['scores']
    fT2_m  = model['T2'] > model['T2_UCL']
    fQ_m   = model['Q']  > model['Q_UCL']
    flag_m = fT2_m | fQ_m

    # Indice variabili
    with st.expander("📋 Indice variabili"):
        st.dataframe(
            pd.DataFrame({'Indice': range(1, len(fn)+1), 'Variabile': fn}),
            use_container_width=True, hide_index=True
        )

    # Loading plots
    st.markdown("**Loading plots**")
    for i in range(k_m):
        st.plotly_chart(chart_loadings(P, fn, i, evr_m),
                        use_container_width=True,
                        key=f'load_{i}')
        top5 = np.argsort(np.abs(P[:, i]))[::-1][:5]
        with st.expander(f"Top 5 variabili PC{i+1}"):
            st.dataframe(pd.DataFrame({
                'Indice':    top5+1,
                'Variabile': [fn[j] for j in top5],
                'Loading':   [round(float(P[j,i]),4) for j in top5],
                'Direzione': ['↑' if P[j,i]>0 else '↓' for j in top5]
            }), use_container_width=True, hide_index=True)

    # Score plots
    st.markdown("**Score plots — coppie di PC**")
    pairs = [(i, i+1) for i in range(0, k_m-1, 2)]
    for pc_i, pc_j in pairs:
        st.plotly_chart(
            chart_scores(T_m, lam_m, model['T2_UCL'],
                         evr_m, pc_i, pc_j, flag_m),
            use_container_width=True,
            key=f'score_{pc_i}_{pc_j}'
        )


# ════════════════════════════════════════════════════════════
#  SEZIONE 5 — MONITORAGGIO (PHASE II)
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-header"><span class="step-number">5</span>Monitoraggio — Phase II</div>',
            unsafe_allow_html=True)

if st.session_state.model is None:
    st.info("⬆️ Costruisci prima il modello nella sezione 3.")
else:
    model = st.session_state.model
    fn    = model['feature_names']

    uploaded_test = st.file_uploader(
        "Carica nuovi dati dalla USB",
        type=['csv','xlsx','xls'], key='upload_test'
    )

    if uploaded_test:
        if uploaded_test.name.endswith('.csv'):
            df_test_raw = pd.read_csv(uploaded_test)
        else:
            df_test_raw = pd.read_excel(uploaded_test)
        df_test_raw.columns = df_test_raw.columns.astype(str)

        miss_c = [c for c in fn if c not in df_test_raw.columns]
        if miss_c:
            st.error(f"Colonne mancanti: {miss_c}")
        else:
            df_test_X = df_test_raw[fn].copy()
            df_test_X.fillna(df_test_X.mean(), inplace=True)
            mon = monitor_new(model, df_test_X.values)
            st.session_state.mon       = mon
            st.session_state.df_test_X = df_test_X

            n_test  = len(df_test_X)
            n_t2    = int(mon['T2_flag'].sum())
            n_q     = int(mon['Q_flag'].sum())
            n_any   = int((mon['T2_flag'] | mon['Q_flag']).sum())
            pct_any = n_any / n_test * 100

            st.success(f"✅ {n_test} cicli analizzati")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Cicli",      n_test)
            c2.metric("T² anomali", f"{n_t2} ({n_t2/n_test*100:.1f}%)")
            c3.metric("Q anomali",  f"{n_q} ({n_q/n_test*100:.1f}%)")
            stato = ("🟢 STABILE" if pct_any < 5
                     else "🟡 ATTENZIONE" if pct_any < 15
                     else "🔴 ANOMALIE")
            c4.metric("Stato", stato)

            # Control charts interattivi
            st.markdown("**Control charts — clicca un punto per analizzarlo**")

            fig_t2 = go.Figure()
            fig_t2.add_scatter(x=np.arange(n_test).tolist(), y=mon['T2'].tolist(),
                               mode='lines+markers',
                               line=dict(color='#2980b9', width=1.2),
                               marker=dict(size=6, color=['#e74c3c' if f else '#2980b9'
                                                          for f in mon['T2_flag']]),
                               hovertemplate='Ciclo %{x}<br>T²: %{y:.3f}<extra></extra>',
                               name='T²')
            fig_t2.add_hline(y=model['T2_UCL'], line_dash='dash',
                             line_color='#e74c3c', line_width=1.5,
                             annotation_text=f"UCL={model['T2_UCL']:.3f}",
                             annotation_position='right')
            fig_t2.update_layout(
                title=dict(text='Phase II — Hotelling T²',
                           font=dict(family='IBM Plex Mono', size=13)),
                xaxis_title='Ciclo', height=300,
                margin=dict(l=10,r=10,t=40,b=30),
                plot_bgcolor='#fafafa', paper_bgcolor='white',
                font=dict(family='IBM Plex Sans'),
                clickmode='event+select'
            )

            fig_q = go.Figure()
            fig_q.add_scatter(x=np.arange(n_test).tolist(), y=mon['Q'].tolist(),
                              mode='lines+markers',
                              line=dict(color='#27ae60', width=1.2),
                              marker=dict(size=6, color=['#e74c3c' if f else '#27ae60'
                                                         for f in mon['Q_flag']]),
                              hovertemplate='Ciclo %{x}<br>Q: %{y:.3f}<extra></extra>',
                              name='Q')
            fig_q.add_hline(y=model['Q_UCL'], line_dash='dash',
                            line_color='#e74c3c', line_width=1.5,
                            annotation_text=f"UCL={model['Q_UCL']:.3f}",
                            annotation_position='right')
            fig_q.update_layout(
                title=dict(text='Phase II — Q (SPE)',
                           font=dict(family='IBM Plex Mono', size=13)),
                xaxis_title='Ciclo', height=300,
                margin=dict(l=10,r=10,t=40,b=30),
                plot_bgcolor='#fafafa', paper_bgcolor='white',
                font=dict(family='IBM Plex Sans')
            )

            st.plotly_chart(fig_t2, use_container_width=True, key='p2_t2')
            st.plotly_chart(fig_q,  use_container_width=True, key='p2_q')

            # Selezione ciclo per analisi
            flagged_idx = np.where(mon['T2_flag'] | mon['Q_flag'])[0]

            if len(flagged_idx) == 0:
                st.success("✅ Nessuna anomalia — processo in controllo.")
            else:
                st.markdown(f"### 🔍 Analisi anomalie — {len(flagged_idx)} cicli fuori controllo")

                obs_choice = st.selectbox(
                    "Seleziona ciclo da analizzare",
                    options=flagged_idx.tolist(),
                    format_func=lambda x: (
                        f"Ciclo {x}  —  "
                        f"T²: {mon['T2'][x]:.2f} ({mon['T2'][x]/model['T2_UCL']:.1f}×UCL)  |  "
                        f"Q: {mon['Q'][x]:.2f} ({mon['Q'][x]/model['Q_UCL']:.1f}×UCL)"
                    ),
                    key='obs_select'
                )

                t2_obs = float(mon['T2'][obs_choice])
                q_obs  = float(mon['Q'][obs_choice])
                ratio  = max(t2_obs/model['T2_UCL'], q_obs/model['Q_UCL'])

                if ratio < 1.5:
                    stato_obs = "⚠️ ATTENZIONE"
                    cls_obs   = "status-warn"
                else:
                    stato_obs = "🔴 ANOMALIA"
                    cls_obs   = "status-alarm"

                st.markdown(
                    f"<p class='{cls_obs}'>Ciclo {obs_choice} — {stato_obs} — "
                    f"severità: {ratio:.2f}× UCL</p>",
                    unsafe_allow_html=True
                )

                # Contribution plots
                c_t2_v = (mon['Xn_s'][obs_choice] *
                          (model['loadings'] @
                           (mon['Tn'][obs_choice] / model['eigenvalues'])))
                c_q_v  = mon['En'][obs_choice]

                top_t2 = np.argsort(np.abs(c_t2_v))[::-1][:3].tolist()
                top_q  = np.argsort(np.abs(c_q_v))[::-1][:3].tolist()

                col_l, col_r = st.columns(2)
                with col_l:
                    fig_ct2, exceed_t2 = chart_contribution(
                        c_t2_v,
                        model['T2contrib_UCL'],
                        model['T2contrib_LCL'],
                        fn, f'T² Contribution — Ciclo {obs_choice}'
                    )
                    st.plotly_chart(fig_ct2, use_container_width=True,
                                    key=f'ct2_{obs_choice}')
                    if exceed_t2:
                        st.markdown("**Variabili fuori limite — T²:**")
                        st.dataframe(
                            pd.DataFrame(exceed_t2,
                                         columns=['Idx','Variabile','Valore','LCL','UCL']),
                            use_container_width=True, hide_index=True
                        )

                with col_r:
                    fig_cq, exceed_q = chart_contribution(
                        c_q_v,
                        model['Qcontrib_UCL'],
                        model['Qcontrib_LCL'],
                        fn, f'Q Contribution — Ciclo {obs_choice}'
                    )
                    st.plotly_chart(fig_cq, use_container_width=True,
                                    key=f'cq_{obs_choice}')
                    if exceed_q:
                        st.markdown("**Variabili fuori limite — Q:**")
                        st.dataframe(
                            pd.DataFrame(exceed_q,
                                         columns=['Idx','Variabile','Valore','LCL','UCL']),
                            use_container_width=True, hide_index=True
                        )

                # LLM spiegazione anomalia
                st.markdown("**🤖 Spiegazione per il tecnico**")
                if st.button("Genera spiegazione con AI", key='llm_anomaly'):
                    with st.spinner("Analisi in corso..."):
                        testo_an, err_an = llm_explain_anomaly(
                            t2_obs, q_obs,
                            model['T2_UCL'], model['Q_UCL'],
                            top_t2, top_q, fn
                        )
                    if err_an:
                        st.error(f"Errore API: {err_an}")
                    else:
                        st.markdown(
                            f"<div class='llm-box'>{testo_an.replace(chr(10),'<br>')}</div>",
                            unsafe_allow_html=True
                        )

                # Log intervento
                st.divider()
                st.markdown("**📝 Registra intervento**")
                with st.form("log_form"):
                    azione = st.text_area(
                        "Descrivi l'azione correttiva",
                        height=80,
                        placeholder="Es: aumentata contropressione da 80 a 95 bar"
                    )
                    if st.form_submit_button("💾 Salva", use_container_width=True):
                        if azione:
                            st.session_state.anomaly_log.append({
                                'Ciclo':      obs_choice,
                                'T²':         round(t2_obs, 3),
                                'Q':          round(q_obs, 3),
                                'Severità':   f"{ratio:.2f}×UCL",
                                'Intervento': azione
                            })
                            st.success("✅ Salvato.")

                if st.session_state.anomaly_log:
                    st.markdown("**📋 Log interventi**")
                    st.dataframe(
                        pd.DataFrame(st.session_state.anomaly_log),
                        use_container_width=True
                    )
