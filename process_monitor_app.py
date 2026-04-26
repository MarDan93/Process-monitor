import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f, norm

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Process Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    .metric-card {
        background: #f8f9fa; border: 1px solid #e0e0e0;
        border-left: 4px solid #1a1a2e; padding: 16px 20px;
        border-radius: 4px; margin-bottom: 12px;
    }
    .metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 28px; font-weight: 500; color: #1a1a2e; }
    .metric-label { font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.08em; }
    .status-ok    { border-left-color: #2ecc71; }
    .status-warn  { border-left-color: #f39c12; }
    .status-alarm { border-left-color: #e74c3c; }
    .explain-box {
        background: #1a1a2e; color: #e0e0e0; padding: 20px 24px;
        border-radius: 4px; font-size: 14px; line-height: 1.7; margin-top: 16px;
    }
    .explain-box strong { color: #f39c12; }
    .var-pill {
        display: inline-block; background: #e74c3c22; color: #c0392b;
        border: 1px solid #e74c3c44; padding: 2px 10px; border-radius: 20px;
        font-size: 12px; margin: 2px; font-family: 'IBM Plex Mono', monospace;
    }
    .section-title {
        font-family: 'IBM Plex Mono', monospace; font-size: 11px;
        text-transform: uppercase; letter-spacing: 0.12em; color: #999;
        margin-bottom: 8px; padding-bottom: 4px; border-bottom: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
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
    h0     = 1 - (2*t1*t3)/(3*t2**2)
    z      = norm.ppf(alpha)
    Q_UCL  = t1 * ((z*np.sqrt(2*t2*h0**2)/t1)+1+(t2*h0*(h0-1)/t1**2))**(1/h0)
    return {
        'scaler': scaler, 'pca': pca, 'k': k,
        'scores': T, 'loadings': P, 'eigenvalues': lam,
        'T2': T2, 'Q': Q, 'T2_UCL': T2_UCL, 'Q_UCL': Q_UCL,
        'X_scaled': Xs, 'E': E,
        'evr': pca.explained_variance_ratio_ * 100,
        'feature_names': []
    }


def compute_contrib_limits(model, df_X):
    fn  = model['feature_names']
    sc  = model['scaler']
    pca = model['pca']
    P   = model['loadings']
    lam = model['eigenvalues']
    Xs  = sc.transform(df_X[fn].values)
    T   = pca.transform(Xs)
    E   = Xs - pca.inverse_transform(T)
    z   = norm.ppf(0.975)
    mu_e = E.mean(0); sd_e = E.std(0, ddof=1)
    model['Qcontrib_LCL'] = mu_e - z*sd_e
    model['Qcontrib_UCL'] = mu_e + z*sd_e
    W = T/lam; cT2 = Xs*(W@P.T)
    mu_c = cT2.mean(0); sd_c = cT2.std(0, ddof=1)
    model['T2contrib_LCL'] = mu_c - z*sd_c
    model['T2contrib_UCL'] = mu_c + z*sd_c
    return model


def monitor_new(model, X_new_raw):
    sc  = model['scaler']; pca = model['pca']; lam = model['eigenvalues']
    Xns = sc.transform(X_new_raw); Tn = pca.transform(Xns)
    T2n = np.sum((Tn**2)/lam, axis=1)
    En  = Xns - pca.inverse_transform(Tn); Qn = np.sum(En**2, axis=1)
    return {
        'T2': T2n, 'Q': Qn,
        'T2_flag': T2n > model['T2_UCL'],
        'Q_flag':  Qn  > model['Q_UCL'],
        'Xn_s': Xns, 'Tn': Tn, 'En': En
    }


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
    """Rimuove outlier iterativamente dal train set fino a convergenza."""
    mask    = np.ones(len(X_raw), dtype=bool)
    log     = []
    for it in range(1, max_iter+1):
        X_it   = X_raw[mask]
        n_it   = len(X_it)
        sc_it  = StandardScaler()
        Xs_it  = sc_it.fit_transform(X_it)
        k_it   = min(k_clean, Xs_it.shape[1]-1, n_it-2)
        res_it = fit_pca_spc(X_it, k_it, alpha_clean)
        flag   = (res_it['T2'] > res_it['T2_UCL']) | (res_it['Q'] > res_it['Q_UCL'])
        n_rem  = int(flag.sum())
        log.append({'Iterazione': it, 'Cicli prima': n_it,
                    'Rimossi': n_rem, 'Cicli dopo': n_it - n_rem})
        if n_rem == 0:
            break
        idx_clean = np.where(mask)[0]
        mask[idx_clean[flag]] = False
    return mask, pd.DataFrame(log)


def severity_label(t2, q, T2_UCL, Q_UCL):
    ratio = max(t2/T2_UCL, q/Q_UCL)
    if ratio < 1.0:   return "✅ IN CONTROLLO", "ok"
    elif ratio < 1.5: return "⚠️ ATTENZIONE",   "warn"
    else:             return "🔴 ANOMALIA",      "alarm"


def explain_obs(t2, q, T2_UCL, Q_UCL, top_t2, top_q, fn):
    lines = []
    if t2 > T2_UCL:
        lines.append(f"Il <strong>T²</strong> è {t2/T2_UCL:.1f}× il limite — "
                     "il ciclo si discosta dalla baseline nello spazio del modello.")
    if q > Q_UCL:
        lines.append(f"Il <strong>Q</strong> è {q/Q_UCL:.1f}× il limite — "
                     "sono presenti variazioni non catturate dal modello.")
    if top_t2:
        v = ", ".join(f"<span class='var-pill'>{fn[i] if fn else f'Var {i+1}'}</span>"
                      for i in top_t2)
        lines.append(f"Variabili che causano il T²: {v}")
    if top_q:
        v = ", ".join(f"<span class='var-pill'>{fn[i] if fn else f'Var {i+1}'}</span>"
                      for i in top_q)
        lines.append(f"Variabili che causano il Q: {v}")
    return "<br><br>".join(lines) if lines else "<em>Ciclo in controllo.</em>"


def chart_line(values, ucl, title, color, flags=None):
    fig = go.Figure()
    fig.add_scatter(x=np.arange(len(values)), y=values,
                    mode='lines+markers',
                    line=dict(color=color, width=1.2),
                    marker=dict(size=4),
                    hovertemplate='Ciclo %{x}<br>%{y:.3f}<extra></extra>')
    fig.add_hline(y=ucl, line_dash='dash', line_color='#e74c3c',
                  line_width=1.5, annotation_text=f'UCL={ucl:.3f}',
                  annotation_position='right')
    if flags is not None and flags.any():
        fi = np.where(flags)[0]
        fig.add_scatter(x=fi, y=values[fi], mode='markers',
                        marker=dict(size=10, color='#e74c3c', symbol='x',
                                    line=dict(width=2)),
                        name='Anomalia',
                        hovertemplate='Ciclo %{x}<br>%{y:.3f}<extra></extra>')
    fig.update_layout(
        title=dict(text=title, font=dict(family='IBM Plex Mono', size=13)),
        xaxis_title='Ciclo', height=280,
        margin=dict(l=10, r=10, t=40, b=30),
        plot_bgcolor='#fafafa', paper_bgcolor='white',
        font=dict(family='IBM Plex Sans'),
        showlegend=False, hovermode='x unified'
    )
    return fig


def chart_contribution(contrib, ucl_v, lcl_v, fn, title):
    p      = len(contrib)
    xax    = list(range(1, p+1))
    colors = ['#e74c3c' if (contrib[i] > ucl_v[i] or contrib[i] < lcl_v[i])
              else '#2c3e50' for i in range(p)]
    fig = go.Figure()
    fig.add_bar(x=xax, y=contrib, marker_color=colors,
                customdata=fn,
                hovertemplate='%{customdata}<br>Contributo: %{y:.4f}<extra></extra>')
    # UCL e LCL per variabile come linee step
    fig.add_scatter(x=xax, y=ucl_v, mode='lines',
                    line=dict(color='#e74c3c', dash='dash', width=1.5),
                    name='UCL', hoverinfo='skip')
    fig.add_scatter(x=xax, y=lcl_v, mode='lines',
                    line=dict(color='#e74c3c', dash='dash', width=1.5),
                    name='LCL', hoverinfo='skip')
    fig.add_hline(y=0, line_color='black', line_width=0.8)
    # Tabella variabili fuori limite sotto il grafico
    exceed = [(i+1, fn[i] if fn else f'Var {i+1}', round(float(contrib[i]),4),
               round(float(lcl_v[i]),4), round(float(ucl_v[i]),4))
              for i in range(p)
              if contrib[i] > ucl_v[i] or contrib[i] < lcl_v[i]]
    fig.update_layout(
        title=dict(text=title, font=dict(family='IBM Plex Mono', size=12)),
        xaxis=dict(title='Variabile (indice)', tickvals=xax,
                   ticktext=[str(x) for x in xax]),
        yaxis_title='Contributo (signed)',
        height=300, margin=dict(l=10, r=10, t=40, b=30),
        plot_bgcolor='#fafafa', paper_bgcolor='white',
        font=dict(family='IBM Plex Sans'),
        legend=dict(orientation='h', y=-0.3)
    )
    return fig, exceed


def chart_scores(T, lam, T2_UCL, evr, pc_i, pc_j, flagged):
    ang = np.linspace(0, 2*np.pi, 400)
    a   = np.sqrt(T2_UCL * lam[pc_i])
    b   = np.sqrt(T2_UCL * lam[pc_j])
    fig = go.Figure()
    ok  = ~flagged
    fig.add_scatter(x=T[ok, pc_i], y=T[ok, pc_j], mode='markers',
                    marker=dict(size=5, color='#2c3e50', opacity=0.6),
                    name='In controllo',
                    hovertemplate=f'PC{pc_i+1}: %{{x:.3f}}<br>PC{pc_j+1}: %{{y:.3f}}<extra></extra>')
    if flagged.any():
        fig.add_scatter(x=T[flagged, pc_i], y=T[flagged, pc_j], mode='markers',
                        marker=dict(size=9, color='#e74c3c', symbol='x',
                                    line=dict(width=2)),
                        name='Anomalia',
                        hovertemplate=f'PC{pc_i+1}: %{{x:.3f}}<br>PC{pc_j+1}: %{{y:.3f}}<extra></extra>')
    fig.add_scatter(x=a*np.cos(ang), y=b*np.sin(ang),
                    mode='lines', line=dict(color='#7f8c8d', dash='dash', width=1.2),
                    name='UCL T² ellisse', hoverinfo='skip')
    fig.add_hline(y=0, line_color='lightgray', line_width=0.8)
    fig.add_vline(x=0, line_color='lightgray', line_width=0.8)
    fig.update_layout(
        title=dict(text=f'Score plot PC{pc_i+1} vs PC{pc_j+1}',
                   font=dict(family='IBM Plex Mono', size=12)),
        xaxis_title=f'PC{pc_i+1} ({evr[pc_i]:.1f}%)',
        yaxis_title=f'PC{pc_j+1} ({evr[pc_j]:.1f}%)',
        height=320, margin=dict(l=10, r=10, t=40, b=30),
        plot_bgcolor='#fafafa', paper_bgcolor='white',
        font=dict(family='IBM Plex Sans'),
        legend=dict(orientation='h', y=-0.3)
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
                    customdata=fn,
                    hovertemplate='%{customdata}<br>Loading: %{y:.4f}<extra></extra>')
    fig.add_hline(y=0, line_color='black', line_width=0.8)
    fig.update_layout(
        title=dict(text=f'PC{pc_i+1} Loadings ({evr[pc_i]:.1f}% varianza)',
                   font=dict(family='IBM Plex Mono', size=12)),
        xaxis=dict(title='Variabile (indice)', tickvals=xax,
                   ticktext=[str(x) for x in xax]),
        yaxis_title='Loading',
        height=280, margin=dict(l=10, r=10, t=40, b=30),
        plot_bgcolor='#fafafa', paper_bgcolor='white',
        font=dict(family='IBM Plex Sans'), showlegend=False
    )
    return fig


# ═════════════════════════════════════════════════════════════
#  SESSION STATE
# ═════════════════════════════════════════════════════════════
for key, val in [('model', None), ('feature_names', []),
                 ('df_train', None), ('anomaly_log', [])]:
    if key not in st.session_state:
        st.session_state[key] = val


# ═════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🏭 Process Monitor")
    st.markdown("**PCA-SPC · Injection Molding**")
    st.divider()
    st.markdown('<div class="section-title">Configurazione</div>', unsafe_allow_html=True)
    alpha = st.slider("Confidenza UCL", 0.90, 0.99, 0.95, 0.01)
    y_cols_input = st.text_area("Variabili Y (qualità) — una per riga",
                                placeholder="Quota cuscino\nTempo ciclo\nQuota minimo cuscino")
    y_cols = [c.strip() for c in y_cols_input.split('\n') if c.strip()]
    exclude_input = st.text_area("Colonne da escludere — una per riga",
                                 placeholder="1\ntimestamp")
    exclude_cols = [c.strip() for c in exclude_input.split('\n') if c.strip()]
    st.divider()
    st.caption("La selezione del numero di PC avviene nella tab Phase I.")


# ═════════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📥 Phase I — Calibrazione",
                             "📊 Phase II — Monitoraggio",
                             "📋 Variabili & Loadings"])


# ════════════════════════════════════
#  TAB 1 — PHASE I
# ════════════════════════════════════
with tab1:
    st.markdown("### Carica dati di calibrazione")
    st.caption("Cicli raccolti in condizioni stabili dopo setup stampo / manutenzione")

    uploaded_train = st.file_uploader(
        "Trascina il file CSV o Excel dalla USB (Phase I)",
        type=['csv', 'xlsx', 'xls'], key='train_upload'
    )

    if uploaded_train:
        # Load
        if uploaded_train.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_train)
        else:
            df_raw = pd.read_excel(uploaded_train)
        df_raw.columns = df_raw.columns.astype(str)

        # Drop excluded
        drop_cols = [c for c in exclude_cols if c in df_raw.columns]
        if drop_cols:
            df_raw.drop(columns=drop_cols, inplace=True)

        # Separate X / Y
        y_valid = [c for c in y_cols if c in df_raw.columns]
        df_num  = df_raw.select_dtypes(include=[np.number]).copy()
        df_num.fillna(df_num.mean(), inplace=True)
        const   = df_num.columns[df_num.std() == 0].tolist()
        if const:
            df_num.drop(columns=const, inplace=True)
        x_cols = [c for c in df_num.columns if c not in y_valid]
        df_X   = df_num[x_cols].reset_index(drop=True)
        df_Y   = df_num[y_valid].reset_index(drop=True) if y_valid else pd.DataFrame()

        st.success(f"✅ {len(df_X)} cicli · {len(x_cols)} variabili X · {len(y_valid)} variabili Y")

        c1, c2, c3 = st.columns(3)
        c1.metric("Cicli", len(df_X))
        c2.metric("Variabili X", len(x_cols))
        c3.metric("Variabili Y", len(y_valid) if y_valid else "—")

        with st.expander("Anteprima dati"):
            st.dataframe(df_X.head(10), use_container_width=True)

        # ── Analisi esplorativa per selezione k ───────────────
        X_raw  = df_X.values
        sc_tmp = StandardScaler()
        Xs_tmp = sc_tmp.fit_transform(X_raw)
        n_obs, n_vars = Xs_tmp.shape
        max_k  = min(20, n_vars-1, n_obs-2)

        pca_tmp = PCA(n_components=max_k, svd_solver='full', random_state=42)
        pca_tmp.fit(Xs_tmp)
        eigs   = pca_tmp.explained_variance_
        evr_all = pca_tmp.explained_variance_ratio_ * 100
        cum    = np.cumsum(evr_all)
        ks     = list(range(1, max_k+1))

        k_kaiser = max(2, min(int(np.sum(eigs > 1)), max_k))
        k_90     = int(np.argmax(cum >= 90.0)) + 1
        k_95     = int(np.argmax(cum >= 95.0)) + 1

        st.markdown("---")
        st.markdown("### 📐 Selezione numero di componenti principali")
        st.caption("Usa i tre grafici per scegliere k, poi conferma in fondo.")

        with st.spinner("Calcolo RMSECV in corso..."):
            best_k_cv, press_cv, rmsecv_cv = compute_rmsecv(Xs_tmp, max_k)

        # Grafico 1 — Scree plot (curva)
        st.markdown("**1 · Scree plot** — cerca il gomito della curva")
        fig1 = go.Figure()
        fig1.add_scatter(x=ks, y=evr_all.tolist(), mode='lines+markers',
                         line=dict(color='#2c3e50', width=2),
                         marker=dict(size=6, color='#2c3e50'),
                         name='Varianza per PC',
                         hovertemplate='PC%{x}<br>Varianza: %{y:.2f}%<extra></extra>')
        fig1.add_hline(y=float(100/n_vars), line_dash='dot', line_color='#7f8c8d',
                       annotation_text=f'Soglia Kaiser ({100/n_vars:.1f}%)',
                       annotation_position='right')
        fig1.add_vline(x=k_kaiser, line_dash='dash', line_color='#e74c3c',
                       annotation_text=f'Kaiser k={k_kaiser}',
                       annotation_position='top right')
        fig1.update_layout(xaxis_title='PC', yaxis_title='Varianza spiegata (%)',
                           height=280, margin=dict(l=10,r=10,t=20,b=30),
                           plot_bgcolor='#fafafa', paper_bgcolor='white',
                           font=dict(family='IBM Plex Sans'), showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        st.info(f"💡 Kaiser suggerisce **k = {k_kaiser}**")

        # Grafico 2 — Varianza cumulativa
        st.markdown("**2 · Varianza cumulativa** — quota di processo spiegata")
        fig2 = go.Figure()
        fig2.add_scatter(x=ks, y=cum.tolist(), mode='lines+markers',
                         line=dict(color='#2980b9', width=2),
                         marker=dict(size=6),
                         hovertemplate='PC%{x}<br>Cumulativa: %{y:.1f}%<extra></extra>')
        fig2.add_hline(y=90, line_dash='dash', line_color='#f39c12',
                       annotation_text='90%', annotation_position='right')
        fig2.add_hline(y=95, line_dash='dash', line_color='#e74c3c',
                       annotation_text='95%', annotation_position='right')
        fig2.add_vline(x=k_90, line_dash='dot', line_color='#f39c12',
                       annotation_text=f'k={k_90}', annotation_position='top left')
        fig2.add_vline(x=k_95, line_dash='dot', line_color='#e74c3c',
                       annotation_text=f'k={k_95}', annotation_position='top right')
        fig2.update_layout(xaxis_title='Numero PC', yaxis_title='Varianza cumulativa (%)',
                           yaxis=dict(range=[0,105]),
                           height=280, margin=dict(l=10,r=10,t=20,b=30),
                           plot_bgcolor='#fafafa', paper_bgcolor='white',
                           font=dict(family='IBM Plex Sans'), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.info(f"💡 90% varianza → **k = {k_90}**  |  95% varianza → **k = {k_95}**")

        # Grafico 3 — RMSECV
        st.markdown("**3 · RMSECV** — errore di ricostruzione in cross-validation")
        fig3 = go.Figure()
        fig3.add_scatter(x=ks, y=rmsecv_cv.tolist(), mode='lines+markers',
                         line=dict(color='#16a085', width=2),
                         marker=dict(
                             size=[12 if i+1==best_k_cv else 6 for i in range(max_k)],
                             color=['#e74c3c' if i+1==best_k_cv else '#16a085'
                                    for i in range(max_k)]),
                         hovertemplate='PC%{x}<br>RMSECV: %{y:.4f}<extra></extra>')
        fig3.add_vline(x=best_k_cv, line_dash='dash', line_color='#e74c3c',
                       annotation_text=f'Minimo k={best_k_cv}',
                       annotation_position='top right')
        fig3.update_layout(xaxis_title='Numero PC', yaxis_title='RMSECV',
                           height=280, margin=dict(l=10,r=10,t=20,b=30),
                           plot_bgcolor='#fafafa', paper_bgcolor='white',
                           font=dict(family='IBM Plex Sans'), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        st.info(f"💡 RMSECV minimo → **k = {best_k_cv}**")

        # Riepilogo e scelta finale
        st.markdown("---")
        st.markdown("**Riepilogo suggerimenti**")
        cs1, cs2, cs3 = st.columns(3)
        cs1.metric("Kaiser",  f"k = {k_kaiser}")
        cs2.metric("90% var", f"k = {k_90}")
        cs3.metric("RMSECV",  f"k = {best_k_cv}")

        k = st.number_input("Scegli il numero di PC da usare nel modello",
                            min_value=2, max_value=max_k,
                            value=int(best_k_cv), step=1)
        st.success(f"✅ Modello sarà costruito con **k = {k}** PC — "
                   f"varianza spiegata: **{cum[k-1]:.1f}%**")

        # ── Pulizia dati Phase I ───────────────────────────────
        st.markdown("---")
        st.markdown("### 🧹 Pulizia dati Phase I")
        st.caption("Rimuove cicli anomali dal set di calibrazione per ottenere un modello più robusto.")

        use_cleaning = st.toggle(
            "Attiva pulizia iterativa degli outlier",
            value=False,
            help="Consigliato quando non sei sicuro che tutti i cicli fossero stabili"
        )

        clean_mask = np.ones(len(X_raw), dtype=bool)
        cleaning_log_df = None

        if use_cleaning:
            alpha_clean = st.slider(
                "Confidenza per la pulizia (più alto = più conservativo)",
                min_value=0.95, max_value=0.999,
                value=0.99, step=0.001,
                format="%.3f",
                help="0.99 = rimuove solo cicli chiaramente anomali (consigliato)"
            )
            k_clean = max(2, min(k_kaiser, max_k))

            if st.button("▶ Esegui pulizia", use_container_width=False):
                with st.spinner("Pulizia iterativa in corso..."):
                    clean_mask, cleaning_log_df = iterative_cleaning(
                        X_raw, k_clean, alpha_clean
                    )
                n_removed = int((~clean_mask).sum())
                pct_rem   = n_removed / len(X_raw) * 100
                st.success(f"✅ Pulizia completata — rimossi **{n_removed} cicli** "
                           f"({pct_rem:.1f}%) · Cicli puliti: **{clean_mask.sum()}**")
                st.dataframe(cleaning_log_df, use_container_width=True, hide_index=True)
                if pct_rem > 20:
                    st.warning("⚠️ Più del 20% dei cicli rimossi — il periodo di "
                               "calibrazione potrebbe non essere stato stabile. "
                               "Valuta di raccogliere nuovi dati in condizioni più controllate.")
        else:
            st.caption("Pulizia disattivata — tutti i cicli verranno usati per la calibrazione.")

        # ── Costruisci modello ─────────────────────────────────
        st.markdown("---")
        if st.button("🔧 Costruisci modello Phase I", type="primary",
                     use_container_width=True):
            X_clean = X_raw[clean_mask]
            with st.spinner("Fitting PCA-SPC..."):
                model = fit_pca_spc(X_clean, int(k), alpha)
                model['feature_names'] = x_cols
                model = compute_contrib_limits(model, pd.DataFrame(X_clean, columns=x_cols))
                st.session_state.model        = model
                st.session_state.feature_names = x_cols
                st.session_state.df_train     = pd.DataFrame(X_clean, columns=x_cols)

            st.success("✅ Modello costruito e salvato.")

            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("T² UCL", f"{model['T2_UCL']:.3f}")
            cm2.metric("Q UCL",  f"{model['Q_UCL']:.3f}")
            flagged_tr = ((model['T2'] > model['T2_UCL']) |
                          (model['Q']  > model['Q_UCL'])).sum()
            pct_f = flagged_tr / len(X_clean) * 100
            cm3.metric("Flag residui Phase I", f"{flagged_tr} ({pct_f:.1f}%)")

            # Control charts Phase I
            fT2  = model['T2'] > model['T2_UCL']
            fQ   = model['Q']  > model['Q_UCL']
            flagged_arr = fT2 | fQ
            st.plotly_chart(chart_line(model['T2'], model['T2_UCL'],
                                       'Phase I — Hotelling T²', '#2c3e50', fT2),
                            use_container_width=True)
            st.plotly_chart(chart_line(model['Q'], model['Q_UCL'],
                                       'Phase I — Q (SPE)', '#16a085', fQ),
                            use_container_width=True)

            # Score plots per ogni coppia di PC
            st.markdown("#### Score plots — coppie di PC")
            evr_m = model['evr']
            lam_m = model['eigenvalues']
            T_m   = model['scores']
            k_m   = model['k']
            pairs = [(i, i+1) for i in range(0, k_m-1, 2)]
            for pc_i, pc_j in pairs:
                st.plotly_chart(
                    chart_scores(T_m, lam_m, model['T2_UCL'],
                                 evr_m, pc_i, pc_j, flagged_arr),
                    use_container_width=True
                )


# ════════════════════════════════════
#  TAB 2 — PHASE II
# ════════════════════════════════════
with tab2:
    if st.session_state.model is None:
        st.warning("⚠️ Prima costruisci il modello in **Phase I — Calibrazione**.")
    else:
        model = st.session_state.model
        fn    = st.session_state.feature_names

        st.markdown("### Carica nuovi dati dalla USB")
        uploaded_test = st.file_uploader(
            "Trascina il file CSV o Excel dalla USB (Phase II)",
            type=['csv', 'xlsx', 'xls'], key='test_upload'
        )

        if uploaded_test:
            if uploaded_test.name.endswith('.csv'):
                df_test_raw = pd.read_csv(uploaded_test)
            else:
                df_test_raw = pd.read_excel(uploaded_test)
            df_test_raw.columns = df_test_raw.columns.astype(str)

            missing_cols = [c for c in fn if c not in df_test_raw.columns]
            if missing_cols:
                st.error(f"Colonne mancanti: {missing_cols}")
            else:
                df_test_X = df_test_raw[fn].copy()
                df_test_X.fillna(df_test_X.mean(), inplace=True)
                mon = monitor_new(model, df_test_X.values)

                n_test  = len(df_test_X)
                n_t2    = mon['T2_flag'].sum()
                n_q     = mon['Q_flag'].sum()
                n_any   = (mon['T2_flag'] | mon['Q_flag']).sum()
                pct_any = n_any / n_test * 100

                st.success(f"✅ {n_test} cicli analizzati")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Cicli totali", n_test)
                c2.metric("T² anomali",  f"{n_t2} ({n_t2/n_test*100:.1f}%)")
                c3.metric("Q anomali",   f"{n_q} ({n_q/n_test*100:.1f}%)")
                stato = ("🟢 STABILE" if pct_any < 5
                         else "🟡 ATTENZIONE" if pct_any < 15
                         else "🔴 ANOMALIE")
                c4.metric("Stato processo", stato)

                st.plotly_chart(chart_line(mon['T2'], model['T2_UCL'],
                                           'Phase II — Hotelling T²',
                                           '#2980b9', mon['T2_flag']),
                                use_container_width=True)
                st.plotly_chart(chart_line(mon['Q'], model['Q_UCL'],
                                           'Phase II — Q (SPE)',
                                           '#27ae60', mon['Q_flag']),
                                use_container_width=True)

                flagged_idx = np.where(mon['T2_flag'] | mon['Q_flag'])[0]

                if len(flagged_idx) == 0:
                    st.success("✅ Nessuna anomalia — processo in controllo.")
                else:
                    st.markdown(f"### 🔍 Analisi anomalie — {len(flagged_idx)} cicli")

                    obs_choice = st.selectbox(
                        "Seleziona ciclo da analizzare",
                        options=flagged_idx,
                        format_func=lambda x: (
                            f"Ciclo {x}  —  "
                            f"T²: {mon['T2'][x]:.2f} ({mon['T2'][x]/model['T2_UCL']:.1f}×UCL)  |  "
                            f"Q: {mon['Q'][x]:.2f} ({mon['Q'][x]/model['Q_UCL']:.1f}×UCL)"
                        )
                    )

                    t2_obs = float(mon['T2'][obs_choice])
                    q_obs  = float(mon['Q'][obs_choice])
                    label, status_cls = severity_label(t2_obs, q_obs,
                                                       model['T2_UCL'], model['Q_UCL'])

                    st.markdown(
                        f"<div class='metric-card status-{status_cls}'>"
                        f"<div class='metric-label'>Ciclo {obs_choice}</div>"
                        f"<div class='metric-value'>{label}</div>"
                        f"</div>", unsafe_allow_html=True
                    )

                    # Tabella variabili indice → nome
                    with st.expander("📋 Indice variabili (riferimento per i grafici)"):
                        st.dataframe(
                            pd.DataFrame({'Indice': range(1, len(fn)+1),
                                          'Variabile': fn}),
                            use_container_width=True, hide_index=True
                        )

                    # Contribution plots con UCL/LCL per variabile
                    c_t2, c_q = (
                        mon['Xn_s'][obs_choice] * (
                            model['loadings'] @
                            (mon['Tn'][obs_choice] / model['eigenvalues'])
                        ),
                        mon['En'][obs_choice]
                    )

                    top_t2 = np.argsort(np.abs(c_t2))[::-1][:3].tolist()
                    top_q  = np.argsort(np.abs(c_q))[::-1][:3].tolist()

                    col_l, col_r = st.columns(2)
                    with col_l:
                        fig_ct2, exceed_t2 = chart_contribution(
                            c_t2, model['T2contrib_UCL'], model['T2contrib_LCL'],
                            fn, f'T² Contribution — Ciclo {obs_choice}'
                        )
                        st.plotly_chart(fig_ct2, use_container_width=True)
                        if exceed_t2:
                            st.markdown("**Variabili fuori limite (T²):**")
                            st.dataframe(
                                pd.DataFrame(exceed_t2,
                                             columns=['Idx','Variabile','Valore','LCL','UCL']),
                                use_container_width=True, hide_index=True
                            )

                    with col_r:
                        fig_cq, exceed_q = chart_contribution(
                            c_q, model['Qcontrib_UCL'], model['Qcontrib_LCL'],
                            fn, f'Q Contribution — Ciclo {obs_choice}'
                        )
                        st.plotly_chart(fig_cq, use_container_width=True)
                        if exceed_q:
                            st.markdown("**Variabili fuori limite (Q):**")
                            st.dataframe(
                                pd.DataFrame(exceed_q,
                                             columns=['Idx','Variabile','Valore','LCL','UCL']),
                                use_container_width=True, hide_index=True
                            )

                    # Spiegazione italiana
                    st.markdown(
                        f"<div class='explain-box'>"
                        f"<strong>Spiegazione per il tecnico</strong><br><br>"
                        f"{explain_obs(t2_obs, q_obs, model['T2_UCL'], model['Q_UCL'], top_t2, top_q, fn)}"
                        f"</div>", unsafe_allow_html=True
                    )

                    # Log intervento
                    st.divider()
                    st.markdown("#### 📝 Registra intervento")
                    with st.form("log_form"):
                        azione = st.text_area(
                            "Descrivi l'intervento effettuato", height=80,
                            placeholder="Es: aumentata contropressione da 80 a 95 bar"
                        )
                        if st.form_submit_button("💾 Salva nel log",
                                                 use_container_width=True):
                            if azione:
                                st.session_state.anomaly_log.append({
                                    'Ciclo': obs_choice,
                                    'T²': round(t2_obs, 3),
                                    'Q':  round(q_obs, 3),
                                    'Intervento': azione,
                                })
                                st.success("✅ Intervento registrato.")

                    if st.session_state.anomaly_log:
                        st.markdown("#### 📋 Log interventi")
                        st.dataframe(
                            pd.DataFrame(st.session_state.anomaly_log),
                            use_container_width=True
                        )


# ════════════════════════════════════
#  TAB 3 — VARIABILI & LOADINGS
# ════════════════════════════════════
with tab3:
    if st.session_state.model is None:
        st.warning("⚠️ Carica prima i dati in Phase I.")
    else:
        model = st.session_state.model
        fn    = st.session_state.feature_names
        P     = model['loadings']
        k_m   = model['k']
        evr_m = model['evr']
        lam_m = model['eigenvalues']
        T_m   = model['scores']

        # Tabella variabili
        st.markdown("### Indice variabili")
        st.dataframe(
            pd.DataFrame({'Indice': range(1, len(fn)+1), 'Variabile': fn}),
            use_container_width=True, hide_index=True
        )

        st.divider()

        # Loading plots — una curva per ogni PC
        st.markdown("### Loading plots")
        st.caption("Seleziona una PC per vedere il contributo di ogni variabile.")
        pc_sel = st.selectbox(
            "Componente principale",
            options=range(k_m),
            format_func=lambda i: f"PC{i+1} — {evr_m[i]:.2f}% varianza"
        )
        st.plotly_chart(chart_loadings(P, fn, pc_sel, evr_m),
                        use_container_width=True)

        # Top 5 per questa PC
        top5 = np.argsort(np.abs(P[:, pc_sel]))[::-1][:5]
        st.markdown("**Top 5 variabili per questa PC:**")
        st.dataframe(pd.DataFrame({
            'Indice':    top5 + 1,
            'Variabile': [fn[i] for i in top5],
            'Loading':   [round(float(P[i, pc_sel]), 4) for i in top5],
            'Direzione': ['↑ positivo' if P[i, pc_sel] > 0 else '↓ negativo'
                          for i in top5]
        }), use_container_width=True, hide_index=True)

        st.divider()

        # Score plots — tutte le coppie
        st.markdown("### Score plots — coppie di PC")
        fT2_tr = model['T2'] > model['T2_UCL']
        fQ_tr  = model['Q']  > model['Q_UCL']
        flagged_tr = fT2_tr | fQ_tr

        pairs = [(i, i+1) for i in range(0, k_m-1, 2)]
        for pc_i, pc_j in pairs:
            st.plotly_chart(
                chart_scores(T_m, lam_m, model['T2_UCL'],
                             evr_m, pc_i, pc_j, flagged_tr),
                use_container_width=True
            )
