import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f, norm
import anthropic

st.set_page_config(page_title="Process Monitor", page_icon="🏭",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
h1,h2,h3{font-family:'IBM Plex Mono',monospace;}
.llm-box{background:#f0f4ff;border:1px solid #c5d0f5;border-left:4px solid #3b5bdb;
  padding:16px 20px;border-radius:4px;font-size:14px;line-height:1.7;margin-top:12px;}
.alarm-box{background:#fff5f5;border:1px solid #ffc9c9;border-left:4px solid #e74c3c;
  padding:14px 18px;border-radius:4px;font-size:14px;margin-top:8px;}
.ok-box{background:#f0fff4;border:1px solid #b2f2bb;border-left:4px solid #2ecc71;
  padding:14px 18px;border-radius:4px;font-size:14px;margin-top:8px;}
.var-pill{display:inline-block;background:#e74c3c22;color:#c0392b;
  border:1px solid #e74c3c44;padding:2px 8px;border-radius:20px;
  font-size:12px;margin:2px;font-family:'IBM Plex Mono',monospace;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def fit_pca_spc(X_raw, k, alpha=0.95):
    n, p = X_raw.shape
    sc   = StandardScaler(); Xs = sc.fit_transform(X_raw)
    pca  = PCA(n_components=k, svd_solver='full', random_state=42)
    T    = pca.fit_transform(Xs); P = pca.components_.T; lam = pca.explained_variance_
    T2   = np.sum((T**2)/lam, axis=1)
    T2_UCL = (k*(n-1)/(n-k))*f.ppf(alpha, k, n-k)
    E    = Xs - pca.inverse_transform(T); Q = np.sum(E**2, axis=1)
    pf   = PCA(n_components=min(p,n-1), svd_solver='full', random_state=42); pf.fit(Xs)
    re   = pf.explained_variance_[k:]
    t1,t2,t3 = re.sum(),(re**2).sum(),(re**3).sum()
    h0   = 1-(2*t1*t3)/(3*t2**2); z = norm.ppf(alpha)
    Q_UCL = t1*((z*np.sqrt(2*t2*h0**2)/t1)+1+(t2*h0*(h0-1)/t1**2))**(1/h0)
    # contribution limits (±1.96σ on train)
    zc   = norm.ppf(0.975)
    mu_e = E.mean(0); sd_e = E.std(0,ddof=1)
    W    = T/lam; cT2 = Xs*(W@P.T)
    mu_c = cT2.mean(0); sd_c = cT2.std(0,ddof=1)
    return dict(
        scaler=sc, pca=pca, k=k, scores=T, loadings=P, eigenvalues=lam,
        T2=T2, Q=Q, T2_UCL=T2_UCL, Q_UCL=Q_UCL, X_scaled=Xs, E=E,
        evr=pca.explained_variance_ratio_*100, feature_names=[],
        Qcontrib_LCL=mu_e-zc*sd_e, Qcontrib_UCL=mu_e+zc*sd_e,
        T2contrib_LCL=mu_c-zc*sd_c, T2contrib_UCL=mu_c+zc*sd_c,
    )


def monitor_new(model, X_new):
    sc=model['scaler']; pca=model['pca']; lam=model['eigenvalues']
    Xns=sc.transform(X_new); Tn=pca.transform(Xns)
    T2n=np.sum((Tn**2)/lam,axis=1)
    En=Xns-pca.inverse_transform(Tn); Qn=np.sum(En**2,axis=1)
    return dict(T2=T2n, Q=Qn,
                T2_flag=T2n>model['T2_UCL'], Q_flag=Qn>model['Q_UCL'],
                Xn_s=Xns, Tn=Tn, En=En)


def compute_rmsecv(X_s, max_k, G=10):
    n,p = X_s.shape; max_k=min(max_k,p,n-1)
    idx=np.arange(n); sg=[idx[g::G] for g in range(G)]
    vg=[np.array([j]) for j in range(p)]
    press=np.zeros(max_k); rmsecv=np.zeros(max_k)
    for k in range(1,max_k+1):
        PRESS=0.0; COUNT=0
        for ti in sg:
            tri=np.setdiff1d(idx,ti); Xtr=X_s[tri]; Xte=X_s[ti]
            pc=PCA(n_components=min(k,p),svd_solver='full',random_state=42)
            pc.fit(Xtr); P=pc.components_.T
            for mc in vg:
                oc=np.setdiff1d(np.arange(p),mc); kk=min(k,P[oc,:].shape[0]-1)
                if kk<1: continue
                Th,*_=np.linalg.lstsq(P[oc,:kk],Xte[:,oc].T,rcond=None)
                Xh=(Th.T)@P[:,:kk].T; r=Xte[:,mc]-Xh[:,mc]
                PRESS+=np.sum(r**2); COUNT+=r.size
        press[k-1]=PRESS; rmsecv[k-1]=np.sqrt(PRESS/COUNT) if COUNT>0 else np.inf
    return int(np.argmin(rmsecv))+1, press, rmsecv


def iterative_cleaning(X_raw, k_clean, alpha_clean, max_iter=10):
    mask=np.ones(len(X_raw),dtype=bool); log=[]
    for it in range(1,max_iter+1):
        X_it=X_raw[mask]; n_it=len(X_it)
        sc_it=StandardScaler(); Xs_it=sc_it.fit_transform(X_it)
        k_it=min(k_clean,Xs_it.shape[1]-1,n_it-2)
        res=fit_pca_spc(X_it,k_it,alpha_clean)
        flag=(res['T2']>res['T2_UCL'])|(res['Q']>res['Q_UCL'])
        n_rem=int(flag.sum())
        log.append(dict(Iter=it,Prima=n_it,Rimossi=n_rem,Dopo=n_it-n_rem))
        if n_rem==0: break
        idx_c=np.where(mask)[0]; mask[idx_c[flag]]=False
    return mask, pd.DataFrame(log)


def llm_describe(df_X, x_cols, y_cols):
    try:
        key=st.secrets.get("ANTHROPIC_API_KEY","")
        if not key: return None,"API key mancante nei Secrets."
        desc=df_X.describe().round(3).to_string()
        miss=df_X.isnull().sum(); miss_info=miss[miss>0].to_string() if miss.any() else "Nessuno"
        corr=df_X.corr().abs(); np.fill_diagonal(corr.values,0)
        top=corr.unstack().nlargest(5)
        corr_str="\n".join(f"  {a}↔{b}: {v:.2f}" for (a,b),v in top.items())
        prompt=f"""Sei un esperto di stampaggio a iniezione e analisi dati industriali.
Dataset: {len(df_X)} osservazioni, {len(x_cols)} variabili X, {len(y_cols)} variabili Y ({', '.join(y_cols) if y_cols else 'nessuna'}).
Statistiche:\n{desc}\nValori mancanti:\n{miss_info}\nTop correlazioni:\n{corr_str}
Fornisci in italiano:
1. Panoramica generale (2 righe)
2. Variabili con alta variabilità (CV%)
3. Valori anomali o sospetti da verificare
4. Correlazioni rilevanti per la PCA
5. Raccomandazioni prima dell'analisi
Sii conciso e pratico."""
        client=anthropic.Anthropic(api_key=key)
        r=client.messages.create(model="claude-sonnet-4-20250514",max_tokens=600,
                                  messages=[{"role":"user","content":prompt}])
        return r.content[0].text, None
    except Exception as e:
        return None, str(e)


def llm_anomaly(t2, q, T2_UCL, Q_UCL, top_t2, top_q, fn):
    try:
        key=st.secrets.get("ANTHROPIC_API_KEY","")
        if not key: return None,"API key mancante."
        v_t2=[fn[i] if fn else f'Var {i+1}' for i in top_t2]
        v_q =[fn[i] if fn else f'Var {i+1}' for i in top_q]
        prompt=f"""Sei un esperto di stampaggio a iniezione.
T²={t2:.3f} (UCL={T2_UCL:.3f}, {t2/T2_UCL:.2f}×), Q={q:.3f} (UCL={Q_UCL:.3f}, {q/Q_UCL:.2f}×).
Variabili T²: {', '.join(v_t2)}. Variabili Q: {', '.join(v_q)}.
Spiega in italiano a un capoturno:
1. Cosa sta succedendo nel processo
2. Possibili cause fisiche
3. Cosa controllare / azione correttiva
Max 120 parole. Niente statistica, solo processo fisico."""
        client=anthropic.Anthropic(api_key=key)
        r=client.messages.create(model="claude-sonnet-4-20250514",max_tokens=350,
                                  messages=[{"role":"user","content":prompt}])
        return r.content[0].text, None
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════

def make_line_chart(values, ucl, title, color, flags=None):
    fig=go.Figure()
    fig.add_scatter(x=np.arange(len(values)).tolist(), y=values.tolist(),
                    mode='lines+markers', line=dict(color=color,width=1.2),
                    marker=dict(size=5,color=color),
                    hovertemplate='Ciclo %{x}<br>%{y:.3f}<extra></extra>',name='Valore')
    fig.add_hline(y=ucl,line_dash='dash',line_color='#e74c3c',line_width=1.5,
                  annotation_text=f'UCL={ucl:.3f}',annotation_position='right')
    if flags is not None and flags.any():
        fi=np.where(flags)[0]
        fig.add_scatter(x=fi.tolist(),y=values[fi].tolist(),mode='markers',
                        marker=dict(size=10,color='#e74c3c',symbol='x',line=dict(width=2)),
                        name='Anomalia',hovertemplate='Ciclo %{x}<br>%{y:.3f}<extra></extra>')
    fig.update_layout(title=dict(text=title,font=dict(family='IBM Plex Mono',size=13)),
                      xaxis_title='Ciclo',height=290,
                      margin=dict(l=10,r=10,t=40,b=30),
                      plot_bgcolor='#fafafa',paper_bgcolor='white',
                      font=dict(family='IBM Plex Sans'),
                      legend=dict(orientation='h',y=-0.25),hovermode='x unified')
    return fig


def make_contribution_chart(contrib, ucl_v, lcl_v, fn, title):
    p=len(contrib); xax=list(range(1,p+1))
    labels=fn if fn else [f'Var {i+1}' for i in range(p)]
    colors=['#e74c3c' if (contrib[i]>ucl_v[i] or contrib[i]<lcl_v[i])
            else '#2c3e50' for i in range(p)]
    fig=go.Figure()
    fig.add_bar(x=xax,y=contrib.tolist(),marker_color=colors,
                customdata=labels,
                hovertemplate='%{customdata}<br>Contributo: %{y:.4f}<extra></extra>',
                name='Contributo')
    fig.add_scatter(x=xax,y=ucl_v.tolist(),mode='lines',
                    line=dict(color='#e74c3c',dash='dash',width=1.5),
                    name='UCL',hoverinfo='skip')
    fig.add_scatter(x=xax,y=lcl_v.tolist(),mode='lines',
                    line=dict(color='#e74c3c',dash='dash',width=1.5),
                    name='LCL',hoverinfo='skip')
    fig.add_hline(y=0,line_color='black',line_width=0.8)
    fig.update_layout(title=dict(text=title,font=dict(family='IBM Plex Mono',size=12)),
                      xaxis=dict(title='Variabile (indice)',
                                 tickvals=xax,ticktext=[str(x) for x in xax]),
                      yaxis_title='Contributo (signed)',height=310,
                      margin=dict(l=10,r=10,t=40,b=30),
                      plot_bgcolor='#fafafa',paper_bgcolor='white',
                      font=dict(family='IBM Plex Sans'),
                      legend=dict(orientation='h',y=-0.3))
    exceed=[(i+1, labels[i], round(float(contrib[i]),4),
             round(float(lcl_v[i]),4), round(float(ucl_v[i]),4))
            for i in range(p) if contrib[i]>ucl_v[i] or contrib[i]<lcl_v[i]]
    return fig, exceed


def make_score_chart(T, lam, T2_UCL, evr, pc_i, pc_j, flagged):
    ang=np.linspace(0,2*np.pi,400)
    a=np.sqrt(T2_UCL*lam[pc_i]); b=np.sqrt(T2_UCL*lam[pc_j])
    ok=~flagged
    fig=go.Figure()
    fig.add_scatter(x=T[ok,pc_i].tolist(),y=T[ok,pc_j].tolist(),mode='markers',
                    marker=dict(size=5,color='#2c3e50',opacity=0.6),name='In controllo',
                    hovertemplate=f'PC{pc_i+1}: %{{x:.3f}}<br>PC{pc_j+1}: %{{y:.3f}}<extra></extra>')
    if flagged.any():
        fi=np.where(flagged)[0]
        fig.add_scatter(x=T[fi,pc_i].tolist(),y=T[fi,pc_j].tolist(),mode='markers',
                        marker=dict(size=9,color='#e74c3c',symbol='x',line=dict(width=2)),
                        name='Anomalia',
                        hovertemplate=f'PC{pc_i+1}: %{{x:.3f}}<br>PC{pc_j+1}: %{{y:.3f}}<extra></extra>')
    fig.add_scatter(x=(a*np.cos(ang)).tolist(),y=(b*np.sin(ang)).tolist(),
                    mode='lines',line=dict(color='#95a5a6',dash='dash',width=1.2),
                    name='UCL ellisse',hoverinfo='skip')
    fig.add_hline(y=0,line_color='#ecf0f1',line_width=0.8)
    fig.add_vline(x=0,line_color='#ecf0f1',line_width=0.8)
    fig.update_layout(
        title=dict(text=f'Score plot  PC{pc_i+1} ({evr[pc_i]:.1f}%)  vs  PC{pc_j+1} ({evr[pc_j]:.1f}%)',
                   font=dict(family='IBM Plex Mono',size=12)),
        xaxis_title=f'PC{pc_i+1} ({evr[pc_i]:.1f}%)',
        yaxis_title=f'PC{pc_j+1} ({evr[pc_j]:.1f}%)',
        height=380,margin=dict(l=10,r=10,t=40,b=30),
        plot_bgcolor='#fafafa',paper_bgcolor='white',
        font=dict(family='IBM Plex Sans'),legend=dict(orientation='h',y=-0.2))
    return fig


def make_loading_pair_chart(P, fn, evr, pc_i, pc_j):
    """Loading plot for a pair of PCs — PC_i on x axis, PC_j on y axis (biplot style)."""
    p=P.shape[0]; labels=fn if fn else [f'Var {i+1}' for i in range(p)]
    fig=go.Figure()
    # Arrows from origin to each variable
    for idx in range(p):
        fig.add_scatter(x=[0, float(P[idx,pc_i])],
                        y=[0, float(P[idx,pc_j])],
                        mode='lines',line=dict(color='#2980b9',width=1),
                        showlegend=False,hoverinfo='skip')
    # Labels at tip
    fig.add_scatter(x=P[:,pc_i].tolist(), y=P[:,pc_j].tolist(),
                    mode='markers+text',
                    marker=dict(size=8,color='#2980b9'),
                    text=[str(i+1) for i in range(p)],
                    textposition='top center',textfont=dict(size=9),
                    customdata=labels,
                    hovertemplate='%{customdata}<br>PC%{meta[0]}: %{x:.4f}<br>PC%{meta[1]}: %{y:.4f}<extra></extra>',
                    meta=[pc_i+1, pc_j+1],
                    name='Variabili',showlegend=False)
    fig.add_hline(y=0,line_color='#bdc3c7',line_width=0.8)
    fig.add_vline(x=0,line_color='#bdc3c7',line_width=0.8)
    fig.update_layout(
        title=dict(text=f'Loading plot  PC{pc_i+1} ({evr[pc_i]:.1f}%)  vs  PC{pc_j+1} ({evr[pc_j]:.1f}%)',
                   font=dict(family='IBM Plex Mono',size=12)),
        xaxis_title=f'Loading PC{pc_i+1}',
        yaxis_title=f'Loading PC{pc_j+1}',
        height=420,margin=dict(l=10,r=10,t=40,b=30),
        plot_bgcolor='#fafafa',paper_bgcolor='white',
        font=dict(family='IBM Plex Sans'))
    return fig


def make_pc_curve_chart(ks, values, vline_x, vline_label, xlabel, ylabel, title, color):
    fig=go.Figure()
    fig.add_scatter(x=ks,y=values,mode='lines+markers',
                    line=dict(color=color,width=2),
                    marker=dict(size=6,color=[('#e74c3c' if x==vline_x else color) for x in ks]),
                    hovertemplate=f'PC%{{x}}<br>{ylabel}: %{{y:.4f}}<extra></extra>')
    fig.add_vline(x=vline_x,line_dash='dash',line_color='#e74c3c',
                  annotation_text=vline_label,annotation_position='top right')
    fig.update_layout(title=dict(text=title,font=dict(family='IBM Plex Mono',size=13)),
                      xaxis_title=xlabel,yaxis_title=ylabel,height=300,
                      margin=dict(l=10,r=10,t=40,b=30),
                      plot_bgcolor='#fafafa',paper_bgcolor='white',
                      font=dict(family='IBM Plex Sans'),showlegend=False)
    return fig


# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
for key,val in [('model',None),('feature_names',[]),('y_names',[]),
                ('df_X',None),('df_Y',None),('k_chosen',None),
                ('mon',None),('df_test_X',None),('anomaly_log',[]),
                ('rmsecv_computed',False),('rmsecv_result',None)]:
    if key not in st.session_state:
        st.session_state[key]=val


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🏭 Process Monitor")
    st.markdown("**PCA-SPC · Injection Molding**")
    st.divider()
    alpha=st.slider("Confidenza UCL (α)",0.90,0.99,0.95,0.01)
    y_cols_raw=st.text_area("Variabili Y — una per riga",
                            placeholder="Quota cuscino\nTempo ciclo\nQuota minimo cuscino")
    y_cols=[c.strip() for c in y_cols_raw.split('\n') if c.strip()]
    excl_raw=st.text_area("Colonne da escludere — una per riga",
                          placeholder="1\ntimestamp")
    excl_cols=[c.strip() for c in excl_raw.split('\n') if c.strip()]


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
st.markdown("# 🏭 Process Monitor")
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📂 Dataset",
    "📐 Selezione PC",
    "🔧 Calibrazione",
    "📊 Loadings & Scores",
    "🔍 Monitoraggio"
])


# ════════════════════════════════════════════
#  TAB 1 — DATASET
# ════════════════════════════════════════════
with tab1:
    st.markdown("### Carica dati")
    st.caption("File CSV o Excel esportato dalla USB della pressa.")

    up=st.file_uploader("Trascina il file qui",type=['csv','xlsx','xls'],key='up_main')
    if up:
        df_raw=(pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up))
        df_raw.columns=df_raw.columns.astype(str)
        drop_c=[c for c in excl_cols if c in df_raw.columns]
        if drop_c: df_raw.drop(columns=drop_c,inplace=True)
        y_valid=[c for c in y_cols if c in df_raw.columns]
        df_num=df_raw.select_dtypes(include=[np.number]).copy()
        const_c=df_num.columns[df_num.std()==0].tolist()
        if const_c: df_num.drop(columns=const_c,inplace=True)
        x_cols=[c for c in df_num.columns if c not in y_valid]
        df_X=df_num[x_cols].copy()
        df_Y=df_num[y_valid].copy() if y_valid else pd.DataFrame()

        st.session_state.df_X=df_X.copy()
        st.session_state.df_Y=df_Y.copy()
        st.session_state.feature_names=x_cols
        st.session_state.y_names=y_valid

        # KPI
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Cicli",len(df_X))
        c2.metric("Variabili X",len(x_cols))
        c3.metric("Variabili Y",len(y_valid) if y_valid else "—")
        miss=int(df_X.isnull().sum().sum())
        c4.metric("Valori mancanti",miss,
                  delta="⚠️" if miss>0 else "✅ nessuno",delta_color="off")

        # Indice variabili
        st.markdown("#### Indice variabili")
        col_a,col_b=st.columns(2)
        with col_a:
            st.markdown("**X — Variabili di processo**")
            st.dataframe(pd.DataFrame({'Indice':range(1,len(x_cols)+1),'Variabile':x_cols}),
                         use_container_width=True,hide_index=True)
        with col_b:
            if y_valid:
                st.markdown("**Y — Variabili di qualità**")
                st.dataframe(pd.DataFrame({'Variabile':y_valid}),
                             use_container_width=True,hide_index=True)

        # Statistiche descrittive
        st.markdown("#### Statistiche descrittive")
        desc=df_X.describe().T.round(3)
        desc['cv%']=(desc['std']/desc['mean'].abs()*100).round(1)
        st.dataframe(desc,use_container_width=True)

        # Valori mancanti
        if miss>0:
            st.markdown("#### Valori mancanti")
            miss_df=df_X.isnull().sum().reset_index()
            miss_df.columns=['Variabile','N mancanti']
            miss_df=miss_df[miss_df['N mancanti']>0]
            miss_df['%']=(miss_df['N mancanti']/len(df_X)*100).round(1)
            st.dataframe(miss_df,use_container_width=True,hide_index=True)
            st.info("ℹ️ I valori mancanti verranno sostituiti con la media della colonna.")

        # LLM
        st.markdown("#### Descrizione automatica con AI")
        if st.button("🤖 Genera descrizione dataset",key='llm_ds'):
            df_X_filled=df_X.fillna(df_X.mean())
            with st.spinner("Analisi in corso..."):
                txt,err=llm_describe(df_X_filled,x_cols,y_valid)
            if err: st.error(f"Errore: {err}")
            else:
                st.markdown(f"<div class='llm-box'>{txt.replace(chr(10),'<br>')}</div>",
                            unsafe_allow_html=True)
        st.caption("Richiede ANTHROPIC_API_KEY nei Secrets di Streamlit Cloud.")


# ════════════════════════════════════════════
#  TAB 2 — SELEZIONE PC
# ════════════════════════════════════════════
with tab2:
    if st.session_state.df_X is None:
        st.info("⬆️ Carica prima il dataset nella tab Dataset.")
    else:
        df_X=st.session_state.df_X.fillna(st.session_state.df_X.mean())
        X_raw=df_X.values
        sc_tmp=StandardScaler(); Xs_tmp=sc_tmp.fit_transform(X_raw)
        n_obs,n_vars=Xs_tmp.shape; max_k=min(20,n_vars-1,n_obs-2)
        pca_tmp=PCA(n_components=max_k,svd_solver='full',random_state=42)
        pca_tmp.fit(Xs_tmp)
        eigs=pca_tmp.explained_variance_
        evr_all=pca_tmp.explained_variance_ratio_*100
        cum=np.cumsum(evr_all); ks=list(range(1,max_k+1))
        k_kaiser=max(2,min(int(np.sum(eigs>1)),max_k))
        k_90=int(np.argmax(cum>=90.0))+1
        k_95=int(np.argmax(cum>=95.0))+1

        st.markdown("### Seleziona il criterio da visualizzare")
        grafico=st.radio("",
            ["Scree plot (Kaiser)","Varianza cumulativa","RMSECV"],
            horizontal=True,key='pc_radio',label_visibility='collapsed')

        if grafico=="Scree plot (Kaiser)":
            fig=make_pc_curve_chart(ks,evr_all.tolist(),k_kaiser,
                                    f'Kaiser k={k_kaiser}','PC',
                                    'Varianza spiegata (%)','Scree plot','#2c3e50')
            fig.add_hline(y=float(100/n_vars),line_dash='dot',line_color='#7f8c8d',
                         annotation_text=f'Soglia Kaiser ({100/n_vars:.1f}%)',
                         annotation_position='right')
            st.plotly_chart(fig,use_container_width=True,key='chart_scree')
            st.info(f"💡 Kaiser suggerisce **k = {k_kaiser}** PC")

        elif grafico=="Varianza cumulativa":
            fig=go.Figure()
            fig.add_scatter(x=ks,y=cum.tolist(),mode='lines+markers',
                            line=dict(color='#2980b9',width=2),marker=dict(size=6),
                            hovertemplate='PC%{x}<br>Cumulativa: %{y:.1f}%<extra></extra>')
            fig.add_hline(y=90,line_dash='dash',line_color='#f39c12',
                         annotation_text='90%',annotation_position='right')
            fig.add_hline(y=95,line_dash='dash',line_color='#e74c3c',
                         annotation_text='95%',annotation_position='right')
            fig.add_vline(x=k_90,line_dash='dot',line_color='#f39c12',
                         annotation_text=f'k={k_90}',annotation_position='top left')
            fig.add_vline(x=k_95,line_dash='dot',line_color='#e74c3c',
                         annotation_text=f'k={k_95}',annotation_position='top right')
            fig.update_layout(xaxis_title='Numero PC',yaxis_title='Varianza cumulativa (%)',
                             yaxis=dict(range=[0,105]),height=300,
                             margin=dict(l=10,r=10,t=20,b=30),
                             plot_bgcolor='#fafafa',paper_bgcolor='white',
                             font=dict(family='IBM Plex Sans'),showlegend=False)
            st.plotly_chart(fig,use_container_width=True,key='chart_cumvar')
            st.info(f"💡 90% varianza → **k = {k_90}**  |  95% varianza → **k = {k_95}**")

        else:  # RMSECV
            if not st.session_state.rmsecv_computed:
                if st.button("▶ Calcola RMSECV",key='btn_rmsecv',type='primary'):
                    with st.spinner("RMSECV in corso... (può richiedere qualche minuto)"):
                        bk,_,rcv=compute_rmsecv(Xs_tmp,max_k)
                    st.session_state.rmsecv_result=(bk,rcv)
                    st.session_state.rmsecv_computed=True
                    st.rerun()
                else:
                    st.info("Clicca il pulsante per calcolare il RMSECV.")
            else:
                bk,rcv=st.session_state.rmsecv_result
                fig=make_pc_curve_chart(ks,rcv.tolist(),bk,
                                        f'Minimo k={bk}','Numero PC',
                                        'RMSECV','RMSECV — errore di ricostruzione in cross-validation','#16a085')
                st.plotly_chart(fig,use_container_width=True,key='chart_rmsecv')
                st.info(f"💡 RMSECV minimo → **k = {bk}** PC")
                if st.button("🔄 Ricalcola",key='btn_rmsecv_reset'):
                    st.session_state.rmsecv_computed=False
                    st.rerun()

        # Riepilogo e scelta
        st.markdown("---")
        st.markdown("**Riepilogo suggerimenti:**")
        rs1,rs2,rs3=st.columns(3)
        rs1.metric("Kaiser",f"k = {k_kaiser}")
        rs2.metric("90% var",f"k = {k_90}")
        if st.session_state.rmsecv_computed and st.session_state.rmsecv_result:
            rs3.metric("RMSECV",f"k = {st.session_state.rmsecv_result[0]}")
        else:
            rs3.metric("RMSECV","— (calcola sopra)")

        k_chosen=st.number_input("Numero di PC da usare nel modello",
                                  min_value=2,max_value=max_k,
                                  value=k_kaiser,step=1,key='k_input')
        st.session_state.k_chosen=int(k_chosen)
        st.success(f"✅ k = **{k_chosen}** PC — varianza spiegata: **{cum[k_chosen-1]:.1f}%**")


# ════════════════════════════════════════════
#  TAB 3 — CALIBRAZIONE
# ════════════════════════════════════════════
with tab3:
    if st.session_state.df_X is None:
        st.info("⬆️ Carica il dataset nella tab Dataset.")
    elif st.session_state.k_chosen is None:
        st.info("⬆️ Scegli il numero di PC nella tab Selezione PC.")
    else:
        df_X=st.session_state.df_X.fillna(st.session_state.df_X.mean())
        x_cols=st.session_state.feature_names
        X_raw=df_X.values; k_use=st.session_state.k_chosen

        # Pulizia
        st.markdown("### 🧹 Pulizia dati (opzionale)")
        st.caption("Rimuove cicli anomali dal set di calibrazione prima di fittare il modello. "
                   "Utile quando non sei sicuro che tutti i cicli fossero stabili.")
        use_clean=st.toggle("Attiva pulizia iterativa",value=False,key='toggle_clean')
        clean_mask=np.ones(len(X_raw),dtype=bool)

        if use_clean:
            alpha_clean=st.slider("Confidenza pulizia",0.95,0.999,0.99,0.001,
                                   format="%.3f",
                                   help="0.99 = rimuove solo cicli chiaramente anomali")
            col_btn,col_note=st.columns([1,3])
            with col_btn:
                run_clean=st.button("▶ Esegui pulizia",key='btn_clean')
            if run_clean:
                sc_k=StandardScaler(); Xs_k=sc_k.fit_transform(X_raw)
                eig_k=PCA(svd_solver='full').fit(Xs_k).explained_variance_
                k_cl=max(2,min(int(np.sum(eig_k>1)),X_raw.shape[1]-1,X_raw.shape[0]-2))
                with st.spinner("Pulizia iterativa in corso..."):
                    clean_mask,log_df=iterative_cleaning(X_raw,k_cl,alpha_clean)
                n_rem=int((~clean_mask).sum())
                pct=n_rem/len(X_raw)*100
                st.success(f"✅ Rimossi **{n_rem}** cicli ({pct:.1f}%) · Rimasti: **{clean_mask.sum()}**")
                st.dataframe(log_df,use_container_width=True,hide_index=True)
                if pct>20:
                    st.warning("⚠️ >20% rimossi — il periodo di calibrazione potrebbe "
                               "non essere stato stabile.")

        # Fit
        st.markdown("### 🔧 Costruisci modello")
        if st.button("Costruisci modello Phase I",type="primary",
                     use_container_width=True,key='btn_fit'):
            X_clean=X_raw[clean_mask]
            with st.spinner("Fitting PCA-SPC..."):
                model=fit_pca_spc(X_clean,k_use,alpha)
                model['feature_names']=x_cols
                st.session_state.model=model

            st.success("✅ Modello costruito.")
            cm1,cm2,cm3=st.columns(3)
            cm1.metric("T² UCL",f"{model['T2_UCL']:.3f}")
            cm2.metric("Q UCL",f"{model['Q_UCL']:.3f}")
            n_flag=int(((model['T2']>model['T2_UCL'])|(model['Q']>model['Q_UCL'])).sum())
            cm3.metric("Flag residui Phase I",
                       f"{n_flag} ({n_flag/len(X_clean)*100:.1f}%)",
                       delta="🟢 OK" if n_flag/len(X_clean)<0.05 else "⚠️ Verifica",
                       delta_color="off")

            fT2=model['T2']>model['T2_UCL']
            fQ =model['Q'] >model['Q_UCL']
            st.plotly_chart(make_line_chart(model['T2'],model['T2_UCL'],
                                            'Phase I — Hotelling T²','#2c3e50',fT2),
                            use_container_width=True,key='p1_t2')
            st.plotly_chart(make_line_chart(model['Q'],model['Q_UCL'],
                                            'Phase I — Q (SPE)','#16a085',fQ),
                            use_container_width=True,key='p1_q')

        elif st.session_state.model is not None:
            model=st.session_state.model
            st.info(f"Modello già costruito — k={model['k']} PC | "
                    f"T² UCL={model['T2_UCL']:.3f} | Q UCL={model['Q_UCL']:.3f}")
            if st.button("🔄 Ricostruisci modello",key='btn_refit'):
                st.session_state.model=None
                st.rerun()


# ════════════════════════════════════════════
#  TAB 4 — LOADINGS & SCORES
# ════════════════════════════════════════════
with tab4:
    if st.session_state.model is None:
        st.info("⬆️ Costruisci prima il modello nella tab Calibrazione.")
    else:
        model=st.session_state.model
        fn=model['feature_names']; P=model['loadings']
        k_m=model['k']; evr_m=model['evr']
        lam_m=model['eigenvalues']; T_m=model['scores']
        flag_m=(model['T2']>model['T2_UCL'])|(model['Q']>model['Q_UCL'])

        # Indice variabili sempre visibile
        with st.expander("📋 Indice variabili (riferimento per i grafici)", expanded=False):
            st.dataframe(
                pd.DataFrame({'Indice': range(1, len(fn)+1), 'Variabile': fn}),
                use_container_width=True, hide_index=True
            )

        # Tutte le combinazioni possibili di coppie PC
        from itertools import combinations
        all_pairs = list(combinations(range(k_m), 2))
        pair_labels = [
            f"PC{a+1} vs PC{b+1}  ({evr_m[a]:.1f}% + {evr_m[b]:.1f}%)"
            for a, b in all_pairs
        ]

        st.markdown("### Seleziona il grafico da visualizzare")
        tipo = st.radio("Tipo di grafico", ["Loading plot", "Score plot"],
                        horizontal=True, key='ls_tipo')

        coppia_idx = st.selectbox(
            "Coppia di PC",
            options=range(len(all_pairs)),
            format_func=lambda i: pair_labels[i],
            key='ls_coppia'
        )
        pc_i, pc_j = all_pairs[coppia_idx]

        if tipo == "Loading plot":
            p_count = P.shape[0]
            labels  = fn if fn else [f'Var {i+1}' for i in range(p_count)]

            fig_load = go.Figure()

            # Solo punti con numero indice come etichetta
            fig_load.add_scatter(
                x=P[:, pc_i].tolist(),
                y=P[:, pc_j].tolist(),
                mode='markers+text',
                marker=dict(size=9, color='#2980b9',
                            line=dict(color='white', width=1)),
                text=[str(i+1) for i in range(p_count)],
                textposition='top center',
                textfont=dict(size=9, color='#1a1a2e'),
                hovertemplate=(
                    '<b>%{customdata}</b><br>'
                    f'PC{pc_i+1}: %{{x:.4f}}<br>'
                    f'PC{pc_j+1}: %{{y:.4f}}<extra></extra>'
                ),
                customdata=labels,
                name='Variabili', showlegend=False
            )

            fig_load.add_hline(y=0, line_color='#bdc3c7', line_width=0.8)
            fig_load.add_vline(x=0, line_color='#bdc3c7', line_width=0.8)
            fig_load.update_layout(
                title=dict(
                    text=(f'Loading plot  PC{pc_i+1} ({evr_m[pc_i]:.1f}%)'
                          f'  vs  PC{pc_j+1} ({evr_m[pc_j]:.1f}%)'),
                    font=dict(family='IBM Plex Mono', size=12)
                ),
                xaxis_title=f'PC{pc_i+1} — loading ({evr_m[pc_i]:.1f}% var)',
                yaxis_title=f'PC{pc_j+1} — loading ({evr_m[pc_j]:.1f}% var)',
                height=500,
                margin=dict(l=10, r=10, t=50, b=30),
                plot_bgcolor='#fafafa', paper_bgcolor='white',
                font=dict(family='IBM Plex Sans')
            )
            st.plotly_chart(fig_load, use_container_width=True, key='load_chart')

            st.caption(
                "💡 Il numero sul punto corrisponde all'indice variabile nella tabella sotto. "
                "Hover sul punto per vedere il nome completo. "
                "Variabili vicine = correlate | Variabili opposte = correlate negativamente."
            )

            # Tabella loadings ordinata per distanza dall'origine
            st.markdown("**Tabella variabili — ordinate per influenza su questa coppia di PC:**")
            df_load = pd.DataFrame({
                'Indice':              range(1, p_count+1),
                'Variabile':           labels,
                f'Loading PC{pc_i+1}': P[:, pc_i].round(4),
                f'Loading PC{pc_j+1}': P[:, pc_j].round(4),
                'Distanza origine':    np.sqrt(P[:, pc_i]**2 + P[:, pc_j]**2).round(4),
            }).sort_values('Distanza origine', ascending=False)
            st.dataframe(df_load, use_container_width=True, hide_index=True)

        else:  # Score plot
            st.plotly_chart(
                make_score_chart(T_m, lam_m, model['T2_UCL'],
                                 evr_m, pc_i, pc_j, flag_m),
                use_container_width=True, key='score_chart'
            )
            n_flag = int(flag_m.sum())
            st.caption(f"Punti rossi: {n_flag} cicli anomali su {len(T_m)} totali")


# ════════════════════════════════════════════
#  TAB 5 — MONITORAGGIO
# ════════════════════════════════════════════
with tab5:
    if st.session_state.model is None:
        st.info("⬆️ Costruisci prima il modello nella tab Calibrazione.")
    else:
        model = st.session_state.model
        fn    = model['feature_names']

        st.markdown("### Carica nuovi dati dalla USB")
        up_test = st.file_uploader("Trascina il file qui",
                                   type=['csv','xlsx','xls'], key='up_test')

        if up_test:
            df_tr = (pd.read_csv(up_test) if up_test.name.endswith('.csv')
                     else pd.read_excel(up_test))
            df_tr.columns = df_tr.columns.astype(str)
            miss_c = [c for c in fn if c not in df_tr.columns]
            if miss_c:
                st.error(f"Colonne mancanti nel file: {miss_c}")
            else:
                df_test_X = df_tr[fn].copy().fillna(df_tr[fn].mean())
                mon = monitor_new(model, df_test_X.values)
                st.session_state.mon       = mon
                st.session_state.df_test_X = df_test_X

                n_test = len(df_test_X)
                n_t2   = int(mon['T2_flag'].sum())
                n_q    = int(mon['Q_flag'].sum())
                n_any  = int((mon['T2_flag'] | mon['Q_flag']).sum())
                pct    = n_any / n_test * 100

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Cicli totali", n_test)
                c2.metric("T² anomali",  f"{n_t2} ({n_t2/n_test*100:.1f}%)")
                c3.metric("Q anomali",   f"{n_q} ({n_q/n_test*100:.1f}%)")
                stato = ("🟢 STABILE" if pct < 5
                         else "🟡 ATTENZIONE" if pct < 15
                         else "🔴 ANOMALIE")
                c4.metric("Stato processo", stato)

                # ── Control charts ────────────────────────────
                st.markdown("### Control charts")
                st.caption(
                    "Clicca un punto anomalo (rosso) sul grafico per analizzarlo, "
                    "oppure selezionalo dalla tabella sottostante."
                )

                # T² chart
                fig_t2 = make_line_chart(mon['T2'], model['T2_UCL'],
                                         'Phase II — Hotelling T²',
                                         '#2980b9', mon['T2_flag'])
                ev_t2 = st.plotly_chart(fig_t2, use_container_width=True,
                                        key='p2_t2', on_select='rerun',
                                        selection_mode='points')

                # Q chart
                fig_q = make_line_chart(mon['Q'], model['Q_UCL'],
                                        'Phase II — Q (SPE)',
                                        '#27ae60', mon['Q_flag'])
                ev_q = st.plotly_chart(fig_q, use_container_width=True,
                                       key='p2_q', on_select='rerun',
                                       selection_mode='points')

                # Leggi ciclo cliccato — cerca in entrambi i grafici
                clicked_cycle = None
                try:
                    pts_t2 = (ev_t2.selection.points
                              if ev_t2 and hasattr(ev_t2, 'selection')
                              and ev_t2.selection else [])
                    pts_q  = (ev_q.selection.points
                              if ev_q  and hasattr(ev_q,  'selection')
                              and ev_q.selection  else [])
                    if pts_t2:
                        clicked_cycle = int(pts_t2[0]['x'])
                    elif pts_q:
                        clicked_cycle = int(pts_q[0]['x'])
                except Exception:
                    clicked_cycle = None

                # ── Tabella anomalie ──────────────────────────
                flagged_idx = np.where(mon['T2_flag'] | mon['Q_flag'])[0]

                if len(flagged_idx) == 0:
                    st.markdown(
                        "<div class='ok-box'>✅ Nessuna anomalia — processo in controllo.</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"### 🔍 Cicli anomali — {len(flagged_idx)} fuori controllo")
                    st.caption(
                        "Clicca un punto sul grafico oppure seleziona un ciclo dalla tabella."
                    )

                    # Tabella riassuntiva di tutti i cicli anomali
                    df_anom = pd.DataFrame({
                        'Ciclo':       flagged_idx,
                        'T²':          mon['T2'][flagged_idx].round(3),
                        'T²/UCL':      (mon['T2'][flagged_idx]/model['T2_UCL']).round(2),
                        'Q':           mon['Q'][flagged_idx].round(3),
                        'Q/UCL':       (mon['Q'][flagged_idx]/model['Q_UCL']).round(2),
                        'T² flag':     mon['T2_flag'][flagged_idx],
                        'Q flag':      mon['Q_flag'][flagged_idx],
                        'Severità':    np.maximum(
                            mon['T2'][flagged_idx]/model['T2_UCL'],
                            mon['Q'][flagged_idx]/model['Q_UCL']
                        ).round(2),
                    }).sort_values('Severità', ascending=False).reset_index(drop=True)

                    st.dataframe(df_anom, use_container_width=True, hide_index=True)

                    # Determina ciclo da analizzare:
                    # priorità al click sul grafico, altrimenti primo della tabella
                    if clicked_cycle is not None and clicked_cycle in flagged_idx:
                        default_idx = list(flagged_idx).index(clicked_cycle)
                    else:
                        default_idx = 0

                    obs_choice = st.selectbox(
                        "Ciclo selezionato per l'analisi",
                        options=flagged_idx.tolist(),
                        index=default_idx,
                        format_func=lambda x: (
                            f"Ciclo {x}  —  "
                            f"T²: {mon['T2'][x]:.2f} ({mon['T2'][x]/model['T2_UCL']:.1f}×UCL)  |  "
                            f"Q: {mon['Q'][x]:.2f} ({mon['Q'][x]/model['Q_UCL']:.1f}×UCL)"
                        ),
                        key='obs_sel'
                    )

                    t2_obs = float(mon['T2'][obs_choice])
                    q_obs  = float(mon['Q'][obs_choice])
                    ratio  = max(t2_obs/model['T2_UCL'], q_obs/model['Q_UCL'])

                    st.markdown(
                        f"<div class='alarm-box'>"
                        f"<strong>Ciclo {obs_choice} — "
                        f"{'🔴 ANOMALIA' if ratio>=1.5 else '⚠️ ATTENZIONE'}</strong><br>"
                        f"T² = {t2_obs:.3f} ({t2_obs/model['T2_UCL']:.2f}× UCL) &nbsp;|&nbsp; "
                        f"Q = {q_obs:.3f} ({q_obs/model['Q_UCL']:.2f}× UCL)"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    # ── Contribution plots ────────────────────
                    st.markdown("#### Contribution plots")

                    c_t2v = (mon['Xn_s'][obs_choice] *
                             (model['loadings'] @
                              (mon['Tn'][obs_choice] / model['eigenvalues'])))
                    c_qv  = mon['En'][obs_choice]
                    top_t2 = np.argsort(np.abs(c_t2v))[::-1][:3].tolist()
                    top_q  = np.argsort(np.abs(c_qv))[::-1][:3].tolist()

                    # Indice variabili sopra i grafici
                    with st.expander("📋 Indice variabili"):
                        st.dataframe(
                            pd.DataFrame({'Indice': range(1,len(fn)+1), 'Variabile': fn}),
                            use_container_width=True, hide_index=True
                        )

                    col_l, col_r = st.columns(2)

                    with col_l:
                        st.markdown(f"**T² Contribution — Ciclo {obs_choice}**")
                        fig_ct2, exc_t2 = make_contribution_chart(
                            c_t2v, model['T2contrib_UCL'], model['T2contrib_LCL'],
                            fn, f'T² Contribution — Ciclo {obs_choice}'
                        )
                        st.plotly_chart(fig_ct2, use_container_width=True,
                                        key=f'ct2_{obs_choice}')

                        # Tabella completa UCL/LCL per ogni variabile
                        p_count = len(fn)
                        df_ct2 = pd.DataFrame({
                            'Idx':       range(1, p_count+1),
                            'Variabile': fn,
                            'Contributo': c_t2v.round(4),
                            'LCL':        model['T2contrib_LCL'].round(4),
                            'UCL':        model['T2contrib_UCL'].round(4),
                            'Fuori':      ['🔴 SÌ' if (c_t2v[i]>model['T2contrib_UCL'][i]
                                                        or c_t2v[i]<model['T2contrib_LCL'][i])
                                           else '✅' for i in range(p_count)]
                        })
                        with st.expander("Tabella completa T² — UCL/LCL per variabile"):
                            st.dataframe(df_ct2, use_container_width=True, hide_index=True)

                        if exc_t2:
                            st.markdown("**Variabili fuori limite:**")
                            st.dataframe(
                                pd.DataFrame(exc_t2, columns=['Idx','Variabile','Valore','LCL','UCL']),
                                use_container_width=True, hide_index=True
                            )

                    with col_r:
                        st.markdown(f"**Q Contribution — Ciclo {obs_choice}**")
                        fig_cq, exc_q = make_contribution_chart(
                            c_qv, model['Qcontrib_UCL'], model['Qcontrib_LCL'],
                            fn, f'Q Contribution — Ciclo {obs_choice}'
                        )
                        st.plotly_chart(fig_cq, use_container_width=True,
                                        key=f'cq_{obs_choice}')

                        df_cq = pd.DataFrame({
                            'Idx':       range(1, p_count+1),
                            'Variabile': fn,
                            'Contributo': c_qv.round(4),
                            'LCL':        model['Qcontrib_LCL'].round(4),
                            'UCL':        model['Qcontrib_UCL'].round(4),
                            'Fuori':      ['🔴 SÌ' if (c_qv[i]>model['Qcontrib_UCL'][i]
                                                        or c_qv[i]<model['Qcontrib_LCL'][i])
                                           else '✅' for i in range(p_count)]
                        })
                        with st.expander("Tabella completa Q — UCL/LCL per variabile"):
                            st.dataframe(df_cq, use_container_width=True, hide_index=True)

                        if exc_q:
                            st.markdown("**Variabili fuori limite:**")
                            st.dataframe(
                                pd.DataFrame(exc_q, columns=['Idx','Variabile','Valore','LCL','UCL']),
                                use_container_width=True, hide_index=True
                            )

                    # ── LLM spiegazione ───────────────────────
                    st.markdown("#### 🤖 Spiegazione per il tecnico")
                    if st.button("Genera spiegazione con AI",
                                 key=f'llm_an_{obs_choice}'):
                        with st.spinner("Analisi in corso..."):
                            txt_an, err_an = llm_anomaly(
                                t2_obs, q_obs,
                                model['T2_UCL'], model['Q_UCL'],
                                top_t2, top_q, fn
                            )
                        if err_an:
                            st.error(f"Errore: {err_an}")
                        else:
                            st.markdown(
                                f"<div class='llm-box'>"
                                f"{txt_an.replace(chr(10),'<br>')}"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                    # ── Log intervento ────────────────────────
                    st.markdown("#### 📝 Registra intervento")
                    with st.form(f"log_{obs_choice}"):
                        azione = st.text_area(
                            "Descrivi l'azione correttiva", height=70,
                            placeholder="Es: aumentata contropressione da 80 a 95 bar"
                        )
                        if st.form_submit_button("💾 Salva nel log",
                                                 use_container_width=True):
                            if azione:
                                st.session_state.anomaly_log.append({
                                    'Ciclo':      obs_choice,
                                    'T²':         round(t2_obs, 3),
                                    'Q':          round(q_obs, 3),
                                    'Severità':   f"{ratio:.2f}×UCL",
                                    'Intervento': azione
                                })
                                st.success("✅ Salvato.")

                    # Log completo
                    if st.session_state.anomaly_log:
                        st.markdown("#### 📋 Log interventi")
                        st.dataframe(
                            pd.DataFrame(st.session_state.anomaly_log),
                            use_container_width=True
                        )
