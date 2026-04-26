import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f, norm
import google.generativeai as genai

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
.ai-btn{background:#f0f4ff;border:1px solid #c5d0f5;color:#3b5bdb;
  padding:8px 16px;border-radius:6px;font-size:13px;cursor:pointer;
  font-family:'IBM Plex Mono',monospace;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  GEMINI API HELPER
# ══════════════════════════════════════════════════════════════

def call_gemini(prompt, temperature=0.3):
    """Call Gemini Flash (free tier) and return text response."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            return None, "GEMINI_API_KEY non configurata nei Secrets di Streamlit."
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config=genai.GenerationConfig(
                temperature=temperature, max_output_tokens=600
            )
        )
        response = model.generate_content(prompt)
        return response.text, None
    except Exception as e:
        return None, str(e)


def llm_button(label, prompt, key):
    """Reusable AI explanation button — renders button + result inline."""
    if st.button(f"🤖 {label}", key=key):
        with st.spinner("Analisi AI in corso..."):
            txt, err = call_gemini(prompt)
        if err:
            st.error(f"Errore Gemini: {err}")
        else:
            st.markdown(
                f"<div class='llm-box'>{txt.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True
            )
    st.caption("Powered by Google Gemini 1.5 Flash (gratuito)")


# ══════════════════════════════════════════════════════════════
#  CORE PCA-SPC FUNCTIONS
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
    zc   = norm.ppf(0.975)
    mu_e = E.mean(0); sd_e = E.std(0,ddof=1)
    W    = T/lam; cT2 = Xs*(W@P.T); mu_c = cT2.mean(0); sd_c = cT2.std(0,ddof=1)
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
    return dict(T2=T2n,Q=Qn,
                T2_flag=T2n>model['T2_UCL'],Q_flag=Qn>model['Q_UCL'],
                Xn_s=Xns,Tn=Tn,En=En)


def compute_rmsecv(X_s, max_k, G=10):
    n,p=X_s.shape; max_k=min(max_k,p,n-1)
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
    return int(np.argmin(rmsecv))+1,press,rmsecv


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
    return mask,pd.DataFrame(log)


# ══════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════

def make_line_chart(values, ucl, title, color, flags=None):
    fig=go.Figure()
    fig.add_scatter(x=np.arange(len(values)).tolist(),y=values.tolist(),
                    mode='lines+markers',line=dict(color=color,width=1.2),
                    marker=dict(size=5,color=color),
                    hovertemplate='Ciclo %{x}<br>%{y:.3f}<extra></extra>',name='Valore')
    fig.add_hline(y=ucl,line_dash='dash',line_color='#e74c3c',line_width=1.5,
                  annotation_text=f'UCL={ucl:.3f}',annotation_position='right')
    if flags is not None and flags.any():
        fi=np.where(flags)[0]
        fig.add_scatter(x=fi.tolist(),y=values[fi].tolist(),mode='markers',
                        marker=dict(size=10,color='#e74c3c',symbol='x',line=dict(width=2)),
                        name='Anomalia',
                        hovertemplate='Ciclo %{x}<br>%{y:.3f}<extra></extra>')
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
                hovertemplate='[%{x}] %{customdata}<br>Contributo: %{y:.4f}<extra></extra>',
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
                      yaxis_title='Contributo (signed)',height=320,
                      margin=dict(l=10,r=10,t=40,b=30),
                      plot_bgcolor='#fafafa',paper_bgcolor='white',
                      font=dict(family='IBM Plex Sans'),
                      legend=dict(orientation='h',y=-0.3))
    exceed=[(i+1,labels[i],round(float(contrib[i]),4),
             round(float(lcl_v[i]),4),round(float(ucl_v[i]),4))
            for i in range(p) if contrib[i]>ucl_v[i] or contrib[i]<lcl_v[i]]
    return fig,exceed


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
        title=dict(text=f'Score plot PC{pc_i+1} ({evr[pc_i]:.1f}%) vs PC{pc_j+1} ({evr[pc_j]:.1f}%)',
                   font=dict(family='IBM Plex Mono',size=12)),
        xaxis_title=f'PC{pc_i+1} ({evr[pc_i]:.1f}%)',
        yaxis_title=f'PC{pc_j+1} ({evr[pc_j]:.1f}%)',
        height=380,margin=dict(l=10,r=10,t=40,b=30),
        plot_bgcolor='#fafafa',paper_bgcolor='white',
        font=dict(family='IBM Plex Sans'),legend=dict(orientation='h',y=-0.2))
    return fig


# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
for key,val in [('model',None),('feature_names',[]),('y_names',[]),
                ('df_X',None),('df_Y',None),('k_chosen',None),
                ('mon',None),('df_test_X',None),('anomaly_log',[]),
                ('rmsecv_computed',False),('rmsecv_result',None),
                ('selected_cycle',None)]:
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
    st.divider()
    st.markdown("**AI (Gemini)**")
    st.caption("Aggiungi GEMINI_API_KEY nei Secrets di Streamlit Cloud per abilitare le spiegazioni AI.")


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
st.markdown("# 🏭 Process Monitor")
tab1,tab2,tab3,tab4,tab5=st.tabs([
    "📂 Dataset","📐 Selezione PC","🔧 Calibrazione",
    "📊 Loadings & Scores","🔍 Monitoraggio"])


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

        c1,c2,c3,c4=st.columns(4)
        c1.metric("Cicli",len(df_X))
        c2.metric("Variabili X",len(x_cols))
        c3.metric("Variabili Y",len(y_valid) if y_valid else "—")
        miss=int(df_X.isnull().sum().sum())
        c4.metric("Valori mancanti",miss,
                  delta="⚠️" if miss>0 else "✅ nessuno",delta_color="off")

        col_a,col_b=st.columns(2)
        with col_a:
            st.markdown("**Variabili X (processo)**")
            st.dataframe(pd.DataFrame({'Indice':range(1,len(x_cols)+1),'Variabile':x_cols}),
                         use_container_width=True,hide_index=True)
        with col_b:
            if y_valid:
                st.markdown("**Variabili Y (qualità)**")
                st.dataframe(pd.DataFrame({'Variabile':y_valid}),
                             use_container_width=True,hide_index=True)

        st.markdown("#### Statistiche descrittive")
        desc=df_X.describe().T.round(3)
        desc['cv%']=(desc['std']/desc['mean'].abs()*100).round(1)
        st.dataframe(desc,use_container_width=True)

        if miss>0:
            st.markdown("#### Valori mancanti")
            miss_df=df_X.isnull().sum().reset_index()
            miss_df.columns=['Variabile','N']
            miss_df=miss_df[miss_df['N']>0]
            miss_df['%']=(miss_df['N']/len(df_X)*100).round(1)
            st.dataframe(miss_df,use_container_width=True,hide_index=True)
            st.info("ℹ️ Valori mancanti sostituiti con la media della colonna.")

        # AI — descrizione dataset
        st.markdown("---")
        st.markdown("#### 🤖 Analisi AI del dataset")
        desc_str=df_X.fillna(df_X.mean()).describe().round(3).to_string()
        miss_str=df_X.isnull().sum().to_string()
        corr=df_X.corr().abs()
        corr_np=corr.values.copy(); np.fill_diagonal(corr_np,0)
        corr_filled=pd.DataFrame(corr_np,index=corr.index,columns=corr.columns)
        top_corr="\n".join(f"  {a}↔{b}: {v:.2f}"
                           for (a,b),v in corr_filled.unstack().nlargest(5).items())
        prompt_ds=f"""Sei un esperto di stampaggio a iniezione e analisi dati industriali.
Dataset: {len(df_X)} cicli, {len(x_cols)} variabili X, {len(y_valid)} variabili Y ({', '.join(y_valid) if y_valid else 'nessuna'}).
Statistiche:\n{desc_str}\nValori mancanti:\n{miss_str}\nTop correlazioni:\n{top_corr}
Fornisci in italiano:
1. Panoramica generale (2 righe)
2. Variabili con alta variabilità (CV% elevato)
3. Valori anomali o sospetti
4. Correlazioni rilevanti per la PCA
5. Raccomandazioni prima dell'analisi
Sii conciso e pratico. Max 200 parole."""
        llm_button("Genera descrizione dataset", prompt_ds, key='ai_ds')


# ════════════════════════════════════════════
#  TAB 2 — SELEZIONE PC
# ════════════════════════════════════════════
with tab2:
    if st.session_state.df_X is None:
        st.info("⬆️ Carica prima il dataset.")
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
        grafico=st.radio("",["Scree plot","Varianza cumulativa","RMSECV"],
                         horizontal=True,key='pc_radio',label_visibility='collapsed')

        if grafico=="Scree plot":
            fig=go.Figure()
            fig.add_scatter(x=ks,y=evr_all.tolist(),mode='lines+markers',
                            line=dict(color='#2c3e50',width=2),marker=dict(size=6),
                            hovertemplate='PC%{x}<br>Varianza: %{y:.2f}%<extra></extra>')
            fig.add_hline(y=float(100/n_vars),line_dash='dot',line_color='#7f8c8d',
                         annotation_text=f'Soglia Kaiser ({100/n_vars:.1f}%)',
                         annotation_position='right')
            fig.add_vline(x=k_kaiser,line_dash='dash',line_color='#e74c3c',
                         annotation_text=f'Kaiser k={k_kaiser}',
                         annotation_position='top right')
            fig.update_layout(xaxis_title='PC',yaxis_title='Varianza spiegata (%)',
                             height=320,margin=dict(l=10,r=10,t=20,b=30),
                             plot_bgcolor='#fafafa',paper_bgcolor='white',
                             font=dict(family='IBM Plex Sans'),showlegend=False)
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
                             yaxis=dict(range=[0,105]),height=320,
                             margin=dict(l=10,r=10,t=20,b=30),
                             plot_bgcolor='#fafafa',paper_bgcolor='white',
                             font=dict(family='IBM Plex Sans'),showlegend=False)
            st.plotly_chart(fig,use_container_width=True,key='chart_cumvar')
            st.info(f"💡 90% varianza → **k = {k_90}**  |  95% varianza → **k = {k_95}**")

        else:
            if not st.session_state.rmsecv_computed:
                if st.button("▶ Calcola RMSECV",key='btn_rmsecv',type='primary'):
                    with st.spinner("RMSECV in corso..."):
                        bk,_,rcv=compute_rmsecv(Xs_tmp,max_k)
                    st.session_state.rmsecv_result=(bk,rcv)
                    st.session_state.rmsecv_computed=True
                    st.rerun()
                else:
                    st.info("Clicca per calcolare il RMSECV.")
            else:
                bk,rcv=st.session_state.rmsecv_result
                fig=go.Figure()
                fig.add_scatter(x=ks,y=rcv.tolist(),mode='lines+markers',
                                line=dict(color='#16a085',width=2),
                                marker=dict(size=[12 if i+1==bk else 6 for i in range(max_k)],
                                            color=['#e74c3c' if i+1==bk else '#16a085'
                                                   for i in range(max_k)]),
                                hovertemplate='PC%{x}<br>RMSECV: %{y:.4f}<extra></extra>')
                fig.add_vline(x=bk,line_dash='dash',line_color='#e74c3c',
                             annotation_text=f'Minimo k={bk}',
                             annotation_position='top right')
                fig.update_layout(xaxis_title='Numero PC',yaxis_title='RMSECV',
                                 height=320,margin=dict(l=10,r=10,t=20,b=30),
                                 plot_bgcolor='#fafafa',paper_bgcolor='white',
                                 font=dict(family='IBM Plex Sans'),showlegend=False)
                st.plotly_chart(fig,use_container_width=True,key='chart_rmsecv')
                st.info(f"💡 RMSECV minimo → **k = {bk}** PC")
                if st.button("🔄 Ricalcola",key='btn_rmsecv_reset'):
                    st.session_state.rmsecv_computed=False; st.rerun()

        st.markdown("---")
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

        # AI — interpretazione scree plot
        st.markdown("---")
        st.markdown("#### 🤖 Interpreta i risultati")
        prompt_pc=f"""Sei un esperto di PCA per il controllo qualità in stampaggio a iniezione.
Dataset: {n_obs} cicli, {n_vars} variabili di processo.
Criteri selezione PC:
- Kaiser: k={k_kaiser} PC
- 90% varianza: k={k_90} PC
- 95% varianza: k={k_95} PC
- Varianza spiegata con k={k_chosen}: {cum[k_chosen-1]:.1f}%
Spiega in italiano a un process engineer:
1. Cosa significa questo numero di PC per il processo
2. Se la scelta di k={k_chosen} è ragionevole
3. Cosa si perde e cosa si guadagna con più o meno PC
Max 150 parole."""
        llm_button("Interpreta selezione PC", prompt_pc, key='ai_pc')


# ════════════════════════════════════════════
#  TAB 3 — CALIBRAZIONE
# ════════════════════════════════════════════
with tab3:
    if st.session_state.df_X is None:
        st.info("⬆️ Carica il dataset.")
    elif st.session_state.k_chosen is None:
        st.info("⬆️ Scegli il numero di PC nella tab Selezione PC.")
    else:
        df_X=st.session_state.df_X.fillna(st.session_state.df_X.mean())
        x_cols=st.session_state.feature_names
        X_raw=df_X.values; k_use=st.session_state.k_chosen

        st.markdown("### 🧹 Pulizia dati (opzionale)")
        st.caption("Rimuove cicli anomali dal set di calibrazione. "
                   "Utile se non sei sicuro che tutti i cicli fossero stabili.")
        use_clean=st.toggle("Attiva pulizia iterativa",value=False,key='toggle_clean')
        clean_mask=np.ones(len(X_raw),dtype=bool)

        if use_clean:
            alpha_clean=st.slider("Confidenza pulizia",0.95,0.999,0.99,0.001,format="%.3f")
            if st.button("▶ Esegui pulizia",key='btn_clean'):
                sc_k=StandardScaler(); Xs_k=sc_k.fit_transform(X_raw)
                eig_k=PCA(svd_solver='full').fit(Xs_k).explained_variance_
                k_cl=max(2,min(int(np.sum(eig_k>1)),X_raw.shape[1]-1,X_raw.shape[0]-2))
                with st.spinner("Pulizia iterativa..."):
                    clean_mask,log_df=iterative_cleaning(X_raw,k_cl,alpha_clean)
                n_rem=int((~clean_mask).sum()); pct=n_rem/len(X_raw)*100
                st.success(f"✅ Rimossi **{n_rem}** cicli ({pct:.1f}%) · Rimasti: **{clean_mask.sum()}**")
                st.dataframe(log_df,use_container_width=True,hide_index=True)
                if pct>20:
                    st.warning("⚠️ >20% rimossi — il periodo di calibrazione potrebbe non essere stabile.")

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
            pct_f=n_flag/len(X_clean)*100
            cm3.metric("Flag residui",f"{n_flag} ({pct_f:.1f}%)",
                       delta="🟢 OK" if pct_f<5 else "⚠️ Verifica",delta_color="off")

            fT2=model['T2']>model['T2_UCL']; fQ=model['Q']>model['Q_UCL']
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
            if st.button("🔄 Ricostruisci",key='btn_refit'):
                st.session_state.model=None; st.rerun()

        # AI — interpretazione calibrazione
        if st.session_state.model is not None:
            model=st.session_state.model
            n_flag=int(((model['T2']>model['T2_UCL'])|(model['Q']>model['Q_UCL'])).sum())
            st.markdown("---")
            st.markdown("#### 🤖 Interpreta il modello di calibrazione")
            prompt_cal=f"""Sei un esperto di SPC multivariato per stampaggio a iniezione.
Modello PCA-SPC calibrato su {len(model['T2'])} cicli con k={model['k']} PC.
T² UCL={model['T2_UCL']:.3f} (α={alpha})
Q UCL={model['Q_UCL']:.3f} (α={alpha})
Cicli residui fuori controllo in Phase I: {n_flag} ({n_flag/len(model['T2'])*100:.1f}%)
Variabili: {', '.join(model['feature_names'][:8])}{'...' if len(model['feature_names'])>8 else ''}
Spiega in italiano:
1. Cosa rappresentano T² UCL e Q UCL per questo processo
2. Se i {n_flag} cicli residui fuori limite sono accettabili
3. Come interpretare i grafici Phase I
Max 150 parole."""
            llm_button("Interpreta modello calibrazione", prompt_cal, key='ai_cal')


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

        with st.expander("📋 Indice variabili",expanded=False):
            st.dataframe(pd.DataFrame({'Indice':range(1,len(fn)+1),'Variabile':fn}),
                         use_container_width=True,hide_index=True)

        from itertools import combinations
        all_pairs=list(combinations(range(k_m),2))
        pair_labels=[f"PC{a+1} vs PC{b+1}  ({evr_m[a]:.1f}% + {evr_m[b]:.1f}%)"
                     for a,b in all_pairs]

        st.markdown("### Seleziona il grafico")
        tipo=st.radio("Tipo",["Loading plot","Score plot"],
                      horizontal=True,key='ls_tipo')
        coppia_idx=st.selectbox("Coppia di PC",options=range(len(all_pairs)),
                                format_func=lambda i: pair_labels[i],key='ls_coppia')
        pc_i,pc_j=all_pairs[coppia_idx]

        if tipo=="Loading plot":
            p_count=P.shape[0]
            labels=fn if fn else [f'Var {i+1}' for i in range(p_count)]
            fig_load=go.Figure()
            fig_load.add_scatter(
                x=P[:,pc_i].tolist(),y=P[:,pc_j].tolist(),
                mode='markers+text',
                marker=dict(size=9,color='#2980b9',line=dict(color='white',width=1)),
                text=[str(i+1) for i in range(p_count)],
                textposition='top center',textfont=dict(size=9,color='#1a1a2e'),
                hovertemplate=(f'<b>%{{customdata}}</b><br>'
                               f'PC{pc_i+1}: %{{x:.4f}}<br>'
                               f'PC{pc_j+1}: %{{y:.4f}}<extra></extra>'),
                customdata=labels,name='Variabili',showlegend=False)
            fig_load.add_hline(y=0,line_color='#bdc3c7',line_width=0.8)
            fig_load.add_vline(x=0,line_color='#bdc3c7',line_width=0.8)
            fig_load.update_layout(
                title=dict(text=f'Loading plot PC{pc_i+1} ({evr_m[pc_i]:.1f}%) vs PC{pc_j+1} ({evr_m[pc_j]:.1f}%)',
                           font=dict(family='IBM Plex Mono',size=12)),
                xaxis_title=f'PC{pc_i+1} loading',yaxis_title=f'PC{pc_j+1} loading',
                height=500,margin=dict(l=10,r=10,t=50,b=30),
                plot_bgcolor='#fafafa',paper_bgcolor='white',
                font=dict(family='IBM Plex Sans'))
            st.plotly_chart(fig_load,use_container_width=True,key='load_chart')
            st.caption("💡 Numero = indice variabile. Hover per il nome completo. "
                       "Variabili vicine = correlate | Opposte = correlate negativamente.")

            df_load=pd.DataFrame({
                'Indice':range(1,p_count+1),'Variabile':labels,
                f'Loading PC{pc_i+1}':P[:,pc_i].round(4),
                f'Loading PC{pc_j+1}':P[:,pc_j].round(4),
                'Distanza origine':np.sqrt(P[:,pc_i]**2+P[:,pc_j]**2).round(4),
            }).sort_values('Distanza origine',ascending=False)
            st.dataframe(df_load,use_container_width=True,hide_index=True)

        else:
            st.plotly_chart(make_score_chart(T_m,lam_m,model['T2_UCL'],
                                             evr_m,pc_i,pc_j,flag_m),
                            use_container_width=True,key='score_chart')
            st.caption(f"Punti rossi: {int(flag_m.sum())} cicli anomali su {len(T_m)}")

        # AI — interpretazione loadings
        st.markdown("---")
        st.markdown("#### 🤖 Interpreta questo grafico")
        top_vars=np.argsort(np.sqrt(P[:,pc_i]**2+P[:,pc_j]**2))[::-1][:5]
        top_str=", ".join(f"{fn[i]} ({P[i,pc_i]:.3f}/{P[i,pc_j]:.3f})" for i in top_vars)
        prompt_load=f"""Sei un esperto di stampaggio a iniezione e PCA.
Loading plot PC{pc_i+1} ({evr_m[pc_i]:.1f}% varianza) vs PC{pc_j+1} ({evr_m[pc_j]:.1f}% varianza).
Tipo grafico: {'loading plot' if tipo=='Loading plot' else 'score plot'}.
Top 5 variabili per influenza su questa coppia: {top_str}
Spiega in italiano a un process engineer:
1. Cosa mostra questo grafico
2. Come interpretare la posizione delle variabili principali
3. Quali correlazioni tra variabili sono evidenti
Max 150 parole."""
        llm_button(f"Interpreta {'loading' if tipo=='Loading plot' else 'score'} plot",
                   prompt_load, key='ai_load')


# ════════════════════════════════════════════
#  TAB 5 — MONITORAGGIO
# ════════════════════════════════════════════
with tab5:
    if st.session_state.model is None:
        st.info("⬆️ Costruisci prima il modello nella tab Calibrazione.")
    else:
        model=st.session_state.model; fn=model['feature_names']

        st.markdown("### Carica nuovi dati dalla USB")
        up_test=st.file_uploader("Trascina il file qui",
                                  type=['csv','xlsx','xls'],key='up_test')

        if up_test:
            df_tr=(pd.read_csv(up_test) if up_test.name.endswith('.csv')
                   else pd.read_excel(up_test))
            df_tr.columns=df_tr.columns.astype(str)
            miss_c=[c for c in fn if c not in df_tr.columns]
            if miss_c:
                st.error(f"Colonne mancanti: {miss_c}")
            else:
                df_test_X=df_tr[fn].copy().fillna(df_tr[fn].mean())
                mon=monitor_new(model,df_test_X.values)
                st.session_state.mon=mon
                st.session_state.df_test_X=df_test_X

                n_test=len(df_test_X)
                n_t2=int(mon['T2_flag'].sum()); n_q=int(mon['Q_flag'].sum())
                n_any=int((mon['T2_flag']|mon['Q_flag']).sum())
                pct=n_any/n_test*100

                c1,c2,c3,c4=st.columns(4)
                c1.metric("Cicli",n_test)
                c2.metric("T² anomali",f"{n_t2} ({n_t2/n_test*100:.1f}%)")
                c3.metric("Q anomali",f"{n_q} ({n_q/n_test*100:.1f}%)")
                stato=("🟢 STABILE" if pct<5 else "🟡 ATTENZIONE" if pct<15 else "🔴 ANOMALIE")
                c4.metric("Stato",stato)

                # AI — overview processo
                st.markdown("---")
                prompt_ov=f"""Processo: stampaggio a iniezione. Phase II monitoring.
{n_test} cicli analizzati. T² anomali: {n_t2} ({n_t2/n_test*100:.1f}%). Q anomali: {n_q} ({n_q/n_test*100:.1f}%). Stato: {stato}.
T² UCL={model['T2_UCL']:.3f}, Q UCL={model['Q_UCL']:.3f}, k={model['k']} PC.
Spiega in italiano a un capoturno cosa sta succedendo al processo e se c'è motivo di preoccupazione. Max 100 parole."""
                llm_button("Interpreta stato processo", prompt_ov, key='ai_overview')

                # Control charts
                st.markdown("---")
                st.markdown("### Control charts")
                st.caption("I punti rossi sono cicli fuori controllo.")
                st.plotly_chart(make_line_chart(mon['T2'],model['T2_UCL'],
                                                'Phase II — Hotelling T²','#2980b9',mon['T2_flag']),
                                use_container_width=True,key='p2_t2')
                st.plotly_chart(make_line_chart(mon['Q'],model['Q_UCL'],
                                                'Phase II — Q (SPE)','#27ae60',mon['Q_flag']),
                                use_container_width=True,key='p2_q')

                # Tabella anomalie cliccabile
                flagged_idx=np.where(mon['T2_flag']|mon['Q_flag'])[0]

                if len(flagged_idx)==0:
                    st.markdown("<div class='ok-box'>✅ Nessuna anomalia — processo in controllo.</div>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(f"### 🔍 Cicli anomali — {len(flagged_idx)} fuori controllo")
                    st.caption("Seleziona un ciclo dalla tabella per vedere i contribution plots.")

                    # Tabella ordinata per severità decrescente
                    df_anom=pd.DataFrame({
                        'Ciclo':       flagged_idx,
                        'T²':          mon['T2'][flagged_idx].round(3),
                        'T²/UCL':      (mon['T2'][flagged_idx]/model['T2_UCL']).round(2),
                        'Q':           mon['Q'][flagged_idx].round(3),
                        'Q/UCL':       (mon['Q'][flagged_idx]/model['Q_UCL']).round(2),
                        'T² flag':     mon['T2_flag'][flagged_idx],
                        'Q flag':      mon['Q_flag'][flagged_idx],
                        'Severità×UCL':np.maximum(
                            mon['T2'][flagged_idx]/model['T2_UCL'],
                            mon['Q'][flagged_idx]/model['Q_UCL']
                        ).round(2),
                    }).sort_values('Severità×UCL',ascending=False).reset_index(drop=True)

                    # Tabella con selezione riga
                    sel=st.dataframe(df_anom,use_container_width=True,
                                     hide_index=True,
                                     on_select='rerun',
                                     selection_mode='single-row',
                                     key='anom_table')

                    # Determina ciclo selezionato
                    obs_choice=None
                    if sel and sel.selection and sel.selection.get('rows'):
                        row_idx=sel.selection['rows'][0]
                        obs_choice=int(df_anom.iloc[row_idx]['Ciclo'])
                    elif len(flagged_idx)>0:
                        obs_choice=int(df_anom.iloc[0]['Ciclo'])

                    if obs_choice is not None:
                        t2_obs=float(mon['T2'][obs_choice])
                        q_obs =float(mon['Q'][obs_choice])
                        ratio =max(t2_obs/model['T2_UCL'],q_obs/model['Q_UCL'])

                        st.markdown(
                            f"<div class='alarm-box'>"
                            f"<strong>Ciclo {obs_choice} — "
                            f"{'🔴 ANOMALIA' if ratio>=1.5 else '⚠️ ATTENZIONE'}</strong><br>"
                            f"T² = {t2_obs:.3f} ({t2_obs/model['T2_UCL']:.2f}× UCL) &nbsp;|&nbsp; "
                            f"Q = {q_obs:.3f} ({q_obs/model['Q_UCL']:.2f}× UCL)"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                        # Contribution plots
                        c_t2v=(mon['Xn_s'][obs_choice]*
                               (model['loadings']@(mon['Tn'][obs_choice]/model['eigenvalues'])))
                        c_qv =mon['En'][obs_choice]
                        top_t2=np.argsort(np.abs(c_t2v))[::-1][:3].tolist()
                        top_q =np.argsort(np.abs(c_qv))[::-1][:3].tolist()

                        st.markdown("#### Contribution plots")
                        with st.expander("📋 Indice variabili"):
                            st.dataframe(
                                pd.DataFrame({'Indice':range(1,len(fn)+1),'Variabile':fn}),
                                use_container_width=True,hide_index=True)

                        p_count=len(fn)
                        col_l,col_r=st.columns(2)

                        with col_l:
                            st.markdown(f"**T² — Ciclo {obs_choice}**")
                            fig_ct2,exc_t2=make_contribution_chart(
                                c_t2v,model['T2contrib_UCL'],model['T2contrib_LCL'],
                                fn,f'T² Contribution — Ciclo {obs_choice}')
                            st.plotly_chart(fig_ct2,use_container_width=True,
                                            key=f'ct2_{obs_choice}')
                            df_ct2=pd.DataFrame({
                                'Idx':range(1,p_count+1),'Variabile':fn,
                                'Contributo':c_t2v.round(4),
                                'LCL':model['T2contrib_LCL'].round(4),
                                'UCL':model['T2contrib_UCL'].round(4),
                                'Fuori':['🔴' if (c_t2v[i]>model['T2contrib_UCL'][i]
                                                  or c_t2v[i]<model['T2contrib_LCL'][i])
                                         else '✅' for i in range(p_count)]
                            })
                            with st.expander("Tabella completa T²"):
                                st.dataframe(df_ct2,use_container_width=True,hide_index=True)
                            if exc_t2:
                                st.markdown("**Fuori limite T²:**")
                                st.dataframe(pd.DataFrame(exc_t2,
                                             columns=['Idx','Variabile','Valore','LCL','UCL']),
                                             use_container_width=True,hide_index=True)

                        with col_r:
                            st.markdown(f"**Q — Ciclo {obs_choice}**")
                            fig_cq,exc_q=make_contribution_chart(
                                c_qv,model['Qcontrib_UCL'],model['Qcontrib_LCL'],
                                fn,f'Q Contribution — Ciclo {obs_choice}')
                            st.plotly_chart(fig_cq,use_container_width=True,
                                            key=f'cq_{obs_choice}')
                            df_cq=pd.DataFrame({
                                'Idx':range(1,p_count+1),'Variabile':fn,
                                'Contributo':c_qv.round(4),
                                'LCL':model['Qcontrib_LCL'].round(4),
                                'UCL':model['Qcontrib_UCL'].round(4),
                                'Fuori':['🔴' if (c_qv[i]>model['Qcontrib_UCL'][i]
                                                  or c_qv[i]<model['Qcontrib_LCL'][i])
                                         else '✅' for i in range(p_count)]
                            })
                            with st.expander("Tabella completa Q"):
                                st.dataframe(df_cq,use_container_width=True,hide_index=True)
                            if exc_q:
                                st.markdown("**Fuori limite Q:**")
                                st.dataframe(pd.DataFrame(exc_q,
                                             columns=['Idx','Variabile','Valore','LCL','UCL']),
                                             use_container_width=True,hide_index=True)

                        # AI — spiegazione anomalia
                        st.markdown("---")
                        st.markdown("#### 🤖 Spiegazione per il tecnico")
                        v_t2=[fn[i] for i in top_t2]; v_q=[fn[i] for i in top_q]
                        prompt_an=f"""Sei un esperto di stampaggio a iniezione.
Ciclo anomalo {obs_choice}: T²={t2_obs:.3f} ({t2_obs/model['T2_UCL']:.2f}×UCL), Q={q_obs:.3f} ({q_obs/model['Q_UCL']:.2f}×UCL).
Variabili che contribuiscono al T²: {', '.join(v_t2)}.
Variabili che contribuiscono al Q: {', '.join(v_q)}.
Spiega in italiano a un capoturno:
1. Cosa sta succedendo nel processo (in termini fisici)
2. Possibili cause
3. Cosa controllare sulla macchina / azione correttiva immediata
Max 120 parole. Niente statistica, solo processo fisico."""
                        llm_button("Spiega anomalia al tecnico", prompt_an,
                                   key=f'ai_an_{obs_choice}')

                        # Log
                        st.markdown("#### 📝 Registra intervento")
                        with st.form(f"log_{obs_choice}"):
                            azione=st.text_area("Azione correttiva adottata",height=70,
                                               placeholder="Es: aumentata contropressione da 80 a 95 bar")
                            if st.form_submit_button("💾 Salva",use_container_width=True):
                                if azione:
                                    st.session_state.anomaly_log.append({
                                        'Ciclo':obs_choice,'T²':round(t2_obs,3),
                                        'Q':round(q_obs,3),
                                        'Severità':f"{ratio:.2f}×UCL",'Intervento':azione})
                                    st.success("✅ Salvato.")

                    if st.session_state.anomaly_log:
                        st.markdown("#### 📋 Log interventi")
                        st.dataframe(pd.DataFrame(st.session_state.anomaly_log),
                                     use_container_width=True)
