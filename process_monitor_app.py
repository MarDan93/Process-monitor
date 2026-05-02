import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
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
.llm-box{background:#1e3a5f;border:2px solid #4a90d9;border-left:6px solid #4a90d9;
  padding:14px 18px;border-radius:6px;font-size:14px;line-height:1.7;
  margin-top:10px;color:#ffffff !important;}
.alarm-box{background:#5c1a1a;border:2px solid #e74c3c;border-left:6px solid #e74c3c;
  padding:12px 16px;border-radius:6px;font-size:14px;margin-top:8px;color:#ffffff !important;}
.alarm-box strong{color:#ff9999;}
.ok-box{background:#1a3d2b;border:2px solid #2ecc71;border-left:6px solid #2ecc71;
  padding:12px 16px;border-radius:6px;font-size:14px;margin-top:8px;color:#ffffff !important;}
.ctx-box{background:#2a1f4f;border:2px solid #8b5cf6;border-left:6px solid #8b5cf6;
  padding:12px 16px;border-radius:6px;font-size:13px;margin-top:8px;color:#e2d9ff !important;}
.chat-user{background:#1e3a5f;border-radius:8px 8px 0 8px;padding:10px 14px;
  margin:6px 0;font-size:13px;color:#fff;text-align:right;}
.chat-ai{background:#1a3d2b;border-radius:8px 8px 8px 0;padding:10px 14px;
  margin:6px 0;font-size:13px;color:#fff;}
.file-tag{display:inline-block;padding:2px 8px;border-radius:4px;
  font-size:11px;font-family:'IBM Plex Mono',monospace;margin:2px;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  CONTEXT
# ═══════════════════════════════════════════

def get_process_context():
    ctx = st.session_state.get('process_context','').strip()
    obj = st.session_state.get('analysis_objective','').strip()
    out = ""
    if ctx: out += f"\n=== PROCESS CONTEXT ===\n{ctx}\n=== END CONTEXT ===\n"
    if obj: out += f"\n=== ANALYSIS OBJECTIVE ===\n{obj}\n=== END OBJECTIVE ===\n"
    return out


# ═══════════════════════════════════════════
#  AI — MULTI-MODEL WITH FALLBACK
#  Priority: Gemini 2.5 Flash → Gemini 2.0 Flash → Claude Haiku
# ═══════════════════════════════════════════

MAX_TOKENS = 8000

def call_gemini_model(prompt, model_name):
    key = st.secrets.get("GEMINI_API_KEY","")
    if not key:
        raise ValueError("GEMINI_API_KEY not configured.")
    genai.configure(api_key=key)
    m = genai.GenerativeModel(
        model_name,
        generation_config=genai.GenerationConfig(
            temperature=0.3, max_output_tokens=MAX_TOKENS)
    )
    return m.generate_content(prompt).text


def call_claude_haiku(prompt):
    try:
        import anthropic
        key = st.secrets.get("ANTHROPIC_API_KEY","")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not configured.")
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=MAX_TOKENS,
            messages=[{"role":"user","content":prompt}]
        )
        return msg.content[0].text
    except ImportError:
        raise ValueError("anthropic library not installed — add it to requirements.txt")


def call_ai(prompt):
    """
    Multi-model fallback (cheapest/free first):
    1. Gemini 2.5 Flash      (free, 10 RPM, 250 RPD)
    2. Gemini 2.5 Flash-Lite (free, 15 RPM, 1000 RPD) ← higher daily quota
    3. Claude Haiku           (paid, ~$0.001/call)
    Returns (text, model_used, error)
    """
    RATE_CODES = ["429","quota","rate","limit","resource","exhausted","overload","unavailable"]

    for model_name, label in [
        ("gemini-2.5-flash",      "Gemini 2.5 Flash"),
        ("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite"),
    ]:
        try:
            txt = call_gemini_model(prompt, model_name)
            return txt, label, None
        except Exception as e:
            if not any(c in str(e).lower() for c in RATE_CODES):
                return None, None, f"{label} error: {e}"
            # rate-limited → try next model

    # Final fallback: Claude Haiku (paid)
    try:
        txt = call_claude_haiku(prompt)
        return txt, "Claude Haiku (paid fallback)", None
    except Exception as e:
        return None, None, f"All models failed. Last error: {e}"


def llm_button(label, prompt, key):
    if st.button(f"🤖 {label}", key=key):
        full_prompt = get_process_context() + "\n" + prompt
        with st.spinner("AI analysis..."):
            txt, model_used, err = call_ai(full_prompt)
        if err:
            st.error(f"AI error: {err}")
        else:
            st.markdown(
                f"<div class='llm-box'>{txt.replace(chr(10),'<br>')}</div>",
                unsafe_allow_html=True)
            st.caption(f"Powered by {model_used}")



# ═══════════════════════════════════════════
#  PCA-SPC CORE
# ═══════════════════════════════════════════

def fit_pca_spc(X_raw, k, alpha=0.95):
    n, p = X_raw.shape
    sc = StandardScaler(); Xs = sc.fit_transform(X_raw)
    pca = PCA(n_components=k, svd_solver='full', random_state=42)
    T = pca.fit_transform(Xs); P = pca.components_.T; lam = pca.explained_variance_
    T2 = np.sum((T**2)/lam, axis=1)
    T2_UCL = (k*(n-1)/(n-k))*f.ppf(alpha, k, n-k)
    E = Xs - pca.inverse_transform(T); Q = np.sum(E**2, axis=1)
    pf = PCA(n_components=min(p,n-1), svd_solver='full', random_state=42); pf.fit(Xs)
    re = pf.explained_variance_[k:]
    t1,t2,t3 = re.sum(),(re**2).sum(),(re**3).sum()
    h0 = 1-(2*t1*t3)/(3*t2**2); z = norm.ppf(alpha)
    Q_UCL = t1*((z*np.sqrt(2*t2*h0**2)/t1)+1+(t2*h0*(h0-1)/t1**2))**(1/h0)
    zc = norm.ppf(0.975)
    mu_e = E.mean(0); sd_e = E.std(0,ddof=1)
    W = T/lam; cT2 = Xs*(W@P.T); mu_c = cT2.mean(0); sd_c = cT2.std(0,ddof=1)
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
        log.append(dict(Iter=it,Before=n_it,Removed=n_rem,After=n_it-n_rem))
        if n_rem==0: break
        idx_c=np.where(mask)[0]; mask[idx_c[flag]]=False
    return mask, pd.DataFrame(log)


# ═══════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════

FILE_COLORS = ['#2980b9','#e74c3c','#27ae60','#f39c12','#8e44ad',
               '#16a085','#d35400','#2c3e50','#c0392b','#1abc9c']

def chart_line_multi(values, ucl, title, color, flags=None,
                     file_labels=None, file_boundaries=None):
    """Line chart supporting multi-file source indicators."""
    fig = go.Figure()
    x = np.arange(len(values))

    # Shade regions by file source
    if file_boundaries and len(file_boundaries) > 1:
        for fi, (start, end) in enumerate(zip(file_boundaries[:-1], file_boundaries[1:])):
            fc = FILE_COLORS[fi % len(FILE_COLORS)]
            fig.add_vrect(x0=start, x1=end-1,
                          fillcolor=fc, opacity=0.07, line_width=0,
                          annotation_text=file_labels[fi] if file_labels else f"File {fi+1}",
                          annotation_position="top left",
                          annotation_font_size=9)

    fig.add_scatter(x=x.tolist(), y=values.tolist(),
                    mode='lines+markers', line=dict(color=color, width=1.2),
                    marker=dict(size=4, color=color),
                    hovertemplate='Cycle %{x}<br>%{y:.3f}<extra></extra>', name='Value')
    fig.add_hline(y=ucl, line_dash='dash', line_color='#e74c3c', line_width=1.5,
                  annotation_text=f'UCL={ucl:.3f}', annotation_position='right')
    if flags is not None and flags.any():
        fi = np.where(flags)[0]
        fig.add_scatter(x=fi.tolist(), y=values[fi].tolist(), mode='markers',
                        marker=dict(size=10, color='#e74c3c', symbol='x',
                                    line=dict(width=2)),
                        name='Anomaly',
                        hovertemplate='Cycle %{x}<br>%{y:.3f}<extra></extra>')
    fig.update_layout(title=dict(text=title, font=dict(family='IBM Plex Mono', size=13)),
                      xaxis_title='Cycle', height=300,
                      margin=dict(l=10,r=10,t=40,b=30),
                      plot_bgcolor='#fafafa', paper_bgcolor='white',
                      font=dict(family='IBM Plex Sans'),
                      legend=dict(orientation='h', y=-0.25), hovermode='x unified')
    return fig


def chart_line(values, ucl, title, color, flags=None):
    return chart_line_multi(values, ucl, title, color, flags)


def chart_contribution(contrib, ucl_v, lcl_v, fn, title):
    p=len(contrib); xax=list(range(1,p+1))
    labels=fn if fn else [f'Var {i+1}' for i in range(p)]
    colors=['#e74c3c' if (contrib[i]>ucl_v[i] or contrib[i]<lcl_v[i])
            else '#2c3e50' for i in range(p)]
    fig=go.Figure()
    fig.add_bar(x=xax,y=contrib.tolist(),marker_color=colors,customdata=labels,
                hovertemplate='[%{x}] %{customdata}<br>Contribution: %{y:.4f}<extra></extra>',
                name='Contribution')
    fig.add_scatter(x=xax,y=ucl_v.tolist(),mode='lines',
                    line=dict(color='#e74c3c',dash='dash',width=1.5),name='UCL',hoverinfo='skip')
    fig.add_scatter(x=xax,y=lcl_v.tolist(),mode='lines',
                    line=dict(color='#e74c3c',dash='dash',width=1.5),name='LCL',hoverinfo='skip')
    fig.add_hline(y=0,line_color='black',line_width=0.8)
    fig.update_layout(title=dict(text=title,font=dict(family='IBM Plex Mono',size=12)),
                      xaxis=dict(title='Variable (index)',tickvals=xax,
                                 ticktext=[str(x) for x in xax]),
                      yaxis_title='Contribution (signed)',height=320,
                      margin=dict(l=10,r=10,t=40,b=30),
                      plot_bgcolor='#fafafa',paper_bgcolor='white',
                      font=dict(family='IBM Plex Sans'),legend=dict(orientation='h',y=-0.3))
    exceed=[(i+1,labels[i],round(float(contrib[i]),4),
             round(float(lcl_v[i]),4),round(float(ucl_v[i]),4))
            for i in range(p) if contrib[i]>ucl_v[i] or contrib[i]<lcl_v[i]]
    return fig,exceed


def chart_score(T, lam, T2_UCL_global, evr, pc_i, pc_j, flagged, alpha=0.95, n_train=None):
    """
    Score plot PC_i vs PC_j.
    Point color based on 2D confidence ellipse (k=2, F distribution).
    Black = inside ellipse | Red = outside ellipse.
    """
    ang = np.linspace(0, 2*np.pi, 400)

    # 2D confidence limit for this specific pair of PCs
    n = n_train if n_train is not None else max(T.shape[0], 10)
    T2_UCL_2d = (2*(n-1)/(n-2)) * f.ppf(alpha, 2, n-2)
    a = np.sqrt(lam[pc_i] * T2_UCL_2d)
    b = np.sqrt(lam[pc_j] * T2_UCL_2d)

    # Flag based on 2D ellipse: (score_i/a)^2 + (score_j/b)^2 > 1
    outside_2d = (T[:,pc_i]/a)**2 + (T[:,pc_j]/b)**2 > 1
    inside_2d  = ~outside_2d

    fig = go.Figure()

    # Inside ellipse — black
    if inside_2d.any():
        fig.add_scatter(
            x=T[inside_2d, pc_i].tolist(),
            y=T[inside_2d, pc_j].tolist(),
            mode='markers',
            marker=dict(size=5, color='#1a1a2e', opacity=0.7),
            name=f'Inside {int(alpha*100)}% ellipse',
            hovertemplate=f'PC{pc_i+1}: %{{x:.3f}}<br>PC{pc_j+1}: %{{y:.3f}}<extra></extra>'
        )

    # Outside ellipse — red filled circle (no X)
    if outside_2d.any():
        fig.add_scatter(
            x=T[outside_2d, pc_i].tolist(),
            y=T[outside_2d, pc_j].tolist(),
            mode='markers',
            marker=dict(size=8, color='#e74c3c', opacity=0.85,
                        line=dict(color='#c0392b', width=1)),
            name=f'Outside {int(alpha*100)}% ellipse',
            hovertemplate=f'PC{pc_i+1}: %{{x:.3f}}<br>PC{pc_j+1}: %{{y:.3f}}<extra></extra>'
        )

    # Confidence ellipse — red dashed
    fig.add_scatter(
        x=(a*np.cos(ang)).tolist(), y=(b*np.sin(ang)).tolist(),
        mode='lines',
        line=dict(color='#e74c3c', dash='dash', width=1.8),
        name=f'{int(alpha*100)}% confidence ellipse',
        hoverinfo='skip'
    )

    fig.add_hline(y=0, line_color='#bdc3c7', line_width=0.8)
    fig.add_vline(x=0, line_color='#bdc3c7', line_width=0.8)

    n_out = int(outside_2d.sum())
    fig.update_layout(
        title=dict(
            text=(f'Score plot  PC{pc_i+1} ({evr[pc_i]:.1f}%)'
                  f'  vs  PC{pc_j+1} ({evr[pc_j]:.1f}%)'
                  f'  —  {int(alpha*100)}% confidence  |  {n_out}/{len(T)} outside'),
            font=dict(family='IBM Plex Mono', size=12)
        ),
        xaxis_title=f'PC{pc_i+1} ({evr[pc_i]:.1f}%)',
        yaxis_title=f'PC{pc_j+1} ({evr[pc_j]:.1f}%)',
        height=440,
        margin=dict(l=10, r=10, t=55, b=30),
        plot_bgcolor='#fafafa', paper_bgcolor='white',
        font=dict(family='IBM Plex Sans'),
        legend=dict(orientation='h', y=-0.18)
    )
    return fig


def chart_loading(P,fn,evr,pc_i,pc_j):
    p=P.shape[0]; labels=fn if fn else [f'Var {i+1}' for i in range(p)]
    fig=go.Figure()
    fig.add_scatter(x=P[:,pc_i].tolist(),y=P[:,pc_j].tolist(),
                    mode='markers+text',
                    marker=dict(size=9,color='#2980b9',line=dict(color='white',width=1)),
                    text=[str(i+1) for i in range(p)],
                    textposition='top center',textfont=dict(size=9,color='#1a1a2e'),
                    customdata=labels,
                    hovertemplate='<b>%{customdata}</b><br>'
                                  f'PC{pc_i+1}: %{{x:.4f}}<br>PC{pc_j+1}: %{{y:.4f}}<extra></extra>',
                    name='Variables',showlegend=False)
    fig.add_hline(y=0,line_color='#bdc3c7',line_width=0.8)
    fig.add_vline(x=0,line_color='#bdc3c7',line_width=0.8)
    fig.update_layout(
        title=dict(text=f'Loading plot PC{pc_i+1} ({evr[pc_i]:.1f}%) vs PC{pc_j+1} ({evr[pc_j]:.1f}%)',
                   font=dict(family='IBM Plex Mono',size=12)),
        xaxis_title=f'PC{pc_i+1} loading',yaxis_title=f'PC{pc_j+1} loading',
        height=500,margin=dict(l=10,r=10,t=50,b=30),
        plot_bgcolor='#fafafa',paper_bgcolor='white',font=dict(family='IBM Plex Sans'))
    return fig


def show_contribution_block(model,Xs,E,T,obs_idx,fn,key_suffix):
    lam=model['eigenvalues']; P=model['loadings']
    c_t2=Xs[obs_idx]*(P@(T[obs_idx]/lam)); c_q=E[obs_idx]
    top_t2=np.argsort(np.abs(c_t2))[::-1][:3].tolist()
    top_q=np.argsort(np.abs(c_q))[::-1][:3].tolist()
    p_count=len(fn)
    with st.expander("📋 Variable Index",expanded=False):
        st.dataframe(pd.DataFrame({'Index':range(1,p_count+1),'Variable':fn}),
                     use_container_width=True,hide_index=True)
    col_l,col_r=st.columns(2)
    with col_l:
        fig_t2,exc_t2=chart_contribution(c_t2,model['T2contrib_UCL'],model['T2contrib_LCL'],
                                          fn,f'T² Contribution — Obs {obs_idx}')
        st.plotly_chart(fig_t2,use_container_width=True,key=f'ct2_{key_suffix}')
        df_t2=pd.DataFrame({'Idx':range(1,p_count+1),'Variable':fn,
                             'Contribution':c_t2.round(4),
                             'LCL':model['T2contrib_LCL'].round(4),
                             'UCL':model['T2contrib_UCL'].round(4),
                             'Out':['🔴' if (c_t2[i]>model['T2contrib_UCL'][i]
                                             or c_t2[i]<model['T2contrib_LCL'][i])
                                    else '✅' for i in range(p_count)]})
        with st.expander("T² Table"):
            st.dataframe(df_t2,use_container_width=True,hide_index=True)
        if exc_t2:
            st.markdown("**Out-of-limit — T²:**")
            st.dataframe(pd.DataFrame(exc_t2,columns=['Idx','Variable','Value','LCL','UCL']),
                         use_container_width=True,hide_index=True)
    with col_r:
        fig_q,exc_q=chart_contribution(c_q,model['Qcontrib_UCL'],model['Qcontrib_LCL'],
                                        fn,f'Q Contribution — Obs {obs_idx}')
        st.plotly_chart(fig_q,use_container_width=True,key=f'cq_{key_suffix}')
        df_q=pd.DataFrame({'Idx':range(1,p_count+1),'Variable':fn,
                            'Contribution':c_q.round(4),
                            'LCL':model['Qcontrib_LCL'].round(4),
                            'UCL':model['Qcontrib_UCL'].round(4),
                            'Out':['🔴' if (c_q[i]>model['Qcontrib_UCL'][i]
                                            or c_q[i]<model['Qcontrib_LCL'][i])
                                   else '✅' for i in range(p_count)]})
        with st.expander("Q Table"):
            st.dataframe(df_q,use_container_width=True,hide_index=True)
        if exc_q:
            st.markdown("**Out-of-limit — Q:**")
            st.dataframe(pd.DataFrame(exc_q,columns=['Idx','Variable','Value','LCL','UCL']),
                         use_container_width=True,hide_index=True)
    return top_t2,top_q


def show_anomaly_table_and_contrib(model,T2_arr,Q_arr,Xs,E,T_arr,fn,table_key,prefix):
    flagged_idx=np.where((T2_arr>model['T2_UCL'])|(Q_arr>model['Q_UCL']))[0]
    if len(flagged_idx)==0:
        st.markdown("<div class='ok-box'>✅ No anomalies detected.</div>",
                    unsafe_allow_html=True)
        return None
    st.caption(f"{len(flagged_idx)} out-of-control observations — click a row to analyse it.")
    df_anom=pd.DataFrame({
        'Cycle':        flagged_idx,
        'T²':           T2_arr[flagged_idx].round(3),
        'T²/UCL':       (T2_arr[flagged_idx]/model['T2_UCL']).round(2),
        'Q':            Q_arr[flagged_idx].round(3),
        'Q/UCL':        (Q_arr[flagged_idx]/model['Q_UCL']).round(2),
        'T² flag':      T2_arr[flagged_idx]>model['T2_UCL'],
        'Q flag':       Q_arr[flagged_idx]>model['Q_UCL'],
        'Severity×UCL': np.maximum(T2_arr[flagged_idx]/model['T2_UCL'],
                                   Q_arr[flagged_idx]/model['Q_UCL']).round(2),
    }).sort_values('Severity×UCL',ascending=False).reset_index(drop=True)
    sel=st.dataframe(df_anom,use_container_width=True,hide_index=True,
                     on_select='rerun',selection_mode='single-row',key=table_key)
    obs=None
    if sel and sel.selection and sel.selection.get('rows'):
        obs=int(df_anom.iloc[sel.selection['rows'][0]]['Cycle'])
    else:
        obs=int(df_anom.iloc[0]['Cycle'])
    t2_obs=float(T2_arr[obs]); q_obs=float(Q_arr[obs])
    ratio=max(t2_obs/model['T2_UCL'],q_obs/model['Q_UCL'])
    st.markdown(
        f"<div class='alarm-box'><strong>Cycle {obs} — "
        f"{'🔴 ANOMALY' if ratio>=1.5 else '⚠️ WARNING'}</strong><br>"
        f"T²={t2_obs:.3f} ({t2_obs/model['T2_UCL']:.2f}×UCL) | "
        f"Q={q_obs:.3f} ({q_obs/model['Q_UCL']:.2f}×UCL)</div>",
        unsafe_allow_html=True)
    st.markdown("#### Contribution plots")
    top_t2,top_q=show_contribution_block(model,Xs,E,T_arr,obs,fn,f'{prefix}_{obs}')
    return obs,t2_obs,q_obs,ratio,top_t2,top_q


# ═══════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════
for k,v in [('model',None),('feature_names',[]),('y_names',[]),
            ('df_X',None),('df_Y',None),('k_chosen',None),
            ('mon',None),('mon_files',[]),
            ('anomaly_log',[]),
            ('rmsecv_computed',False),('rmsecv_result',None),
            ('process_context',''),('analysis_objective',''),('context_saved',False),
            ('split_method','Temporal'),('split_ratio',0.7),('split_row',None),
            ('df_train',None),('df_test_builtin',None),
            ('global_chat',[])]:   # single persistent chat
    if k not in st.session_state:
        st.session_state[k]=v


# ═══════════════════════════════════════════
#  SIDEBAR — settings + global persistent chat
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🏭 Process Monitor")
    st.markdown("**PCA-SPC · Industrial Monitoring**")
    st.divider()

    # Settings (collapsed to save space)
    with st.expander("⚙️ Settings", expanded=False):
        if st.session_state.context_saved and st.session_state.process_context:
            st.markdown("**Process context loaded ✅**")
            obj=st.session_state.get('analysis_objective','')
            if obj:
                st.caption(f"Objective: {obj[:60]}{'...' if len(obj)>60 else ''}")
        alpha=st.slider("UCL Confidence (α)",0.90,0.99,0.95,0.01)
        y_cols_raw=st.text_area("Y Variables — one per line")
        y_cols=[c.strip() for c in y_cols_raw.split('\n') if c.strip()]
        excl_raw=st.text_area("Columns to exclude — one per line")
        excl_cols=[c.strip() for c in excl_raw.split('\n') if c.strip()]
        st.caption("AI: Gemini 2.5 Flash → Flash-Lite → Claude Haiku")

    st.divider()

    # ── Global AI Chat ────────────────────────────────────
    st.markdown("### 💬 AI Assistant")
    st.caption("Ask anything — the AI knows your process, model and monitoring results.")

    # Build live project context for the chat
    def build_project_snapshot():
        """Compact snapshot of current app state — injected into every chat message."""
        lines = []
        if st.session_state.process_context:
            lines.append(f"Process: {st.session_state.process_context[:200]}")
        if st.session_state.analysis_objective:
            lines.append(f"Objective: {st.session_state.analysis_objective}")
        if st.session_state.df_X is not None:
            fn = st.session_state.feature_names
            lines.append(f"Dataset: {len(st.session_state.df_X)} cycles, "
                         f"{len(fn)} X vars ({', '.join(fn[:6])}{'...' if len(fn)>6 else ''})")
        if st.session_state.model is not None:
            m = st.session_state.model
            n_flag = int(((m['T2']>m['T2_UCL'])|(m['Q']>m['Q_UCL'])).sum())
            lines.append(f"Model: k={m['k']} PCs, T²UCL={m['T2_UCL']:.3f}, "
                         f"QUCL={m['Q_UCL']:.3f}, Phase I flags={n_flag}")
        if st.session_state.mon_files:
            all_t2f = np.concatenate([f['mon']['T2_flag'] for f in st.session_state.mon_files])
            all_qf  = np.concatenate([f['mon']['Q_flag']  for f in st.session_state.mon_files])
            n_any = int((all_t2f|all_qf).sum()); n_tot = len(all_t2f)
            files = ", ".join(f['name'] for f in st.session_state.mon_files)
            lines.append(f"Monitoring: {n_tot} cycles from [{files}], "
                         f"{n_any} anomalies ({n_any/n_tot*100:.1f}%)")
        if st.session_state.anomaly_log:
            lines.append(f"Interventions logged: {len(st.session_state.anomaly_log)}")
        return "\n".join(lines) if lines else "No data loaded yet."

    # Render chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.global_chat:
            if msg['role'] == 'user':
                st.markdown(f"<div class='chat-user'>🧑 {msg['text']}</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-ai'>🤖 {msg['text']}</div>",
                            unsafe_allow_html=True)

    # Input
    user_q = st.text_input("", key="global_chat_input",
                           label_visibility="collapsed",
                           placeholder="Ask anything about your analysis...")

    col_send, col_clear = st.columns([3,1])
    with col_send:
        if st.button("Send", key="global_chat_send", use_container_width=True) \
                and user_q.strip():
            snapshot = build_project_snapshot()
            history_str = "\n".join(
                f"{'User' if m['role']=='user' else 'AI'}: {m['text']}"
                for m in st.session_state.global_chat[-8:]  # last 4 exchanges
            )
            full_prompt = (
                get_process_context() + "\n"
                + f"=== PROJECT STATE ===\n{snapshot}\n"
                + (f"=== CHAT HISTORY ===\n{history_str}\n" if history_str else "")
                + f"User: {user_q}\n"
                + "Answer concisely and practically. "
                  "If you need more information to answer well, ask one specific question."
            )
            with st.spinner("Thinking..."):
                reply, model_used, err = call_ai(full_prompt)
            if err:
                reply = f"Error: {err}"
            st.session_state.global_chat.append({'role':'user', 'text':user_q})
            st.session_state.global_chat.append({'role':'ai',   'text':reply or ""})
            st.rerun()
    with col_clear:
        if st.button("Clear", key="global_chat_clear", use_container_width=True):
            st.session_state.global_chat = []
            st.rerun()


# ═══════════════════════════════════════════
#  WORKFLOW DETECTION
# ═══════════════════════════════════════════

def get_workflow():
    """
    Returns 'spc', 'diagnostic', or 'exploratory' based on saved objective.
    Defaults to 'spc' if context not saved yet.
    """
    obj = st.session_state.get('analysis_objective', '').lower()
    if 'diagnostic' in obj:
        return 'diagnostic'
    if 'exploratory' in obj:
        return 'exploratory'
    return 'spc'


WORKFLOW_LABELS = {
    'spc':         ('🔵 SPC',         '#2980b9', 'Statistical Process Control'),
    'diagnostic':  ('🟠 Diagnostics', '#e67e22', 'Diagnostics & Root Cause'),
    'exploratory': ('🟢 Exploratory', '#27ae60', 'Exploratory Analysis'),
}

WORKFLOW_TAB_GUIDE = {
    'spc': {
        'Dataset':         '1 · Load data and optionally split train/test',
        'PC Selection':    '2 · Choose number of PCs',
        'Calibration':     '3 · Build Phase I model on train set',
        'Loadings/Scores': '4 · Explore model structure',
        'Monitoring':      '5 · Upload new production data → detect anomalies',
        'Summary':         '6 · Root cause analysis on Phase II anomalies',
    },
    'diagnostic': {
        'Dataset':         '1 · Load the full historical dataset',
        'PC Selection':    '2 · Choose number of PCs',
        'Calibration':     '3 · Fit model on full dataset — anomalies = internal flags',
        'Loadings/Scores': '4 · Explore variable structure',
        'Monitoring':      '— Not needed (data already in Calibration)',
        'Summary':         '5 · Root cause analysis on internal anomalies',
    },
    'exploratory': {
        'Dataset':         '1 · Load dataset and explore statistics',
        'PC Selection':    '2 · Choose number of PCs',
        'Calibration':     '3 · Fit model to explore structure',
        'Loadings/Scores': '4 · Main section — loading and score plots',
        'Monitoring':      '— Optional',
        'Summary':         '5 · Variable patterns and correlations',
    },
}


st.markdown("# 🏭 Process Monitor")

# Workflow banner
wf = get_workflow()
wf_label, wf_color, wf_name = WORKFLOW_LABELS[wf]
st.markdown(
    f"<div style='background:{wf_color}22;border:1px solid {wf_color}55;"
    f"border-left:5px solid {wf_color};padding:10px 16px;border-radius:6px;"
    f"margin-bottom:12px;font-size:13px;'>"
    f"<strong>{wf_label} — {wf_name}</strong>"
    + (f" &nbsp;·&nbsp; Set in Process Context tab"
       if st.session_state.context_saved
       else " &nbsp;·&nbsp; ⚙️ Set your objective in the Process Context tab to activate the guided workflow")
    + "</div>",
    unsafe_allow_html=True
)
if st.session_state.context_saved:
    with st.expander("📋 Workflow guide for this objective", expanded=False):
        for tab_name, step in WORKFLOW_TAB_GUIDE[wf].items():
            icon = "⏭️" if step.startswith("—") else "✅"
            st.markdown(f"{icon} **{tab_name}** — {step}")

tab0,tab1,tab2,tab3,tab4,tab5,tab6=st.tabs([
    "⚙️ Process Context","📂 Dataset","📐 PC Selection",
    "🔧 Calibration","📊 Loadings & Scores","🔍 Monitoring",
    "📋 Summary & Root Cause"])


# ═══════════════════════════════════════════
#  TAB 0 — PROCESS CONTEXT
# ═══════════════════════════════════════════
with tab0:
    st.markdown("### ⚙️ Process Context")
    st.markdown("Describe your process and analysis objective. "
                "The AI will use this to contextualise every response.")
    st.markdown("---")
    ctx_input=st.text_area("Process description",
                            value=st.session_state.process_context,
                            height=180,key='ctx_textarea')
    st.markdown("**Analysis objective**")
    obj_options=["Statistical Process Control — build a model and monitor production",
                 "Diagnostics — analyse existing data to find anomalies and root causes",
                 "Exploratory analysis — understand process structure and variable correlations",
                 "Other (describe below)"]
    obj_sel=st.selectbox("Select objective",obj_options,key='obj_select')
    obj_extra=""
    if obj_sel=="Other (describe below)":
        obj_extra=st.text_area("Describe your objective",height=80,key='obj_extra')
    final_obj=obj_extra.strip() if obj_sel=="Other (describe below)" else obj_sel
    col_save,col_clear=st.columns([2,1])
    with col_save:
        if st.button("💾 Save context",type="primary",use_container_width=True):
            st.session_state.process_context=ctx_input.strip()
            st.session_state.analysis_objective=final_obj
            st.session_state.context_saved=True
            st.success("✅ Context saved.")
    with col_clear:
        if st.button("🗑️ Clear",use_container_width=True):
            st.session_state.process_context=''
            st.session_state.analysis_objective=''
            st.session_state.context_saved=False
            st.rerun()
    if st.session_state.context_saved and st.session_state.process_context:
        st.markdown("---")
        prompt_ctx=(
            f"The user described this process:\n{st.session_state.process_context}\n"
            f"Analysis objective: {st.session_state.get('analysis_objective','')}\n\n"
            "Provide a structured summary:\n"
            "1. Process type and key characteristics\n"
            "2. Most critical variables likely to drive quality or anomalies\n"
            "3. Most common failure modes in this type of process\n"
            "4. What PCA-SPC can and cannot detect here\n"
            "5. Specific recommendations given the stated objective\n"
            "Be specific. No generic statements."
        )
        llm_button("Generate process summary",prompt_ctx,key='ai_ctx')


# ═══════════════════════════════════════════
#  TAB 1 — DATASET
# ═══════════════════════════════════════════
with tab1:
    st.markdown("### Load Data")
    up=st.file_uploader("CSV or Excel file",type=['csv','xlsx','xls'],key='up_main')
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
        df_X=df_num[x_cols].copy(); df_Y=df_num[y_valid].copy() if y_valid else pd.DataFrame()
        st.session_state.df_X=df_X; st.session_state.df_Y=df_Y
        st.session_state.feature_names=x_cols; st.session_state.y_names=y_valid

        c1,c2,c3,c4=st.columns(4)
        c1.metric("Cycles",len(df_X)); c2.metric("X Variables",len(x_cols))
        c3.metric("Y Variables",len(y_valid) if y_valid else "—")
        miss=int(df_X.isnull().sum().sum())
        c4.metric("Missing values",miss,delta="⚠️" if miss>0 else "✅",delta_color="off")

        col_a,col_b=st.columns(2)
        with col_a:
            st.markdown("**X Variables (process)**")
            st.dataframe(pd.DataFrame({'Index':range(1,len(x_cols)+1),'Variable':x_cols}),
                         use_container_width=True,hide_index=True)
        with col_b:
            if y_valid:
                st.markdown("**Y Variables (quality)**")
                st.dataframe(pd.DataFrame({'Variable':y_valid}),
                             use_container_width=True,hide_index=True)

        st.markdown("#### Descriptive Statistics")
        desc=df_X.describe().T.round(3)
        desc['cv%']=(desc['std']/desc['mean'].abs()*100).round(1)
        st.dataframe(desc,use_container_width=True)
        if miss>0:
            st.warning("Missing values detected — will be replaced with column mean.")

        # ── TRAIN / TEST SPLIT — only for SPC workflow ─────
        st.markdown("---")
        wf_ds = get_workflow()
        if wf_ds == 'spc':
            st.markdown("#### Train / Test Split")
            st.caption("Split into Phase I (calibration) and Phase II (test). "
                       "Skip if you prefer to upload separate monitoring files.")
            split_method=st.radio("Split method",["Temporal","Random"],
                                   horizontal=True,key='split_radio')
            st.session_state.split_method=split_method
            n_total=len(df_X)
            if split_method=="Temporal":
                split_ratio=st.slider("Train size (%)",50,90,70,5,key='split_ratio_slider')
                st.session_state.split_ratio=split_ratio/100
                split_row=int(n_total*(split_ratio/100))
                st.session_state.split_row=split_row
                st.info(f"Train: first **{split_row}** cycles | Test: last **{n_total-split_row}** cycles")
                fig_split=go.Figure()
                fig_split.add_vrect(x0=0,x1=split_row,fillcolor='#2980b9',opacity=0.12,
                                    line_width=0,annotation_text=f'Train ({split_row})',
                                    annotation_position='top left',annotation_font_size=10)
                fig_split.add_vrect(x0=split_row,x1=n_total,fillcolor='#e74c3c',opacity=0.12,
                                    line_width=0,annotation_text=f'Test ({n_total-split_row})',
                                    annotation_position='top right',annotation_font_size=10)
                fig_split.add_vline(x=split_row,line_dash='dash',line_color='#e74c3c',line_width=2)
                first_col=x_cols[0]
                fig_split.add_scatter(x=list(range(n_total)),
                                      y=df_X[first_col].fillna(df_X[first_col].mean()).tolist(),
                                      mode='lines',line=dict(color='#2c3e50',width=1),
                                      name=first_col)
                fig_split.update_layout(height=200,margin=dict(l=10,r=10,t=10,b=30),
                                        plot_bgcolor='#fafafa',paper_bgcolor='white',
                                        showlegend=False,xaxis_title='Cycle')
                st.plotly_chart(fig_split,use_container_width=True,key='split_preview')
            else:
                split_ratio=st.slider("Train size (%)",50,90,70,5,key='split_ratio_slider_r')
                st.session_state.split_ratio=split_ratio/100
                split_row=int(n_total*(split_ratio/100))
                st.session_state.split_row=split_row
                st.info(f"Train: **{split_row}** random cycles | Test: **{n_total-split_row}** random cycles")
            if st.button("Apply split",key='btn_split',type='primary'):
                df_Xf=df_X.fillna(df_X.mean())
                if split_method=="Temporal":
                    sr=st.session_state.split_row
                    df_tr=df_Xf.iloc[:sr].reset_index(drop=True)
                    df_te=df_Xf.iloc[sr:].reset_index(drop=True)
                else:
                    rng=np.random.default_rng(42)
                    idx=rng.permutation(len(df_Xf))
                    sr=st.session_state.split_row
                    df_tr=df_Xf.iloc[idx[:sr]].reset_index(drop=True)
                    df_te=df_Xf.iloc[idx[sr:]].reset_index(drop=True)
                st.session_state.df_train=df_tr
                st.session_state.df_test_builtin=df_te
                st.success(f"✅ Split applied — Train: {len(df_tr)} | Test: {len(df_te)}")
        else:
            # Diagnostic / Exploratory — use full dataset, no split needed
            st.markdown(
                f"<div class='ok-box'>"
                f"ℹ️ <strong>{WORKFLOW_LABELS[wf_ds][2]}</strong> workflow — "
                f"train/test split not required. "
                f"The full dataset will be used for calibration.</div>",
                unsafe_allow_html=True)
            # Clear any previous split so calibration uses full data
            st.session_state.df_train = None
            st.session_state.df_test_builtin = None

        st.markdown("---")
        df_Xf2=df_X.fillna(df_X.mean())

        # Compact but complete dataset summary for AI
        desc2 = df_Xf2.describe().T
        desc2['cv%'] = (desc2['std'] / desc2['mean'].abs() * 100).round(1)
        desc2['range'] = (desc2['max'] - desc2['min']).round(3)
        desc2['skew'] = df_Xf2.skew().round(2)

        # Top 5 by CV%
        top_cv = desc2.nlargest(5,'cv%')[['mean','std','cv%','min','max']].round(3)
        top_cv_str = "\n".join(
            f"  {v}: mean={r['mean']}, std={r['std']}, CV={r['cv%']}%, "
            f"min={r['min']}, max={r['max']}"
            for v,r in top_cv.iterrows()
        )
        # Variables with suspicious min or max (beyond 3 sigma)
        outlier_vars = []
        for v in df_Xf2.columns:
            mu,s = df_Xf2[v].mean(), df_Xf2[v].std()
            if s > 0:
                if df_Xf2[v].min() < mu-4*s or df_Xf2[v].max() > mu+4*s:
                    outlier_vars.append(f"{v}(min={df_Xf2[v].min():.2f}, max={df_Xf2[v].max():.2f})")
        outlier_str = ", ".join(outlier_vars[:5]) if outlier_vars else "none"

        miss_str = ", ".join(f"{c}:{n}" for c,n in df_X.isnull().sum().items() if n>0) or "none"

        corr = df_Xf2.corr().abs()
        corr_np = corr.values.copy(); np.fill_diagonal(corr_np, 0)
        corr_f = pd.DataFrame(corr_np, index=corr.index, columns=corr.columns)
        top_corr = ", ".join(f"{a}↔{b}:{v:.2f}"
                             for (a,b),v in corr_f.unstack().nlargest(5).items())

        prompt_ds=(
            f"Dataset: {len(df_X)} obs, {len(x_cols)} X vars, "
            f"{len(y_valid)} Y ({', '.join(y_valid) if y_valid else 'none'}).\n"
            f"Top 5 variables by variability (CV%):\n{top_cv_str}\n"
            f"Possible outliers (beyond 4σ): {outlier_str}\n"
            f"Missing values: {miss_str}\n"
            f"Top correlations: {top_corr}\n\n"
            "5 bullet points:\n"
            "• Data quality issues (missing, outliers)\n"
            "• Variables with unusual variability — is it expected for this process?\n"
            "• Key correlations relevant for PCA\n"
            "• Any concern before modelling\n"
            "• One specific recommendation\n"
            "Reference specific variable names. Be direct."
        )
        llm_button("Analyse dataset", prompt_ds, key='ai_ds')


# ═══════════════════════════════════════════
#  TAB 2 — PC SELECTION
# ═══════════════════════════════════════════
with tab2:
    if st.session_state.df_X is None:
        st.info("⬆️ Load the dataset first.")
    else:
        df_X=st.session_state.df_X.fillna(st.session_state.df_X.mean())
        # Use train split if available
        if st.session_state.df_train is not None:
            df_for_pca=st.session_state.df_train
            st.info(f"ℹ️ Using train split ({len(df_for_pca)} cycles) for PC selection.")
        else:
            df_for_pca=df_X
        X_raw=df_for_pca.values
        sc_tmp=StandardScaler(); Xs_tmp=sc_tmp.fit_transform(X_raw)
        n_obs,n_vars=Xs_tmp.shape; max_k=min(20,n_vars-1,n_obs-2)
        pca_tmp=PCA(n_components=max_k,svd_solver='full',random_state=42); pca_tmp.fit(Xs_tmp)
        eigs=pca_tmp.explained_variance_
        evr_all=pca_tmp.explained_variance_ratio_*100
        cum=np.cumsum(evr_all); ks=list(range(1,max_k+1))
        k_kaiser=max(2,min(int(np.sum(eigs>1)),max_k))
        k_90=int(np.argmax(cum>=90.0))+1; k_95=int(np.argmax(cum>=95.0))+1

        st.markdown("### Select Criterion")
        grafico=st.radio("",["Scree plot","Cumulative variance","RMSECV"],
                         horizontal=True,key='pc_radio',label_visibility='collapsed')
        if grafico=="Scree plot":
            fig=go.Figure()
            fig.add_scatter(x=ks,y=evr_all.tolist(),mode='lines+markers',
                            line=dict(color='#2c3e50',width=2),marker=dict(size=6),
                            hovertemplate='PC%{x}<br>%{y:.2f}%<extra></extra>')
            fig.add_hline(y=float(100/n_vars),line_dash='dot',line_color='#7f8c8d',
                         annotation_text=f'Kaiser ({100/n_vars:.1f}%)',annotation_position='right')
            fig.add_vline(x=k_kaiser,line_dash='dash',line_color='#e74c3c',
                         annotation_text=f'k={k_kaiser}',annotation_position='top right')
            fig.update_layout(xaxis_title='PC',yaxis_title='Variance (%)',height=320,
                             margin=dict(l=10,r=10,t=20,b=30),
                             plot_bgcolor='#fafafa',paper_bgcolor='white',showlegend=False)
            st.plotly_chart(fig,use_container_width=True,key='chart_scree')
            st.info(f"💡 Kaiser: **k = {k_kaiser}** PCs")
        elif grafico=="Cumulative variance":
            fig=go.Figure()
            fig.add_scatter(x=ks,y=cum.tolist(),mode='lines+markers',
                            line=dict(color='#2980b9',width=2),marker=dict(size=6),
                            hovertemplate='PC%{x}<br>%{y:.1f}%<extra></extra>')
            fig.add_hline(y=90,line_dash='dash',line_color='#f39c12',
                         annotation_text='90%',annotation_position='right')
            fig.add_hline(y=95,line_dash='dash',line_color='#e74c3c',
                         annotation_text='95%',annotation_position='right')
            fig.add_vline(x=k_90,line_dash='dot',line_color='#f39c12',
                         annotation_text=f'k={k_90}',annotation_position='top left')
            fig.add_vline(x=k_95,line_dash='dot',line_color='#e74c3c',
                         annotation_text=f'k={k_95}',annotation_position='top right')
            fig.update_layout(xaxis_title='Number of PCs',yaxis_title='Cumulative variance (%)',
                             yaxis=dict(range=[0,105]),height=320,
                             margin=dict(l=10,r=10,t=20,b=30),
                             plot_bgcolor='#fafafa',paper_bgcolor='white',showlegend=False)
            st.plotly_chart(fig,use_container_width=True,key='chart_cumvar')
            st.info(f"💡 90% → k={k_90} | 95% → k={k_95}")
        else:
            if not st.session_state.rmsecv_computed:
                if st.button("▶ Compute RMSECV",type='primary',key='btn_rmsecv'):
                    with st.spinner("Computing RMSECV..."):
                        bk,_,rcv=compute_rmsecv(Xs_tmp,max_k)
                    st.session_state.rmsecv_result=(bk,rcv)
                    st.session_state.rmsecv_computed=True; st.rerun()
                else:
                    st.info("Click to compute RMSECV.")
            else:
                bk,rcv=st.session_state.rmsecv_result
                fig=go.Figure()
                fig.add_scatter(x=ks,y=rcv.tolist(),mode='lines+markers',
                                line=dict(color='#16a085',width=2),
                                marker=dict(size=[12 if i+1==bk else 6 for i in range(max_k)],
                                            color=['#e74c3c' if i+1==bk else '#16a085'
                                                   for i in range(max_k)]),
                                hovertemplate='PC%{x}<br>RMSECV:%{y:.4f}<extra></extra>')
                fig.add_vline(x=bk,line_dash='dash',line_color='#e74c3c',
                             annotation_text=f'min k={bk}',annotation_position='top right')
                fig.update_layout(xaxis_title='Number of PCs',yaxis_title='RMSECV',height=320,
                                 margin=dict(l=10,r=10,t=20,b=30),
                                 plot_bgcolor='#fafafa',paper_bgcolor='white',showlegend=False)
                st.plotly_chart(fig,use_container_width=True,key='chart_rmsecv')
                st.info(f"💡 RMSECV minimum: **k = {bk}** PCs")
                if st.button("🔄 Recalculate",key='btn_rmsecv_reset'):
                    st.session_state.rmsecv_computed=False; st.rerun()

        st.markdown("---")
        rs1,rs2,rs3=st.columns(3)
        rs1.metric("Kaiser",f"k={k_kaiser}"); rs2.metric("90% var",f"k={k_90}")
        rs3.metric("RMSECV",f"k={st.session_state.rmsecv_result[0]}"
                   if st.session_state.rmsecv_computed else "— compute above")
        k_chosen=st.number_input("Number of PCs for the model",min_value=2,max_value=max_k,
                                  value=k_kaiser,step=1,key='k_input')
        st.session_state.k_chosen=int(k_chosen)
        st.success(f"✅ k = **{k_chosen}** PCs — explained variance: **{cum[k_chosen-1]:.1f}%**")
        st.markdown("---")
        prompt_pc=(
            f"PCA: {n_obs} obs, {n_vars} vars. "
            f"Kaiser k={k_kaiser}, 90%→k={k_90}, 95%→k={k_95}. "
            f"Selected: k={k_chosen} ({cum[k_chosen-1]:.1f}% var).\n"
            "3 bullets: (1) Is k appropriate? (2) What phenomena do first PCs capture? "
            "(3) Any concern? Be specific."
        )
        llm_button("Interpret PC selection",prompt_pc,key='ai_pc')


# ═══════════════════════════════════════════
#  TAB 3 — CALIBRATION
# ═══════════════════════════════════════════
with tab3:
    if st.session_state.df_X is None:
        st.info("⬆️ Load the dataset first.")
    elif st.session_state.k_chosen is None:
        st.info("⬆️ Choose the number of PCs first.")
    else:
        # Determine training data source
        if st.session_state.df_train is not None:
            df_cal=st.session_state.df_train.copy()
            st.info(f"ℹ️ Using train split: **{len(df_cal)}** cycles")
        else:
            df_cal=st.session_state.df_X.fillna(st.session_state.df_X.mean()).copy()
            st.caption("No split applied — using full dataset for calibration.")
        x_cols=st.session_state.feature_names
        X_raw=df_cal[x_cols].values if set(x_cols).issubset(df_cal.columns) else df_cal.values
        k_use=st.session_state.k_chosen

        st.markdown("### 🧹 Data Cleaning (optional)")
        use_clean=st.toggle("Enable iterative cleaning",value=False,key='toggle_clean')
        clean_mask=np.ones(len(X_raw),dtype=bool)
        if use_clean:
            alpha_clean=st.slider("Cleaning confidence",0.95,0.999,0.99,0.001,format="%.3f")
            if st.button("▶ Run cleaning",key='btn_clean'):
                sc_k=StandardScaler(); Xs_k=sc_k.fit_transform(X_raw)
                eig_k=PCA(svd_solver='full').fit(Xs_k).explained_variance_
                k_cl=max(2,min(int(np.sum(eig_k>1)),X_raw.shape[1]-1,X_raw.shape[0]-2))
                with st.spinner("Iterative cleaning..."):
                    clean_mask,log_df=iterative_cleaning(X_raw,k_cl,alpha_clean)
                n_rem=int((~clean_mask).sum()); pct=n_rem/len(X_raw)*100
                st.success(f"✅ Removed **{n_rem}** cycles ({pct:.1f}%) — Remaining: **{clean_mask.sum()}**")
                st.dataframe(log_df,use_container_width=True,hide_index=True)
                if pct>20:
                    st.warning("⚠️ >20% removed — calibration period may not have been stable.")

        st.markdown("### 🔧 Build Model")
        if st.button("Build Phase I Model",type="primary",use_container_width=True,key='btn_fit'):
            X_clean=X_raw[clean_mask]
            with st.spinner("Fitting PCA-SPC..."):
                mdl=fit_pca_spc(X_clean,k_use,alpha)
                mdl['feature_names']=x_cols
                st.session_state.model=mdl
                # If built-in test set exists, pre-run monitoring
                if st.session_state.df_test_builtin is not None:
                    df_te=st.session_state.df_test_builtin
                    X_te=df_te[x_cols].values if set(x_cols).issubset(df_te.columns) else df_te.values
                    mon_te=monitor_new(mdl,X_te)
                    st.session_state.mon_files=[{
                        'name':'Built-in test set','n_rows':len(X_te),'mon':mon_te}]
            st.rerun()

        if st.session_state.model is not None:
            mdl=st.session_state.model
            if st.button("🔄 Rebuild model",key='btn_refit'):
                st.session_state.model=None; st.session_state.mon=None
                st.session_state.mon_files=[]; st.rerun()
            n_flag=int(((mdl['T2']>mdl['T2_UCL'])|(mdl['Q']>mdl['Q_UCL'])).sum())
            pct_f=n_flag/len(mdl['T2'])*100
            st.success("✅ Model built.")
            cm1,cm2,cm3=st.columns(3)
            cm1.metric("T² UCL",f"{mdl['T2_UCL']:.3f}")
            cm2.metric("Q UCL",f"{mdl['Q_UCL']:.3f}")
            cm3.metric("Residual flags",f"{n_flag} ({pct_f:.1f}%)",
                       delta="🟢 OK" if pct_f<5 else "⚠️ Check",delta_color="off")
            fT2=mdl['T2']>mdl['T2_UCL']; fQ=mdl['Q']>mdl['Q_UCL']
            st.plotly_chart(chart_line(mdl['T2'],mdl['T2_UCL'],
                                       'Phase I — Hotelling T²','#2c3e50',fT2),
                            use_container_width=True,key='p1_t2')
            st.plotly_chart(chart_line(mdl['Q'],mdl['Q_UCL'],
                                       'Phase I — Q (SPE)','#16a085',fQ),
                            use_container_width=True,key='p1_q')
            st.markdown("#### Contribution plots — Phase I anomalous cycles")
            result_p1=show_anomaly_table_and_contrib(
                mdl,mdl['T2'],mdl['Q'],mdl['X_scaled'],mdl['E'],mdl['scores'],
                mdl['feature_names'],'p1_table','p1')
            if result_p1:
                obs_p1,t2_p1,q_p1,ratio_p1,top_t2_p1,top_q_p1=result_p1
                st.markdown("---")
                v_t2=[x_cols[i] for i in top_t2_p1]; v_q=[x_cols[i] for i in top_q_p1]
                prompt_p1=(
                    f"Phase I outlier {obs_p1}: T²={t2_p1:.2f}×UCL, Q={q_p1:.2f}×UCL. "
                    f"T² vars: {', '.join(v_t2)}. Q vars: {', '.join(v_q)}.\n"
                    "3 bullets: (1) Remove it? (2) What does it suggest? (3) Action."
                )
                llm_button("Interpret Phase I anomaly",prompt_p1,key=f'ai_p1_{obs_p1}')
            else:
                st.success("✅ No anomalies in training set — clean calibration.")
            st.markdown("---")
            prompt_cal=(
                f"Phase I model: {len(mdl['T2'])} cycles, k={mdl['k']} PCs, "
                f"T²UCL={mdl['T2_UCL']:.3f}, QUCL={mdl['Q_UCL']:.3f}, "
                f"α={alpha}, flags={n_flag} ({pct_f:.1f}%).\n"
                "3 bullets: (1) UCL values reasonable? (2) Flags acceptable? "
                "(3) Ready for Phase II? Be specific."
            )
            llm_button("Interpret calibration model",prompt_cal,key='ai_cal')


# ═══════════════════════════════════════════
#  TAB 4 — LOADINGS & SCORES
# ═══════════════════════════════════════════
with tab4:
    if st.session_state.model is None:
        st.info("⬆️ Build the model first.")
    else:
        mdl=st.session_state.model
        fn=mdl['feature_names']; P=mdl['loadings']
        k_m=mdl['k']; evr_m=mdl['evr']; lam_m=mdl['eigenvalues']
        T_m=mdl['scores']; flag_m=(mdl['T2']>mdl['T2_UCL'])|(mdl['Q']>mdl['Q_UCL'])
        with st.expander("📋 Variable Index",expanded=False):
            st.dataframe(pd.DataFrame({'Index':range(1,len(fn)+1),'Variable':fn}),
                         use_container_width=True,hide_index=True)
        all_pairs=list(combinations(range(k_m),2))
        pair_labels=[f"PC{a+1} vs PC{b+1} ({evr_m[a]:.1f}%+{evr_m[b]:.1f}%)"
                     for a,b in all_pairs]
        st.markdown("### Select Chart")
        tipo=st.radio("Type",["Loading plot","Score plot"],horizontal=True,key='ls_tipo')
        coppia_idx=st.selectbox("PC pair",options=range(len(all_pairs)),
                                format_func=lambda i: pair_labels[i],key='ls_coppia')
        pc_i,pc_j=all_pairs[coppia_idx]
        if tipo=="Loading plot":
            st.plotly_chart(chart_loading(P,fn,evr_m,pc_i,pc_j),
                            use_container_width=True,key='load_chart')
            st.caption("Number = variable index. Hover to see full name. "
                       "Close = correlated | Opposite = negatively correlated.")
            df_load=pd.DataFrame({
                'Index':range(1,P.shape[0]+1),'Variable':fn,
                f'PC{pc_i+1}':P[:,pc_i].round(4),f'PC{pc_j+1}':P[:,pc_j].round(4),
                'Distance':np.sqrt(P[:,pc_i]**2+P[:,pc_j]**2).round(4),
            }).sort_values('Distance',ascending=False)
            st.dataframe(df_load,use_container_width=True,hide_index=True)
        else:
            st.plotly_chart(chart_score(T_m, lam_m, mdl['T2_UCL'], evr_m,
                                        pc_i, pc_j, flag_m,
                                        alpha=alpha,
                                        n_train=len(T_m)),
                            use_container_width=True,key='score_chart')
            st.caption(
                f"⚫ Inside {int(alpha*100)}% ellipse  |  "
                f"🔴 Outside {int(alpha*100)}% ellipse  |  "
                f"Ellipse = bivariate F limit (k=2, n={len(T_m)}, α={alpha})."
            )
        st.markdown("---")
        top5=np.argsort(np.sqrt(P[:,pc_i]**2+P[:,pc_j]**2))[::-1][:5]
        top_str=", ".join(f"{fn[i]}({P[i,pc_i]:.3f}/{P[i,pc_j]:.3f})" for i in top5)
        prompt_load=(
            f"{'Loading' if tipo=='Loading plot' else 'Score'} plot "
            f"PC{pc_i+1} ({evr_m[pc_i]:.1f}%) vs PC{pc_j+1} ({evr_m[pc_j]:.1f}%). "
            f"Top vars: {top_str}.\n"
            "3 bullets: (1) What phenomena do these PCs represent? "
            "(2) What do top vars tell us? (3) Notable correlations?"
        )
        llm_button("Interpret chart",prompt_load,key='ai_load')


# ═══════════════════════════════════════════
#  TAB 5 — MONITORING
# ═══════════════════════════════════════════
with tab5:
    if st.session_state.model is None:
        st.info("⬆️ Build the model first.")
    else:
        mdl=st.session_state.model; fn=mdl['feature_names']
        wf_mon = get_workflow()

        # In diagnostic/exploratory — redirect to Calibration Phase I results
        if wf_mon in ('diagnostic', 'exploratory'):
            st.markdown(
                f"<div class='ctx-box'>"
                f"<strong>ℹ️ {WORKFLOW_LABELS[wf_mon][2]} workflow</strong><br>"
                f"In this workflow the anomalies are already identified in the "
                f"<strong>Calibration</strong> tab (Phase I flags). "
                f"Go to <strong>Summary & Root Cause</strong> to see the full analysis.<br><br>"
                f"You can still upload additional files here if you want to compare "
                f"a second dataset against the same model."
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown("---")
            st.markdown("#### Optional: compare additional files against this model")
        st.caption("Upload one or more files. Each file is added to the monitoring session "
                   "without overwriting the others. Data is concatenated chronologically.")

        up_test=st.file_uploader("Add CSV or Excel file",
                                  type=['csv','xlsx','xls'],key='up_test')
        if up_test:
            df_tr=(pd.read_csv(up_test) if up_test.name.endswith('.csv')
                   else pd.read_excel(up_test))
            df_tr.columns=df_tr.columns.astype(str)
            miss_c=[c for c in fn if c not in df_tr.columns]
            if miss_c:
                st.error(f"Missing columns: {miss_c}")
            else:
                df_new=df_tr[fn].copy().fillna(df_tr[fn].mean())
                mon_new_=monitor_new(mdl,df_new.values)
                fname=up_test.name
                # Avoid adding the same file twice
                existing_names=[f['name'] for f in st.session_state.mon_files]
                if fname not in existing_names:
                    st.session_state.mon_files.append({
                        'name': fname,
                        'n_rows': len(df_new),
                        'mon': mon_new_
                    })
                    st.success(f"✅ Added: **{fname}** ({len(df_new)} cycles)")
                else:
                    st.info(f"ℹ️ File **{fname}** already loaded.")

        # File manager
        if st.session_state.mon_files:
            st.markdown("#### Loaded files")
            for fi, fobj in enumerate(st.session_state.mon_files):
                col_f, col_rm = st.columns([5,1])
                col_f.markdown(
                    f"<span class='file-tag' style='background:{FILE_COLORS[fi%len(FILE_COLORS)]}22;"
                    f"border:1px solid {FILE_COLORS[fi%len(FILE_COLORS)]}44;'>"
                    f"●</span> **{fobj['name']}** — {fobj['n_rows']} cycles",
                    unsafe_allow_html=True)
                if col_rm.button("✕", key=f'rm_file_{fi}'):
                    st.session_state.mon_files.pop(fi)
                    st.rerun()

        if st.session_state.mon_files:
            # Concatenate all monitoring results
            all_t2=np.concatenate([f['mon']['T2']   for f in st.session_state.mon_files])
            all_q =np.concatenate([f['mon']['Q']    for f in st.session_state.mon_files])
            all_t2f=np.concatenate([f['mon']['T2_flag'] for f in st.session_state.mon_files])
            all_qf =np.concatenate([f['mon']['Q_flag']  for f in st.session_state.mon_files])
            all_Xns=np.concatenate([f['mon']['Xn_s'] for f in st.session_state.mon_files])
            all_En =np.concatenate([f['mon']['En']   for f in st.session_state.mon_files])
            all_Tn =np.concatenate([f['mon']['Tn']   for f in st.session_state.mon_files])

            # File boundaries for chart shading
            boundaries=[0]
            for fobj in st.session_state.mon_files:
                boundaries.append(boundaries[-1]+fobj['n_rows'])
            file_names=[f['name'] for f in st.session_state.mon_files]

            n_test=len(all_t2)
            n_t2=int(all_t2f.sum()); n_q=int(all_qf.sum())
            n_any=int((all_t2f|all_qf).sum()); pct=n_any/n_test*100
            stato=("🟢 STABLE" if pct<5 else "🟡 WARNING" if pct<15 else "🔴 ANOMALIES")

            c1,c2,c3,c4=st.columns(4)
            c1.metric("Total cycles",n_test)
            c2.metric("T² anomalies",f"{n_t2} ({n_t2/n_test*100:.1f}%)")
            c3.metric("Q anomalies",f"{n_q} ({n_q/n_test*100:.1f}%)")
            c4.metric("Status",stato)

            # Per-file summary table
            st.markdown("#### Results per file")
            rows_pf=[]
            for fobj in st.session_state.mon_files:
                m=fobj['mon']
                nf=len(m['T2']); nt2=int(m['T2_flag'].sum()); nq=int(m['Q_flag'].sum())
                na=int((m['T2_flag']|m['Q_flag']).sum())
                rows_pf.append({
                    'File': fobj['name'], 'Cycles': nf,
                    'T² anom.': f"{nt2} ({nt2/nf*100:.1f}%)",
                    'Q anom.':  f"{nq}  ({nq/nf*100:.1f}%)",
                    'Any anom.':f"{na}  ({na/nf*100:.1f}%)",
                    'Status':   "🟢" if na/nf<0.05 else "🟡" if na/nf<0.15 else "🔴"
                })
            st.dataframe(pd.DataFrame(rows_pf),use_container_width=True,hide_index=True)

            st.markdown("---")
            prompt_ov=(
                f"Phase II: {n_test} cycles across {len(st.session_state.mon_files)} files. "
                f"T² anomalies: {n_t2} ({n_t2/n_test*100:.1f}%). "
                f"Q anomalies: {n_q} ({n_q/n_test*100:.1f}%). Status: {stato}.\n\n"
                "2-3 lines: is the process stable? Any drift or trend across files? "
                "What should the shift supervisor do right now? Be direct."
            )
            llm_button("Interpret process status",prompt_ov,key='ai_overview')

            st.markdown("---")
            st.markdown("### Control Charts — all files")
            st.caption("Shaded regions indicate different files.")
            st.plotly_chart(
                chart_line_multi(all_t2,mdl['T2_UCL'],'Phase II — Hotelling T²',
                                 '#2980b9',all_t2f,file_names,boundaries),
                use_container_width=True,key='p2_t2')
            st.plotly_chart(
                chart_line_multi(all_q,mdl['Q_UCL'],'Phase II — Q (SPE)',
                                 '#27ae60',all_qf,file_names,boundaries),
                use_container_width=True,key='p2_q')

            st.markdown("### 🔍 Anomaly Analysis")
            result=show_anomaly_table_and_contrib(
                mdl,all_t2,all_q,all_Xns,all_En,all_Tn,
                fn,'p2_table','p2')
            if result:
                obs,t2_obs,q_obs,ratio,top_t2,top_q=result
                # Identify which file this cycle belongs to
                file_of_obs="unknown"
                for fi,(start,end) in enumerate(zip(boundaries[:-1],boundaries[1:])):
                    if start<=obs<end:
                        file_of_obs=file_names[fi]; break
                st.caption(f"Cycle {obs} — from file: **{file_of_obs}**")
                st.markdown("---")
                v_t2=[fn[i] for i in top_t2]; v_q=[fn[i] for i in top_q]
                prompt_an=(
                    f"Anomaly cycle {obs}: T²={t2_obs:.2f}×UCL, Q={q_obs:.2f}×UCL. "
                    f"T² vars: {', '.join(v_t2)}. Q vars: {', '.join(v_q)}.\n"
                    "3 bullets for supervisor: (1) What is happening physically? "
                    "(2) Most likely cause? (3) Immediate action? No statistics."
                )
                llm_button("Explain anomaly to technician",prompt_an,key=f'ai_an_{obs}')
                st.markdown("#### 📝 Log Intervention")
                with st.form(f"log_{obs}"):
                    azione=st.text_area("Corrective action taken",height=70)
                    if st.form_submit_button("💾 Save",use_container_width=True):
                        if azione:
                            st.session_state.anomaly_log.append({
                                'Cycle':obs,'File':file_of_obs,
                                'T²':round(t2_obs,3),'Q':round(q_obs,3),
                                'Severity':f"{ratio:.2f}×UCL",'Action':azione})
                            st.success("✅ Saved.")
            if st.session_state.anomaly_log:
                st.markdown("#### 📋 Intervention Log")
                st.dataframe(pd.DataFrame(st.session_state.anomaly_log),
                             use_container_width=True)


# ═══════════════════════════════════════════
#  TAB 6 — SUMMARY & ROOT CAUSE
# ═══════════════════════════════════════════
with tab6:
    st.markdown("### 📋 Summary & Root Cause Analysis")
    if st.session_state.model is None:
        st.info("⬆️ Build the calibration model first.")
    else:
        mdl=st.session_state.model; fn=mdl['feature_names']
        wf_sum = get_workflow()

        st.markdown("#### Model summary")
        s1,s2,s3,s4=st.columns(4)
        s1.metric("k PCs",mdl['k'])
        s2.metric("T² UCL",f"{mdl['T2_UCL']:.3f}")
        s3.metric("Q UCL",f"{mdl['Q_UCL']:.3f}")
        n_flag_p1=int(((mdl['T2']>mdl['T2_UCL'])|(mdl['Q']>mdl['Q_UCL'])).sum())
        s4.metric("Phase I flags",f"{n_flag_p1} ({n_flag_p1/len(mdl['T2'])*100:.1f}%)")

        # For diagnostic/exploratory: use Phase I data directly
        # For spc: require Phase II monitoring data
        if wf_sum in ('diagnostic', 'exploratory'):
            # Use Phase I flags as the anomaly source
            all_t2   = mdl['T2'];        all_q   = mdl['Q']
            all_t2f  = mdl['T2']>mdl['T2_UCL']
            all_qf   = mdl['Q'] >mdl['Q_UCL']
            all_Xns  = mdl['X_scaled'];  all_En  = mdl['E']
            all_Tn   = mdl['scores']
            source_label = "Phase I (full dataset)"
            st.markdown(
                f"<div class='ctx-box'>ℹ️ <strong>{WORKFLOW_LABELS[wf_sum][2]}</strong> — "
                f"showing anomalies from Phase I calibration on the full dataset.</div>",
                unsafe_allow_html=True)
        elif st.session_state.mon_files:
            all_t2  = np.concatenate([f['mon']['T2']       for f in st.session_state.mon_files])
            all_q   = np.concatenate([f['mon']['Q']        for f in st.session_state.mon_files])
            all_t2f = np.concatenate([f['mon']['T2_flag']  for f in st.session_state.mon_files])
            all_qf  = np.concatenate([f['mon']['Q_flag']   for f in st.session_state.mon_files])
            all_Xns = np.concatenate([f['mon']['Xn_s']     for f in st.session_state.mon_files])
            all_En  = np.concatenate([f['mon']['En']       for f in st.session_state.mon_files])
            all_Tn  = np.concatenate([f['mon']['Tn']       for f in st.session_state.mon_files])
            source_label = "Phase II monitoring"
        else:
            st.info("⬆️ Load Phase II monitoring data in the Monitoring tab.")
            st.stop()
            all_t2=all_q=all_t2f=all_qf=all_Xns=all_En=all_Tn=None
            source_label=""

        if all_t2 is not None:
            n_test=len(all_t2); n_t2=int(all_t2f.sum()); n_q=int(all_qf.sum())
            n_any=int((all_t2f|all_qf).sum()); pct=n_any/n_test*100
            stato=("🟢 STABLE" if pct<5 else "🟡 WARNING" if pct<15 else "🔴 ANOMALIES")

            st.markdown(f"#### Analysis results — {source_label}")
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Total cycles",n_test)
            m2.metric("T² anomalies",f"{n_t2} ({n_t2/n_test*100:.1f}%)")
            m3.metric("Q anomalies",f"{n_q} ({n_q/n_test*100:.1f}%)")
            m4.metric("Status",stato)

            flagged_idx=np.where(all_t2f|all_qf)[0]
            if len(flagged_idx)>0:
                st.markdown("#### Top anomalies")
                df_top=pd.DataFrame({
                    'Cycle':        flagged_idx,
                    'T²':           all_t2[flagged_idx].round(3),
                    'T²/UCL':       (all_t2[flagged_idx]/mdl['T2_UCL']).round(2),
                    'Q':            all_q[flagged_idx].round(3),
                    'Q/UCL':        (all_q[flagged_idx]/mdl['Q_UCL']).round(2),
                    'Severity×UCL': np.maximum(all_t2[flagged_idx]/mdl['T2_UCL'],
                                               all_q[flagged_idx]/mdl['Q_UCL']).round(2),
                }).sort_values('Severity×UCL',ascending=False).head(10).reset_index(drop=True)
                st.dataframe(df_top,use_container_width=True,hide_index=True)

                st.markdown("#### Most recurrent variables in anomalies")
                P=mdl['loadings']; lam=mdl['eigenvalues']
                t2_ucl_v=mdl['T2contrib_UCL']; t2_lcl_v=mdl['T2contrib_LCL']
                q_ucl_v=mdl['Qcontrib_UCL'];   q_lcl_v=mdl['Qcontrib_LCL']
                t2_cnt=np.zeros(len(fn)); q_cnt=np.zeros(len(fn))
                for obs in flagged_idx:
                    c_t2=all_Xns[obs]*(P@(all_Tn[obs]/lam)); c_q=all_En[obs]
                    for i in range(len(fn)):
                        if c_t2[i]>t2_ucl_v[i] or c_t2[i]<t2_lcl_v[i]: t2_cnt[i]+=1
                        if c_q[i]>q_ucl_v[i]   or c_q[i]<q_lcl_v[i]:   q_cnt[i]+=1
                df_vars=pd.DataFrame({
                    'Index':range(1,len(fn)+1),'Variable':fn,
                    'T² exceedances':t2_cnt.astype(int),
                    'Q exceedances': q_cnt.astype(int),
                    'Total':(t2_cnt+q_cnt).astype(int),
                }).sort_values('Total',ascending=False)
                df_vars_top=df_vars[df_vars['Total']>0].head(10)
                if len(df_vars_top)>0:
                    st.dataframe(df_vars_top,use_container_width=True,hide_index=True)
                    fig_var=go.Figure()
                    fig_var.add_bar(x=df_vars_top['Variable'].tolist(),
                                    y=df_vars_top['T² exceedances'].tolist(),
                                    name='T² exceedances',marker_color='#2980b9')
                    fig_var.add_bar(x=df_vars_top['Variable'].tolist(),
                                    y=df_vars_top['Q exceedances'].tolist(),
                                    name='Q exceedances',marker_color='#e74c3c')
                    fig_var.update_layout(barmode='stack',
                        title=dict(text='Variable exceedance frequency',
                                   font=dict(family='IBM Plex Mono',size=13)),
                        xaxis_title='Variable',yaxis_title='N° anomalous cycles',
                        height=320,margin=dict(l=10,r=10,t=40,b=80),
                        plot_bgcolor='#fafafa',paper_bgcolor='white',
                        font=dict(family='IBM Plex Sans'),
                        legend=dict(orientation='h',y=-0.35),xaxis=dict(tickangle=-45))
                    st.plotly_chart(fig_var,use_container_width=True,key='summary_var_chart')

                st.markdown("---")
                top_overall=df_vars.nlargest(5,'Total')['Variable'].tolist()
                top_t2v=df_vars.nlargest(5,'T² exceedances')['Variable'].tolist()
                top_qv =df_vars.nlargest(5,'Q exceedances')['Variable'].tolist()
                prompt_rc=(
                    f"Phase II: {n_test} cycles, {n_any} anomalies ({pct:.1f}%).\n"
                    f"T² top vars: {', '.join(top_t2v)}.\n"
                    f"Q top vars: {', '.join(top_qv)}.\n"
                    f"Files analysed: {', '.join(f['name'] for f in st.session_state.mon_files)}.\n\n"
                    "Structured Root Cause Analysis:\n\n"
                    "**1. DIAGNOSIS** — What is happening physically?\n"
                    "**2. PROBABLE ROOT CAUSES** — ranked by probability, reference variables\n"
                    "**3. IMMEDIATE ACTIONS** — what to do right now\n"
                    "**4. MEDIUM-TERM ACTIONS** — what to investigate to prevent recurrence\n"
                    "**5. WHAT TO MONITOR** — variables and limits to watch\n\n"
                    "Be specific. No generic statements."
                )
                llm_button("Generate Root Cause Analysis & Action Plan",
                           prompt_rc,key='ai_rootcause')
            else:
                st.markdown("<div class='ok-box'>✅ No anomalies — process is stable.</div>",
                            unsafe_allow_html=True)

        if st.session_state.anomaly_log:
            st.markdown("---")
            st.markdown("#### 📋 Intervention Log")
            st.dataframe(pd.DataFrame(st.session_state.anomaly_log),
                         use_container_width=True)
