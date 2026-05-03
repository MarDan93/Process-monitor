import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f, norm
import google.generativeai as genai
import pickle, json, io, re

# Language detection — graceful fallback if not installed
try:
    from langdetect import detect as _detect_lang
    def detect_language(text):
        try:
            code = _detect_lang(text[:500])
            names = {'it':'Italian','en':'English','de':'German','fr':'French',
                     'es':'Spanish','pt':'Portuguese','nl':'Dutch','pl':'Polish'}
            return names.get(code, 'English')
        except Exception:
            return 'English'
except ImportError:
    def detect_language(text):
        return 'English'

st.set_page_config(page_title="Process Monitor", page_icon="🏭",
                   layout="wide", initial_sidebar_state="expanded")

# ═══════════════════════════════════════════
#  DESIGN SYSTEM
# ═══════════════════════════════════════════
COLORS = dict(
    primary   = "#2563EB",   # blue
    success   = "#16A34A",   # green
    warning   = "#D97706",   # amber
    danger    = "#DC2626",   # red
    neutral   = "#374151",   # gray-700
    muted     = "#6B7280",   # gray-500
    surface   = "#F9FAFB",   # gray-50
    border    = "#E5E7EB",   # gray-200
    white     = "#FFFFFF",
)
CHART_PALETTE = ["#2563EB","#DC2626","#16A34A","#D97706","#7C3AED","#0891B2","#BE185D"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {COLORS['neutral']};
}}
h1, h2, h3 {{ font-family: 'Inter', sans-serif; font-weight: 600; }}
code, pre {{ font-family: 'JetBrains Mono', monospace; }}

/* Clean metric cards */
[data-testid="metric-container"] {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
[data-testid="stMetricLabel"] {{
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {COLORS['muted']} !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 22px !important;
    font-weight: 700 !important;
    color: {COLORS['neutral']} !important;
}}

/* Tab styling */
[data-testid="stTabs"] [role="tab"] {{
    font-size: 13px;
    font-weight: 500;
    padding: 8px 16px;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {COLORS['primary']};
    font-weight: 600;
}}

/* Buttons */
[data-testid="baseButton-primary"] {{
    background: {COLORS['primary']} !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}}

/* Expander */
[data-testid="stExpander"] {{
    border: 1px solid {COLORS['border']} !important;
    border-radius: 8px !important;
}}

/* AI response box */
.ai-response {{
    background: {COLORS['white']};
    border: 1px solid #BFDBFE;
    border-left: 4px solid {COLORS['primary']};
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 14px;
    line-height: 1.75;
    color: {COLORS['neutral']};
    margin-top: 12px;
    box-shadow: 0 1px 3px rgba(37,99,235,0.08);
}}
.ai-response strong {{ color: {COLORS['primary']}; }}

/* Status boxes */
.box-ok {{
    background: #F0FDF4; border: 1px solid #BBF7D0;
    border-left: 4px solid {COLORS['success']};
    border-radius: 8px; padding: 12px 16px;
    font-size: 14px; color: #166534; margin-top: 8px;
}}
.box-warn {{
    background: #FFFBEB; border: 1px solid #FDE68A;
    border-left: 4px solid {COLORS['warning']};
    border-radius: 8px; padding: 12px 16px;
    font-size: 14px; color: #92400E; margin-top: 8px;
}}
.box-alarm {{
    background: #FEF2F2; border: 1px solid #FECACA;
    border-left: 4px solid {COLORS['danger']};
    border-radius: 8px; padding: 12px 16px;
    font-size: 14px; color: #991B1B; margin-top: 8px;
}}
.box-alarm strong {{ color: #7F1D1D; }}
.box-info {{
    background: #EFF6FF; border: 1px solid #BFDBFE;
    border-left: 4px solid {COLORS['primary']};
    border-radius: 8px; padding: 12px 16px;
    font-size: 14px; color: #1E40AF; margin-top: 8px;
}}

/* Chat bubbles */
.chat-user {{
    background: {COLORS['primary']}; color: white;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 14px; margin: 6px 0 6px 40px;
    font-size: 13px; line-height: 1.5;
}}
.chat-ai {{
    background: {COLORS['surface']}; color: {COLORS['neutral']};
    border: 1px solid {COLORS['border']};
    border-radius: 16px 16px 16px 4px;
    padding: 10px 14px; margin: 6px 40px 6px 0;
    font-size: 13px; line-height: 1.5;
}}

/* Workflow badge */
.wf-badge {{
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 600;
    letter-spacing: 0.04em;
}}

/* File tag */
.file-tag {{
    display: inline-block; padding: 2px 10px;
    border-radius: 20px; font-size: 11px;
    font-family: 'JetBrains Mono', monospace; margin: 2px;
}}

/* Section header */
.section-header {{
    font-size: 15px; font-weight: 600;
    color: {COLORS['neutral']}; margin-bottom: 4px;
    padding-bottom: 8px;
    border-bottom: 2px solid {COLORS['border']};
}}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════

def box(text, kind='info'):
    css = {'ok':'box-ok','warn':'box-warn','alarm':'box-alarm','info':'box-info'}.get(kind,'box-info')
    st.markdown(f"<div class='{css}'>{text}</div>", unsafe_allow_html=True)


def ai_box(text, model_label=None):
    """Render AI response with professional formatting."""
    # Convert markdown to readable HTML
    html = text

    # **bold** → styled span
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#1E40AF">\1</strong>', html)

    # Numbered sections like "1. TITLE" or "**1. TITLE**"
    html = re.sub(
        r'(?m)^(\d+)\.\s+([A-ZÀÁÈÉÌÍÒÓÙÚ ]{3,})\s*$',
        r'<div style="margin:14px 0 6px 0;font-weight:700;font-size:13px;'
        r'color:#1E3A8A;text-transform:uppercase;letter-spacing:0.05em;'
        r'border-left:3px solid #2563EB;padding-left:10px">'
        r'\1. \2</div>',
        html
    )

    # Bullet lines starting with • or -
    html = re.sub(
        r'(?m)^[•\-]\s+(.+)$',
        r'<div style="display:flex;gap:8px;margin:5px 0;padding-left:4px">'
        r'<span style="color:#2563EB;font-weight:700;flex-shrink:0">›</span>'
        r'<span>\1</span></div>',
        html
    )

    # Double newlines → paragraph break
    html = re.sub(r'\n{2,}', '<div style="margin:8px 0"></div>', html)
    html = html.replace('\n', '<br>')

    # Wrap and render
    st.markdown(
        f"""<div style="
            background:#F8FAFF;
            border:1px solid #DBEAFE;
            border-left:4px solid #2563EB;
            border-radius:10px;
            padding:18px 22px;
            margin-top:12px;
            font-size:14px;
            line-height:1.75;
            color:#1E293B;
            box-shadow:0 2px 8px rgba(37,99,235,0.07);
        ">{html}</div>""",
        unsafe_allow_html=True
    )
    if model_label:
        st.caption(f"✨ via {model_label}")


# ═══════════════════════════════════════════
#  CONTEXT
# ═══════════════════════════════════════════

def get_process_context():
    ctx = st.session_state.get('process_context','').strip()
    obj = st.session_state.get('analysis_objective','').strip()
    out = ""
    if ctx: out += f"\n=== PROCESS CONTEXT ===\n{ctx}\n=== END ===\n"
    if obj: out += f"\n=== OBJECTIVE ===\n{obj}\n=== END ===\n"
    return out


# ═══════════════════════════════════════════
#  AI — MULTI-MODEL FALLBACK
# ═══════════════════════════════════════════

MAX_TOKENS = 8000
RATE_CODES = ["429","quota","rate","limit","resource","exhausted","overload","unavailable"]


def call_gemini_model(prompt, model_name):
    key = st.secrets.get("GEMINI_API_KEY","")
    if not key: raise ValueError("GEMINI_API_KEY not configured.")
    genai.configure(api_key=key)
    m = genai.GenerativeModel(model_name,
        generation_config=genai.GenerationConfig(temperature=0.3, max_output_tokens=MAX_TOKENS))
    return m.generate_content(prompt).text


def call_claude_haiku(prompt):
    try:
        import anthropic
        key = st.secrets.get("ANTHROPIC_API_KEY","")
        if not key: raise ValueError("ANTHROPIC_API_KEY not configured.")
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=MAX_TOKENS,
                                     messages=[{"role":"user","content":prompt}])
        return msg.content[0].text
    except ImportError:
        raise ValueError("anthropic library not installed.")


def get_response_language():
    """Detect language from process context, fallback to English."""
    ctx = st.session_state.get('process_context','').strip()
    if ctx:
        return detect_language(ctx)
    return 'English'


def call_ai(prompt):
    lang = get_response_language()
    full_prompt = f"Always respond in {lang}.\n\n" + prompt
    for model_name, label in [("gemini-2.5-flash","Gemini 2.5 Flash"),
                               ("gemini-2.5-flash-lite","Gemini 2.5 Flash-Lite")]:
        try:
            return call_gemini_model(full_prompt, model_name), label, None
        except Exception as e:
            if not any(c in str(e).lower() for c in RATE_CODES):
                return None, None, f"{label}: {e}"
    try:
        return call_claude_haiku(full_prompt), "Claude Haiku", None
    except Exception as e:
        return None, None, f"All models failed: {e}"


def llm_button(label, prompt, key):
    if st.button(f"✨ {label}", key=key):
        with st.spinner("Generating AI analysis..."):
            txt, model, err = call_ai(get_process_context() + "\n" + prompt)
        if err:
            st.error(f"AI error: {err}")
        else:
            ai_box(txt, model)


# ═══════════════════════════════════════════
#  AI CHAT POPUP (st.dialog)
# ═══════════════════════════════════════════

@st.dialog("💬 AI Assistant", width="large")
def chat_popup():
    def build_snapshot():
        lines = []
        if st.session_state.process_context:
            lines.append(f"Process: {st.session_state.process_context[:200]}")
        if st.session_state.analysis_objective:
            lines.append(f"Objective: {st.session_state.analysis_objective}")
        if st.session_state.df_X is not None:
            fn = st.session_state.feature_names
            lines.append(f"Dataset: {len(st.session_state.df_X)} cycles, {len(fn)} X vars "
                         f"({', '.join(fn[:5])}{'...' if len(fn)>5 else ''})")
        if st.session_state.model is not None:
            m = st.session_state.model
            nf = int(((m['T2']>m['T2_UCL'])|(m['Q']>m['Q_UCL'])).sum())
            lines.append(f"Model: k={m['k']} PCs, T²UCL={m['T2_UCL']:.3f}, "
                         f"QUCL={m['Q_UCL']:.3f}, flags={nf}")
        if st.session_state.mon_files:
            t2f = np.concatenate([f['mon']['T2_flag'] for f in st.session_state.mon_files])
            qf  = np.concatenate([f['mon']['Q_flag']  for f in st.session_state.mon_files])
            na  = int((t2f|qf).sum()); nt = len(t2f)
            lines.append(f"Monitoring: {nt} cycles, {na} anomalies ({na/nt*100:.1f}%)")
        return "\n".join(lines) or "No data loaded yet."

    lang = get_response_language()

    # Render chat history as bubbles
    chat_html = ""
    for msg in st.session_state.global_chat:
        if msg['role'] == 'user':
            chat_html += f"""
            <div style="display:flex;justify-content:flex-end;margin:8px 0">
              <div style="background:#2563EB;color:white;border-radius:16px 16px 4px 16px;
                          padding:10px 16px;max-width:80%;font-size:13px;line-height:1.5;
                          box-shadow:0 1px 3px rgba(37,99,235,0.3)">
                {msg['text']}
              </div>
            </div>"""
        else:
            # Format AI message
            txt = msg['text']
            txt = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', txt)
            txt = txt.replace('\n','<br>')
            chat_html += f"""
            <div style="display:flex;justify-content:flex-start;margin:8px 0">
              <div style="background:#F1F5F9;color:#1E293B;border-radius:16px 16px 16px 4px;
                          padding:10px 16px;max-width:85%;font-size:13px;line-height:1.6;
                          border:1px solid #E2E8F0">
                <div style="font-size:10px;color:#64748B;margin-bottom:4px;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.05em">AI Assistant</div>
                {txt}
              </div>
            </div>"""

    if not st.session_state.global_chat:
        chat_html = """
        <div style="text-align:center;padding:40px 20px;color:#94A3B8">
          <div style="font-size:32px;margin-bottom:12px">💬</div>
          <div style="font-size:14px">Ask anything about your process, model, or analysis results.</div>
          <div style="font-size:12px;margin-top:6px">The AI knows your current project state.</div>
        </div>"""

    # Scrollable chat area
    st.markdown(
        f"""<div style="
            height:420px;overflow-y:auto;
            border:1px solid #E2E8F0;border-radius:10px;
            padding:16px;background:#FAFAFA;
            margin-bottom:12px;
        " id="chat-area">{chat_html}</div>""",
        unsafe_allow_html=True
    )

    # Input row
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_q = st.text_input(
            "", key="popup_chat_input",
            label_visibility="collapsed",
            placeholder="Scrivi un messaggio...",
        )
    with col_send:
        send = st.button("➤", key="popup_send", use_container_width=True, type="primary")

    col_info, col_clear = st.columns([3,1])
    with col_info:
        st.caption(f"🌍 Language: {lang}  ·  ✨ AI: Gemini 2.5 Flash")
    with col_clear:
        if st.button("🗑️ Clear", key="popup_clear", use_container_width=True):
            st.session_state.global_chat = []
            st.rerun()

    # Send on button click or Enter (text_input triggers rerun on Enter)
    if (send or user_q) and user_q.strip():
        # Only process if it's a new message (not already the last user message)
        last_user = next((m['text'] for m in reversed(st.session_state.global_chat)
                         if m['role']=='user'), None)
        if user_q.strip() != last_user:
            history = "\n".join(
                f"{'User' if m['role']=='user' else 'AI'}: {m['text']}"
                for m in st.session_state.global_chat[-8:])
            prompt = (
                get_process_context() + "\n"
                + f"=== PROJECT STATE ===\n{build_snapshot()}\n"
                + (f"=== HISTORY ===\n{history}\n" if history else "")
                + f"User: {user_q}\n"
                + "Answer concisely and practically. "
                  "Ask a follow-up question if you need more info."
            )
            with st.spinner("Generating response..."):
                reply, model, err = call_ai(prompt)
            if err: reply = f"Errore: {err}"
            st.session_state.global_chat.append({'role':'user', 'text': user_q})
            st.session_state.global_chat.append({'role':'ai',   'text': reply or ""})
            st.rerun()


# ═══════════════════════════════════════════
#  MODEL PERSISTENCE — SAVE / LOAD
# ═══════════════════════════════════════════

def serialize_model(model):
    """Convert model dict to bytes for download."""
    return pickle.dumps(model)


def deserialize_model(data):
    """Load model from bytes."""
    return pickle.loads(data)


def save_session_bundle():
    """Bundle model + context + log into a single downloadable file."""
    bundle = {
        'model':             st.session_state.model,
        'feature_names':     st.session_state.feature_names,
        'y_names':           st.session_state.y_names,
        'process_context':   st.session_state.process_context,
        'analysis_objective':st.session_state.analysis_objective,
        'anomaly_log':       st.session_state.anomaly_log,
        'k_chosen':          st.session_state.k_chosen,
    }
    return pickle.dumps(bundle)


def load_session_bundle(data):
    """Restore session from bundle bytes."""
    bundle = pickle.loads(data)
    for key in ['model','feature_names','y_names','process_context',
                'analysis_objective','anomaly_log','k_chosen']:
        if key in bundle:
            st.session_state[key] = bundle[key]
    if bundle.get('process_context'):
        st.session_state.context_saved = True


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
    return dict(scaler=sc, pca=pca, k=k, scores=T, loadings=P, eigenvalues=lam,
                T2=T2, Q=Q, T2_UCL=T2_UCL, Q_UCL=Q_UCL, X_scaled=Xs, E=E,
                evr=pca.explained_variance_ratio_*100, feature_names=[],
                n_train=n,
                Qcontrib_LCL=mu_e-zc*sd_e, Qcontrib_UCL=mu_e+zc*sd_e,
                T2contrib_LCL=mu_c-zc*sd_c, T2contrib_UCL=mu_c+zc*sd_c)


def monitor_new(model, X_new):
    sc=model['scaler']; pca=model['pca']; lam=model['eigenvalues']
    Xns=sc.transform(X_new); Tn=pca.transform(Xns)
    T2n=np.sum((Tn**2)/lam,axis=1)
    En=Xns-pca.inverse_transform(Tn); Qn=np.sum(En**2,axis=1)
    return dict(T2=T2n,Q=Qn,T2_flag=T2n>model['T2_UCL'],Q_flag=Qn>model['Q_UCL'],
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

def chart_line_multi(values, ucl, title, color, flags=None,
                     file_labels=None, file_boundaries=None):
    fig = go.Figure()
    if file_boundaries and len(file_boundaries)>1:
        for fi,(s,e) in enumerate(zip(file_boundaries[:-1],file_boundaries[1:])):
            fc=CHART_PALETTE[fi%len(CHART_PALETTE)]
            fig.add_vrect(x0=s,x1=e-1,fillcolor=fc,opacity=0.06,line_width=0,
                          annotation_text=file_labels[fi] if file_labels else f"File {fi+1}",
                          annotation_position="top left",annotation_font_size=9)
    fig.add_scatter(x=np.arange(len(values)).tolist(),y=values.tolist(),
                    mode='lines',line=dict(color=color,width=1.5),
                    hovertemplate='Cycle %{x}<br>%{y:.3f}<extra></extra>',name='Value')
    fig.add_hline(y=ucl,line_dash='dash',line_color=COLORS['danger'],line_width=1.5,
                  annotation_text=f'UCL={ucl:.3f}',annotation_position='right',
                  annotation_font_color=COLORS['danger'],annotation_font_size=11)
    if flags is not None and flags.any():
        fi=np.where(flags)[0]
        fig.add_scatter(x=fi.tolist(),y=values[fi].tolist(),mode='markers',
                        marker=dict(size=8,color=COLORS['danger'],
                                    line=dict(color='white',width=1)),
                        name='Anomaly',
                        hovertemplate='Cycle %{x}<br>%{y:.3f}<extra></extra>')
    fig.update_layout(
        title=dict(text=title,font=dict(family='Inter',size=13,color=COLORS['neutral'])),
        xaxis_title='Cycle',height=280,
        margin=dict(l=10,r=80,t=40,b=30),
        plot_bgcolor=COLORS['surface'],paper_bgcolor=COLORS['white'],
        font=dict(family='Inter'),
        legend=dict(orientation='h',y=-0.28,font=dict(size=11)),
        hovermode='x unified',
        xaxis=dict(gridcolor=COLORS['border'],gridwidth=1),
        yaxis=dict(gridcolor=COLORS['border'],gridwidth=1))
    return fig


def chart_line(values, ucl, title, color, flags=None):
    return chart_line_multi(values,ucl,title,color,flags)


def chart_contribution(contrib, ucl_v, lcl_v, fn, title):
    p=len(contrib); xax=list(range(1,p+1))
    labels=fn if fn else [f'Var {i+1}' for i in range(p)]
    colors=[COLORS['danger'] if (contrib[i]>ucl_v[i] or contrib[i]<lcl_v[i])
            else COLORS['primary'] for i in range(p)]
    fig=go.Figure()
    fig.add_bar(x=xax,y=contrib.tolist(),marker_color=colors,
                marker_line_width=0,
                customdata=labels,
                hovertemplate='[%{x}] %{customdata}<br>%{y:.4f}<extra></extra>',
                name='Contribution')
    fig.add_scatter(x=xax,y=ucl_v.tolist(),mode='lines',
                    line=dict(color=COLORS['danger'],dash='dot',width=1.5),
                    name='UCL±',hoverinfo='skip')
    fig.add_scatter(x=xax,y=lcl_v.tolist(),mode='lines',
                    line=dict(color=COLORS['danger'],dash='dot',width=1.5),
                    showlegend=False,hoverinfo='skip')
    fig.add_hline(y=0,line_color=COLORS['neutral'],line_width=0.8)
    fig.update_layout(
        title=dict(text=title,font=dict(family='Inter',size=12,color=COLORS['neutral'])),
        xaxis=dict(title='Variable index',tickvals=xax,ticktext=[str(x) for x in xax],
                   gridcolor=COLORS['border']),
        yaxis=dict(title='Contribution (signed)',gridcolor=COLORS['border']),
        height=300,margin=dict(l=10,r=10,t=40,b=30),
        plot_bgcolor=COLORS['surface'],paper_bgcolor=COLORS['white'],
        font=dict(family='Inter'),legend=dict(orientation='h',y=-0.32,font=dict(size=11)))
    exceed=[(i+1,labels[i],round(float(contrib[i]),4),
             round(float(lcl_v[i]),4),round(float(ucl_v[i]),4))
            for i in range(p) if contrib[i]>ucl_v[i] or contrib[i]<lcl_v[i]]
    return fig,exceed


def chart_score(T, lam, T2_UCL_global, evr, pc_i, pc_j, flagged,
                alpha=0.95, n_train=None):
    ang=np.linspace(0,2*np.pi,400)
    n=n_train if n_train else max(T.shape[0],10)
    T2_UCL_2d=(2*(n-1)/(n-2))*f.ppf(alpha,2,n-2)
    a=np.sqrt(lam[pc_i]*T2_UCL_2d); b=np.sqrt(lam[pc_j]*T2_UCL_2d)
    outside=(T[:,pc_i]/a)**2+(T[:,pc_j]/b)**2>1
    inside=~outside
    fig=go.Figure()
    if inside.any():
        fig.add_scatter(x=T[inside,pc_i].tolist(),y=T[inside,pc_j].tolist(),
                        mode='markers',
                        marker=dict(size=5,color=COLORS['neutral'],opacity=0.6),
                        name=f'Inside {int(alpha*100)}% ellipse',
                        hovertemplate=f'PC{pc_i+1}:%{{x:.3f}}<br>PC{pc_j+1}:%{{y:.3f}}<extra></extra>')
    if outside.any():
        fig.add_scatter(x=T[outside,pc_i].tolist(),y=T[outside,pc_j].tolist(),
                        mode='markers',
                        marker=dict(size=8,color=COLORS['danger'],
                                    line=dict(color='white',width=1)),
                        name=f'Outside {int(alpha*100)}% ellipse',
                        hovertemplate=f'PC{pc_i+1}:%{{x:.3f}}<br>PC{pc_j+1}:%{{y:.3f}}<extra></extra>')
    fig.add_scatter(x=(a*np.cos(ang)).tolist(),y=(b*np.sin(ang)).tolist(),
                    mode='lines',line=dict(color=COLORS['danger'],dash='dash',width=1.8),
                    name=f'{int(alpha*100)}% confidence',hoverinfo='skip')
    fig.add_hline(y=0,line_color=COLORS['border'],line_width=0.8)
    fig.add_vline(x=0,line_color=COLORS['border'],line_width=0.8)
    fig.update_layout(
        title=dict(text=(f'Score plot  PC{pc_i+1} ({evr[pc_i]:.1f}%)'
                         f'  vs  PC{pc_j+1} ({evr[pc_j]:.1f}%)'
                         f'  —  {int(outside.sum())}/{len(T)} outside {int(alpha*100)}% ellipse'),
                   font=dict(family='Inter',size=12,color=COLORS['neutral'])),
        xaxis_title=f'PC{pc_i+1} ({evr[pc_i]:.1f}%)',
        yaxis_title=f'PC{pc_j+1} ({evr[pc_j]:.1f}%)',
        height=420,margin=dict(l=10,r=10,t=55,b=30),
        plot_bgcolor=COLORS['surface'],paper_bgcolor=COLORS['white'],
        font=dict(family='Inter'),
        xaxis=dict(gridcolor=COLORS['border']),
        yaxis=dict(gridcolor=COLORS['border']),
        legend=dict(orientation='h',y=-0.18,font=dict(size=11)))
    return fig


def chart_loading(P, fn, evr, pc_i, pc_j):
    p=P.shape[0]; labels=fn if fn else [f'Var {i+1}' for i in range(p)]
    fig=go.Figure()
    fig.add_scatter(x=P[:,pc_i].tolist(),y=P[:,pc_j].tolist(),
                    mode='markers+text',
                    marker=dict(size=9,color=COLORS['primary'],
                                line=dict(color='white',width=1.5)),
                    text=[str(i+1) for i in range(p)],
                    textposition='top center',
                    textfont=dict(size=9,color=COLORS['muted']),
                    customdata=labels,
                    hovertemplate='<b>%{customdata}</b><br>'
                                  f'PC{pc_i+1}: %{{x:.4f}}<br>'
                                  f'PC{pc_j+1}: %{{y:.4f}}<extra></extra>',
                    showlegend=False)
    fig.add_hline(y=0,line_color=COLORS['border'],line_width=1)
    fig.add_vline(x=0,line_color=COLORS['border'],line_width=1)
    fig.update_layout(
        title=dict(text=f'Loading plot  PC{pc_i+1} ({evr[pc_i]:.1f}%)  vs  PC{pc_j+1} ({evr[pc_j]:.1f}%)',
                   font=dict(family='Inter',size=12,color=COLORS['neutral'])),
        xaxis_title=f'PC{pc_i+1} loading',yaxis_title=f'PC{pc_j+1} loading',
        height=480,margin=dict(l=10,r=10,t=50,b=30),
        plot_bgcolor=COLORS['surface'],paper_bgcolor=COLORS['white'],
        font=dict(family='Inter'),
        xaxis=dict(gridcolor=COLORS['border']),
        yaxis=dict(gridcolor=COLORS['border']))
    return fig


def show_contribution_block(model,Xs,E,T,obs_idx,fn,key_suffix):
    lam=model['eigenvalues']; P=model['loadings']
    c_t2=Xs[obs_idx]*(P@(T[obs_idx]/lam)); c_q=E[obs_idx]
    top_t2=np.argsort(np.abs(c_t2))[::-1][:3].tolist()
    top_q=np.argsort(np.abs(c_q))[::-1][:3].tolist()
    p_count=len(fn)
    with st.expander("📋 Variable index",expanded=False):
        st.dataframe(pd.DataFrame({'#':range(1,p_count+1),'Variable':fn}),
                     use_container_width=True,hide_index=True)
    col_l,col_r=st.columns(2)
    with col_l:
        st.markdown("**T² Contribution**")
        fig_t2,exc_t2=chart_contribution(c_t2,model['T2contrib_UCL'],
                                          model['T2contrib_LCL'],fn,
                                          f'T² — Obs {obs_idx}')
        st.plotly_chart(fig_t2,use_container_width=True,key=f'ct2_{key_suffix}')
        if exc_t2:
            st.markdown("**Out-of-limit variables:**")
            st.dataframe(pd.DataFrame(exc_t2,columns=['#','Variable','Value','LCL','UCL']),
                         use_container_width=True,hide_index=True)
        with st.expander("Full T² table"):
            st.dataframe(pd.DataFrame({
                '#':range(1,p_count+1),'Variable':fn,
                'Contribution':c_t2.round(4),
                'LCL':model['T2contrib_LCL'].round(4),
                'UCL':model['T2contrib_UCL'].round(4),
                'Status':['🔴' if (c_t2[i]>model['T2contrib_UCL'][i]
                                   or c_t2[i]<model['T2contrib_LCL'][i])
                          else '✅' for i in range(p_count)]
            }),use_container_width=True,hide_index=True)
    with col_r:
        st.markdown("**Q Contribution**")
        fig_q,exc_q=chart_contribution(c_q,model['Qcontrib_UCL'],
                                        model['Qcontrib_LCL'],fn,
                                        f'Q — Obs {obs_idx}')
        st.plotly_chart(fig_q,use_container_width=True,key=f'cq_{key_suffix}')
        if exc_q:
            st.markdown("**Out-of-limit variables:**")
            st.dataframe(pd.DataFrame(exc_q,columns=['#','Variable','Value','LCL','UCL']),
                         use_container_width=True,hide_index=True)
        with st.expander("Full Q table"):
            st.dataframe(pd.DataFrame({
                '#':range(1,p_count+1),'Variable':fn,
                'Contribution':c_q.round(4),
                'LCL':model['Qcontrib_LCL'].round(4),
                'UCL':model['Qcontrib_UCL'].round(4),
                'Status':['🔴' if (c_q[i]>model['Qcontrib_UCL'][i]
                                   or c_q[i]<model['Qcontrib_LCL'][i])
                          else '✅' for i in range(p_count)]
            }),use_container_width=True,hide_index=True)
    return top_t2,top_q


def show_anomaly_table_and_contrib(model,T2_arr,Q_arr,Xs,E,T_arr,fn,table_key,prefix):
    flagged_idx=np.where((T2_arr>model['T2_UCL'])|(Q_arr>model['Q_UCL']))[0]
    if len(flagged_idx)==0:
        box("✅ No anomalies detected — process in control.","ok")
        return None
    st.caption(f"{len(flagged_idx)} out-of-control observations — click a row to analyse it.")
    st.caption("💡 **Severity×UCL** = max(T²/UCL, Q/UCL) — how far above the control limit. "
               "1.0 = just outside | 2.0 = twice the limit | 5.0+ = severe anomaly.")
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
    kind='alarm' if ratio>=1.5 else 'warn'
    box(f"<strong>Cycle {obs} — {'🔴 ANOMALY' if ratio>=1.5 else '⚠️ WARNING'}</strong> &nbsp;·&nbsp; "
        f"T²={t2_obs:.3f} ({t2_obs/model['T2_UCL']:.2f}×UCL) &nbsp;|&nbsp; "
        f"Q={q_obs:.3f} ({q_obs/model['Q_UCL']:.2f}×UCL)", kind)
    st.markdown("#### Contribution plots")
    top_t2,top_q=show_contribution_block(model,Xs,E,T_arr,obs,fn,f'{prefix}_{obs}')
    return obs,t2_obs,q_obs,ratio,top_t2,top_q


# ═══════════════════════════════════════════
#  WORKFLOW
# ═══════════════════════════════════════════

def get_workflow():
    obj=st.session_state.get('analysis_objective','').lower()
    if 'diagnostic' in obj: return 'diagnostic'
    if 'exploratory' in obj: return 'exploratory'
    return 'spc'

WF_STYLES={
    'spc':         ('#EFF6FF','#2563EB','🔵 SPC','Statistical Process Control'),
    'diagnostic':  ('#FFF7ED','#D97706','🟠 Diagnostics','Diagnostics & Root Cause'),
    'exploratory': ('#F0FDF4','#16A34A','🟢 Exploratory','Exploratory Analysis'),
}
WF_GUIDE={
    'spc':        {'Dataset':'Load data + optional split','PC Selection':'Choose k',
                   'Calibration':'Build Phase I model','Loadings & Scores':'Explore structure',
                   'Monitoring':'Upload new data → detect anomalies','Summary':'Root cause analysis'},
    'diagnostic': {'Dataset':'Load full historical dataset','PC Selection':'Choose k',
                   'Calibration':'Fit on full dataset — anomalies = internal flags',
                   'Loadings & Scores':'Explore variable structure',
                   'Monitoring':'— Not required','Summary':'Root cause on Phase I flags'},
    'exploratory':{'Dataset':'Load dataset','PC Selection':'Choose k',
                   'Calibration':'Fit model','Loadings & Scores':'★ Main section',
                   'Monitoring':'— Optional','Summary':'Variable patterns'},
}


# ═══════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════
for k,v in [('model',None),('feature_names',[]),('y_names',[]),
            ('df_X',None),('df_Y',None),('k_chosen',None),
            ('mon',None),('mon_files',[]),('anomaly_log',[]),
            ('rmsecv_computed',False),('rmsecv_result',None),
            ('process_context',''),('analysis_objective',''),('context_saved',False),
            ('df_train',None),('df_test_builtin',None),
            ('global_chat',[])]:
    if k not in st.session_state:
        st.session_state[k]=v


# ═══════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏭 Process Monitor")
    st.caption("PCA-SPC · Industrial Monitoring")
    st.divider()

    # Chat button
    if st.button("💬 Open AI Chat", use_container_width=True, type="primary"):
        chat_popup()

    st.divider()

    # Settings
    with st.expander("⚙️ Settings", expanded=True):
        alpha=st.slider("UCL Confidence (α)",0.90,0.99,0.95,0.01)
        y_cols_raw=st.text_area("Y Variables — one per line")
        y_cols=[c.strip() for c in y_cols_raw.split('\n') if c.strip()]
        excl_raw=st.text_area("Columns to exclude — one per line")
        excl_cols=[c.strip() for c in excl_raw.split('\n') if c.strip()]

    st.divider()

    # Model persistence
    st.markdown("**💾 Model persistence**")
    up_bundle=st.file_uploader("Load saved session (.pkl)",
                                type=['pkl'],key='up_bundle')
    if up_bundle:
        try:
            load_session_bundle(up_bundle.read())
            st.success("✅ Session restored.")
            st.rerun()
        except Exception as e:
            st.error(f"Load error: {e}")

    if st.session_state.model is not None:
        bundle_bytes=save_session_bundle()
        st.download_button("⬇️ Save session",data=bundle_bytes,
                           file_name="process_monitor_session.pkl",
                           mime="application/octet-stream",
                           use_container_width=True)
        st.caption("Save and reload this file to resume your session.")

    st.divider()
    if st.session_state.context_saved:
        wf=get_workflow()
        bg,col,lbl,name=WF_STYLES[wf]
        st.markdown(
            f"<div class='wf-badge' style='background:{bg};color:{col};"
            f"border:1px solid {col}44'>{lbl} {name}</div>",
            unsafe_allow_html=True)
    st.caption("AI: Gemini 2.5 Flash → Flash-Lite → Claude Haiku")


# ═══════════════════════════════════════════
#  MAIN — header + workflow banner
# ═══════════════════════════════════════════
st.markdown("# 🏭 Process Monitor")

wf=get_workflow()
bg,col,lbl,name=WF_STYLES[wf]
if st.session_state.context_saved:
    st.markdown(
        f"<div style='background:{bg};border:1px solid {col}33;"
        f"border-left:4px solid {col};padding:10px 16px;border-radius:8px;"
        f"margin-bottom:12px;font-size:13px;font-weight:500;color:{col}'>"
        f"{lbl} {name}"
        f"</div>", unsafe_allow_html=True)
    with st.expander("📋 Workflow guide", expanded=False):
        for tab_n,step in WF_GUIDE[wf].items():
            icon="⏭️" if step.startswith("—") else ("⭐" if "★" in step else "→")
            st.markdown(f"{icon} **{tab_n}** — {step}")
else:
    st.markdown(
        f"<div style='background:#FFF7ED;border-left:4px solid #D97706;"
        f"padding:10px 16px;border-radius:8px;margin-bottom:12px;"
        f"font-size:13px;color:#92400E'>"
        f"⚙️ Start by setting your <strong>Process Context</strong> and objective "
        f"to activate the guided workflow.</div>",
        unsafe_allow_html=True)

tab0,tab1,tab2,tab3,tab4,tab5,tab6=st.tabs([
    "⚙️ Context","📂 Dataset","📐 PC Selection",
    "🔧 Calibration","📊 Loadings & Scores","🔍 Monitoring",
    "📋 Summary"])


# ═══════════════════════════════════════════
#  TAB 0 — PROCESS CONTEXT
# ═══════════════════════════════════════════
with tab0:
    st.markdown('<div class="section-header">Process Context & Objective</div>',
                unsafe_allow_html=True)
    st.markdown("The AI uses this information to contextualise every analysis response.")
    st.markdown("")

    ctx_input=st.text_area("Process description",value=st.session_state.process_context,
                            height=160,key='ctx_textarea')
    st.markdown("**Analysis objective**")
    obj_options=["Statistical Process Control — build a model and monitor production",
                 "Diagnostics — analyse existing data to find anomalies and root causes",
                 "Exploratory analysis — understand process structure and variable correlations",
                 "Other (describe below)"]
    obj_sel=st.selectbox("",obj_options,key='obj_select',label_visibility='collapsed')
    obj_extra=""
    if obj_sel=="Other (describe below)":
        obj_extra=st.text_area("Describe your objective",height=80,key='obj_extra')
    final_obj=obj_extra.strip() if obj_sel=="Other (describe below)" else obj_sel

    col_s,col_c=st.columns([2,1])
    with col_s:
        if st.button("💾 Save context",type="primary",use_container_width=True):
            st.session_state.process_context=ctx_input.strip()
            st.session_state.analysis_objective=final_obj
            st.session_state.context_saved=True
            st.success("✅ Context saved — all AI responses will be contextualised.")
    with col_c:
        if st.button("🗑️ Clear",use_container_width=True):
            st.session_state.process_context=''
            st.session_state.analysis_objective=''
            st.session_state.context_saved=False
            st.rerun()

    if st.session_state.context_saved and st.session_state.process_context:
        st.markdown("---")
        prompt_ctx=(
            f"Process: {st.session_state.process_context}\n"
            f"Objective: {st.session_state.get('analysis_objective','')}\n\n"
            "5 bullet points: (1) process type & key characteristics, "
            "(2) most critical variables, (3) common failure modes, "
            "(4) what PCA-SPC can/cannot detect here, (5) recommendations for stated objective. "
            "Be specific, no generic statements."
        )
        llm_button("Generate process summary",prompt_ctx,key='ai_ctx')


# ═══════════════════════════════════════════
#  TAB 1 — DATASET
# ═══════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Load & Explore Data</div>',
                unsafe_allow_html=True)
    st.markdown("")
    up=st.file_uploader("Upload CSV or Excel",type=['csv','xlsx','xls'],key='up_main')
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
        c4.metric("Missing values",miss)

        col_a,col_b=st.columns(2)
        with col_a:
            st.markdown("**X Variables**")
            st.dataframe(pd.DataFrame({'#':range(1,len(x_cols)+1),'Variable':x_cols}),
                         use_container_width=True,hide_index=True,height=200)
        with col_b:
            if y_valid:
                st.markdown("**Y Variables**")
                st.dataframe(pd.DataFrame({'Variable':y_valid}),
                             use_container_width=True,hide_index=True)

        st.markdown("**Descriptive Statistics**")
        desc=df_X.describe().T.round(3)
        desc['cv%']=(desc['std']/desc['mean'].abs()*100).round(1)
        st.dataframe(desc,use_container_width=True)
        if miss>0:
            box("⚠️ Missing values detected — will be imputed with column mean.","warn")

        st.markdown("---")
        wf_ds=get_workflow()
        if wf_ds=='spc':
            st.markdown("**Train / Test Split**")
            split_method=st.radio("Method",["Temporal","Random"],
                                   horizontal=True,key='split_radio')
            n_total=len(df_X)
            sr_key='split_ratio_slider' if split_method=="Temporal" else 'split_ratio_slider_r'
            split_ratio=st.slider("Train size (%)", 5, 95, 70, 1, key=sr_key)
            split_row=int(n_total*(split_ratio/100))
            st.info(f"Train: **{split_row}** observations ({split_ratio}%) | "
                    f"Test: **{n_total-split_row}** observations ({100-split_ratio}%)")
            if split_method=="Temporal":
                fc=x_cols[0]
                fig_s=go.Figure()
                fig_s.add_vrect(x0=0,x1=split_row,fillcolor=COLORS['primary'],opacity=0.08,
                                line_width=0,annotation_text=f'Train ({split_ratio}%)',
                                annotation_position='top left',annotation_font_size=9)
                fig_s.add_vrect(x0=split_row,x1=n_total,fillcolor=COLORS['danger'],opacity=0.08,
                                line_width=0,annotation_text=f'Test ({100-split_ratio}%)',
                                annotation_position='top right',annotation_font_size=9)
                fig_s.add_vline(x=split_row,line_dash='dash',line_color=COLORS['danger'],line_width=1.5)
                fig_s.add_scatter(x=list(range(n_total)),
                                  y=df_X[fc].fillna(df_X[fc].mean()).tolist(),
                                  mode='lines',line=dict(color=COLORS['neutral'],width=1),name=fc)
                fig_s.update_layout(
                    title=dict(text=f'Split preview — variable: {fc}',
                               font=dict(family='Inter',size=11,color=COLORS['muted'])),
                    height=200,margin=dict(l=10,r=10,t=30,b=30),
                    plot_bgcolor=COLORS['surface'],paper_bgcolor=COLORS['white'],
                    showlegend=False,xaxis_title='Observation index',
                    xaxis=dict(gridcolor=COLORS['border']),
                    yaxis=dict(gridcolor=COLORS['border'],title=fc))
                st.plotly_chart(fig_s,use_container_width=True,key='split_preview')
                st.caption(
                    f"🔵 Blue region = **train set** ({split_row} observations) — "
                    f"used to calibrate the PCA-SPC model (Phase I). "
                    f"🔴 Red region = **test set** ({n_total-split_row} observations) — "
                    f"used to validate the model in Monitoring (Phase II). "
                    f"The line shows the first process variable ({fc}) to visualise where the split falls."
                )
            if st.button("Apply split",key='btn_split',type='primary'):
                df_Xf=df_X.fillna(df_X.mean())
                if split_method=="Temporal":
                    df_tr=df_Xf.iloc[:split_row].reset_index(drop=True)
                    df_te=df_Xf.iloc[split_row:].reset_index(drop=True)
                else:
                    rng=np.random.default_rng(42)
                    idx=rng.permutation(n_total)
                    df_tr=df_Xf.iloc[idx[:split_row]].reset_index(drop=True)
                    df_te=df_Xf.iloc[idx[split_row:]].reset_index(drop=True)
                st.session_state.df_train=df_tr
                st.session_state.df_test_builtin=df_te
                st.success(f"✅ Split applied — Train: {len(df_tr)} | Test: {len(df_te)}")
        else:
            box(f"ℹ️ <strong>{WF_STYLES[wf_ds][3]}</strong> workflow — "
                f"full dataset will be used for calibration. No split needed.","info")
            st.session_state.df_train=None; st.session_state.df_test_builtin=None

        # AI analysis
        st.markdown("---")
        df_Xf2=df_X.fillna(df_X.mean())
        desc2=df_Xf2.describe().T
        desc2['cv%']=(desc2['std']/desc2['mean'].abs()*100).round(1)
        top_cv=desc2.nlargest(5,'cv%')[['mean','std','cv%','min','max']].round(3)
        top_cv_str="\n".join(f"  {v}: CV={r['cv%']}%, mean={r['mean']}, "
                             f"min={r['min']}, max={r['max']}"
                             for v,r in top_cv.iterrows())
        outlier_vars=[]
        for v in df_Xf2.columns:
            mu,s=df_Xf2[v].mean(),df_Xf2[v].std()
            if s>0 and (df_Xf2[v].min()<mu-4*s or df_Xf2[v].max()>mu+4*s):
                outlier_vars.append(f"{v}(min={df_Xf2[v].min():.2f},max={df_Xf2[v].max():.2f})")
        corr=df_Xf2.corr().abs(); corr_np=corr.values.copy()
        np.fill_diagonal(corr_np,0)
        corr_f=pd.DataFrame(corr_np,index=corr.index,columns=corr.columns)
        top_corr=", ".join(f"{a}↔{b}:{v:.2f}"
                           for (a,b),v in corr_f.unstack().nlargest(5).items())
        miss_str=", ".join(f"{c}:{n}" for c,n in df_X.isnull().sum().items() if n>0) or "none"
        prompt_ds=(
            f"Dataset: {len(df_X)} obs, {len(x_cols)} X vars, "
            f"{len(y_valid)} Y ({', '.join(y_valid) if y_valid else 'none'}).\n"
            f"Top 5 by variability:\n{top_cv_str}\n"
            f"Possible outliers (4σ): {', '.join(outlier_vars[:5]) or 'none'}\n"
            f"Missing: {miss_str}\nTop correlations: {top_corr}\n\n"
            "5 bullets: (1) data quality, (2) unusual variability, "
            "(3) key correlations for PCA, (4) concerns, (5) specific recommendation. "
            "Reference variable names."
        )
        llm_button("Analyse dataset",prompt_ds,key='ai_ds')


# ═══════════════════════════════════════════
#  TAB 2 — PC SELECTION
# ═══════════════════════════════════════════
with tab2:
    if st.session_state.df_X is None:
        box("⬆️ Load the dataset first.","info")
    else:
        st.markdown('<div class="section-header">Principal Component Selection</div>',
                    unsafe_allow_html=True)
        st.markdown("")
        df_X=st.session_state.df_X.fillna(st.session_state.df_X.mean())
        df_for_pca=st.session_state.df_train if st.session_state.df_train is not None else df_X
        if st.session_state.df_train is not None:
            box(f"ℹ️ Using train split ({len(df_for_pca)} cycles).","info")
        X_raw=df_for_pca.values
        sc_tmp=StandardScaler(); Xs_tmp=sc_tmp.fit_transform(X_raw)
        n_obs,n_vars=Xs_tmp.shape; max_k=min(20,n_vars-1,n_obs-2)
        pca_tmp=PCA(n_components=max_k,svd_solver='full',random_state=42)
        pca_tmp.fit(Xs_tmp); eigs=pca_tmp.explained_variance_
        evr_all=pca_tmp.explained_variance_ratio_*100
        cum=np.cumsum(evr_all); ks=list(range(1,max_k+1))
        k_kaiser=max(2,min(int(np.sum(eigs>1)),max_k))
        k_90=int(np.argmax(cum>=90.0))+1; k_95=int(np.argmax(cum>=95.0))+1

        grafico=st.radio("Criterion",["Scree plot","Cumulative variance","RMSECV"],
                         horizontal=True,key='pc_radio')
        if grafico=="Scree plot":
            fig=go.Figure()
            fig.add_scatter(x=ks,y=evr_all.tolist(),mode='lines+markers',
                            line=dict(color=COLORS['primary'],width=2),
                            marker=dict(size=6,color=COLORS['primary']),
                            hovertemplate='PC%{x}: %{y:.2f}%<extra></extra>')
            fig.add_hline(y=float(100/n_vars),line_dash='dot',line_color=COLORS['muted'],
                         annotation_text=f'Kaiser threshold',annotation_position='right',
                         annotation_font_size=10)
            fig.add_vline(x=k_kaiser,line_dash='dash',line_color=COLORS['danger'],
                         annotation_text=f'k={k_kaiser}',annotation_position='top right')
            fig.update_layout(xaxis_title='PC',yaxis_title='Variance explained (%)',
                             height=300,margin=dict(l=10,r=80,t=20,b=30),
                             plot_bgcolor=COLORS['surface'],paper_bgcolor=COLORS['white'],
                             showlegend=False,font=dict(family='Inter'),
                             xaxis=dict(gridcolor=COLORS['border']),
                             yaxis=dict(gridcolor=COLORS['border']))
            st.plotly_chart(fig,use_container_width=True,key='chart_scree')
            box(f"💡 Kaiser criterion suggests <strong>k = {k_kaiser}</strong> PCs","info")
        elif grafico=="Cumulative variance":
            fig=go.Figure()
            fig.add_scatter(x=ks,y=cum.tolist(),mode='lines+markers',
                            line=dict(color=COLORS['primary'],width=2),marker=dict(size=6))
            for pct,c,lbl in [(90,COLORS['warning'],'90%'),(95,COLORS['danger'],'95%')]:
                fig.add_hline(y=pct,line_dash='dash',line_color=c,
                             annotation_text=lbl,annotation_position='right')
            fig.update_layout(xaxis_title='Number of PCs',yaxis_title='Cumulative variance (%)',
                             yaxis=dict(range=[0,105],gridcolor=COLORS['border']),
                             xaxis=dict(gridcolor=COLORS['border']),
                             height=300,margin=dict(l=10,r=80,t=20,b=30),
                             plot_bgcolor=COLORS['surface'],paper_bgcolor=COLORS['white'],
                             showlegend=False,font=dict(family='Inter'))
            st.plotly_chart(fig,use_container_width=True,key='chart_cumvar')
            box(f"💡 90% variance → <strong>k={k_90}</strong> &nbsp;|&nbsp; "
                f"95% variance → <strong>k={k_95}</strong>","info")
        else:
            if not st.session_state.rmsecv_computed:
                if st.button("▶ Compute RMSECV",type='primary',key='btn_rmsecv'):
                    with st.spinner("Computing RMSECV..."):
                        bk,_,rcv=compute_rmsecv(Xs_tmp,max_k)
                    st.session_state.rmsecv_result=(bk,rcv)
                    st.session_state.rmsecv_computed=True; st.rerun()
                else:
                    box("Click the button to compute RMSECV cross-validation.","info")
            else:
                bk,rcv=st.session_state.rmsecv_result
                fig=go.Figure()
                fig.add_scatter(x=ks,y=rcv.tolist(),mode='lines+markers',
                                line=dict(color=COLORS['success'],width=2),
                                marker=dict(size=[10 if i+1==bk else 5 for i in range(max_k)],
                                            color=[COLORS['danger'] if i+1==bk
                                                   else COLORS['success'] for i in range(max_k)]))
                fig.add_vline(x=bk,line_dash='dash',line_color=COLORS['danger'],
                             annotation_text=f'min k={bk}',annotation_position='top right')
                fig.update_layout(xaxis_title='Number of PCs',yaxis_title='RMSECV',
                                 height=300,margin=dict(l=10,r=80,t=20,b=30),
                                 plot_bgcolor=COLORS['surface'],paper_bgcolor=COLORS['white'],
                                 showlegend=False,font=dict(family='Inter'),
                                 xaxis=dict(gridcolor=COLORS['border']),
                                 yaxis=dict(gridcolor=COLORS['border']))
                st.plotly_chart(fig,use_container_width=True,key='chart_rmsecv')
                box(f"💡 RMSECV minimum → <strong>k = {bk}</strong> PCs","info")
                if st.button("🔄 Recalculate",key='btn_rmsecv_reset'):
                    st.session_state.rmsecv_computed=False; st.rerun()

        st.markdown("---")
        rs1,rs2,rs3=st.columns(3)
        rs1.metric("Kaiser",f"k = {k_kaiser}"); rs2.metric("90% var",f"k = {k_90}")
        rs3.metric("RMSECV",f"k = {st.session_state.rmsecv_result[0]}"
                   if st.session_state.rmsecv_computed else "—")
        k_chosen=st.number_input("Number of PCs for the model",min_value=2,max_value=max_k,
                                  value=k_kaiser,step=1,key='k_input')
        st.session_state.k_chosen=int(k_chosen)
        box(f"✅ <strong>k = {k_chosen}</strong> PCs selected — "
            f"<strong>{cum[k_chosen-1]:.1f}%</strong> of variance explained","ok")
        st.markdown("---")
        prompt_pc=(
            f"PCA: {n_obs} obs, {n_vars} vars. Kaiser k={k_kaiser}, "
            f"90%→k={k_90}, 95%→k={k_95}. Selected: k={k_chosen} "
            f"({cum[k_chosen-1]:.1f}% var).\n"
            "3 bullets: (1) Is k appropriate? (2) What process phenomena do first PCs capture? "
            "(3) Any concern?"
        )
        llm_button("Interpret PC selection",prompt_pc,key='ai_pc')


# ═══════════════════════════════════════════
#  TAB 3 — CALIBRATION
# ═══════════════════════════════════════════
with tab3:
    if st.session_state.df_X is None:
        box("⬆️ Load the dataset first.","info")
    elif st.session_state.k_chosen is None:
        box("⬆️ Choose the number of PCs first.","info")
    else:
        st.markdown('<div class="section-header">Phase I — Model Calibration</div>',
                    unsafe_allow_html=True)
        st.markdown("")

        df_X=st.session_state.df_X.fillna(st.session_state.df_X.mean())
        x_cols=st.session_state.feature_names
        if st.session_state.df_train is not None:
            df_cal=st.session_state.df_train.copy()
            box(f"ℹ️ Using train split: <strong>{len(df_cal)}</strong> cycles","info")
        else:
            df_cal=df_X.copy()
        X_raw=df_cal[x_cols].values if set(x_cols).issubset(df_cal.columns) else df_cal.values
        k_use=st.session_state.k_chosen

        with st.expander("🧹 Iterative data cleaning (optional)"):
            use_clean=st.toggle("Enable",value=False,key='toggle_clean')
            clean_mask=np.ones(len(X_raw),dtype=bool)
            if use_clean:
                alpha_clean=st.slider("Cleaning confidence",0.95,0.999,0.99,0.001,format="%.3f")
                if st.button("▶ Run cleaning",key='btn_clean'):
                    sc_k=StandardScaler(); Xs_k=sc_k.fit_transform(X_raw)
                    eig_k=PCA(svd_solver='full').fit(Xs_k).explained_variance_
                    k_cl=max(2,min(int(np.sum(eig_k>1)),X_raw.shape[1]-1,X_raw.shape[0]-2))
                    with st.spinner("Running iterative cleaning..."):
                        clean_mask,log_df=iterative_cleaning(X_raw,k_cl,alpha_clean)
                    n_rem=int((~clean_mask).sum()); pct=n_rem/len(X_raw)*100
                    box(f"✅ Removed <strong>{n_rem}</strong> cycles ({pct:.1f}%) — "
                        f"Remaining: <strong>{clean_mask.sum()}</strong>","ok")
                    st.dataframe(log_df,use_container_width=True,hide_index=True)
                    if pct>20:
                        box("⚠️ >20% removed — calibration data may not have been stable.","warn")

        st.markdown("**Build model**")
        if st.button("Build Phase I Model",type="primary",use_container_width=True,key='btn_fit'):
            X_clean=X_raw[clean_mask] if 'clean_mask' in dir() else X_raw
            with st.spinner("Fitting PCA-SPC model..."):
                mdl=fit_pca_spc(X_clean,k_use,alpha)
                mdl['feature_names']=x_cols
                st.session_state.model=mdl
                if st.session_state.df_test_builtin is not None:
                    df_te=st.session_state.df_test_builtin
                    X_te=df_te[x_cols].values if set(x_cols).issubset(df_te.columns) else df_te.values
                    st.session_state.mon_files=[{
                        'name':'Built-in test set','n_rows':len(X_te),
                        'mon':monitor_new(mdl,X_te)}]
            st.rerun()

        if st.session_state.model is not None:
            mdl=st.session_state.model
            if st.button("🔄 Rebuild model",key='btn_refit'):
                st.session_state.model=None; st.session_state.mon_files=[]; st.rerun()

            n_flag=int(((mdl['T2']>mdl['T2_UCL'])|(mdl['Q']>mdl['Q_UCL'])).sum())
            pct_f=n_flag/len(mdl['T2'])*100
            box("✅ Model built successfully.","ok")
            cm1,cm2,cm3,cm4=st.columns(4)
            cm1.metric("k PCs",mdl['k'])
            cm2.metric("T² UCL",f"{mdl['T2_UCL']:.3f}")
            cm3.metric("Q UCL",f"{mdl['Q_UCL']:.3f}")
            cm4.metric("Phase I flags",f"{n_flag} ({pct_f:.1f}%)")

            fT2=mdl['T2']>mdl['T2_UCL']; fQ=mdl['Q']>mdl['Q_UCL']
            st.plotly_chart(chart_line(mdl['T2'],mdl['T2_UCL'],
                                       'Phase I — Hotelling T²',COLORS['primary'],fT2),
                            use_container_width=True,key='p1_t2')
            st.plotly_chart(chart_line(mdl['Q'],mdl['Q_UCL'],
                                       'Phase I — Q (SPE)',COLORS['success'],fQ),
                            use_container_width=True,key='p1_q')

            st.markdown("**Phase I anomalous cycles**")
            result_p1=show_anomaly_table_and_contrib(
                mdl,mdl['T2'],mdl['Q'],mdl['X_scaled'],mdl['E'],mdl['scores'],
                mdl['feature_names'],'p1_table','p1')
            if result_p1:
                obs_p1,t2_p1,q_p1,_,top_t2_p1,top_q_p1=result_p1
                v_t2=[x_cols[i] for i in top_t2_p1]; v_q=[x_cols[i] for i in top_q_p1]
                prompt_p1=(
                    f"Phase I outlier {obs_p1}: T²={t2_p1:.2f}×UCL, Q={q_p1:.2f}×UCL. "
                    f"T² vars: {', '.join(v_t2)}. Q vars: {', '.join(v_q)}.\n"
                    "3 bullets: (1) Remove it? (2) What does it suggest? (3) Action."
                )
                llm_button("Interpret Phase I anomaly",prompt_p1,key=f'ai_p1_{obs_p1}')
            else:
                box("✅ No anomalies in training set — clean calibration.","ok")
            st.markdown("---")
            prompt_cal=(
                f"Phase I: {len(mdl['T2'])} cycles, k={mdl['k']} PCs, "
                f"T²UCL={mdl['T2_UCL']:.3f}, QUCL={mdl['Q_UCL']:.3f}, α={alpha}. "
                f"Flags: {n_flag} ({pct_f:.1f}%).\n"
                "3 bullets: (1) UCL values reasonable? (2) Flags acceptable? (3) Ready for Phase II?"
            )
            llm_button("Interpret calibration model",prompt_cal,key='ai_cal')


# ═══════════════════════════════════════════
#  TAB 4 — LOADINGS & SCORES
# ═══════════════════════════════════════════
with tab4:
    if st.session_state.model is None:
        box("⬆️ Build the model first.","info")
    else:
        st.markdown('<div class="section-header">Loadings & Score Plots</div>',
                    unsafe_allow_html=True)
        st.markdown("")
        mdl=st.session_state.model
        fn=mdl['feature_names']; P=mdl['loadings']
        k_m=mdl['k']; evr_m=mdl['evr']; lam_m=mdl['eigenvalues']
        T_m=mdl['scores']; flag_m=(mdl['T2']>mdl['T2_UCL'])|(mdl['Q']>mdl['Q_UCL'])

        with st.expander("📋 Variable index"):
            st.dataframe(pd.DataFrame({'#':range(1,len(fn)+1),'Variable':fn}),
                         use_container_width=True,hide_index=True,height=200)

        all_pairs=list(combinations(range(k_m),2))
        pair_labels=[f"PC{a+1} vs PC{b+1} ({evr_m[a]:.1f}% + {evr_m[b]:.1f}%)"
                     for a,b in all_pairs]
        col_tipo,col_coppia=st.columns([1,2])
        with col_tipo:
            tipo=st.radio("Chart type",["Loading plot","Score plot"],key='ls_tipo')
        with col_coppia:
            coppia_idx=st.selectbox("PC pair",options=range(len(all_pairs)),
                                    format_func=lambda i: pair_labels[i],key='ls_coppia')
        pc_i,pc_j=all_pairs[coppia_idx]

        if tipo=="Loading plot":
            st.plotly_chart(chart_loading(P,fn,evr_m,pc_i,pc_j),
                            use_container_width=True,key='load_chart')
            st.caption("Number = variable index. Hover for full name. "
                       "⚫ Close = correlated | Opposite = negatively correlated.")
            df_load=pd.DataFrame({
                '#':range(1,P.shape[0]+1),'Variable':fn,
                f'PC{pc_i+1}':P[:,pc_i].round(4),f'PC{pc_j+1}':P[:,pc_j].round(4),
                'Distance':np.sqrt(P[:,pc_i]**2+P[:,pc_j]**2).round(4),
            }).sort_values('Distance',ascending=False)
            st.dataframe(df_load,use_container_width=True,hide_index=True)
        else:
            n_tr=mdl.get('n_train',len(T_m))
            st.plotly_chart(chart_score(T_m,lam_m,mdl['T2_UCL'],evr_m,
                                        pc_i,pc_j,flag_m,alpha=alpha,n_train=n_tr),
                            use_container_width=True,key='score_chart')
            st.caption(f"⚫ Inside {int(alpha*100)}% ellipse  |  "
                       f"🔴 Outside {int(alpha*100)}% ellipse  |  "
                       f"Ellipse based on F distribution (k=2, n={n_tr}, α={alpha})")

        st.markdown("---")
        top5=np.argsort(np.sqrt(P[:,pc_i]**2+P[:,pc_j]**2))[::-1][:5]
        top_str=", ".join(f"{fn[i]}({P[i,pc_i]:.3f}/{P[i,pc_j]:.3f})" for i in top5)
        prompt_load=(
            f"{'Loading' if tipo=='Loading plot' else 'Score'} plot "
            f"PC{pc_i+1} ({evr_m[pc_i]:.1f}%) vs PC{pc_j+1} ({evr_m[pc_j]:.1f}%). "
            f"Top vars: {top_str}.\n"
            "3 bullets: (1) What process phenomena do these PCs represent? "
            "(2) What do top variables tell us? (3) Notable correlations?"
        )
        llm_button("Interpret chart",prompt_load,key='ai_load')


# ═══════════════════════════════════════════
#  TAB 5 — MONITORING
# ═══════════════════════════════════════════
with tab5:
    if st.session_state.model is None:
        box("⬆️ Build the model first.","info")
    else:
        mdl=st.session_state.model; fn=mdl['feature_names']
        wf_mon=get_workflow()
        st.markdown('<div class="section-header">Phase II — Process Monitoring</div>',
                    unsafe_allow_html=True)
        st.markdown("")

        if wf_mon in ('diagnostic','exploratory'):
            box(f"ℹ️ <strong>{WF_STYLES[wf_mon][3]}</strong> workflow — anomalies are identified "
                f"in Calibration (Phase I flags). See <strong>Summary</strong> for root cause analysis. "
                f"You can still load additional files below for comparison.","info")
            st.markdown("**Optional: compare additional datasets**")
        else:
            # SPC workflow — offer built-in test set if split was applied
            if st.session_state.df_test_builtin is not None:
                already_loaded = 'Built-in test set' in [
                    f['name'] for f in st.session_state.mon_files]
                if not already_loaded:
                    box("ℹ️ A train/test split was applied in the Dataset tab. "
                        "You can use the built-in test set directly, or upload separate files.",
                        "info")
                    if st.button("▶ Use built-in test set",
                                 key='btn_use_builtin', type='primary'):
                        df_te = st.session_state.df_test_builtin
                        X_te = (df_te[fn].values
                                if set(fn).issubset(df_te.columns)
                                else df_te.values)
                        st.session_state.mon_files.append({
                            'name': 'Built-in test set',
                            'n_rows': len(df_te),
                            'mon': monitor_new(mdl, X_te)
                        })
                        st.rerun()
                else:
                    box("✅ Built-in test set already loaded in monitoring.",
                        "ok")
            st.markdown("**Or upload additional files from USB**")

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
                fname=up_test.name
                if fname not in [f['name'] for f in st.session_state.mon_files]:
                    st.session_state.mon_files.append({
                        'name':fname,'n_rows':len(df_new),
                        'mon':monitor_new(mdl,df_new.values)})
                    box(f"✅ Added: <strong>{fname}</strong> ({len(df_new)} cycles)","ok")
                else:
                    box(f"ℹ️ File <strong>{fname}</strong> already loaded.","info")

        if st.session_state.mon_files:
            st.markdown("**Loaded files**")
            for fi,fobj in enumerate(st.session_state.mon_files):
                col_f,col_rm=st.columns([5,1])
                col_f.markdown(
                    f"<span class='file-tag' style='background:{CHART_PALETTE[fi%len(CHART_PALETTE)]}18;"
                    f"color:{CHART_PALETTE[fi%len(CHART_PALETTE)]};"
                    f"border:1px solid {CHART_PALETTE[fi%len(CHART_PALETTE)]}44'>"
                    f"● {fobj['name']}</span> — {fobj['n_rows']} cycles",
                    unsafe_allow_html=True)
                if col_rm.button("✕",key=f'rm_{fi}'):
                    st.session_state.mon_files.pop(fi); st.rerun()

            all_t2 =np.concatenate([f['mon']['T2']      for f in st.session_state.mon_files])
            all_q  =np.concatenate([f['mon']['Q']       for f in st.session_state.mon_files])
            all_t2f=np.concatenate([f['mon']['T2_flag'] for f in st.session_state.mon_files])
            all_qf =np.concatenate([f['mon']['Q_flag']  for f in st.session_state.mon_files])
            all_Xns=np.concatenate([f['mon']['Xn_s']    for f in st.session_state.mon_files])
            all_En =np.concatenate([f['mon']['En']       for f in st.session_state.mon_files])
            all_Tn =np.concatenate([f['mon']['Tn']       for f in st.session_state.mon_files])

            boundaries=[0]+list(np.cumsum([f['n_rows'] for f in st.session_state.mon_files]))
            file_names=[f['name'] for f in st.session_state.mon_files]

            n_test=len(all_t2); n_t2=int(all_t2f.sum()); n_q=int(all_qf.sum())
            n_any=int((all_t2f|all_qf).sum()); pct=n_any/n_test*100
            stato="🟢 STABLE" if pct<5 else "🟡 WARNING" if pct<15 else "🔴 ANOMALIES"

            st.markdown("---")
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Total cycles",n_test)
            c2.metric("T² anomalies",f"{n_t2} ({n_t2/n_test*100:.1f}%)")
            c3.metric("Q anomalies",f"{n_q} ({n_q/n_test*100:.1f}%)")
            c4.metric("Status",stato)

            # Per-file table
            rows_pf=[]
            for fobj in st.session_state.mon_files:
                m=fobj['mon']; nf=len(m['T2'])
                nt2=int(m['T2_flag'].sum()); nq=int(m['Q_flag'].sum())
                na=int((m['T2_flag']|m['Q_flag']).sum())
                rows_pf.append({'File':fobj['name'],'Cycles':nf,
                                'T² anom':f"{nt2} ({nt2/nf*100:.1f}%)",
                                'Q anom': f"{nq} ({nq/nf*100:.1f}%)",
                                'Any':    f"{na} ({na/nf*100:.1f}%)",
                                'Status': "🟢" if na/nf<0.05 else "🟡" if na/nf<0.15 else "🔴"})
            st.dataframe(pd.DataFrame(rows_pf),use_container_width=True,hide_index=True)

            prompt_ov=(
                f"Phase II: {n_test} cycles, {n_any} anomalies ({pct:.1f}%), status={stato}. "
                f"Files: {', '.join(file_names)}.\n"
                "2 lines: is the process stable? Any drift across files? What should supervisor do?"
            )
            llm_button("Interpret process status",prompt_ov,key='ai_overview')

            st.markdown("---")
            st.markdown("**Control charts**")
            st.plotly_chart(
                chart_line_multi(all_t2,mdl['T2_UCL'],'Phase II — Hotelling T²',
                                 COLORS['primary'],all_t2f,file_names,boundaries),
                use_container_width=True,key='p2_t2')
            st.plotly_chart(
                chart_line_multi(all_q,mdl['Q_UCL'],'Phase II — Q (SPE)',
                                 COLORS['success'],all_qf,file_names,boundaries),
                use_container_width=True,key='p2_q')

            st.markdown("**Anomaly analysis**")
            result=show_anomaly_table_and_contrib(
                mdl,all_t2,all_q,all_Xns,all_En,all_Tn,fn,'p2_table','p2')
            if result:
                obs,t2_obs,q_obs,ratio,top_t2,top_q=result
                file_of_obs="unknown"
                for fi,(s,e) in enumerate(zip(boundaries[:-1],boundaries[1:])):
                    if s<=obs<e: file_of_obs=file_names[fi]; break
                st.caption(f"Cycle {obs} — from: **{file_of_obs}**")
                v_t2=[fn[i] for i in top_t2]; v_q=[fn[i] for i in top_q]
                prompt_an=(
                    f"Anomaly cycle {obs} from {file_of_obs}: "
                    f"T²={t2_obs:.2f}×UCL, Q={q_obs:.2f}×UCL. "
                    f"T² vars: {', '.join(v_t2)}. Q vars: {', '.join(v_q)}.\n"
                    "3 bullets for supervisor: (1) What is happening physically? "
                    "(2) Most likely cause? (3) Immediate action? No statistics."
                )
                llm_button("Explain anomaly to technician",prompt_an,key=f'ai_an_{obs}')
                st.markdown("**📝 Log intervention**")
                with st.form(f"log_{obs}"):
                    azione=st.text_area("Corrective action taken",height=70)
                    if st.form_submit_button("💾 Save",use_container_width=True):
                        if azione:
                            st.session_state.anomaly_log.append({
                                'Cycle':obs,'File':file_of_obs,
                                'T²':round(t2_obs,3),'Q':round(q_obs,3),
                                'Severity':f"{ratio:.2f}×UCL",'Action':azione})
                            box("✅ Saved.","ok")
            if st.session_state.anomaly_log:
                st.markdown("**📋 Intervention log**")
                st.dataframe(pd.DataFrame(st.session_state.anomaly_log),
                             use_container_width=True)


# ═══════════════════════════════════════════
#  TAB 6 — SUMMARY & ROOT CAUSE
# ═══════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Summary & Root Cause Analysis</div>',
                unsafe_allow_html=True)
    st.markdown("")
    if st.session_state.model is None:
        box("⬆️ Build the calibration model first.","info")
    else:
        mdl=st.session_state.model; fn=mdl['feature_names']
        wf_sum=get_workflow()

        cm1,cm2,cm3,cm4=st.columns(4)
        cm1.metric("k PCs",mdl['k'])
        cm2.metric("T² UCL",f"{mdl['T2_UCL']:.3f}")
        cm3.metric("Q UCL",f"{mdl['Q_UCL']:.3f}")
        nf_p1=int(((mdl['T2']>mdl['T2_UCL'])|(mdl['Q']>mdl['Q_UCL'])).sum())
        cm4.metric("Phase I flags",f"{nf_p1} ({nf_p1/len(mdl['T2'])*100:.1f}%)")

        # Data source based on workflow
        if wf_sum in ('diagnostic','exploratory'):
            all_t2=mdl['T2']; all_q=mdl['Q']
            all_t2f=mdl['T2']>mdl['T2_UCL']; all_qf=mdl['Q']>mdl['Q_UCL']
            all_Xns=mdl['X_scaled']; all_En=mdl['E']; all_Tn=mdl['scores']
            source_label="Phase I (full dataset)"
            box(f"ℹ️ <strong>{WF_STYLES[wf_sum][3]}</strong> — showing anomalies from Phase I.","info")
        elif st.session_state.mon_files:
            all_t2 =np.concatenate([f['mon']['T2']      for f in st.session_state.mon_files])
            all_q  =np.concatenate([f['mon']['Q']       for f in st.session_state.mon_files])
            all_t2f=np.concatenate([f['mon']['T2_flag'] for f in st.session_state.mon_files])
            all_qf =np.concatenate([f['mon']['Q_flag']  for f in st.session_state.mon_files])
            all_Xns=np.concatenate([f['mon']['Xn_s']    for f in st.session_state.mon_files])
            all_En =np.concatenate([f['mon']['En']       for f in st.session_state.mon_files])
            all_Tn =np.concatenate([f['mon']['Tn']       for f in st.session_state.mon_files])
            source_label="Phase II monitoring"
        else:
            box("⬆️ Load Phase II monitoring data in the Monitoring tab, "
                "or switch to Diagnostics/Exploratory objective to analyse Phase I data.","info")
            st.stop()
            all_t2=None

        n_test=len(all_t2); n_t2=int(all_t2f.sum()); n_q=int(all_qf.sum())
        n_any=int((all_t2f|all_qf).sum()); pct=n_any/n_test*100
        stato="🟢 STABLE" if pct<5 else "🟡 WARNING" if pct<15 else "🔴 ANOMALIES"

        st.markdown(f"**Analysis results — {source_label}**")
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Total cycles",n_test)
        m2.metric("T² anomalies",f"{n_t2} ({n_t2/n_test*100:.1f}%)")
        m3.metric("Q anomalies",f"{n_q} ({n_q/n_test*100:.1f}%)")
        m4.metric("Status",stato)

        flagged_idx=np.where(all_t2f|all_qf)[0]
        if len(flagged_idx)==0:
            box("✅ No anomalies detected — process is stable.","ok")
        else:
            # Top anomalies
            st.markdown("**Top anomalies by severity**")
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

            # Variable frequency
            st.markdown("**Most recurrent variables in anomalies**")
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
                '#':range(1,len(fn)+1),'Variable':fn,
                'T² exceed':t2_cnt.astype(int),'Q exceed':q_cnt.astype(int),
                'Total':(t2_cnt+q_cnt).astype(int),
            }).sort_values('Total',ascending=False)
            df_vt=df_vars[df_vars['Total']>0].head(10)
            if len(df_vt)>0:
                st.dataframe(df_vt,use_container_width=True,hide_index=True)
                fig_var=go.Figure()
                fig_var.add_bar(x=df_vt['Variable'].tolist(),
                                y=df_vt['T² exceed'].tolist(),
                                name='T² exceedances',marker_color=COLORS['primary'],
                                marker_line_width=0)
                fig_var.add_bar(x=df_vt['Variable'].tolist(),
                                y=df_vt['Q exceed'].tolist(),
                                name='Q exceedances',marker_color=COLORS['danger'],
                                marker_line_width=0)
                fig_var.update_layout(
                    barmode='stack',
                    title=dict(text='Variable exceedance frequency',
                               font=dict(family='Inter',size=12)),
                    xaxis=dict(title='Variable',tickangle=-40,
                               gridcolor=COLORS['border']),
                    yaxis=dict(title='N° anomalous cycles',gridcolor=COLORS['border']),
                    height=300,margin=dict(l=10,r=10,t=40,b=80),
                    plot_bgcolor=COLORS['surface'],paper_bgcolor=COLORS['white'],
                    font=dict(family='Inter'),
                    legend=dict(orientation='h',y=-0.45,font=dict(size=11)))
                st.plotly_chart(fig_var,use_container_width=True,key='summary_var_chart')

            st.markdown("---")
            top_ov=df_vars.nlargest(5,'Total')['Variable'].tolist()
            top_t2v=df_vars.nlargest(5,'T² exceed')['Variable'].tolist()
            top_qv =df_vars.nlargest(5,'Q exceed')['Variable'].tolist()
            files_str=", ".join(f['name'] for f in st.session_state.mon_files) if st.session_state.mon_files else "Phase I data"
            prompt_rc=(
                f"Analysis: {n_test} cycles, {n_any} anomalies ({pct:.1f}%), {stato}.\n"
                f"T² top variables: {', '.join(top_t2v)}.\n"
                f"Q top variables: {', '.join(top_qv)}.\n"
                f"Source: {files_str}.\n\n"
                "Structured root cause analysis:\n"
                "**1. DIAGNOSIS** — what is happening physically?\n"
                "**2. PROBABLE ROOT CAUSES** — ranked, reference variables\n"
                "**3. IMMEDIATE ACTIONS** — what to do now\n"
                "**4. MEDIUM-TERM ACTIONS** — prevent recurrence\n"
                "**5. WHAT TO MONITOR** — variables and limits going forward\n"
                "Be specific. No generic statements."
            )
            llm_button("Generate Root Cause Analysis & Action Plan",
                       prompt_rc,key='ai_rootcause')

            # ── Y VALIDATION — only if Y columns are defined ────────
            y_names = st.session_state.get('y_names', [])
            df_Y    = st.session_state.get('df_Y', None)

            if y_names and df_Y is not None and len(df_Y) > 0:
                st.markdown("---")
                st.markdown("### 🎯 Y Validation")
                st.markdown(
                    "Compares PCA-SPC anomaly flags against your defined Y variables. "
                    "Helps assess whether the model detects process conditions "
                    "that actually lead to quality issues or failures."
                )

                # Align Y with the observations used in the analysis
                # For SPC (Phase II): Y needs to match test set indices
                # For Diagnostic: Y matches full dataset
                if wf_sum in ('diagnostic','exploratory'):
                    # Full dataset — Y aligns directly
                    df_Y_aligned = df_Y.reset_index(drop=True)
                    n_y = len(df_Y_aligned)
                    if n_y != n_test:
                        box("⚠️ Y variable length does not match the analysed dataset. "
                            "Cannot perform validation.","warn")
                        df_Y_aligned = None
                else:
                    # SPC — Y for the test portion only
                    if st.session_state.df_test_builtin is not None:
                        # Get original full df_Y and slice test portion
                        df_X_full = st.session_state.df_X
                        n_full = len(df_X_full)
                        test_size = len(st.session_state.df_test_builtin)
                        train_size = n_full - test_size
                        df_Y_aligned = df_Y.iloc[train_size:].reset_index(drop=True)
                        if len(df_Y_aligned) != n_test:
                            df_Y_aligned = None
                            box("⚠️ Y variable length does not match test set. "
                                "Cannot perform validation.","warn")
                    else:
                        df_Y_aligned = None
                        box("ℹ️ Y validation for externally loaded files is not "
                            "currently supported. Use the built-in train/test split.","info")

                if df_Y_aligned is not None:
                    for y_col in y_names:
                        if y_col not in df_Y_aligned.columns:
                            continue

                        y_vals = df_Y_aligned[y_col].values
                        pca_flags = (all_t2f | all_qf)

                        # Check if Y looks binary or continuous
                        unique_vals = np.unique(y_vals[~np.isnan(y_vals)])
                        is_binary = len(unique_vals) <= 2

                        st.markdown(f"**Y variable: `{y_col}`**")

                        if is_binary:
                            # Confusion matrix style analysis
                            y_bin = (y_vals > 0).astype(int)
                            pca_bin = pca_flags.astype(int)

                            tp = int(((pca_bin==1) & (y_bin==1)).sum())
                            fp = int(((pca_bin==1) & (y_bin==0)).sum())
                            fn_ = int(((pca_bin==0) & (y_bin==1)).sum())
                            tn = int(((pca_bin==0) & (y_bin==0)).sum())

                            n_y1 = int(y_bin.sum())
                            n_flagged = int(pca_bin.sum())

                            precision = tp/(tp+fp) if (tp+fp)>0 else 0
                            recall    = tp/(tp+fn_) if (tp+fn_)>0 else 0
                            f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

                            # KPI row
                            v1,v2,v3,v4 = st.columns(4)
                            v1.metric("Y=1 events", f"{n_y1} ({n_y1/n_test*100:.1f}%)")
                            v2.metric("PCA flagged", f"{n_flagged} ({n_flagged/n_test*100:.1f}%)")
                            v3.metric("Overlap (TP)", f"{tp}",
                                      help="Cycles flagged by PCA-SPC that also have Y=1")
                            v4.metric("Missed (FN)", f"{fn_}",
                                      help="Y=1 events NOT flagged by PCA-SPC")

                            # Confusion matrix as table
                            st.markdown("**Confusion matrix:**")
                            df_cm = pd.DataFrame({
                                '': ['PCA flag = 1', 'PCA flag = 0'],
                                'Y = 1': [tp, fn_],
                                'Y = 0': [fp, tn]
                            })
                            st.dataframe(df_cm, use_container_width=False,
                                         hide_index=True)

                            # Metrics
                            m1,m2,m3 = st.columns(3)
                            m1.metric("Precision",f"{precision:.1%}",
                                      help="Of PCA-flagged cycles, how many had Y=1?")
                            m2.metric("Recall",f"{recall:.1%}",
                                      help="Of Y=1 events, how many were flagged by PCA?")
                            m3.metric("F1 Score",f"{f1:.2f}")

                            # Interpretation
                            if recall > 0.7 and precision > 0.5:
                                interp_kind = "ok"
                                interp = (f"✅ Strong agreement — the PCA-SPC model detects "
                                          f"{recall:.0%} of {y_col} events with "
                                          f"{precision:.0%} precision. "
                                          f"The process variables carry meaningful signal.")
                            elif recall > 0.4:
                                interp_kind = "warn"
                                interp = (f"⚠️ Partial agreement — the model detects "
                                          f"{recall:.0%} of {y_col} events. "
                                          f"{fn_} events were missed — they may have causes "
                                          f"outside the monitored X variables.")
                            else:
                                interp_kind = "warn"
                                interp = (f"⚠️ Low overlap — only {recall:.0%} of {y_col} events "
                                          f"are flagged by PCA-SPC. The process variables may not "
                                          f"fully explain this outcome, or the model needs "
                                          f"recalibration on a more representative period.")
                            box(interp, interp_kind)

                            # AI interpretation
                            prompt_yval = (
                                f"PCA-SPC model validated against Y='{y_col}'.\n"
                                f"Total cycles: {n_test}. Y=1 events: {n_y1} ({n_y1/n_test*100:.1f}%).\n"
                                f"PCA flagged: {n_flagged}. True positives: {tp}. "
                                f"False negatives: {fn_}. False positives: {fp}.\n"
                                f"Precision: {precision:.1%}. Recall: {recall:.1%}. F1: {f1:.2f}.\n\n"
                                "In 3 bullet points:\n"
                                "• What does this overlap tell us about the process?\n"
                                "• Why might some Y=1 events be missed by PCA-SPC?\n"
                                "• What does this mean for using this model in production?"
                            )
                            llm_button(f"Interpret Y validation — {y_col}",
                                       prompt_yval, key=f'ai_yval_{y_col}')

                        else:
                            # Continuous Y — compare means
                            y_anomalous = y_vals[pca_flags]
                            y_normal    = y_vals[~pca_flags]

                            if len(y_anomalous) > 0 and len(y_normal) > 0:
                                c1,c2,c3 = st.columns(3)
                                c1.metric("Y mean — normal cycles",
                                          f"{y_normal.mean():.3f}",
                                          help="Mean of Y for cycles inside control limits")
                                c2.metric("Y mean — anomalous cycles",
                                          f"{y_anomalous.mean():.3f}",
                                          delta=f"{y_anomalous.mean()-y_normal.mean():+.3f}",
                                          help="Mean of Y for cycles flagged by PCA-SPC")
                                diff_pct = abs(y_anomalous.mean()-y_normal.mean()) / \
                                           (abs(y_normal.mean())+1e-9) * 100
                                c3.metric("Difference", f"{diff_pct:.1f}%")

                                if diff_pct > 10:
                                    box(f"✅ Anomalous cycles show a meaningful difference "
                                        f"in <strong>{y_col}</strong> ({diff_pct:.1f}% shift). "
                                        f"The PCA-SPC model captures process variation "
                                        f"that affects this quality metric.","ok")
                                else:
                                    box(f"⚠️ Anomalous cycles show little difference "
                                        f"in <strong>{y_col}</strong> ({diff_pct:.1f}% shift). "
                                        f"The flagged process variations may not directly "
                                        f"impact this quality variable.","warn")

                        st.markdown("")  # spacing between Y variables

        if st.session_state.anomaly_log:
            st.markdown("---")
            st.markdown("**📋 Intervention log**")
            st.dataframe(pd.DataFrame(st.session_state.anomaly_log),
                         use_container_width=True)
