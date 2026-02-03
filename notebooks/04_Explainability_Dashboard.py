"""Credit Risk Explainability Dashboard"""
import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Credit Risk Explainability Dashboard", layout="wide")

# Setting Custom CSS
st.markdown("""
<style>
:root{--bg:#071018;--muted:#9ca3af;--accent:#10b981;--neg:#ef4444;}
html,body,[class*="stApp"]{background:var(--bg);color:#e6eef8}
.block-container{padding-top:22px;padding-left:18px;padding-right:18px;}
.title-center{text-align:center;font-weight:800;font-size:36px;margin:6px 0;}
.subtitle-center{text-align:center;color:var(--muted);margin:0 0 10px 0;font-size:0.95rem;}
.small{color:var(--muted);font-size:0.88rem}
.big{font-weight:700;font-size:1.15rem;margin-top:6px}
.prob{font-weight:900;font-size:1.9rem;color:var(--accent);margin-top:6px}
.good{color:var(--accent);}.bad{color:var(--neg);}.neutral{color:var(--muted);}
.summary{background:rgba(255,255,255,0.02);padding:10px 12px;border-radius:8px;color:#e6eef8;line-height:1.3;font-size:0.92rem;margin:0;}
.summary .val{color:#10b981;font-weight:700;}
.summary ul{margin:4px 0 4px 18px;padding:0;}
.summary li{margin:2px 0;line-height:1.3;}
.summary strong{display:block;margin-bottom:3px;}
.stImage img{background:transparent!important;}
</style>
""", unsafe_allow_html=True)
#Loading artifacts
ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")

def load(name):
    p = os.path.join(ARTIFACTS, name)
    if not os.path.exists(p):
        st.error(f"Missing: {name}")
        st.stop()
    return joblib.load(p)

rf = load("rf_model.joblib")
X_test_df = load("X_test_df.joblib")
X_test_trans = load("X_test_transformed_rf.joblib")
y_test = load("y_test.joblib")
X_lime_train = load("X_lime_train.joblib")
feature_names = list(load("feature_names.joblib"))
class_names = load("class_names.joblib")
lime_meta = load("lime_meta.joblib")
shap_vals = load("shap_values_class0_rf.joblib")
try:
    shap_base = load("shap_expected_value0.joblib")
except:
    shap_base = 0.0

# Cleaning and formatting names/values
def clean_name(s):
    """Clean feature name"""
    if not s: return ""
    s = str(s)
    for p in ("cat__", "cat_", "feature__", "f__"):
        if s.startswith(p): s = s[len(p):]
    s = s.replace("_", " ").replace("/", " / ")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:37] + "..." if len(s) > 40 else s

def find_val(row, fname):
    """Find feature value in row"""
    if not fname: return "N/A"
    fname = str(fname).strip()
    if fname in row.index: return row[fname]
    for p in ("cat__", "cat_", "feature__", "f__"):
        if fname.startswith(p):
            t = fname[len(p):]
            if t in row.index: return row[t]
    clean_target = clean_name(fname).lower()
    for col in row.index:
        if clean_name(col).lower() == clean_target:
            return row[col]
    return "N/A"

def fmt_val(v):
    """Format value for display"""
    try:
        if v is None: return "N/A"
        if isinstance(v, (float, np.floating, int, np.integer)):
            return f"{float(v):.4f}"
        return f"{float(v):.4f}"
    except:
        s = str(v)
        return s if len(s) <= 60 else s[:57] + "..."

def get_state(v):
    """Get value state (good/bad/neutral)"""
    try:
        if v is None: return "neutral"
        if isinstance(v, (int, np.integer)):
            if int(v) == 1: return "good"
            if int(v) == 0: return "bad"
        s = str(v).strip()
        if s in ("1", "1.0"): return "good"
        if s in ("0", "0.0"): return "bad"
    except: pass
    s = str(v).lower()
    if "good" in s: return "good"
    if "bad" in s: return "bad"
    return "neutral"

def get_label(v):
    """Get display label"""
    s = get_state(v)
    return "Good" if s == "good" else ("Bad" if s == "bad" else str(v))

# Adding SHAP and LIME summaries
def shap_summary(vals, fnames, row, k=3, pred=None):
    """Generate SHAP summary HTML"""
    try:
        arr = np.asarray(vals).ravel()
        if arr.size != len(fnames):
            return f"<strong>Prediction:</strong> {get_label(pred)}<br><em>SHAP unavailable</em>"
    except:
        return f"<strong>Prediction:</strong> {get_label(pred)}<br><em>SHAP unavailable</em>"
    
    pairs = sorted(zip(fnames, arr), key=lambda x: abs(x[1]), reverse=True)
    pos = [p for p in pairs if p[1] > 0][:k]
    neg = [p for p in pairs if p[1] < 0][:k]
    
    html = [f"<strong>Prediction: {get_label(pred)}</strong>"]
    
    if pos:
        html.append("<strong>Pushers toward:</strong><ul>")
        for fn, sc in pos:
            cn = clean_name(fn)
            v = find_val(row, fn)
            html.append(f"<li>{cn}: <span class='val'>{fmt_val(v)}</span> ({sc:+.2f})</li>")
        html.append("</ul>")
    
    if neg:
        html.append("<strong>Pushers away:</strong><ul>")
        for fn, sc in neg:
            cn = clean_name(fn)
            v = find_val(row, fn)
            html.append(f"<li>{cn}: <span class='val'>{fmt_val(v)}</span> ({sc:+.2f})</li>")
        html.append("</ul>")
    
    return "".join(html)

def lime_summary(exp_list, row, k=3, pred=None):
    """Generate LIME summary HTML"""
    try:
        pos = [p for p in exp_list if p[1] > 0][:k]
        neg = [p for p in exp_list if p[1] < 0][:k]
    except:
        return f"<strong>Prediction:</strong> {get_label(pred)}<br><em>LIME unavailable</em>"
    
    html = [f"<strong>Prediction: {get_label(pred)}</strong>"]
    
    if pos:
        html.append("<strong>Pushers toward:</strong><ul>")
        for desc, wt in pos:
            tok = re.split(r"\s*(<=|>=|<|>|=| in |:)\s*", desc)[0].strip()
            cn = clean_name(tok)
            v = find_val(row, tok)
            html.append(f"<li>{cn}: <span class='val'>{fmt_val(v)}</span> ({wt:+.2f})</li>")
        html.append("</ul>")
    
    if neg:
        html.append("<strong>Pushers away:</strong><ul>")
        for desc, wt in neg:
            tok = re.split(r"\s*(<=|>=|<|>|=| in |:)\s*", desc)[0].strip()
            cn = clean_name(tok)
            v = find_val(row, tok)
            html.append(f"<li>{cn}: <span class='val'>{fmt_val(v)}</span> ({wt:+.2f})</li>")
        html.append("</ul>")
    
    return "".join(html)

st.markdown('<div class="title-center">Credit Risk Explainability Dashboard</div>', unsafe_allow_html=True)

n = X_test_df.shape[0]
_, mid, _ = st.columns([1, 6, 1])
with mid:
    idx = st.slider("Sample index", 0, n - 1, 0)

# Getting prediction and probabilities
row = X_test_df.iloc[[idx]]
try:
    pred = rf.predict(row)[0]
    probs = rf.predict_proba(row)[0]
except:
    pred = None
    probs = np.zeros(len(getattr(rf, "classes_", [0])))

try:
    actual = y_test[idx]
except:
    actual = None

actual_state = get_state(actual)
pred_state = get_state(pred)
actual_label = get_label(actual)
pred_label = get_label(pred)

# Good credit probability
try:
    good_idx = next(i for i, c in enumerate(class_names) if "good" in str(c).lower())
except:
    try:
        good_idx = list(rf.classes_).index(pred) if pred in list(rf.classes_) else 0
    except:
        good_idx = 0
good_prob = float(probs[good_idx]) if len(probs) > good_idx else 0.0
pct = f"{good_prob*100:.0f}%"

# Metric cards
c1, c2, c3, c4 = st.columns(4, gap="small")
with c1:
    st.markdown(f'<div class="card"><div class="small">Sample</div><div class="big">{idx}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="card"><div class="small">Actual</div><div class="big"><span class="{actual_state}">{actual_label}</span></div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="card"><div class="small">Prediction</div><div class="big"><span class="{pred_state}">{pred_label}</span></div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="card"><div class="small">Good credit probability</div><div class="prob">{pct}</div></div>', unsafe_allow_html=True)

# SHAP 
left, right = st.columns([6, 4], gap="medium")

with left:
    st.markdown('<div class="card" style="padding:8px">', unsafe_allow_html=True)
    st.markdown("<strong>SHAP — local waterfall</strong>", unsafe_allow_html=True)
    try:
        local_shap = np.asarray(shap_vals[idx])
        base = float(shap_base) if shap_base is not None else 0.0
        short_names = [clean_name(f) for f in feature_names]
        expl = shap.Explanation(values=local_shap, base_values=base, feature_names=short_names)
        
        plt.style.use("dark_background")
        plt.rcParams['font.size'] = 10
        fig = plt.figure(figsize=(9, 4.0))
        fig.patch.set_facecolor("#071018")
        shap.waterfall_plot(expl, max_display=8, show=False)
        plt.tight_layout()
        plt.subplots_adjust(left=0.22, right=0.98, top=0.95, bottom=0.05)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"SHAP error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card" style="padding:8px">', unsafe_allow_html=True)
    st.markdown("<strong>Findings — SHAP</strong>", unsafe_allow_html=True)
    try:
        shap_html = shap_summary(local_shap, feature_names, X_test_df.iloc[idx], k=3, pred=pred)
    except:
        shap_html = f"<strong>Prediction:</strong> {pred_label or 'N/A'}<br><em>SHAP unavailable</em>"
    st.markdown(f'<div class="summary">{shap_html}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

#LIME
left2, right2 = st.columns([6, 4], gap="medium")

with left2:
    st.markdown('<div class="card" style="padding:8px">', unsafe_allow_html=True)
    st.markdown("<strong>LIME — local surrogate</strong>", unsafe_allow_html=True)
    try:
        X_lime_np = np.array(X_lime_train)
        lime_exp_obj = LimeTabularExplainer(
            training_data=X_lime_np,
            feature_names=feature_names,
            class_names=list(map(str, class_names)),
            mode=lime_meta.get("mode", "classification"),
            random_state=lime_meta.get("random_state", None)
        )
        
        try:
            target_idx = list(rf.classes_).index(pred) if pred in list(rf.classes_) else 0
        except:
            target_idx = 0
        
        try:
            row_trans = np.array(X_test_trans[idx])
        except:
            row_trans = np.array(X_test_df.iloc[idx])
        
        lime_exp = lime_exp_obj.explain_instance(
            data_row=row_trans,
            predict_fn=rf.predict_proba,
            labels=(target_idx,),
            num_features=12,
            num_samples=4000
        )
        
        fig_lime = lime_exp.as_pyplot_figure(label=target_idx)
        try:
            ax = fig_lime.axes[0]
            weights = [w for (_, w) in lime_exp.as_list(label=target_idx)]
            for patch, w in zip(ax.patches, weights):
                patch.set_facecolor("#10b981" if w >= 0 else "#ef4444")
                patch.set_edgecolor("black")
                patch.set_linewidth(0.3)
        except:
            pass
        fig_lime.set_size_inches(9, 3.6)
        st.pyplot(fig_lime)
        plt.close(fig_lime)
    except Exception as e:
        st.error(f"LIME error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

with right2:
    st.markdown('<div class="card" style="padding:8px">', unsafe_allow_html=True)
    st.markdown("<strong>Findings — LIME</strong>", unsafe_allow_html=True)
    try:
        lime_list = lime_exp.as_list(label=target_idx)
        lime_html = lime_summary(lime_list, X_test_df.iloc[idx], k=3, pred=pred)
    except:
        lime_html = f"<strong>Prediction:</strong> {pred_label or 'N/A'}<br><em>LIME unavailable</em>"
    st.markdown(f'<div class="summary">{lime_html}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div style="color:#9ca3af;font-size:0.85rem;margin-top:8px">Summaries are based on predefined definitions and configurations.</div>', unsafe_allow_html=True)