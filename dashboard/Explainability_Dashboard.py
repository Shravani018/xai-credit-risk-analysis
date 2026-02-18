"""Credit Risk Explainability Dashboard"""
import os
import re
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings("ignore")
#Setting configuration of the page
st.set_page_config(
    page_title="Credit Risk Explainability Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# Custom CSS for dark theme and styling
st.markdown("""
<style>
:root {
    --bg:      #071018;
    --surface: #0d1a26;
    --muted:   #6b859e;
    --text:    #d8e8f5;
    --accent:  #10b981;
    --danger:  #ef4444;
    --blue:    #38bdf8;
}

html, body, [class*="stApp"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'Segoe UI', Arial, sans-serif;
}

.block-container { padding: 20px 24px 40px 24px !important; max-width: 1400px !important; }
h1 { text-align: center !important; }
[data-testid="stCaptionContainer"] { text-align: center !important; }
.dash-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 800;
    color: var(--text);
    margin: 4px 0 2px 0;
}
.dash-title span { color: var(--accent); }
.dash-subtitle {
    text-align: center;
    color: var(--muted);
    font-size: 0.85rem;
    margin: 0 0 18px 0;
}

.card {
    background: var(--surface);
    border-radius: 10px;
    padding: 14px 18px;
}
.card-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-bottom: 6px;
}
.card-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text);
}
.card-value.good      { color: var(--accent); }
.card-value.bad       { color: var(--danger); }
.card-value.neutral   { color: var(--blue); }
.card-value.prob-bad  { color: var(--danger); font-size: 1.8rem; }
.card-value.prob-good { color: var(--accent); font-size: 1.8rem; }

.section-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin: 0 0 8px 0;
}

.findings {
    background: var(--surface);
    border-radius: 10px;
    padding: 14px 16px;
    height: 100%;
    box-sizing: border-box;
}
.findings-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin: 0 0 10px 0;
}
.finding-pred {
    font-size: 0.85rem;
    color: var(--muted);
    margin: 0 0 10px 0;
}
.finding-pred .good { color: var(--accent); font-weight: 700; }
.finding-pred .bad  { color: var(--danger); font-weight: 700; }
.finding-section {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 10px 0 5px 0;
}
.finding-section.toward { color: var(--accent); }
.finding-section.away   { color: var(--danger); }
.finding-item {
    display: flex;
    align-items: baseline;
    gap: 6px;
    padding: 4px 0;
    font-size: 0.82rem;
    line-height: 1.4;
}
.finding-item .feat      { color: var(--text); flex: 1; }
.finding-item .val       { color: var(--accent); font-size: 0.78rem; font-family: monospace; }
.finding-item .val.na    { color: var(--muted); }
.finding-item .score-pos { color: var(--accent); font-size: 0.75rem; font-family: monospace; }
.finding-item .score-neg { color: var(--danger);  font-size: 0.75rem; font-family: monospace; }

.dash-footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.75rem;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)
#Loading artifacts
ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "artifacts")
# Helper function to load artifacts with error handling
def load(name):
    p = os.path.join(ARTIFACTS, name)
    if not os.path.exists(p):
        st.error(f"Missing artifact: {name}")
        st.stop()
    return joblib.load(p)

rf            = load("rf_model.joblib")
X_test_df     = load("X_test_df.joblib")
X_test_trans  = load("X_test_transformed_rf.joblib")
y_test        = load("y_test.joblib")
X_lime_train  = load("X_lime_train.joblib")
feature_names = list(load("feature_names.joblib"))
class_names   = load("class_names.joblib")
lime_meta     = load("lime_meta.joblib")
shap_vals     = load("shap_values_class0_rf.joblib")
try:
    shap_base = float(load("shap_expected_value0.joblib"))
except:
    shap_base = 0.0
# Utility functions for cleaning names, finding values, formatting, and determining states
def clean_name(s):
    if not s: return ""
    s = str(s)
    for p in ("cat__", "cat_", "feature__", "f__"):
        if s.startswith(p):
            s = s[len(p):]
    s = s.replace("_", " ").replace("/", " / ")
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:37] + "…") if len(s) > 40 else s

def find_val(row, fname):
    if not fname: return None
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
    return None

def fmt_val(v):
    if v is None: return None
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else f"{f:.4f}"
    except:
        s = str(v)
        return s if len(s) <= 60 else s[:57] + "…"

def get_state(v):
    try:
        if v is None: return "neutral"
        if isinstance(v, (int, np.integer)):
            return "good" if int(v) == 1 else ("bad" if int(v) == 0 else "neutral")
        s = str(v).strip()
        if s in ("1", "1.0"): return "good"
        if s in ("0", "0.0"): return "bad"
    except: pass
    sl = str(v).lower()
    if "good" in sl: return "good"
    if "bad"  in sl: return "bad"
    return "neutral"

def get_label(v):
    s = get_state(v)
    return "Good" if s == "good" else ("Bad" if s == "bad" else str(v))

def _finding_items(pairs, row, score_cls):
    html = []
    for fn, sc in pairs:
        cn  = clean_name(fn)
        raw = find_val(row, fn)
        fv  = fmt_val(raw)
        val_html = (f"<span class='val'>{fv}</span>" if fv is not None
                    else "<span class='val na'>N/A</span>")
        sc_html = f"<span class='{score_cls}'>{sc:+.3f}</span>"
        html.append(
            f"<div class='finding-item'>"
            f"<span class='feat'>{cn}</span>"
            f"{val_html}&nbsp;{sc_html}"
            f"</div>"
        )
    return "".join(html)

def shap_findings_html(vals, fnames, row, k=3, pred=None):
    try:
        arr = np.asarray(vals).ravel()
        assert arr.size == len(fnames)
    except:
        return "<p class='finding-pred'>SHAP unavailable</p>"
    ps    = get_state(pred)
    pairs = sorted(zip(fnames, arr), key=lambda x: abs(x[1]), reverse=True)
    pos   = [p for p in pairs if p[1] > 0][:k]
    neg   = [p for p in pairs if p[1] < 0][:k]
    html  = [f"<p class='finding-pred'>Prediction: <span class='{ps}'>{get_label(pred)}</span></p>"]
    if pos:
        html.append("<div class='finding-section toward'>↑ Pushers toward</div>")
        html.append(_finding_items(pos, row, "score-pos"))
    if neg:
        html.append("<div class='finding-section away'>↓ Pushers away</div>")
        html.append(_finding_items(neg, row, "score-neg"))
    return "".join(html)

def lime_findings_html(exp_list, row, k=3, pred=None):
    try:
        pos = [p for p in exp_list if p[1] > 0][:k]
        neg = [p for p in exp_list if p[1] < 0][:k]
    except:
        return "<p class='finding-pred'>LIME unavailable</p>"
    ps   = get_state(pred)
    html = [f"<p class='finding-pred'>Prediction: <span class='{ps}'>{get_label(pred)}</span></p>"]

    def _lime_items(pairs, score_cls):
        out = []
        for desc, wt in pairs:
            tok = re.split(r"\s*(<=|>=|<|>|=| in |:)\s*", desc)[0].strip()
            cn  = clean_name(tok)
            raw = find_val(row, tok)
            fv  = fmt_val(raw)
            val_html = (f"<span class='val'>{fv}</span>" if fv is not None
                        else "<span class='val na'>N/A</span>")
            sc_html = f"<span class='{score_cls}'>{wt:+.3f}</span>"
            out.append(
                f"<div class='finding-item'>"
                f"<span class='feat'>{cn}</span>"
                f"{val_html}&nbsp;{sc_html}"
                f"</div>"
            )
        return "".join(out)

    if pos:
        html.append("<div class='finding-section toward'>↑ Pushers toward</div>")
        html.append(_lime_items(pos, "score-pos"))
    if neg:
        html.append("<div class='finding-section away'>↓ Pushers away</div>")
        html.append(_lime_items(neg, "score-neg"))
    return "".join(html)
# Theme for SHAP and LIME plots
PLOT_BG = "#0d1a26"
plt.rcParams.update({
    "figure.facecolor": PLOT_BG,
    "axes.facecolor":   PLOT_BG,
    "axes.edgecolor":   "#1e3347",
    "axes.labelcolor":  "#6b859e",
    "xtick.color":      "#6b859e",
    "ytick.color":      "#d8e8f5",
    "text.color":       "#d8e8f5",
    "grid.color":       "#1e3347",
    "font.family":      "sans-serif",
    "font.size":        9,
})
#Setup dashboard title and caption
st.title("Credit Risk Explainability Dashboard")
st.caption("Random Forest | SHAP + LIME | Test Set")
# Sample selector
n = X_test_df.shape[0]
_, mid, _ = st.columns([1, 6, 1])
with mid:
    idx = st.slider("Sample index", 0, n - 1, 0)
# Get prediction, probabilities, and actual label for the selected sample
row = X_test_df.iloc[[idx]]
try:
    pred  = rf.predict(row)[0]
    probs = rf.predict_proba(row)[0]
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

try:
    actual = y_test[idx]
except:
    actual = None

# good=1, bad=0
try:
    classes  = list(rf.classes_)
    bad_idx  = classes.index(0)
    good_idx = classes.index(1)
except:
    bad_idx, good_idx = 0, 1

bad_prob  = float(probs[bad_idx])
good_prob = float(probs[good_idx])
bad_pct   = f"{bad_prob  * 100:.0f}%"

actual_state = get_state(actual)
pred_state   = get_state(pred)
actual_label = get_label(actual)
pred_label   = get_label(pred)
prob_class   = "prob-bad" if bad_prob >= 0.5 else "prob-good"
#Adding summary cards for the selected sample
c1, c2, c3, c4, c5 = st.columns(5, gap="small")
for col, label, value, css in [
    (c1, "Sample",           str(idx),     "neutral"),
    (c2, "Total Samples",    str(n),        "neutral"),
    (c3, "Actual",           actual_label,  actual_state),
    (c4, "Prediction",       pred_label,    pred_state),
    (c5, "Bad Credit Prob.", bad_pct,       prob_class),
]:
    with col:
        st.markdown(
            f'<div class="card">'
            f'<div class="card-label">{label}</div>'
            f'<div class="card-value {css}">{value}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
#SHAP Explanation
shap_left, shap_right = st.columns([6, 4], gap="medium")
local_shap = None

with shap_left:
    st.markdown('<p class="section-label">SHAP — Waterfall Explanation</p>', unsafe_allow_html=True)
    try:
        sv = np.asarray(shap_vals)
        if sv.ndim == 1:
            local_shap = np.asarray(shap_vals[idx]).ravel()
        elif sv.ndim == 2:
            local_shap = sv[idx].ravel()
        elif sv.ndim == 3:
            local_shap = sv[bad_idx][idx].ravel()
        else:
            raise ValueError(f"Unexpected shap_vals shape: {sv.shape}")

        if local_shap.shape[0] != len(feature_names):
            raise ValueError(
                f"SHAP length {local_shap.shape[0]} != features {len(feature_names)}"
            )

        short_names = [clean_name(f) for f in feature_names]
        expl = shap.Explanation(
            values=local_shap,
            base_values=shap_base,
            feature_names=short_names
        )
        fig = plt.figure(figsize=(9, 4.2))
        fig.patch.set_facecolor(PLOT_BG)
        shap.waterfall_plot(expl, max_display=9, show=False)
        plt.tight_layout()
        plt.subplots_adjust(left=0.25, right=0.97, top=0.95, bottom=0.08)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.error(f"SHAP plot error: {e}")

with shap_right:
    st.markdown('<div class="findings"><p class="findings-title">Findings — SHAP</p>', unsafe_allow_html=True)
    try:
        shap_html = shap_findings_html(local_shap, feature_names, X_test_df.iloc[idx], k=3, pred=pred)
    except Exception as e:
        shap_html = f"<p class='finding-pred'>SHAP unavailable: {e}</p>"
    st.markdown(shap_html + "</div>", unsafe_allow_html=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
#LIME Explanation
lime_left, lime_right = st.columns([6, 4], gap="medium")
lime_exp   = None
target_idx = 0

with lime_left:
    st.markdown('<p class="section-label">LIME — Local Interpretation</p>', unsafe_allow_html=True)
    try:
        X_lime_np = np.array(X_lime_train)
        lime_explainer = LimeTabularExplainer(
            training_data=X_lime_np,
            feature_names=feature_names,
            class_names=list(map(str, class_names)),
            mode=lime_meta.get("mode", "classification"),
            random_state=lime_meta.get("random_state", 42)
        )

        try:
            target_idx = list(rf.classes_).index(pred) if pred in list(rf.classes_) else 0
        except:
            target_idx = 0

        try:
            row_trans = np.array(X_test_trans[idx])
        except:
            row_trans = np.array(X_test_df.iloc[idx])

        lime_exp = lime_explainer.explain_instance(
            data_row=row_trans,
            predict_fn=rf.predict_proba,
            labels=(target_idx,),
            num_features=10,
            num_samples=4000
        )

        lime_list = lime_exp.as_list(label=target_idx)
        feats  = [clean_name(re.split(r"\s*(<=|>=|<|>|=| in |:)\s*", d)[0].strip()) for d, _ in lime_list]
        scores = [w for _, w in lime_list]
        colors = ["#10b981" if s >= 0 else "#ef4444" for s in scores]

        fig_lime, ax = plt.subplots(figsize=(9, 3.8))
        fig_lime.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        ax.barh(feats[::-1], scores[::-1], color=colors[::-1], height=0.55)
        ax.axvline(0, color="#6b859e", linewidth=0.8)
        ax.set_xlabel("Weight", color="#6b859e", fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3347")
        ax.tick_params(axis="y", labelsize=8, colors="#d8e8f5")
        ax.tick_params(axis="x", labelsize=7, colors="#6b859e")
        plt.tight_layout()
        plt.subplots_adjust(left=0.32, right=0.97, top=0.95, bottom=0.1)
        st.pyplot(fig_lime, use_container_width=True)
        plt.close(fig_lime)
    except Exception as e:
        st.error(f"LIME error: {e}")

with lime_right:
    st.markdown('<div class="findings"><p class="findings-title">Findings — LIME</p>', unsafe_allow_html=True)
    try:
        lime_list_f = lime_exp.as_list(label=target_idx)
        lime_html   = lime_findings_html(lime_list_f, X_test_df.iloc[idx], k=3, pred=pred)
    except Exception as e:
        lime_html = f"<p class='finding-pred'>LIME unavailable: {e}</p>"
    st.markdown(lime_html + "</div>", unsafe_allow_html=True)
# Footer note about the summaries
st.markdown(
    '<p style="text-align:center;color:#6b859e;font-size:0.75rem;margin-top:20px;">SHAP and LIME values are local approximations and should be interpreted alongside domain expertise.</p>',
    unsafe_allow_html=True
)