import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


MODEL_PATHS = [Path("models/ponv_model.pkl"), Path("ponv_model.pkl")]
META_PATHS = [Path("models/ponv_meta.pkl"), Path("ponv_meta.pkl")]


def get_risk_tier(prob, threshold=0.3):
    if prob < 0.15:
        return "LOW", "#35d07f"
    if prob < threshold:
        return "MODERATE", "#f2c94c"
    if prob < 0.5:
        return "HIGH", "#ff795d"
    return "VERY HIGH", "#ff4d67"


def get_explanation(prob, asa, motion, prior, surgery_type):
    factors = []
    if asa >= 2:
        factors.append(f"ASA grade {asa}")
    if motion:
        factors.append("motion-sickness history")
    if prior:
        factors.append("prior PONV")
    if surgery_type in ["Gynae", "Cardiac"]:
        factors.append(f"{surgery_type} surgery")
    if not factors:
        return "No recorded risk factors were identified in the supplied inputs."
    joined = ", ".join(factors)
    action = "consider prophylactic anti-emetics and close monitoring" if prob >= 0.3 else "continue routine post-operative observation"
    return f"The main recorded factors are {joined}. Based on the estimated probability, {action}."


@st.cache_resource
def load_artifacts():
    model = None
    for path in MODEL_PATHS:
        if path.exists():
            try:
                model = joblib.load(path)
                break
            except Exception:
                continue
    meta = {}
    for path in META_PATHS:
        if path.exists():
            try:
                meta = joblib.load(path)
                break
            except Exception:
                continue
    return model, meta if isinstance(meta, dict) else {}


def metric_card(label, value, detail=""):
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div><div class='metric-detail'>{detail}</div></div>",
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="PONV / 48H", page_icon="+", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#e8f0ef; --muted:#91a6a5; --panel:#122322; --line:#23403d; --mint:#72e1be; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
body, .stApp { background:#091514; color:var(--ink); }
.stApp { background-image:radial-gradient(circle at 80% 0%, #173b35 0, #091514 34rem); }
h1, h2, h3 { font-family:'Space Grotesk', sans-serif; letter-spacing:0; }
h1 { font-size:2.8rem !important; margin-bottom:0 !important; }
.eyebrow { color:var(--mint); font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
.subtle, .metric-detail { color:var(--muted); }
.metric-card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px 18px; min-height:92px; }
.metric-label { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { color:var(--ink); font-family:'Space Grotesk'; font-size:1.85rem; font-weight:700; margin-top:5px; }
.metric-detail { font-size:.78rem; margin-top:3px; }
.risk-panel { border-left:4px solid; background:#122322; border-radius:8px; padding:20px 24px; }
.risk-tier { font-family:'Space Grotesk'; font-size:2.6rem; font-weight:700; }
.stButton > button, .stFormSubmitButton > button { background:#72e1be; color:#09201b; border:0; font-weight:700; border-radius:6px; }
.stTabs [data-baseweb="tab"] { color:var(--muted); }
.stTabs [aria-selected="true"] { color:var(--mint); }
section[data-testid="stSidebar"] { background:#0d1d1b; border-right:1px solid var(--line); }
</style>
""", unsafe_allow_html=True)

model, meta = load_artifacts()
dataset = meta.get("dataset", {})
results = meta.get("model_results", {})
url = os.getenv("PONV_APP_URL", "")

st.markdown("<div class='eyebrow'>Clinical analytics console / 48-hour window</div>", unsafe_allow_html=True)
st.title("PONV risk, made legible.")
st.markdown("<span class='subtle'>Post-operative nausea and vomiting prediction for research and decision support.</span>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if not model:
    st.error("No valid model artifact found. Run `python train.py <dataset.xlsx>` first.")
    st.stop()
if meta.get("disclaimer"):
    st.warning(meta["disclaimer"])

overview_tab, predict_tab, models_tab = st.tabs(["COHORT OVERVIEW", "PATIENT PREDICTOR", "MODEL LAB"])

with overview_tab:
    if dataset:
        st.markdown("### Dataset pulse")
        cards = st.columns(4)
        with cards[0]: metric_card("Cohort", f"{dataset.get('rows', 0):,}", "records analysed")
        with cards[1]: metric_card("PONV positive", f"{dataset.get('prevalence', 0):.1%}", f"{dataset.get('positive_cases', 0):,} cases")
        with cards[2]: metric_card("Features", dataset.get("columns", 0), "columns in source")
        with cards[3]: metric_card("Missing cells", f"{dataset.get('missing_cells', 0):,}", "before preprocessing")
        st.markdown("### Cohort statistics")
        left, right = st.columns([1.15, 1])
        with left:
            counts = pd.DataFrame({"Outcome": ["No PONV", "PONV within 48h"], "Patients": [dataset.get("negative_cases", 0), dataset.get("positive_cases", 0)]})
            fig = px.bar(counts, x="Patients", y="Outcome", orientation="h", color="Outcome", color_discrete_sequence=["#527a72", "#72e1be"])
            fig.update_layout(height=250, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce9e7", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            stats = dataset.get("descriptive_statistics", {})
            if stats:
                table = pd.DataFrame(stats).T.rename(columns={"count": "N", "mean": "Mean", "median": "Median", "std": "SD", "min": "Min", "max": "Max"})
                st.dataframe(table.round(2), use_container_width=True, height=250)
            else:
                st.info("Descriptive statistics will appear after retraining with the dataset.")
    else:
        st.info("The model artifact is available, but cohort metadata is not. Retrain with the 1,500-row dataset to populate statistical analysis.")
    st.markdown("### Dossier reference: adjusted odds ratios")
    st.caption("Independent multivariate analysis reported in the supplied research dossier. These are reference findings, not recalculated by this app.")
    st.dataframe(pd.DataFrame([
        {"Predictor": "Motion sickness history", "Adjusted OR": 2.91, "95% CI": "2.11–4.00", "Interpretation": "Independent, strong"},
        {"Predictor": "Previous PONV", "Adjusted OR": 2.52, "95% CI": "1.70–3.75", "Interpretation": "Independent, strong"},
        {"Predictor": "Female sex", "Adjusted OR": 2.10, "95% CI": "1.58–2.80", "Interpretation": "Independent, moderate"},
        {"Predictor": "General anaesthesia", "Adjusted OR": 1.16, "95% CI": "0.88–1.54", "Interpretation": "Not significant"},
        {"Predictor": "Bellville score", "Adjusted OR": None, "95% CI": "Excluded", "Interpretation": "Possible outcome leakage"},
    ]), use_container_width=True, hide_index=True)
    st.markdown("### Access & deployment")
    if url:
        st.link_button("Open deployed predictor", url, use_container_width=False)
    else:
        st.caption("Set `PONV_APP_URL` to the deployed HTTPS URL to expose a shareable browser link. A Google Play or Chrome Web Store listing cannot be created from this local repository alone.")

with predict_tab:
    st.markdown("### New patient assessment")
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Patient profile**")
            age = st.number_input("Age (years)", 0, 120, 45)
            bmi = st.number_input("BMI (kg/m²)", 5.0, 80.0, 25.0, format="%.1f")
            motion = st.checkbox("History of motion sickness")
            prior = st.checkbox("Prior PONV")
            prior_surg = st.checkbox("Prior post-op surgery")
        with col2:
            st.markdown("**Procedure & medication**")
            sex = st.selectbox("Sex", ["Female", "Male", "Other / not recorded"])
            asa_options = [(0, "Minimal"), (1, "Mild"), (2, "Moderate"), (3, "Severe")]
            asa = st.selectbox("ASA severity", asa_options, format_func=lambda item: f"{item[1]} ({item[0]})", index=1)[0]
            surgeries = meta.get("surgery_types") or ["General", "Orthopaedic", "ENT", "Gynae", "Cardiac"]
            anaesthetics = meta.get("anaesthesia_types") or ["GA", "Regional", "MAC"]
            surgery_type = st.selectbox("Surgery type", surgeries)
            anaesthesia_type = st.selectbox("Anaesthesia type", anaesthetics)
            drug_cols = st.columns(2)
            drugs = ["glycopyrrolate", "fentanyl", "propofol", "NMBA", "paracetamol", "ondansetron", "local_anaesthetic"]
            selected = {}
            for index, drug in enumerate(drugs):
                with drug_cols[index % 2]:
                    selected[drug] = st.checkbox(drug.replace("_", " ").title())
        submitted = st.form_submit_button("Calculate 48-hour risk", use_container_width=True)

    if submitted:
        values = {"age": age, "bmi": bmi, "asa": asa, "surgery_type": surgery_type, "anaesthesia_type": anaesthesia_type, "motion_sickness": int(motion), "prior_ponv": int(prior), "history_post_op_surgery": int(prior_surg), "sex": sex, "gender": sex}
        values.update(selected)
        expected_features = meta.get("features", list(values))
        features = {
            name: [values.get(name.lower(), values.get(name, 0))]
            for name in expected_features
        }
        try:
            prob = float(model.predict_proba(pd.DataFrame(features))[:, 1][0])
            threshold = float(meta.get("thresholds_by_asa", {}).get(asa, 0.3))
            tier, color = get_risk_tier(prob, threshold)
            st.markdown("### Assessment")
            st.markdown(f"<div class='risk-panel' style='border-color:{color}'><div class='eyebrow'>Estimated PONV probability / 48h</div><div class='risk-tier' style='color:{color}'>{tier}</div><div style='font-size:2rem;font-family:Space Grotesk'>{prob:.1%}</div><div class='subtle'>ASA {asa} decision threshold: {threshold:.1%}</div></div>", unsafe_allow_html=True)
            st.markdown("#### Plain-language summary")
            st.write(get_explanation(prob, asa, motion, prior, surgery_type))
        except Exception as error:
            st.exception(error)

with models_tab:
    st.markdown("### Validation performance")
    if results:
        rows = [{"Algorithm": key.upper(), **{metric.title(): value for metric, value in values.items() if metric in ["auc", "accuracy", "precision", "recall"]}} for key, values in results.items()]
        table = pd.DataFrame(rows).sort_values("Auc", ascending=False)
        st.dataframe(table.style.format({column: "{:.3f}" for column in ["Auc", "Accuracy", "Precision", "Recall"]}), use_container_width=True, hide_index=True)
        st.success(f"Auto-selected model: {meta.get('best_model_type', '').upper()} by validation AUC")
        roc_fig = go.Figure()
        for key, values in results.items():
            curve = values.get("roc_curve", {})
            if curve:
                roc_fig.add_trace(go.Scatter(x=curve.get("fpr", []), y=curve.get("tpr", []), mode="lines", name=f"{key.upper()} (AUC {values.get('auc', 0):.3f})"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#527a72"), name="Chance"))
        roc_fig.update_layout(title="Validation ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate", height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce9e7")
        st.plotly_chart(roc_fig, use_container_width=True)
        importance = meta.get("feature_importance", {})
        if importance:
            imp = pd.DataFrame({"Feature": list(importance), "Effect": list(importance.values())}).sort_values("Effect")
            fig = px.bar(imp, x="Effect", y="Feature", orientation="h", color="Effect", color_continuous_scale=["#ff795d", "#72e1be"])
            fig.update_layout(title="Top model signals", height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce9e7")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Model comparison and ROC analysis will appear after retraining.")