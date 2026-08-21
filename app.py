import os
import joblib
import streamlit as st
import pandas as pd


def get_risk_tier(prob, threshold=0.3):
    """Classify PONV risk into tiers."""
    if prob < 0.15:
        return "LOW", "🟢", "#00cc00"
    elif prob < threshold:
        return "MODERATE", "🟡", "#ffcc00"
    elif prob < 0.5:
        return "HIGH", "🔴", "#ff6600"
    else:
        return "VERY HIGH", "⛔", "#cc0000"


def get_explanation(prob, asa, motion, prior, bellville, surgery_type):
    """Generate plain-language explanation."""
    factors = []
    
    if asa >= 2:
        factors.append(f"higher ASA grade ({asa})")
    if motion:
        factors.append("history of motion sickness")
    if prior:
        factors.append("prior PONV")
    if bellville > 5:
        factors.append(f"elevated Bellville score ({bellville})")
    if surgery_type in ["Gynae", "Cardiac"]:
        factors.append(f"{surgery_type} surgery (higher risk)")
    
    if not factors:
        return "Patient has low-risk profile for PONV."
    
    risk_factors = ", ".join(factors)
    if prob > 0.4:
        return f"High PONV risk due to: {risk_factors}. Consider prophylactic anti-emetics."
    elif prob > 0.25:
        return f"Moderate PONV risk due to: {risk_factors}. Monitor closely post-op."
    else:
        return f"Mild PONV risk from: {risk_factors}. Routine post-op care."


@st.cache_resource
def load_artifacts():
    model = None
    meta = None
    if os.path.exists("ponv_model.pkl"):
        model = joblib.load("ponv_model.pkl")
    if os.path.exists("ponv_meta.pkl"):
        meta = joblib.load("ponv_meta.pkl")
    return model, meta


st.set_page_config(page_title="PONV Risk Predictor", layout="wide")
st.title("🏥 PONV Risk Predictor Dashboard")
st.markdown("**48-hour Post-Operative Nausea & Vomiting Risk Assessment**")

model, meta = load_artifacts()

if model is None:
    st.warning("⚠️ No trained model found (ponv_model.pkl). Use `train_model.py` to create one.")
    st.stop()

# Show disclaimer and metadata
if meta and isinstance(meta, dict) and meta.get("disclaimer"):
    st.info("📋 " + meta.get("disclaimer"))

# Show model info
if meta and isinstance(meta, dict) and meta.get("best_model_type"):
    st.markdown(f"**Model Algorithm:** {meta.get('best_model_type').upper()}")

with st.form("input_form"):
    st.subheader("Patient & Procedure Inputs")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Patient Demographics**")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=45)
        bmi = st.number_input("BMI (kg/m²)", min_value=5.0, max_value=80.0, value=25.0, format="%.1f")
        bellville = st.number_input("Bellville score", min_value=0, max_value=10, value=0)
        
        st.markdown("**Risk Factors**")
        motion = st.selectbox("History of motion sickness", options=["No", "Yes"]) == "Yes"
        prior = st.selectbox("Prior PONV", options=["No", "Yes"]) == "Yes"
        prior_surg = st.selectbox("Prior post-op surgery", options=["No", "Yes"]) == "Yes"
    
    with col2:
        st.markdown("**Procedure Details**")
        # ASA severity: 0 Minimal, 1 Mild, 2 Moderate, 3 Severe
        asa = st.selectbox("ASA severity", options=[
            (0, "Minimal"),
            (1, "Mild"),
            (2, "Moderate"),
            (3, "Severe"),
        ], format_func=lambda x: f"{x[1]} ({x[0]})", index=1)
        asa = asa[0]
        
        surgery_opts = meta.get("surgery_types") if meta else ["General", "Orthopaedic", "ENT", "Gynae", "Cardiac"]
        surgery_type = st.selectbox("Surgery type", options=surgery_opts)
        
        anaesthesia_opts = meta.get("anaesthesia_types") if meta else ["GA", "Regional", "MAC"]
        anaesthesia_type = st.selectbox("Anaesthesia type", options=anaesthesia_opts)
        
        st.markdown("**Drugs Administered**")
        col2a, col2b = st.columns(2)
        with col2a:
            glyco = st.checkbox("Glycopyrrolate")
            fent = st.checkbox("Fentanyl")
            prop = st.checkbox("Propofol")
            nmba = st.checkbox("NMBA")
        with col2b:
            para = st.checkbox("Paracetamol")
            ondz = st.checkbox("Ondansetron")
            local = st.checkbox("Local anaesthetic")

    submitted = st.form_submit_button("🔍 Predict PONV Risk", use_container_width=True)

if submitted:
    features = {
        "age": [age],
        "BMI": [bmi],
        "bellville_score": [bellville],
        "ASA": [asa],
        "surgery_type": [surgery_type],
        "anaesthesia_type": [anaesthesia_type],
        "motion_sickness": [int(motion)],
        "prior_ponv": [int(prior)],
        "history_post_op_surgery": [int(prior_surg)],
        "glycopyrrolate": [int(glyco)],
        "fentanyl": [int(fent)],
        "propofol": [int(prop)],
        "NMBA": [int(nmba)],
        "paracetamol": [int(para)],
        "ondansetron": [int(ondz)],
        "local_anaesthetic": [int(local)],
    }
    X = pd.DataFrame.from_dict(features)

    try:
        prob = model.predict_proba(X)[:, 1][0]
        tier, emoji, color = get_risk_tier(prob)
        explanation = get_explanation(prob, asa, motion, prior, bellville, surgery_type)
        
        # Display results
        st.markdown("---")
        st.markdown("## Prediction Results")
        
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.markdown(f"<h1 style='color: {color}; text-align: center;'>{emoji} {tier}</h1>", 
                       unsafe_allow_html=True)
        with col_res2:
            st.metric("PONV Probability (48h)", f"{prob:.1%}")
            
            # Determine threshold from metadata per ASA if available
            threshold = 0.3
            if meta and isinstance(meta, dict) and meta.get("thresholds_by_asa"):
                try:
                    thresholds = meta.get("thresholds_by_asa", {})
                    threshold = thresholds.get(int(asa), threshold)
                except Exception:
                    pass
            st.write(f"Decision threshold (ASA {asa}): **{threshold:.1%}**")
        
        st.markdown(f"**Plain-Language Summary:**\n\n{explanation}")
        
        # Model confidence
        if meta and isinstance(meta, dict) and meta.get("model_results"):
            results = meta.get("model_results", {})
            best_auc = results.get(meta.get("best_model_type"), {}).get("auc", 0)
            st.caption(f"Model AUC on validation: {best_auc:.3f}")
        
        # Show model comparison if available
        if meta and isinstance(meta, dict) and meta.get("model_results"):
            st.markdown("---")
            st.markdown("### Model Performance Comparison")
            results_df = pd.DataFrame(meta.get("model_results", {})).T
            st.dataframe(results_df, use_container_width=True)
        
    except Exception as e:
        st.exception(e)
