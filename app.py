from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


DEFAULT_PATH = Path("data") / "Data_1750.xlsx"
TARGET = "PONV48h"
BASE_FEATURES = [
    "Age",
    "Gender",
    "BMI",
    "ASA",
    "Surgery",
    "Anaesthesia",
    "MotionSickness",
    "PreviousPONV",
    "Glycopyrrolate",
    "Fentanyl",
    "Propofol",
    "NMBA",
    "Paracetamol",
    "Ondansetron",
    "LocalAnaesthetic",
]


@st.cache_data
def load_data(source):
    try:
        frame = pd.read_excel(source, sheet_name="Data_Table 1")
    except ValueError:
        frame = pd.read_excel(source)
    frame = frame.rename(
        columns={
            "Previous (Post Operative Nausea and Vomitting) ": "PreviousPONV",
            "Post Operative Nausea Vomitting_48h": TARGET,
            "PONV_48h": TARGET,
            "Bellville score": "BellvilleScore",
            "Bellville": "BellvilleScore",
        }
    )
    frame["BellvilleScore"] = frame["BellvilleScore"].where(
        frame["BellvilleScore"].between(0, 3)
    )
    frame["Target"] = frame[TARGET].map({"No": 0, "Yes": 1})
    return frame


def make_pipeline(frame, features, model):
    numeric = [name for name in features if is_numeric_dtype(frame[name])]
    categorical = [name for name in features if name not in numeric]
    transformer = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline([("preprocess", transformer), ("model", model)])


def evaluate(frame, features, model):
    X = frame[features]
    y = frame["Target"]
    train_index, test_index = train_test_split(
        frame.index, test_size=0.2, stratify=y, random_state=42
    )
    fitted = make_pipeline(frame, features, model)
    fitted.fit(X.loc[train_index], y.loc[train_index])
    probabilities = fitted.predict_proba(X.loc[test_index])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y.loc[test_index], predictions).ravel()
    cv = cross_validate(
        make_pipeline(frame, features, model),
        X,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring=["roc_auc", "average_precision", "recall"],
    )
    metrics = {
        "ROC-AUC": roc_auc_score(y.loc[test_index], probabilities),
        "Average precision": average_precision_score(y.loc[test_index], probabilities),
        "Brier score (lower is better)": brier_score_loss(y.loc[test_index], probabilities),
        "Accuracy": accuracy_score(y.loc[test_index], predictions),
        "Balanced accuracy": balanced_accuracy_score(y.loc[test_index], predictions),
        "Precision": precision_score(y.loc[test_index], predictions, zero_division=0),
        "Sensitivity": recall_score(y.loc[test_index], predictions),
        "F1": f1_score(y.loc[test_index], predictions),
        "5-fold ROC-AUC mean": cv["test_roc_auc"].mean(),
        "5-fold ROC-AUC SD": cv["test_roc_auc"].std(),
        "5-fold sensitivity mean": cv["test_recall"].mean(),
    }
    threshold_rows = []
    for threshold in (0.20, 0.30, 0.40, 0.50):
        threshold_predictions = (probabilities >= threshold).astype(int)
        threshold_rows.append(
            {
                "Threshold": threshold,
                "Sensitivity": recall_score(y.loc[test_index], threshold_predictions),
                "Specificity": recall_score(
                    y.loc[test_index], threshold_predictions, pos_label=0
                ),
                "Positive alerts": int(threshold_predictions.sum()),
            }
        )
    observed, predicted = calibration_curve(
        y.loc[test_index], probabilities, n_bins=5, strategy="quantile"
    )
    calibration = pd.DataFrame({"Observed risk": observed, "Predicted risk": predicted})
    return (
        fitted,
        metrics,
        {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        pd.DataFrame(threshold_rows),
        calibration,
    )


st.set_page_config(page_title="PONV Risk Predictor", page_icon="+", layout="wide")
st.title("PONV Risk Predictor")
st.caption("Research decision-support dashboard for 48-hour post-operative nausea and vomiting")
st.warning(
    "For research and audit use only. This model is not a diagnosis or a substitute for the anesthesiologist's clinical judgment."
)

uploaded = st.sidebar.file_uploader("Upload dataset (.xlsx)", type=["xlsx"])
source = uploaded if uploaded is not None else str(DEFAULT_PATH)
if not Path(source).exists() if isinstance(source, str) else False:
    st.error(f"Workbook not found at {DEFAULT_PATH}. Upload the workbook to continue.")
    st.stop()

data = load_data(source)
data = data.dropna(subset=["Target"]).copy()
use_bellville = st.sidebar.checkbox(
    "Include Bellville score in model", value=False,
    help="Use only if this score is known before the prediction time; otherwise it may leak post-operative information.",
)
features = BASE_FEATURES + (["BellvilleScore"] if use_bellville else [])
model_name = st.sidebar.radio(
    "Model",
    ["Logistic regression", "Random forest", "Decision tree", "Gradient boosting"],
)
probability_threshold = st.sidebar.slider("Probability alert threshold", 0.05, 0.95, 0.30, 0.05)
severity_threshold = st.sidebar.slider("Bellville severity alert threshold", 2, 3, 3)

models = {
    "Logistic regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "Random forest": RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, class_weight="balanced", random_state=42
    ),
    "Decision tree": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=15, class_weight="balanced", random_state=42
    ),
    "Gradient boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42
    ),
}
model_outputs = {
    name: evaluate(data, features, model)
    for name, model in models.items()
}
fitted, metrics, matrix, threshold_table, calibration = model_outputs[model_name]
comparison = pd.DataFrame(
    {
        name: {
            "Holdout ROC-AUC": output[1]["ROC-AUC"],
            "5-fold ROC-AUC mean": output[1]["5-fold ROC-AUC mean"],
            "Average precision": output[1]["Average precision"],
            "Sensitivity": output[1]["Sensitivity"],
            "Brier score": output[1]["Brier score (lower is better)"],
        }
        for name, output in model_outputs.items()
    }
).T.sort_values("Holdout ROC-AUC", ascending=False)


@st.cache_resource
def get_case_database():
    connection = sqlite3.connect("ponv_cases.db", check_same_thread=False)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS case_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            recorded_by TEXT NOT NULL,
            role TEXT NOT NULL,
            note TEXT NOT NULL,
            diet_status TEXT NOT NULL,
            diet_notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.commit()
    return connection


def save_case_note(connection, patient_id, recorded_by, role, note, diet_status, diet_notes):
    connection.execute(
        "INSERT INTO case_notes (patient_id, recorded_by, role, note, diet_status, diet_notes) VALUES (?, ?, ?, ?, ?, ?)",
        (patient_id, recorded_by, role, note, diet_status, diet_notes),
    )
    connection.commit()


def read_case_notes(connection, patient_id=""):
    query = "SELECT patient_id, recorded_by, role, note, diet_status, diet_notes, created_at FROM case_notes"
    parameters = ()
    if patient_id.strip():
        query += " WHERE patient_id = ?"
        parameters = (patient_id.strip(),)
    query += " ORDER BY created_at DESC"
    return pd.read_sql_query(query, connection, params=parameters)


def get_risk_tier(probability, threshold):
    if probability < 0.15:
        return "LOW", "#16803c"
    if probability < threshold:
        return "MODERATE", "#a66a00"
    if probability < 0.50:
        return "HIGH", "#c2410c"
    return "VERY HIGH", "#b42318"

tab_predict, tab_monitor, tab_validation, tab_data = st.tabs(
    ["Predict", "Case monitoring", "Validation", "Data quality"]
)
with tab_predict:
    st.subheader("Patient input")
    left, middle, right = st.columns(3)
    with left:
        age = st.number_input("Age", 0, 120, 45)
        bmi = st.number_input("BMI", 10.0, 80.0, 25.0, 0.1)
        gender = st.selectbox("Gender", sorted(data["Gender"].dropna().unique()))
        asa = st.selectbox("ASA score", [0, 1, 2, 3], index=2)
        surgery = st.selectbox("Surgery", sorted(data["Surgery"].dropna().unique()))
    with middle:
        anaesthesia = st.selectbox("Anaesthesia", sorted(data["Anaesthesia"].dropna().unique()))
        motion = st.selectbox("Motion sickness", ["No", "Yes"])
        previous = st.selectbox("Previous PONV", ["No", "Yes"])
        bellville = st.selectbox("Bellville score", [0, 1, 2, 3, None], format_func=lambda value: "Missing" if value is None else str(value))
    with right:
        drug_names = [
            "Glycopyrrolate", "Fentanyl", "Propofol", "NMBA", "Paracetamol",
            "Ondansetron", "LocalAnaesthetic",
        ]
        drugs = {name: st.selectbox(name, ["No", "Yes"], index=1 if name in ["Propofol", "Fentanyl"] else 0) for name in drug_names}

    row = {
        "Age": age, "Gender": gender, "BMI": bmi, "ASA": asa, "Surgery": surgery,
        "Anaesthesia": anaesthesia, "MotionSickness": motion, "PreviousPONV": previous,
        "BellvilleScore": bellville,
        **drugs,
    }
    probability = fitted.predict_proba(pd.DataFrame([row])[features])[:, 1][0]
    prediction, alert = st.columns(2)
    with prediction:
        st.metric("Predicted PONV probability", f"{probability:.1%}")
        tier, tier_color = get_risk_tier(probability, probability_threshold)
        st.markdown(
            f"<div style='background:{tier_color}; color:white; padding:0.8rem; "
            f"text-align:center; font-size:1.3rem; font-weight:700;'>{tier}</div>",
            unsafe_allow_html=True,
        )
        st.error("Probability alert: elevated risk") if probability >= probability_threshold else st.success("Probability alert: below threshold")
    with alert:
        st.metric("Bellville severity", "Missing" if bellville is None else f"{bellville}/3")
        if bellville is not None and bellville >= severity_threshold:
            st.error("Severity alert: Bellville score meets threshold")
        else:
            st.success("Severity alert: below threshold")

    if probability >= probability_threshold:
        inference_band = "elevated predicted risk"
        inference_action = "The configured probability alert is triggered."
    elif probability >= 0.20:
        inference_band = "intermediate predicted risk"
        inference_action = "The probability is below the configured alert threshold but is not negligible."
    else:
        inference_band = "lower predicted risk"
        inference_action = "The probability is below the configured alert threshold."

    observed_factors = []
    if previous == "Yes":
        observed_factors.append("previous PONV")
    if motion == "Yes":
        observed_factors.append("motion sickness")
    if anaesthesia == "General":
        observed_factors.append("general anaesthesia")
    if surgery in {"Lap Chole", "Thyroidectomy", "Mastectomy", "Lumpectomy"}:
        observed_factors.append("the selected surgery category")
    factor_text = ", ".join(observed_factors) if observed_factors else "no selected history or procedure flag"
    st.subheader("Inference for this prediction")
    st.info(
        f"The {model_name.lower()} estimates a {probability:.1%} probability of PONV within 48 hours, "
        f"placing this patient in the {inference_band}. {inference_action} "
        f"Selected model inputs associated with the estimate include {factor_text}."
    )
    if bellville is not None and bellville >= severity_threshold:
        st.warning(
            f"Bellville severity is {bellville}/3, meeting the configured severity threshold of {severity_threshold}. "
            "This severity signal should be reviewed alongside the probability output and the clinical record."
        )
    st.caption(
        "Inference describes model output, not causation or an individual treatment recommendation. "
        "Confirm inputs and apply clinical judgment before acting."
    )

with tab_monitor:
    st.subheader("Case monitoring")
    st.caption(
        "Notes are stored in ponv_cases.db. For multi-user hosted use, replace this with an encrypted, access-controlled clinical database."
    )
    case_database = get_case_database()
    with st.form("case_note_form", clear_on_submit=True):
        case_patient_id = st.text_input("Patient or case identifier")
        case_recorded_by = st.text_input("Recorded by")
        case_role = st.selectbox("Role", ["Anaesthesiologist", "Surgeon", "Nurse", "Other"])
        case_note = st.text_area("Clinical note")
        case_diet_status = st.selectbox(
            "Pre-operative diet intake",
            ["Not recorded", "Fasting confirmed", "Clear fluids allowed", "Diet taken", "Other"],
        )
        case_diet_notes = st.text_area("Diet intake details or restrictions")
        save_note = st.form_submit_button("Save case note")
    if save_note:
        if not case_patient_id.strip() or not case_recorded_by.strip() or not case_note.strip():
            st.error("Patient or case identifier, recorder, and clinical note are required.")
        else:
            save_case_note(
                case_database, case_patient_id, case_recorded_by, case_role,
                case_note, case_diet_status, case_diet_notes,
            )
            st.success("Case note saved.")
    filter_patient_id = st.text_input("Filter notes by patient or case identifier")
    st.dataframe(read_case_notes(case_database, filter_patient_id), use_container_width=True, hide_index=True)

with tab_validation:
    st.subheader(f"{model_name} validation")
    st.caption("Fixed 80/20 stratified holdout (random state 42) plus 5-fold cross-validation. Threshold metrics are shown separately because alert sensitivity depends on the selected threshold.")
    st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).style.format({"Value": "{:.3f}"}), hide_index=True)
    st.write("Held-out confusion matrix", matrix)
    st.write("Alert-threshold trade-off", threshold_table.style.format({"Threshold": "{:.2f}", "Sensitivity": "{:.3f}", "Specificity": "{:.3f}"}))
    st.write("Held-out calibration bins", calibration.style.format("{:.3f}"))
    st.info("AUC around 0.65-0.70 indicates modest discrimination. Brier score and calibration bins describe probability reliability, but neither replaces external validation or prospective clinical review.")
    st.write("All-model comparison")
    st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)

    fitted_model = fitted.named_steps["model"]
    transformed_names = fitted.named_steps["preprocess"].get_feature_names_out()
    if hasattr(fitted_model, "feature_importances_"):
        importance = pd.DataFrame(
            {"Feature": transformed_names, "Importance": fitted_model.feature_importances_}
        ).sort_values("Importance", ascending=False).head(15)
        st.write("Top model features (associations, not causal effects)")
        st.bar_chart(importance.set_index("Feature"))

with tab_data:
    st.subheader("Data quality checks")
    st.write(f"Rows: {len(data):,} | PONV positive: {int(data['Target'].sum()):,} ({data['Target'].mean():.1%})")
    st.write("ASA values observed", sorted(data["ASA"].dropna().unique().tolist()))
    st.write("Invalid or missing Bellville values treated as missing", int(data["BellvilleScore"].isna().sum()))
    st.write("Drug fields with Yes count", data[BASE_FEATURES[8:]].eq("Yes").sum().sort_values(ascending=False))
    st.warning("The drug variables are observational treatment records. The model must not be interpreted as estimating whether a drug causes or prevents PONV.")
    st.dataframe(data.isna().sum().rename("Missing values").to_frame(), use_container_width=True)
