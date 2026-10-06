"""
PONV Risk Predictor - Training Pipeline
Trains multiple ML models and selects best performer.
"""

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def load_data(path):
    """Load dataset from Excel."""
    try:
        df = pd.read_excel(path, sheet_name="Data_Table 1")
    except ValueError:
        df = pd.read_excel(path)
    return df


def find_col(df, candidates):
    """Find column by flexible name matching."""
    normalized = {
        str(column).strip().lower().replace("_", "").replace(" ", ""): column
        for column in df.columns
    }
    for c in candidates:
        if c in df.columns:
            return c
        key = str(c).strip().lower().replace("_", "").replace(" ", "")
        if key in normalized:
            return normalized[key]
    return None


def map_asa_values(series):
    """Map ASA text to numeric 0-3."""
    mapping = {
        "Minimal": 0, "minimal": 0,
        "Mild": 1, "mild": 1,
        "Moderate": 2, "moderate": 2,
        "Severe": 3, "severe": 3,
    }
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(-1).astype(int)
    else:
        return series.map(lambda v: mapping.get(str(v).strip(), -1)).astype(int)


def build_pipeline(categorical_cols, numeric_cols, model_type="logistic"):
    """Build preprocessing + model pipeline."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols),
    ])

    if model_type == "logistic":
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=-1)
    elif model_type == "dt":
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=42, max_depth=4, min_samples_leaf=20)
    elif model_type == "gb":
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42, max_depth=2)
    elif model_type == "xgb" and HAS_XGBOOST:
        clf = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="logloss", scale_pos_weight=4)
    else:
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe


def main(data_path="data/Data_1750.xlsx", model_type="auto"):
    """Train models and save best performer."""
    
    p = Path(data_path)
    if not p.exists():
        print(f"Data file not found: {data_path}")
        return

    df = load_data(p)
    
    target_col = find_col(
        df, ["PONV_48h", "PONV48h", "Post Operative Nausea Vomitting_48h"]
    )
    if target_col is None:
        raise ValueError("Expected a 48-hour PONV target column in dataset")

    # Flexible column detection
    col_age = find_col(df, ["age", "Age"])
    col_bmi = find_col(df, ["BMI", "bmi"])
    col_bell = find_col(df, ["bellville_score", "Bellville", "bellville", "BellvilleScore", "Belleville scoring", "Bellville score"])
    col_surg = find_col(df, ["surgery_type", "surgery", "SurgeryType"])
    col_anes = find_col(df, ["anaesthesia_type", "anaesthesia", "anaesthesia_administered"])
    col_motion = find_col(df, ["motion_sickness", "MotionSickness", "motionSickness"])
    col_prior_ponv = find_col(df, ["prior_ponv", "priorPONV", "prior_ponv_history", "PreviousPONV"])
    col_prior_surg = find_col(df, ["history_post_op_surgery", "prior_surgery", "previous_surgery"])
    col_asa = find_col(df, ["ASA", "asa", "ASA_score", "asa_score"])
    col_gender = find_col(df, ["Gender", "gender", "Sex"])

    drug_cols = []
    for d in ["glycopyrrolate", "fentanyl", "propofol", "NMBA", "paracetamol", "ondansetron", "local_anaesthetic"]:
        found = find_col(df, [d, d.lower(), d.upper()])
        if found:
            drug_cols.append(found)

    # Bellville score is excluded: it may be recorded post-operatively and would leak the outcome.
    col_bell = None
    numeric_cols = [c for c in [col_age, col_bmi] if c]
    categorical_cols = [c for c in [col_surg, col_anes, col_asa, col_gender] if c]
    bool_cols = [c for c in [col_motion, col_prior_ponv, col_prior_surg] if c]

    features = numeric_cols + categorical_cols + drug_cols + bool_cols

    if len(features) == 0:
        raise ValueError("No feature columns found in dataset.")

    X = df[features].copy()
    y = df[target_col].replace({"No": 0, "Yes": 1}).astype(int)

    # Map ASA textual to numeric
    if col_asa and col_asa in X.columns:
        X[col_asa] = map_asa_values(X[col_asa])

    if col_bell and col_bell in X.columns:
        X[col_bell] = pd.to_numeric(X[col_bell], errors="coerce")
        X.loc[~X[col_bell].between(0, 3), col_bell] = np.nan

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train multiple models
    model_types = ["logistic", "rf", "dt", "gb"]
    if HAS_XGBOOST:
        model_types.append("xgb")

    results = {}
    best_auc = -1
    best_model_type = "logistic"
    best_model = None
    best_prob = None

    for mt in model_types:
        print(f"\n--- Training {mt.upper()} ---")
        try:
            base_pipe = build_pipeline(categorical_cols, numeric_cols, model_type=mt)
            calib = CalibratedClassifierCV(base_pipe, cv=5, method="sigmoid")
            calib.fit(X_train, y_train)

            prob = calib.predict_proba(X_test)[:, 1]
            holdout_auc = roc_auc_score(y_test, prob)
            cv_auc = cross_val_score(
                build_pipeline(categorical_cols, numeric_cols, model_type=mt),
                X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc",
            ).mean()
            # Youden threshold from training data, applied to the untouched test set
            fpr_tr, tpr_tr, th_tr = roc_curve(y_train, calib.predict_proba(X_train)[:, 1])
            thr = float(th_tr[np.argmax(tpr_tr - fpr_tr)])
            pred = (prob >= thr).astype(int)
            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)

            results[mt] = {
                "auc": float(holdout_auc),
                "cv_auc": float(cv_auc),
                "threshold": thr,
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
            }
            print(f"  CV AUC: {cv_auc:.3f}, Holdout AUC: {holdout_auc:.3f}, thr={thr:.3f}, Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}")
            auc = cv_auc

            if auc > best_auc:
                best_auc = auc
                best_model_type = mt
                best_model = calib
                best_prob = prob

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\n✓ Best model: {best_model_type.upper()} (5-fold CV AUC={best_auc:.3f})")
    if best_model is None or best_prob is None:
        raise RuntimeError("No model could be trained. Check the feature and target columns.")

    # Compute per-ASA thresholds
    thresholds_by_asa = {}
    if col_asa and col_asa in X_test.columns:
        for asa_val in sorted(X_test[col_asa].dropna().unique()):
            mask = X_test[col_asa] == asa_val
            if mask.sum() < 20:
                continue
            y_sub = y_test[mask]
            prob_sub = best_prob[mask.values]
            fpr, tpr, th = roc_curve(y_sub, prob_sub)
            j = tpr - fpr
            opt_idx = np.argmax(j)
            thresholds_by_asa[int(asa_val)] = float(th[opt_idx])

    # Save artifacts
    meta = {
        "best_model_type": best_model_type,
        "model_results": results,
        "surgery_types": list(df[col_surg].dropna().unique()) if col_surg else [],
        "anaesthesia_types": list(df[col_anes].dropna().unique()) if col_anes else [],
        "features": features,
        "thresholds_by_asa": thresholds_by_asa,
        "disclaimer": (
            "This PONV predictor is a decision-support prototype. It is NOT a validated clinical tool. "
            "Do not use as sole basis for clinical decisions. Validate externally before deployment."
        ),
    }

    # Create models directory if not exists
    Path("models").mkdir(exist_ok=True)
    
    joblib.dump(best_model, "models/ponv_model.pkl")
    joblib.dump(meta, "models/ponv_meta.pkl")
    joblib.dump(best_model, "ponv_model.pkl")
    joblib.dump(meta, "ponv_meta.pkl")
    print("\n✓ Saved ponv_model.pkl, ponv_meta.pkl and copies in models/")
    print(f"\nModel comparison:\n{pd.DataFrame(results).T.to_string()}")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/Data_1750.xlsx"
    main(data_path)
