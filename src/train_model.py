"""
Medical Insurance Fraud Detection — Easy/Medium Version 2
============================================================
Same simplicity goals as the first easy version, with one important fix:
anomaly detection is now genuine unsupervised MACHINE LEARNING (Isolation
Forest), not a plain statistics rule. This makes the project consistent
with descriptions that say "applied machine learning techniques to
identify anomalous patterns."

Still kept easy to explain:
  - Only 8 simple features (same as v1)
  - Classification model is Logistic Regression (fully interpretable)
  - Anomaly model is Isolation Forest, but explained in one paragraph
    (see the accompanying learning guide)
  - A plain random train/test split (same known simplification as v1,
    clearly disclosed)

Run:
    python3 src/train_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)
RAW_DATA_PATH = Path("/home/claude/data/merged.csv")

FEATURES = [
    "ClaimAmount", "DeductibleAmt", "ClaimDurationDays", "PatientAge",
    "ChronicConditionCount", "NumDiagnosisCodes", "IsInpatient", "ProviderAvgClaimAmt",
]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["ClaimStartDt"] = pd.to_datetime(df["ClaimStartDt"], errors="coerce")
    df["ClaimEndDt"] = pd.to_datetime(df["ClaimEndDt"], errors="coerce")
    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")

    df["ClaimAmount"] = df["InscClaimAmtReimbursed"].fillna(0)
    df["DeductibleAmt"] = df["DeductibleAmtPaid"].fillna(0)
    df["ClaimDurationDays"] = (df["ClaimEndDt"] - df["ClaimStartDt"]).dt.days.clip(lower=0).fillna(0)
    df["PatientAge"] = ((df["ClaimStartDt"] - df["DOB"]).dt.days / 365.25).clip(0, 110)
    df["PatientAge"] = df["PatientAge"].fillna(df["PatientAge"].median())

    chronic_cols = [c for c in df.columns if c.startswith("ChronicCond_")]
    for c in chronic_cols:
        df[c] = (df[c] == 1).astype(int)
    df["ChronicConditionCount"] = df[chronic_cols].sum(axis=1)

    diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11) if f"ClmDiagnosisCode_{i}" in df.columns]
    df["NumDiagnosisCodes"] = df[diag_cols].notna().sum(axis=1)

    df["IsInpatient"] = df["is_Inpatient"].fillna(0).astype(int)

    provider_avg = df.groupby("Provider")["ClaimAmount"].transform("mean")
    df["ProviderAvgClaimAmt"] = provider_avg

    df["label"] = (df["PotentialFraud"].astype(str).str.lower() == "yes").astype(int)

    keep = FEATURES + ["label", "ClaimID", "Provider"]
    return df[keep]


def main():
    print("Loading raw claims data...")
    raw = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    feats = build_features(raw)
    print(f"Feature table shape: {feats.shape}")

    train_df, test_df = train_test_split(
        feats, test_size=0.25, random_state=42, stratify=feats["label"]
    )
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURES])
    X_test = scaler.transform(test_df[FEATURES])
    y_train, y_test = train_df["label"].values, test_df["label"].values

    print("Training Logistic Regression (classification)...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    print("Training Isolation Forest (unsupervised anomaly detection)...")
    anomaly = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
    anomaly.fit(X_train)   # never sees y_train / the fraud label

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metrics = {
        "roc_auc": float(auc), "average_precision": float(ap),
        "confusion_matrix": cm,
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "roc_curve": {"fpr": fpr[::30].tolist(), "tpr": tpr[::30].tolist()},
    }
    print(f"ROC-AUC: {auc:.4f} | Average Precision: {ap:.4f}")
    print("Confusion matrix:", cm)

    coefs = sorted(zip(FEATURES, clf.coef_[0].tolist()), key=lambda x: -abs(x[1]))

    joblib.dump(clf, MODEL_DIR / "fraud_classifier.joblib")
    joblib.dump(anomaly, MODEL_DIR / "anomaly_detector.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(MODEL_DIR / "coefficients.json", "w") as f:
        json.dump(coefs, f, indent=2)

    test_df = test_df.copy()
    test_df["fraud_probability"] = y_prob
    # IsolationForest.score_samples: LOWER = more anomalous, by convention.
    # Flip sign so higher = more anomalous (more intuitive for the dashboard).
    test_df["anomaly_score"] = -anomaly.score_samples(X_test)
    sample = test_df.sample(n=min(3000, len(test_df)), random_state=7)
    sample.to_csv(DATA_DIR / "stream_sample.csv", index=False)

    print("Done. Artifacts saved to model/ and data/stream_sample.csv")


if __name__ == "__main__":
    main()
