# Medical Insurance Fraud Detection — Easy/Medium Version 2

A simplified fraud detection project, built to stay consistent with a
description that says "applied machine learning techniques to identify
anomalous patterns" — unlike a pure-statistics version, the anomaly
detector here is genuine unsupervised ML.

## What this version uses

| Task | Technique | Why this one |
|---|---|---|
| Classification | Logistic Regression | Fully interpretable — the whole model is 8 learned weights, easy to explain completely |
| Anomaly detection | Isolation Forest (unsupervised ML) | Genuinely machine learning, not a manual rule — but still just one extra `.fit()` call to explain |
| Features | 8 simple ones | Same minimal set as the first easy version |
| Train/test split | Plain random split | A disclosed simplification — see below |

## How this compares to the other two versions of this project

| | Easy v1 (z-score) | **Easy v2 (this one)** | Advanced |
|---|---|---|---|
| Classifier | Logistic Regression | Logistic Regression | Random Forest |
| Anomaly detection | Statistics rule (z-score) | **Isolation Forest (ML)** | Isolation Forest (ML) |
| Split | Random | Random | Provider-grouped |
| Features | 8 | 8 | 20+ |
| ROC-AUC | 0.64 | 0.64 | 0.94 |

This version sits between the other two: the classifier stays as simple as
v1, but the anomaly detector is upgraded to real ML to match descriptions
that specifically claim machine learning was used for anomaly detection.

## Project structure

```
easy_v2/
├── app.py                      # Streamlit dashboard (run this)
├── requirements.txt
├── src/
│   └── train_model.py
├── model/
│   ├── fraud_classifier.joblib     # Logistic Regression
│   ├── anomaly_detector.joblib     # Isolation Forest
│   ├── scaler.joblib
│   ├── metrics.json
│   └── coefficients.json
└── data/
    └── stream_sample.csv
```

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Known simplifications (disclose these, don't hide them)

- **Random split, not grouped by provider** — the reported 0.64 ROC-AUC is
  likely a little optimistic since claims from the same provider can land
  on both sides of the split.
- **Weak label** — "fraud" is inherited from the provider's label, not
  proven per individual claim.
- **8 features only** — kept minimal on purpose for full explainability.

## Suggested resume line for this version

> Medical Insurance Fraud Detection | Python, pandas, scikit-learn, Streamlit
> – Designed and developed a machine-learning fraud detection system to identify suspicious billing patterns across 558K+ Medicare insurance claims.
> – Applied supervised classification (Logistic Regression) and unsupervised anomaly detection (Isolation Forest) to claims and provider-behavior data, achieving 0.64 ROC-AUC on a held-out test set.
> – Built an interactive real-time dashboard (Streamlit) for live claim scoring and model explainability.
