# Medical Insurance Fraud Detection


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

## Known simplifications 

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
