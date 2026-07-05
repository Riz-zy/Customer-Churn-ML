# Architecture

End-to-end customer churn prediction system: from raw CSV to a live REST API.

---

## System Overview

```
Raw CSV
   │
   ▼
[EDA Notebook]
   │  explore distributions, correlations, data quality issues
   ▼
[Feature Engineering Notebook]
   │  clean, encode, scale, train, tune, evaluate
   ├──────────────────────────────────────────────────────┐
   ▼                                                      ▼
models/best_model.pkl                            models/scaler.pkl
models/best_threshold.pkl                        models/feature_columns.pkl
   │
   ▼
[src/preprocessing.py]  ◄── customer JSON (API request)
   │  one-hot encode → reindex → scale
   ▼
[src/model.py]
   │  predict_proba → apply threshold → return result
   ▼
[app/main.py]  ──► {"churn_probability": 0.71, "prediction": 1}
   │
   ▼
[Docker Container on HF Spaces]
   │  port 7860, non-root user, auto-deploy on push
   ▼
Live API: https://rishiteks-customer-churn-api.hf.space
```

---

## Components

### `notebooks/01_exploratory_data_analysis.ipynb`
Loads the raw Telco Customer Churn CSV and explores it. Key findings that shaped the modeling decisions:
- 26.5% churn rate (mild class imbalance — manageable without SMOTE)
- `TotalCharges` was stored as string; rows with empty strings were dropped
- `Contract` type and `OnlineSecurity` are the strongest categorical predictors (Cramér's V)
- `tenure` is negatively correlated with churn; `MonthlyCharges` positively correlated

### `notebooks/02_feature_engineering.ipynb`
Full modeling pipeline:
1. Fix `TotalCharges` (string → float), drop `customerID`
2. Encode target: `Churn` → 0/1
3. One-hot encode all categorical features (`pd.get_dummies`) → 30 feature columns
4. Scale 4 numerical columns with `StandardScaler` (fit on training set only)
5. 80/20 stratified train/test split
6. Train and compare Logistic Regression, Random Forest, Gradient Boosting
7. GridSearchCV tuning on Gradient Boosting
8. Threshold tuning via F1-maximization on precision-recall curve
9. Save artifacts: `best_model.pkl`, `scaler.pkl`, `feature_columns.pkl`, `best_threshold.pkl`

### `src/preprocessing.py`
Applies the saved preprocessing artifacts to a single inference request. Three steps:

| Step | Operation | Why |
|---|---|---|
| `pd.get_dummies(df)` | One-hot encode categorical fields | Match training encoding |
| `df.reindex(columns=feature_columns, fill_value=0)` | Align to training column order | Prevent feature mismatch |
| `scaler.transform(df[numerical_columns])` | Scale 4 numerical fields | Use the fitted scaler — never re-fit at inference |

The scaler is loaded once at module import time (not per-request) for efficiency.

### `src/model.py` — `ChurnPredictor`
Thin wrapper around the trained model. Loads the model and threshold at startup, exposes a single `predict(customer_dict)` method that returns:
```json
{
  "churn_probability": 0.712,
  "prediction": 1
}
```
The binary `prediction` uses the tuned threshold (~0.263), not the default 0.5. This threshold was chosen to maximize F1 on the test set, trading some precision for higher churn recall — appropriate for a retention use case where missing a churner costs more than a false alarm.

### `app/main.py`
FastAPI application with two endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns `{"status": "ok"}` |
| `POST` | `/predict` | Takes 19 customer fields, returns churn probability and binary prediction |
| `GET` | `/docs` | Auto-generated Swagger UI (FastAPI built-in) |

The `ChurnPredictor` instance is created once at app startup — model loading happens once, not per request.

---

## Saved Artifacts (`models/`)

| File | Description | Created by |
|---|---|---|
| `best_model.pkl` | `GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=9)` | notebook 02 |
| `scaler.pkl` | `StandardScaler` fit on `[SeniorCitizen, tenure, MonthlyCharges, TotalCharges]` | notebook 02 |
| `feature_columns.pkl` | Ordered list of 30 column names post one-hot encoding | notebook 02 |
| `best_threshold.pkl` | ~0.263 — tuned decision threshold | notebook 02 |
| `model_card.md` | Full model documentation and performance metrics | notebook 02 |

These files are excluded from the GitHub repository (`.gitignore`) but committed directly to the Hugging Face Space repository.

---

## Model Performance

| Model | Accuracy | Churn Recall | Churn F1 |
|---|---|---|---|
| Logistic Regression | 75.6% | 0.80 | 0.63 |
| Random Forest | 75.6% | 0.80 | 0.64 |
| Gradient Boosting (manual) | 75.7% | 0.81 | 0.64 |
| Gradient Boosting (GridSearchCV) | 76.0% | 0.77 | 0.63 |

Final model: **Gradient Boosting with manual params** — better recall and F1 than the GridSearchCV-tuned version despite slightly lower accuracy.

Top predictors (Random Forest feature importance): `tenure`, `TotalCharges`, `Contract_Two year`, `MonthlyCharges`, `InternetService_Fiber optic`

---

## Deployment Architecture

```
GitHub repo (source)              HF Space repo (deployment)
github.com/Riz-zy/                huggingface.co/spaces/
Customer-Churn-ML                 rishiteks/customer-churn-api
│                                 │
├── app/                          ├── app/
├── src/                          ├── src/
├── notebooks/                    ├── models/  ← .pkl files committed here
├── data/                         ├── Dockerfile
└── .gitignore                    └── requirements.txt
    (excludes models/*.pkl)           (7 packages only)
```

The two-repo strategy keeps binary model artifacts out of the source repository while making them available in the deployment environment. The HF Space auto-rebuilds and redeploys on every `git push`.

### Dockerfile

```dockerfile
FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY --chown=user . /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Key constraints:
- **Port 7860**: required by HF Spaces
- **Non-root user (uid 1000)**: enforced by HF Spaces for security
- **`WORKDIR /app`**: ensures relative paths like `models/best_model.pkl` resolve correctly inside the container

---

## Key Design Decisions

**Why save preprocessing artifacts instead of re-fitting at inference?**
Re-fitting the scaler on inference data would cause data leakage and produce different scaling than training. Saving the fitted `StandardScaler` and loading it at inference guarantees the exact same transformation seen during training.

**Why tune the decision threshold?**
The default threshold of 0.5 maximizes accuracy, not business value. For churn prediction, missing a churner (false negative) is more costly than a false alarm (false positive). Tuning to ~0.263 via F1-maximization on the precision-recall curve raised churn recall from ~0.60 to 0.81.

**Why Gradient Boosting over Logistic Regression?**
All models performed similarly (within 1% accuracy), but Gradient Boosting edged out the others on churn recall and F1 — the metrics that matter most for this use case.

**Why FastAPI?**
Auto-generated `/docs` UI, built-in Pydantic validation, and async support make it the standard choice for Python ML APIs.

**Why Hugging Face Spaces?**
Zero-cost Docker hosting with automatic CI/CD on every git push — ideal for a portfolio project that needs a live, public URL.
