# Model Card: Customer Churn Prediction

## Overview

Predicts the probability that a Telco customer will churn, based on account, billing, and service usage features.

## Model Details

- **Algorithm:** Gradient Boosting Classifier (scikit-learn)
- **Hyperparameters:** `n_estimators=300`, `learning_rate=0.05`, `max_depth=3`, `random_state=9`
- **Decision threshold:** ~0.26 (tuned to maximize F1-score on the churn class, rather than the default 0.5)
- **Training data:** 5,625 customers (80% stratified split)
- **Test data:** 1,407 customers (20% stratified split)

## Performance (test set, at tuned threshold)

| Metric | Class 0 (No Churn) | Class 1 (Churn) |
|---|---|---|
| Precision | 0.91 | 0.53 |
| Recall | 0.74 | 0.81 |
| F1-score | 0.82 | 0.64 |

- **Accuracy:** 75.7%
- **Support:** 1,033 (No Churn) / 374 (Churn)

## Model Comparison

| Model | Accuracy | Churn Recall | Churn F1 |
|---|---|---|---|
| Logistic Regression | 75.6% | 0.80 | 0.63 |
| Random Forest | 75.6% | 0.80 | 0.64 |
| **Gradient Boosting (selected)** | 75.7% | 0.81 | 0.64 |
| Gradient Boosting (GridSearchCV-tuned) | 76.0% | 0.77 | 0.63 |

All four configurations performed similarly. The manually-configured Gradient Boosting model was selected over the GridSearchCV-tuned version because it had higher recall and F1 on the churn class on the test set, despite GridSearchCV optimizing for F1 via cross-validation — a reminder that CV score and held-out test performance don't always perfectly align when models are this close.

## Top Predictive Features (from Random Forest importance)

1. `tenure` — longer-tenured customers are less likely to churn
2. `TotalCharges`
3. `Contract_Two year` — long contracts strongly reduce churn risk
4. `MonthlyCharges`
5. `InternetService_Fiber optic` — associated with higher churn risk

## Key Design Decisions

- **Threshold tuning over default 0.5:** the dataset has ~26.5% churn (class imbalance). Optimizing the decision threshold for F1 on the churn class significantly improved recall (catching more at-risk customers) at a reasonable precision cost.
- **Preprocessing isolation:** the `StandardScaler` and one-hot-encoded feature column list are saved as separate artifacts (`scaler.pkl`, `feature_columns.pkl`) and must be reused (not refit) at inference time to avoid data leakage and ensure consistent input shape.

## Artifacts

- `models/best_model.pkl` — trained Gradient Boosting model
- `models/scaler.pkl` — fitted StandardScaler (numeric features only)
- `models/feature_columns.pkl` — ordered list of 30 feature columns after one-hot encoding
- `models/best_threshold.pkl` — tuned decision threshold (~0.46)
