# Customer Churn ML Model and API

An end-to-end machine learning project that predicts customer churn — from raw data to a live REST API.

**Live API:** https://rishiteks-customer-churn-api.hf.space  
**GitHub:** https://github.com/Riz-zy/Customer-Churn-ML

---

## Project Overview

**Goal:** Identify customers at risk of leaving so a business can take retention action before they go.

**Stack:**
- Data & Exploration: Pandas, Jupyter
- ML Models: scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
- API: FastAPI + Uvicorn
- Deployment: Hugging Face Spaces (Docker)

---

## Dataset

Source: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle, blastchar)
- 7,043 customers, 21 features
- Target: binary — did the customer churn? (yes/no)
- Churn rate: 26.5%

---

## Project Structure

```
Customer Churn ML Model and API/
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_deployment_guide.ipynb
│
├── src/
│   ├── preprocessing.py       # Applies saved transformers to a single input
│   └── model.py               # ChurnPredictor class
│
├── app/
│   └── main.py                # FastAPI application
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── best_threshold.pkl
│   └── model_card.md
│
├── data/
│   └── raw/                   # Original CSV — never modified
│
├── tests/
├── ARCHITECTURE.md
├── requirements.txt
└── README.md
```

---

## Quick Start

**Requirements:** Python 3.11+, Git

```bash
# 1. Clone the repo
git clone https://github.com/Riz-zy/Customer-Churn-ML
cd "Customer-Churn-ML"

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate         # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Go to kaggle.com, search "Telco Customer Churn" by blastchar
# Place WA_Fn-UseC_-Telco-Customer-Churn.csv in data/raw/
```

---

## Workflow

### Phase 1: Exploratory Data Analysis ✅

```bash
jupyter notebook
# Open notebooks/01_exploratory_data_analysis.ipynb
```

Key findings:
- `TotalCharges` was stored as a string — converted to float, rows with empty strings dropped
- `Contract` type and `OnlineSecurity` are the strongest predictors of churn
- Longer-tenured customers churn less; higher monthly charges correlate with more churn

---

### Phase 2: Feature Engineering & Modeling ✅

```bash
# Open notebooks/02_feature_engineering.ipynb
```

Steps completed:
- Fixed `TotalCharges`, dropped `customerID`, encoded `Churn` as 0/1
- One-hot encoded all categorical features (21 → 30 columns)
- Scaled 4 numerical columns with `StandardScaler` (fit on training set only)
- 80/20 stratified split — 5,625 train / 1,407 test
- Trained Logistic Regression, Random Forest, and Gradient Boosting
- Tuned Gradient Boosting with GridSearchCV
- Tuned decision threshold via F1-maximization on the precision-recall curve

**Model comparison:**

| Model | Accuracy | Churn Recall | Churn F1 |
|---|---|---|---|
| Logistic Regression | 75.6% | 0.80 | 0.63 |
| Random Forest | 75.6% | 0.80 | 0.64 |
| Gradient Boosting (manual params) | 75.7% | 0.81 | 0.64 |
| Gradient Boosting (GridSearchCV) | 76.0% | 0.77 | 0.63 |

**Final model:** `GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3)`  
Chosen for its higher churn recall and F1 — the GridSearchCV version had marginally better accuracy but worse recall, which matters more for a retention use case.

**Saved artifacts (`models/`):**
- `best_model.pkl` — trained Gradient Boosting model
- `scaler.pkl` — fitted StandardScaler
- `feature_columns.pkl` — ordered list of 30 columns post one-hot encoding
- `best_threshold.pkl` — tuned decision threshold (~0.263)

**Top predictors** (Random Forest feature importance): `tenure`, `TotalCharges`, `Contract_Two year`, `MonthlyCharges`, `InternetService_Fiber optic`

---

### Phase 3: API Development ✅

```bash
# Start the server
uvicorn app.main:app --reload

# Docs at http://localhost:8000/docs
```

Endpoints:
- `GET /health` — returns `{"status": "ok"}`
- `POST /predict` — takes 19 customer fields, returns churn probability and prediction

Example request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 140.70
  }'
```

---

### Phase 4: Deployment ✅

Live at **https://rishiteks-customer-churn-api.hf.space**

- Docker-based Hugging Face Space — auto-redeploys on every `git push`
- Slim production image: 7 runtime packages, `python:3.11-slim` base
- Model binaries committed directly to the HF Space repo (separate from the GitHub source repo)

Endpoints (live):
- `GET /health`
- `POST /predict`
- `GET /docs` — interactive Swagger UI

---

### Phase 5: Documentation ✅

- [ARCHITECTURE.md](ARCHITECTURE.md) — data flow, component breakdown, design decisions
- [notebooks/03_deployment_guide.ipynb](notebooks/03_deployment_guide.ipynb) — step-by-step deployment guide
- [models/model_card.md](models/model_card.md) — model performance, comparison, and rationale

---

## Design Decisions

**Why tune the decision threshold?**  
The default threshold of 0.5 maximizes accuracy. For churn, missing an at-risk customer (false negative) costs more than a false alarm. Tuning to ~0.263 raised churn recall from ~0.60 to 0.81.

**Why save preprocessing artifacts instead of re-fitting at inference?**  
Re-fitting the scaler on new data would produce different scaling than training — the saved `StandardScaler` guarantees the exact same transformation every time.

**Why Gradient Boosting over Logistic Regression?**  
All models performed within 1% of each other, but Gradient Boosting edged out on the metrics that matter (recall and F1). Threshold tuning had more impact on results than model choice.

**Why FastAPI?**  
Auto-generated `/docs` UI, Pydantic input validation, and async support out of the box.

**Why Hugging Face Spaces?**  
Free Docker hosting with automatic CI/CD — no credit card, no server management.

---

## Contact

Rishikesh Tekavade  
- LinkedIn: https://www.linkedin.com/in/rishikesh-tekavade-b865ab1a4/
- GitHub: https://github.com/Riz-zy