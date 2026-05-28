# Customer Churn ML Model and API
An end-to-end machine learning project that predicts customer churn. Build an ML pipeline, train classification models, and deploy a REST API to production. This project showcases ML fundamentals, production-ready code, and cloud deployment—all highly valued on resumes.

## Project Overview

Business Goal: Identify customers at risk of leaving so you can take retention action.

Technical Stack:
- Data & Exploration: Pandas, Jupyter
- ML Models: scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
- API: FastAPI + Uvicorn
- Deployment: Hugging Face Spaces
- Version Control: Git

## Dataset

Source: Telco Customer Churn (Kaggle / UCI ML Repository)
- Target: Binary classification (churn: yes/no)
- Features: Customer demographics, account info, services
- Size: ~7,000 customers, ~20 features
- Class Balance: ~27% churn rate

## Project Structure

```text
data/
├── raw/
│   └── # Original customer data (never modify)
│
├── processed/
│   └── # Cleaned data ready for modeling
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   │   └── # Load, visualize, understand data
│   ├── 02_feature_engineering.ipynb
│   │   └── # Preprocessing, model training
│   └── 03_deployment_guide.ipynb
│       └── # Hugging Face Spaces setup
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   │   └── # Reusable preprocessing functions
│   ├── model.py
│   │   └── # ChurnPredictor class
│   └── utils.py
│       └── # Helper functions
│
├── app/
│   └── main.py
│       └── # FastAPI application
│
├── models/
│   ├── best_model.pkl
│   │   └── # Trained model
│   ├── scaler.pkl
│   │   └── # Fitted StandardScaler
│   ├── encoder.pkl
│   │   └── # Fitted categorical encoder
│   └── model_card.md
│       └── # Model performance metrics
│
├── tests/
│   └── __init__.py
│
├── requirements.txt
│   └── # Python dependencies
│
├── README.md
│   └── # This file
│
└── ARCHITECTURE.md
    └── # Data flow & design decisions
```
    
## Quick Start

Prerequisites

- Python 3.8+
- Git
- pip (Python package manager)

Local Setup

1. Clone the repository:
   git clone <your-repo-url>
   cd "Customer Churn ML Model and API"

2. Create a virtual environment:
   python -m venv venv

3. Activate the virtual environment:
   Windows (PowerShell): .\venv\Scripts\Activate.ps1
   Windows (Command Prompt): venv\Scripts\activate.bat
   Mac/Linux: source venv/bin/activate

4. Install dependencies:
   pip install -r requirements.txt

5. Download the dataset:
   - Download from Kaggle Telco Customer Churn
   - Place WA_Fn-UseC_-Telco-Customer-Churn.csv in data/raw/

## Workflow

Phase 1: Exploratory Data Analysis

jupyter notebook
# Open notebooks/01_exploratory_data_analysis.ipynb

- Load and inspect raw data
- Visualize feature distributions
- Identify missing values and outliers
- Analyze churn patterns and correlations

Phase 2: Feature Engineering & Model Training

jupyter notebook
# Open notebooks/02_feature_engineering.ipynb

- Handle missing values (imputation or removal)
- Encode categorical variables (one-hot, label encoding)
- Scale numerical features (StandardScaler)
- Train & compare models: Logistic Regression, Random Forest, XGBoost
- Perform hyperparameter tuning (GridSearchCV)
- Save best model and preprocessing artifacts

Expected Performance:
- Baseline accuracy: >70%
- Best model: Random Forest or Gradient Boosting
- Metrics to track: Accuracy, Precision, Recall, F1-score, ROC-AUC

Phase 3: API Development & Testing

# Start the API server
uvicorn app.main:app --reload

# Visit http://localhost:8000/docs for interactive API documentation
# Test with sample requests

API Endpoints:
- GET /health — Health check (returns 200 if running)
- POST /predict — Predict churn for a customer
  Input: Customer features (age, tenure, monthly_charges, etc.)
  Output: JSON with churn probability and binary prediction

Phase 4: Deployment to Hugging Face Spaces

# Follow notebooks/03_deployment_guide.ipynb for step-by-step instructions
# Push to Hugging Face Spaces repository
# Auto-deploy on every git push

## Testing

Local API Testing

# Ensure virtual environment is activated
python -c "from app.main import app; print('API imported successfully')"

# Or use curl to test the endpoint:
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "tenure": 24, "monthly_charges": 65.5}'

End-to-End Verification

1. Run notebooks end-to-end (no errors)
2. Train model and save artifacts
3. Start API server
4. Test /predict endpoint with sample data
5. Verify output matches notebook predictions
6. Deploy to Hugging Face Spaces
7. Test deployed endpoint

## Model Performance

[Add metrics after training]

- Best Model: [XGBoost / Random Forest / Logistic Regression]
- Accuracy: [X%]
- Precision: [X%]
- Recall: [X%]
- F1-Score: [X%]
- ROC-AUC: [X%]

## Key Decisions & Rationale

Why Random Forest/XGBoost over Logistic Regression?

- Better handles non-linear relationships and feature interactions
- Higher accuracy on typical customer churn datasets
- Scikit-learn models are beginner-friendly and production-ready

Why FastAPI?

- Industry-standard, fast performance
- Auto-generated API documentation at /docs
- Built-in data validation with Pydantic

Why Hugging Face Spaces?

- Free deployment (no credit card needed)
- Automatic Docker containerization
- Built-in CI/CD (auto-deploys on git push)
- Perfect for portfolio projects

Why separate preprocessing from API?

- Prevents data leakage (never fit transformers on test data)
- Ensures consistency: same preprocessing in notebooks and API
- Makes model production-ready

## Development Workflow

# Create a feature branch
git checkout -b feature/add-xgboost

# Make changes, commit frequently
git add src/model.py
git commit -m "feat: add XGBoost model training"

# Push to GitHub
git push origin feature/add-xgboost

# Create a pull request, get reviewed, then merge

## Further Enhancements (Optional)

- Class Imbalance Handling: SMOTE or class_weight parameter if F1-score < 0.6
- Advanced Features: Polynomial features, interaction terms, domain-specific engineering
- Model Explainability: SHAP values, feature importance plots
- API Enhancements: /feature_importance, /model_info endpoints
- Monitoring: Log predictions, track API performance over time

## Documentation Files

- README.md (this file) — Overview and quick start
- ARCHITECTURE.md — Data flow, design decisions, preprocessing pipeline
- notebooks/03_deployment_guide.ipynb — Step-by-step Hugging Face Spaces deployment
- models/model_card.md — Detailed model performance and metadata

## Contributing

Contributions welcome! Please:
1. Create a feature branch
2. Commit with clear messages
3. Push and create a pull request


## Contact

Rishikesh Tekavade
- LinkedIn: https://www.linkedin.com/in/rishikesh-tekavade-b865ab1a4/
- GitHub: https://github.com/Riz-zy
