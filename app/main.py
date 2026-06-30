from fastapi import FastAPI
from pydantic import BaseModel

from src.model import ChurnPredictor

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)

#Loading the model once when ther app starts
predictor = ChurnPredictor()

class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: CustomerInput):
    result = predictor.predict(customer.dict())
    return result