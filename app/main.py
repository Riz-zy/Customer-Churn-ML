from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


from src.model import ChurnPredictor

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)

#Loading the model once when ther app starts
predictor = ChurnPredictor()


class CustomerInput(BaseModel):


    gender: str = Field(..., description="'Male' or 'Female'")
    SeniorCitizen: int = Field(..., description="1 if the customer is a senior citizen, 0 otherwise")
    Partner: Literal["Yes", "No"] = Field(..., description="'Yes' or 'No'")
    Dependents: Literal["Yes", "No"] = Field(..., description="'Yes' or 'No'")
    tenure: int = Field(..., description="Number of months the customer has stayed with the company")
    PhoneService: Literal["Yes", "No"] = Field(..., description="'Yes' or 'No'")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(..., description="'Yes', 'No', or 'No phone service'")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="'DSL', 'Fiber optic', or 'No'")
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(..., description="'Yes', 'No', or 'No internet service'")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(..., description="'Yes', 'No', or 'No internet service'")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., description="'Yes', 'No', or 'No internet service'")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(..., description="'Yes', 'No', or 'No internet service'")
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(..., description="'Yes', 'No', or 'No internet service'")
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(..., description="'Yes', 'No', or 'No internet service'")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="'Month-to-month', 'One year', or 'Two year'")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="'Yes' or 'No'")
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] = Field(..., description="'Electronic check', 'Mailed check', 'Bank transfer (automatic)', or 'Credit card (automatic)'")
    MonthlyCharges: float = Field(..., description="The amount charged to the customer monthly")
    TotalCharges: float = Field(..., description="The total amount charged to the customer")


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: CustomerInput):
    result = predictor.predict(customer.model_dump())
    return result