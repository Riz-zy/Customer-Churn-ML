sample_1 = {"gender": "Male", 
            "SeniorCitizen": 0, 
            "Partner": "Yes", 
            "Dependents": "No", 
            "tenure": 12, 
            "PhoneService": "No", 
            "MultipleLines": "No", 
            "InternetService": "DSL", 
            "OnlineSecurity": "No", 
            "OnlineBackup": "No", 
            "DeviceProtection": "No", 
            "TechSupport": "No", 
            "StreamingTV": "No", 
            "StreamingMovies": "No", 
            "Contract": "Month-to-month", 
            "PaperlessBilling": "Yes", 
            "PaymentMethod": "Electronic check", 
            "MonthlyCharges": 70.5, 
            "TotalCharges": 846}


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


from src.preprocessing import processing_input_data

#result = processing_input_data(sample_1)

#print(result)
#print(result.shape)
#print(result.columns.tolist())

from src.model import ChurnPredictor

predictor = ChurnPredictor()

result = predictor.predict(sample_1)

print(result)

#Basic validation of the output
assert isinstance(result, dict)
assert "churn_probability" in result
assert "prediction" in result

assert isinstance(result["churn_probability"], float)
assert 0.0 <= result["churn_probability"] <= 1.0

assert result["prediction"] in (0, 1)

print("\n End-to-end prediction pipeline is working!")
