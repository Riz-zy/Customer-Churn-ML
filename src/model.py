import joblib

from src.preprocessing import processing_input_data

class ChurnPredictor:
    def __init__(
            self,
            model_path= 'models/best_model.pkl',
            threshold_path= 'models/best_threshold.pkl'
    ):
        
    #Load the trained model and threshold

        self.model = joblib.load(model_path)
        self.threshold = joblib.load(threshold_path)

    def predict(self, customer):
        #Predicts churn for a single customer
        
        #Preprocessign input
        processed_data = processing_input_data(customer)

        #Probability of class 1
        probability = float(self.model.predict_proba(processed_data)[0][1])


        #Applly saved threshold
        prediction = int(probability >= self.threshold)

        return{
            "churn_probability": probability,
            "prediction": prediction
        }
