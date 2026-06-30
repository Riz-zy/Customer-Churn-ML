# Preprocessing module for data cleaning and transformation

import joblib
import pandas as pd

#Load saved artifacts
scaler = joblib.load('models/scaler.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')


#Numerical columns used for scaling
numerical_columns = [
'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'
]

def processing_input_data(cutomer):
    """
    Preprocess a single customer input data for prediction.

    Parameters:
    cutomer (dict): A dictionary containing customer data.

    Returns:
    pandas.DataFrame: A DataFrame containing the preprocessed customer data ready for prediction.
    """

    #Convert dictionary to DataFrame
    df = pd.DataFrame([cutomer])

    #One-hot encode categorical variables
    df = pd.get_dummies(df)

    #Ensure all expected feature columns are present
    df = df.reindex(columns=feature_columns, fill_value=0)

    #Scale numerical features
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    return df