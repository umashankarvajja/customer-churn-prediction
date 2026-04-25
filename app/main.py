from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict if a telecom customer will churn based on their profile.",
    version="1.0.0"
)

# Load trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract: int          # 0=Month-to-month, 1=One year, 2=Two year
    PaymentMethod: int     # 0=Electronic, 1=Mailed, 2=Bank transfer, 3=Credit card
    InternetService: int   # 0=DSL, 1=Fiber optic, 2=No
    OnlineSecurity: int    # 0=No, 1=Yes, 2=No internet
    TechSupport: int       # 0=No, 1=Yes, 2=No internet
    PaperlessBilling: int  # 0=No, 1=Yes
    SeniorCitizen: int     # 0=No, 1=Yes

class PredictionResult(BaseModel):
    churn_prediction: str
    churn_probability: float
    risk_level: str

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running!",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResult)
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run src/train.py first."
        )
    
    features = pd.DataFrame([customer.dict()])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    if probability >= 0.7:
        risk = "High Risk"
    elif probability >= 0.4:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"
    
    return PredictionResult(
        churn_prediction="Will Churn" if prediction == 1 else "Will Stay",
        churn_probability=round(float(probability), 4),
        risk_level=risk
    )

@app.get("/example")
def example_input():
    return {
        "example_customer": {
            "tenure": 12,
            "MonthlyCharges": 65.5,
            "TotalCharges": 786.0,
            "Contract": 0,
            "PaymentMethod": 0,
            "InternetService": 1,
            "OnlineSecurity": 0,
            "TechSupport": 0,
            "PaperlessBilling": 1,
            "SeniorCitizen": 0
        },
        "usage": "POST this to /predict to get a churn prediction"
    }
