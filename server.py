from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load all components from the pickle file
model_bundle = joblib.load("fraud_detection_pipeline.pkl")
model = model_bundle["model"]
columns = model_bundle["columns"]   # Fix: define this at the top
# scaler and threshold can also be extracted if needed
# scaler = model_bundle["scaler"]
# threshold = model_bundle["threshold"]

app = FastAPI(title="Fraud Detection API")

# Input schema
class FraudInput(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict")
def predict(data: FraudInput):
    input_dict = data.dict()

    # Use correct column order from training
    input_array = np.array([[input_dict[col] for col in columns]])

    # Predict
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0].tolist()

    return {
        "prediction": int(prediction),
        "probability": {
            "non_fraud": probability[0],
            "fraud": probability[1]
        }
    }
