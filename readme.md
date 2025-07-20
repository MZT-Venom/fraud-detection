## 🛠️ Manual: How to Run Your FastAPI Fraud Detection API

### ✅ 1. **Project Structure**

Assume your project folder looks like this:

```
Project/
├── server.py                    # Your FastAPI app
├── fraud_detection_pipeline.pkl # Your saved model
└── sample_input.json            # Optional: for testing
```

------

### ✅ 2. **Install Dependencies**

Make sure you have Python installed (e.g., Python 3.9 or above).
 Then, install required libraries:

```bash
pip install fastapi uvicorn joblib xgboost scikit-learn numpy pydantic
```

------

### ✅ 3. **Contents of `server.py`**

Paste this code inside `server.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load your model and metadata
model_bundle = joblib.load("fraud_detection_pipeline.pkl")
model = model_bundle["model"]
columns = model_bundle["columns"]

app = FastAPI(title="Fraud Detection API")

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
    input_array = np.array([[input_dict[col] for col in columns]])
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0].tolist()

    return {
        "prediction": int(prediction),
        "probability": {
            "non_fraud": probability[0],
            "fraud": probability[1]
        }
    }
```

------

### ✅ 4. **Run the Server**

Open terminal in your project directory and run:

```bash
uvicorn server:app --reload
```

Expected output:

```
Uvicorn running on http://127.0.0.1:8000
```

------

### ✅ 5. **Test the API**

Open your browser and go to:

📍 `http://127.0.0.1:8000/docs`
 This opens the **Swagger UI**, where you can test your API.

------

### ✅ 6. **Send Sample Input**

Use this sample JSON:

```json
{
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536346,
  "V4": 1.378155,
  "V5": -0.338321,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.5516,
  "V12": -0.617801,
  "V13": -0.99139,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053,
  "Amount": 149.62
}
```

------

### ✅ 7. **Example Output**

```json
{
  "prediction": 0,
  "probability": {
    "non_fraud": 0.9973,
    "fraud": 0.0027
  }
}
```

------

