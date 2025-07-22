## 🛠️ Manual: How to Run Your FastAPI Fraud Detection API (with AWS SageMaker)

### ✅ 1. **Project Structure**

Your project folder should look like:

```
Project/
├── server.py              # Your FastAPI app that connects to SageMaker
└── sample_input.json      # Optional: for testing
```

> ✅ No need for local `.pkl` files — the model is hosted and served via Amazon SageMaker.

---

### ✅ 2. **Install Dependencies**

Make sure you have Python installed (e.g., Python 3.9 or above).  
Then, install required libraries:

```bash
pip install fastapi uvicorn boto3 sagemaker pydantic
```

---

### ✅ 3. **Contents of `server.py`**

Replace your existing `server.py` with this:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import boto3
import json

from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# SageMaker endpoint name
ENDPOINT_NAME = "fraud-detector-endpoint-v8"  # Change this if your endpoint has a different name

# Set up SageMaker predictor
predictor = Predictor(
    endpoint_name=ENDPOINT_NAME,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

app = FastAPI(title="Fraud Detection API (via SageMaker)")

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
    input_data = data.dict()
    response = predictor.predict(input_data)
    return response

@app.get("/")
def root():
    return {"message": "Fraud Detection API (via SageMaker) is running"}
```

---

### ✅ 4. **Run the Server**

Open your terminal and run:

```bash
uvicorn server:app --reload
```

---

### ✅ 5. **Test the API**

Go to:

```
http://127.0.0.1:8000/docs
```

This opens the **Swagger UI**, where you can test the `/predict` endpoint interactively.

---

### ✅ 6. **Send Sample Input**

Use this sample JSON input to test the API:

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

---

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

---
