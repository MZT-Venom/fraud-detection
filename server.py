from fastapi import FastAPI
from pydantic import BaseModel
import boto3
import json
import os

app = FastAPI(title="Fraud Detection via SageMaker")

# Read credentials and region from environment (Render will inject them)
aws_access_key = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
region = os.environ.get("AWS_REGION", "us-east-1")
endpoint_name = "fraud-detector-endpoint-v8"

# Boto3 client
runtime_client = boto3.client(
    "sagemaker-runtime",
    region_name=region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

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
    payload = data.dict()

    # Call SageMaker endpoint
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    result = json.loads(response["Body"].read().decode())
    return result

@app.get("/")
def root():
    return {"message": "SageMaker-connected FastAPI is live ✅"}
