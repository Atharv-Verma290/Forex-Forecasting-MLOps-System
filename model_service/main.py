import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from io import StringIO
from datetime import datetime, timedelta
import json 
import joblib
import mlflow.pyfunc
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "60"
    os.environ["MLFLOW_ALLOW_HTTP_REDIRECTS"] = "true"

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5080")
    MODEL_URI="models:/eur_usd_forex_direction_model/Production"

    print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try: 
        global model 
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print("Champion model loaded", flush=True)
    except Exception as e:
        print(f"Failed to load model: {e}")
    
    yield

app = FastAPI(lifespan=lifespan)

class PredictionInput(BaseModel):
    input_data: str 

@app.post("/predict_forex")
async def predict(request: PredictionInput):
    input_df = pd.read_json(StringIO(request.input_data))

    predictors = input_df.columns.drop(["id", "datetime"])
    # model = joblib.load("forex_model.pkl")

    prediction = model.predict(input_df[predictors])

    output = {
        "datetime": str(input_df["datetime"].iloc[0]), 
        "prediction": prediction[0].item()
    }

    return JSONResponse(content=output)



