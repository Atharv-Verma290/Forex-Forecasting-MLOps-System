import os
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from io import StringIO
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from utils import get_predictors

model_metadata = {
    "run_id": None,
    "name": None,
    "version": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "60"
    os.environ["MLFLOW_ALLOW_HTTP_REDIRECTS"] = "true"

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5080")
    MODEL_URI="models:/eur_usd_direction_model/Production"

    print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try: 
        global model 
        model = mlflow.pyfunc.load_model(MODEL_URI)
        
        model_metadata["run_id"] = model.metadata.run_id

        for rm in client.search_registered_models():
            for mv in rm.latest_versions:
                if mv.run_id == model_metadata["run_id"]:
                    print("Model name:", rm.name)
                    model_metadata["name"] = rm.name 
                    print("Version:", mv.version)
                    model_metadata["version"] = mv.version
                    print("Stage", mv.current_stage)

        print("Champion model loaded", flush=True)
    except Exception as e:
        print(f"Failed to load model: {e}")
        model_metadata.clear()
    yield

app = FastAPI(lifespan=lifespan)

class PredictionInput(BaseModel):
    input_data: str 

@app.post("/predict_forex")
async def predict(request: PredictionInput):
    input_df = pd.read_json(StringIO(request.input_data))

    input_df = input_df.drop(columns=["id"], errors='ignore')
    input_df = input_df.set_index('datetime')

    predictors = get_predictors(input_df)
    X = input_df[predictors].astype("float64")

    prediction = model.predict(X)

    output = {
        "datetime": str(input_df.index[0]), 
        "prediction": prediction[0].item(),
        "model_name": model_metadata["name"],
        "model_version": model_metadata["version"]
    }

    return JSONResponse(content=output)



