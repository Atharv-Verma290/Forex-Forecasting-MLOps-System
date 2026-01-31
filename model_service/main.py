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
    """
    Manages the FastAPI application lifecycle and global model state.

    On startup, this function configures the MLflow environment, connects to the 
    tracking server, and attempts to load the current 'Production' model into memory. 
    It also populates global metadata by cross-referencing the loaded model's 
    run ID with the MLflow Model Registry.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Control is yielded back to the FastAPI framework to start the server.
    """
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
    """
    Schema for incoming prediction requests.

    Attributes:
        input_data (str): JSON-serialized string containing the feature set.
    """
    input_data: str 

@app.post("/predict_forex")
async def predict(request: PredictionInput):
    """
    Processes features and returns a model prediction.

    Converts the input JSON string into a DataFrame, standardizes the index 
    and feature set, and executes a prediction using the global champion model.

    Args:
        request (PredictionInput): The container for the raw input data.

    Returns:
        JSONResponse: A dictionary containing the prediction, timestamp, 
                      and source model metadata.
    """
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



