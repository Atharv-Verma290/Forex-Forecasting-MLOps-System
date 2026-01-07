import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import StringIO
import json 
import joblib

app = FastAPI()

class PredictionInput(BaseModel):
    input_data: str 

@app.post("/predict_forex")
async def predict(request: PredictionInput):
    input_df = pd.read_json(StringIO(request.input_data))

    predictors = input_df.columns.drop(["id", "datetime"])
    model = joblib.load("forex_model.pkl")

    prediction = model.predict(input_df[predictors])

    output = {
        "datetime": str(input_df["datetime"].iloc[0]), 
        "prediction": prediction[0].item()
    }

    return JSONResponse(content=output)



