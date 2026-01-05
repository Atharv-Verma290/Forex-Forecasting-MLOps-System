import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json 
import joblib

app = FastAPI()

@app.post("/predict_forex")
async def predict(input_data: pd.DataFrame):
    predictors = input_data.columns.drop(["id", "datetime"])
    model = joblib.load("forex_model.pkl")

    prediction = model.predict(input_data[predictors])

    output = {"datetime": input_data["datetime"], "prediction": prediction}

    return JSONResponse(content=output)



