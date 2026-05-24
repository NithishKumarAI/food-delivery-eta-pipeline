from fastapi import FastAPI
import pandas as pd

from src.predict import predict_eta

app = FastAPI()


@app.get("/")
def home():

    return {"message": "Food Delivery ETA API is running"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = predict_eta(df)

    return {
        "predicted_eta_minutes": prediction
    }