from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="IRIS Prediction API",
    version="1.0"
)

model = joblib.load("artifacts/model.pkl")


class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {
        "message": "IRIS API is running!"
    }


@app.post("/predict")
def predict(data: IrisRequest):

    sample = pd.DataFrame([{
        "sepal_length": data.sepal_length,
        "sepal_width": data.sepal_width,
        "petal_length": data.petal_length,
        "petal_width": data.petal_width,
    }])

    prediction = model.predict(sample)

    return {
        "prediction": int(prediction[0])
    }
