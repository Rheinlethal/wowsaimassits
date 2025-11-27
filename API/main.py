from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model
model = pickle.load(open("offset_predictor.pkl", "rb"))

app = FastAPI()

# Data input dari user
class PredictInput(BaseModel):
    distance: float
    angle: float
    shell_travel_time: float

@app.post("/predict")
def predict(data: PredictInput):
    # Susun data jadi array
    arr = np.array([
        data.distance,
        data.angle,
        data.shell_travel_time,
    ]).reshape(1, -1)

    # Prediksi dari model
    result = model.predict(arr)[0]

    return {"offset_x": float(result)}
