from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="WoWS Offset Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load("offset_predictor.pkl")
except FileNotFoundError:
    raise SystemExit("Error: offset_predictor.pkl not found!")

FEATURES = [
    'shell_travel_time', 'distance', 'angle', 'angle_rad',
    'sin_a', 'cos_a', 'dist_time', 'time_sin', 'time_cos'
]

class PredictInput(BaseModel):
    shell_travel_time: float
    distance: float
    angle: float

@app.post("/api/predict")
def predict(data: PredictInput):
    s = pd.DataFrame([{
        "shell_travel_time": data.shell_travel_time,
        "distance": data.distance,
        "angle": data.angle
    }])
    
    s['angle_rad'] = np.deg2rad(s['angle'])
    s['sin_a'] = np.sin(s['angle_rad'])
    s['cos_a'] = np.cos(s['angle_rad'])
    s['dist_time'] = s['distance'] * s['shell_travel_time']
    s['time_sin'] = s['shell_travel_time'] * s['sin_a']
    s['time_cos'] = s['shell_travel_time'] * s['cos_a']
    
    X = s[FEATURES]
    pred = model.predict(X)[0]
    offset_rounded = int(round(pred))
    
    return {
        "offset_x": offset_rounded,
        "offset_x_raw": float(pred)
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": True}

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    @app.get("/")
    def read_root():
        return FileResponse("static/index.html")
