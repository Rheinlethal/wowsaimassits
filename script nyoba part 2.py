import joblib
import numpy as np
import pandas as pd

# fitur yang sama seperti saat training
FEATURES = [
    'shell_travel_time','distance','angle','angle_rad',
    'sin_a','cos_a','dist_time','time_sin','time_cos'
]

model = joblib.load("offset_predictor.pkl")


def predict_offset(shell_travel_time, distance, angle_deg):
    # buat dataframe satu baris
    s = pd.DataFrame([{
        "shell_travel_time": shell_travel_time,
        "distance": distance,
        "angle": angle_deg
    }])
    
    # fitur tambahan (harus SAMA EXACT seperti training)
    s['angle_rad'] = np.deg2rad(s['angle'])
    s['sin_a'] = np.sin(s['angle_rad'])
    s['cos_a'] = np.cos(s['angle_rad'])
    s['dist_time'] = s['distance'] * s['shell_travel_time']
    s['time_sin'] = s['shell_travel_time'] * s['sin_a']
    s['time_cos'] = s['shell_travel_time'] * s['cos_a']
    
    # ambil subset fitur
    X = s[FEATURES]
    
    # prediksi model ridge
    pred = model.predict(X)[0]
    
    # offset harus bulat → binocular pakai satuan garis
    pred_rounded = int(round(pred))
    
    return pred_rounded
