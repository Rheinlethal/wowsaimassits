import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd


# ---- Load model ----
try:
    model = joblib.load("offset_predictor.pkl")
except:
    raise SystemExit("Error: offset_predictor.pkl tidak ditemukan!")


FEATURES = [
    'shell_travel_time','distance','angle','angle_rad',
    'sin_a','cos_a','dist_time','time_sin','time_cos'
]


# ---- Prediction function ----
def predict_offset(shell_travel_time, distance, angle_deg):
    s = pd.DataFrame([{
        "shell_travel_time": shell_travel_time,
        "distance": distance,
        "angle": angle_deg
    }])

    s['angle_rad'] = np.deg2rad(s['angle'])
    s['sin_a'] = np.sin(s['angle_rad'])
    s['cos_a'] = np.cos(s['angle_rad'])
    s['dist_time'] = s['distance'] * s['shell_travel_time']
    s['time_sin'] = s['shell_travel_time'] * s['sin_a']
    s['time_cos'] = s['shell_travel_time'] * s['cos_a']

    X = s[FEATURES]
    pred = model.predict(X)[0]
    return int(round(pred))


# ---- GUI ----
def run_prediction():
    try:
        t = float(entry_time.get())
        d = float(entry_distance.get())
        a = float(entry_angle.get())
    except:
        messagebox.showerror("Error", "Semua input harus angka.")
        return

    result = predict_offset(t, d, a)
    label_result.config(text=f"Prediksi Offset:  {result}")


root = tk.Tk()
root.title("Offset Predictor GUI")
root.geometry("340x260")
root.resizable(False, False)


# ---- Labels and Entries ----
ttk.Label(root, text="Shell Travel Time").pack(pady=3)
entry_time = ttk.Entry(root)
entry_time.pack()

ttk.Label(root, text="Distance").pack(pady=3)
entry_distance = ttk.Entry(root)
entry_distance.pack()

ttk.Label(root, text="Angle (deg)").pack(pady=3)
entry_angle = ttk.Entry(root)
entry_angle.pack()


# ---- Predict Button ----
ttk.Button(root, text="Predict Offset", command=run_prediction).pack(pady=15)

# ---- Result ----
label_result = ttk.Label(root, text="Prediksi Offset: -", font=("Arial", 14))
label_result.pack(pady=5)


root.mainloop()
