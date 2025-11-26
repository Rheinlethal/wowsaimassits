

# **Offset X Predictor – README**

Model ini dibuat untuk memprediksi **offset_x** pada binocular di World of Warships, sebagai bantuan untuk nembak musuh lebih konsisten tanpa harus ngitung manual setiap kali.
Prediksi ini dipakai khusus dynamic crosshair seperti **Nomogram Classic Top Web by stiv32**.

---

##  **Cara Penggunaan**

### 1️⃣ **Siapkan Dataset**

Gunakan file **`data_tembak.csv`** sebagai dataset utama.
Model akan dilatih menggunakan data ini dan menghasilkan file model **`offset_predictor.pkl`**.

### 2️⃣ **Tes Menggunakan GUI**

Jalankan file **`GUI.py`** untuk mencoba model.
Masukkan input seperti:

* jarak musuh
* angle
* shell travel time
  dan GUI akan menampilkan **prediksi offset_x**.

Angka ini nanti dipakai sebagai patokan untuk mengarahkan tembakan di binocular.

### 3️⃣ **Implementasi ke Game**

Untuk implementasi langsung, gunakan dynamic crosshair:
**Nomogram Classic Top Web (by stiv32)**
Model ini memberi angka offset yang tinggal disesuaikan dengan garis di crosshair.

---

## 📑 **Format File `data_tembak.csv`**

| Kolom                | Penjelasan                                        |
| -------------------- | ------------------------------------------------- |
| `shell_travel_time`  | Waktu peluru buat sampai ke titik aim             |
| `distance`           | Jarak ke kapal musuh                              |
| `angle`              | Sudut antara aim dan arah gerak musuh             |
| `enemy_max_speed`    | Kecepatan maksimum musuh (satuan *knot*)          |
| `enemy_actual_speed` | Kecepatan musuh saat data diambil (hasil rumus)   |
| `offset_x`           | Garis horizontal tempat peluru jatuh di binocular |

### 🧮 Rumus `enemy_actual_speed`

Digunakan untuk perhitungan yang lebih detail:

```
enemy_actual_speed = (distance * offset_x) / (shell_travel_time * sin(radians(angle)))
```

> Variabel speed ini **opsional** untuk Random Forest.
> Dipakai hanya kalau mau hasil prediksi yang lebih presisi.

---

## 🤖 **Tentang Model**

* Model hanya memprediksi **offset_x** (horizontal).
* **Tidak** memprediksi offset_y.
* Ini model prediktif (perkiraan), bukan simulasi fisika penuh.
* Kalau mau super akurat → hitung pakai rumus fisika.

---

## 🚀 **Contoh Penggunaan (Python)**

### 🔹 **Training Model**

```python
# contoh_regresi_offset.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# 1) Load data
df = pd.read_csv("excel datasets tembakan_new_data_nambah actual speed_.csv")

# 2) Basic cleaning: drop NA, optionally remove outliers
df = df.dropna()

# 3) Feature engineering
# convert angle (assumed degrees) to radians; if angle is already radian, skip conversion
df['angle_rad'] = np.deg2rad(df['angle'])
df['sin_a'] = np.sin(df['angle_rad'])
df['cos_a'] = np.cos(df['angle_rad'])
# interaction features
df['dist_time'] = df['distance'] * df['shell_travel_time']
df['time_sin'] = df['shell_travel_time'] * df['sin_a']
df['time_cos'] = df['shell_travel_time'] * df['cos_a']

# Choose features and target
features = ['shell_travel_time','distance','angle','angle_rad','sin_a','cos_a','dist_time','time_sin','time_cos']
X = df[features]
y = df['offset_x']

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5) Pipeline + Ridge (baseline)
pipe_ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])
params_ridge = {
    'ridge__alpha': [0.1, 1.0, 10.0, 50.0]
}
gs_ridge = GridSearchCV(pipe_ridge, params_ridge, cv=5, scoring='neg_mean_absolute_error')
gs_ridge.fit(X_train, y_train)
print("Best Ridge params:", gs_ridge.best_params_)

# 6) Random Forest (nonlinear)
rf = RandomForestRegressor(random_state=42)
params_rf = {
    'n_estimators': [100, 300],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 3, 5]
}
gs_rf = GridSearchCV(rf, params_rf, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
gs_rf.fit(X_train, y_train)
print("Best RF params:", gs_rf.best_params_)

# 7) Evaluate both on test set
models = {
    'ridge': gs_ridge.best_estimator_,
    'rf': gs_rf.best_estimator_
}
for name, m in models.items():
    m.fit(X_train, y_train)  # ensure trained on full training fold
    pred = m.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name} MAE={mae:.3f}, RMSE={rmse:.3f}")

# Choose best model (example: rf)
best_model = models['rf'] if (mean_absolute_error(y_test, models['rf'].predict(X_test))
                              < mean_absolute_error(y_test, models['ridge'].predict(X_test))) else models['ridge']

# 8) Save model
joblib.dump(best_model, "offset_predictor.pkl")

# 9) Example of using model at runtime
def predict_offset(sample):
    # sample: dict with keys shell_travel_time,distance,angle
    s = pd.DataFrame([sample])
    s['angle_rad'] = np.deg2rad(s['angle'])
    s['sin_a'] = np.sin(s['angle_rad'])
    s['cos_a'] = np.cos(s['angle_rad'])
    s['dist_time'] = s['distance'] * s['shell_travel_time']
    s['time_sin'] = s['shell_travel_time'] * s['sin_a']
    s['time_cos'] = s['shell_travel_time'] * s['cos_a']
    Xs = s[features]
    pred = best_model.predict(Xs)[0]
    # Round/clamp to integer binocular ticks (if required)
    pred_rounded = int(round(pred))
    # optionally clamp within observed range
    min_o, max_o = int(df['offset_x'].min()), int(df['offset_x'].max())
    pred_rounded = max(min(pred_rounded, max_o), min_o)
    return pred_rounded

# Example usage:
# print(predict_offset({'shell_travel_time':0.45,'distance':120,'angle':15}))

```

### 🔹 **Predict Offset X**

```python
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

```

### 🔹 **Output Contoh**

```
Prediksi offset_x = 3.42
```

—

## ⭐ **Catatan Penting**

* Model ini dibuat untuk mempermudah gameplay, bukan untuk simulasi ilmiah.
* Hasil prediksi bisa bervariasi tergantung kualitas dataset.
* Dataset sebaiknya berisi kondisi real in-game sebanyak mungkin (jarak berbeda, sudut beda, speed beda).

---
