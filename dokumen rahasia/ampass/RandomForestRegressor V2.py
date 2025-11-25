import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data tembak2_new_data_DATA_.csv")

df = df.rename(columns={
    "shell_time_travel": "shell_time",
    "angel": "angle"
})

# Fitur & Target (speed tebak kamu)
X = df[["shell_time", "distance", "angle", "offset_x"]]
y = df["enemy_speed"]

# ============================================================
# 2. NORMALISASI
# ============================================================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 3. TUNING RANDOM FOREST
# ============================================================

param_grid = {
    "n_estimators": [300, 500, 800],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 4, 6],
    "max_features": ["sqrt", "log2"]
}

rf_base = RandomForestRegressor(random_state=42)

grid = GridSearchCV(
    rf_base,
    param_grid,
    cv=3,
    n_jobs=-1,
    scoring="r2"
)

print("🔧 Melakukan tuning model... sabar bentar...")
grid.fit(X_scaled, y)

best_model = grid.best_estimator_
print("✔ Best Params:", grid.best_params_)

# ============================================================
# 4. TRAIN TEST & EVALUASI MODEL AWAL (SPEED ASUMSI)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

best_model.fit(X_train, y_train)
pred1 = best_model.predict(X_test)

print("\n📌 Model Tahap 1 (Belajar dari speed tebakan awal):")
print("MAE:", mean_absolute_error(y_test, pred1))
print("R² :", r2_score(y_test, pred1))

# ============================================================
# 5. SELF-REFINEMENT (TARGET = PREDIKSI MODEL AWAL)
# ============================================================

df["enemy_speed_refine_target"] = best_model.predict(X_scaled)
y_refined = df["enemy_speed_refine_target"]

# train ulang untuk refine
best_model.fit(X_scaled, y_refined)

# ============================================================
# 6. FINAL PREDIKSI SPEED (SETELAH REFINEMENT)
# ============================================================

df["enemy_speed_final"] = best_model.predict(X_scaled)

# SIMPAN
df.to_csv("hasil_speed_refined V2.csv", index=False)
print("\n📁 File final disimpan: hasil_speed_refined.csv")

# ============================================================
# 7. SIMPAN MODEL + SCALER
# ============================================================

joblib.dump(best_model, "random_forest_speed_refined.pkl")
joblib.dump(scaler, "scaler_speed_refined.pkl")

print("💾 Model + scaler sudah disimpan!")
print("🔥 Model refine siap dipakai buat prediksi offset_x selanjutnya!")
