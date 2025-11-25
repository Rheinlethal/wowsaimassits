import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("data tembak2_new_data_DATA_.csv")

# Pastikan nama kolom cocok
# shell_time_travel,distance,angel,enemy_speed,offset_x

# Rename biar rapi
df = df.rename(columns={
    "shell_time_travel": "shell_time",
    "angel": "angle"
})

# ===============================
# 2. INPUT FEATURES
# ===============================
X = df[["shell_time", "distance", "angle", "offset_x"]]  
y = df["enemy_speed"]  # speed tebakanmu (target sementara)

# ===============================
# 3. NORMALISASI (MinMax)
# ===============================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 4. TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===============================
# 5. RANDOM FOREST REGRESSOR
# ===============================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 6. EVALUASI
# ===============================
pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, pred_test)
r2 = r2_score(y_test, pred_test)

print("MAE:", mae)
print("R² Score:", r2)

# ===============================
# 7. PREDIKSI SPEED BARU
# ===============================
df["enemy_speed_pred"] = model.predict(scaler.transform(X))

# Simpan keluaran
df.to_csv("hasil_prediksi_speed.csv", index=False)
print("File berhasil disimpan: hasil_prediksi_speed.csv")

# ===============================
# 8. SIMPAN MODEL + SCALER
# ===============================
joblib.dump(model, "random_forest_speed_model.pkl")
joblib.dump(scaler, "scaler_speed.pkl")

print("Model + scaler sudah disimpan!")
