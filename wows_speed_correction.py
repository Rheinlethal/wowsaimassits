import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

"""
Script 1: Koreksi Enemy Speed
Tujuan: Mengoreksi nilai enemy_speed yang salah/tidak akurat menjadi actual_speed yang benar
Input: data_tembakan.csv dengan enemy_speed yang mungkin salah
Output: data_tembakan_corrected.csv dengan actual_speed yang sudah dikoreksi
"""

# Load data
print("=" * 60)
print("SCRIPT 1: KOREKSI ENEMY SPEED")
print("=" * 60)
print("\nLoading data...")
df = pd.read_csv('data tembak2_new_data_DATA_.csv')

print(f"Total data: {len(df)} rows")
print("\nContoh data original:")
print(df.head())
print("\nStatistik enemy_speed original:")
print(df['enemy_speed'].describe())

# Feature engineering
print("\n" + "=" * 60)
print("PROSES KOREKSI ACTUAL SPEED")
print("=" * 60)

# Konversi angle ke radian untuk perhitungan
df['angle_rad'] = np.deg2rad(df['angle'])
df['sin_angle'] = np.sin(df['angle_rad'])

# Logika: offset_x berbanding lurus dengan actual_speed * shell_travel_time * sin(angle)
# Kita gunakan ini untuk estimasi awal actual_speed
# Formula: actual_speed ≈ offset_x / (shell_travel_time * sin(angle) * distance_factor)

# Estimasi awal actual speed berdasarkan physics-like relationship
distance_factor = df['distance'] / 10000  # Normalisasi distance
denominator = df['shell_travel_time'] * df['sin_angle'] * distance_factor
denominator = denominator.replace(0, 0.001)  # Hindari division by zero

df['estimated_actual_speed'] = df['offset_x'] / denominator
df['estimated_actual_speed'] = df['estimated_actual_speed'].clip(5, 50)  # Batas realistis

print("\nEstimasi awal actual_speed:")
print(df['estimated_actual_speed'].describe())

# Prepare features untuk Random Forest
# Features yang bisa membantu prediksi actual speed
features = ['shell_travel_time', 'distance', 'angle', 'sin_angle', 
            'enemy_speed', 'offset_x']
target = 'estimated_actual_speed'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest untuk refine estimasi actual speed
print("\nTraining Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Evaluate model
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

print(f"\nPerforma Model:")
print(f"  Train MAE: {mean_absolute_error(y_train, y_pred_train):.4f} knots")
print(f"  Test MAE: {mean_absolute_error(y_test, y_pred_test):.4f} knots")
print(f"  Train R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"  Test R²: {r2_score(y_test, y_pred_test):.4f}")

# Feature importance
print("\nFeature Importance:")
for feature, importance in sorted(zip(features, rf_model.feature_importances_), 
                                 key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {importance:.4f}")

# Predict actual speed untuk semua data
df['actual_speed'] = rf_model.predict(df[features])

# PENTING: actual_speed tidak boleh melebihi enemy_speed (max speed)
# Clip antara 5 knots (minimum realistis) dan enemy_speed (max speed)
df['actual_speed'] = df.apply(
    lambda row: min(max(row['actual_speed'], 5), row['enemy_speed']), 
    axis=1
).round(2)

# Analisis koreksi
df['speed_difference'] = (df['actual_speed'] - df['enemy_speed']).round(2)
df['speed_fraction'] = (df['actual_speed'] / df['enemy_speed']).round(3)
df['speed_error_pct'] = ((df['speed_difference'] / df['enemy_speed']) * 100).round(2)

print("\n" + "=" * 60)
print("HASIL KOREKSI")
print("=" * 60)
print("\nStatistik Actual Speed (setelah koreksi):")
print(df['actual_speed'].describe())
print("\nStatistik Speed Fraction (actual/max):")
print(df['speed_fraction'].describe())
print("\nPerbedaan Enemy Speed vs Actual Speed:")
print(df['speed_difference'].describe())
print("\nPersentase Error:")
print(df['speed_error_pct'].describe())

# Validasi: actual_speed tidak boleh > enemy_speed
invalid_count = (df['actual_speed'] > df['enemy_speed']).sum()
if invalid_count > 0:
    print(f"\n⚠️  Warning: {invalid_count} data memiliki actual_speed > enemy_speed")
    print("    (sudah dikoreksi ke enemy_speed)")
else:
    print("\n✓ Validasi OK: Semua actual_speed <= enemy_speed")

# Tampilkan contoh koreksi
print("\n" + "=" * 60)
print("CONTOH HASIL KOREKSI")
print("=" * 60)
comparison_cols = ['distance', 'angle', 'enemy_speed', 'actual_speed', 
                   'speed_fraction', 'speed_difference', 'offset_x']
print(df[comparison_cols].head(15).to_string(index=False))

# Prepare output file (hanya kolom yang diperlukan)
output_cols = ['shell_travel_time', 'distance', 'angle', 'enemy_speed', 
               'actual_speed', 'offset_x']
df_output = df[output_cols].copy()

# Simpan hasil
output_file = 'data_tembakan_corrected.csv'
df_output.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("✓ DATA YANG SUDAH DIKOREKSI DISIMPAN")
print("=" * 60)
print(f"File: {output_file}")
print(f"Kolom: {', '.join(output_cols)}")
print(f"Total rows: {len(df_output)}")

# Simpan model
model_file = 'speed_correction_model.pkl'
joblib.dump(rf_model, model_file)
print(f"\n✓ MODEL DISIMPAN: {model_file}")

print("\n" + "=" * 60)
print("SELESAI!")
print("=" * 60)
print("Selanjutnya: Jalankan Script 2 untuk prediksi offset_x")
print(f"File input untuk Script 2: {output_file}")