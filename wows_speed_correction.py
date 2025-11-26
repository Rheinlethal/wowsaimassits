import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

"""
Script untuk mengoreksi enemy_speed dari max_speed ke actual_speed
Input: CSV dengan kolom shell_travel_time, distance, angle, enemy_speed, offset_x
Output: CSV dengan kolom tambahan actual_speed dan speed_fraction
"""

# Load data
print("Loading data...")
df = pd.read_csv('data_tembakan.csv')

print(f"Total data: {len(df)} rows")
print("\nContoh data:")
print(df.head())

# Feature engineering untuk mencari actual speed
# Logika: offset_x = actual_speed * shell_travel_time * sin(angle) * correction_factor
# Kita bisa estimasi speed_fraction = actual_speed / max_speed

print("\n=== Menghitung Speed Fraction ===")

# Konversi angle ke radian
df['angle_rad'] = np.deg2rad(df['angle'])

# Estimasi awal speed fraction berdasarkan offset_x
# offset_x ≈ actual_speed * shell_travel_time * sin(angle) / distance_factor
df['estimated_speed_fraction'] = df['offset_x'] / (
    df['enemy_speed'] * df['shell_travel_time'] * np.sin(df['angle_rad']) + 0.001
)

# Normalisasi ke range 0-1
df['estimated_speed_fraction'] = df['estimated_speed_fraction'].clip(0.1, 1.0)

print("Distribusi estimasi speed fraction:")
print(df['estimated_speed_fraction'].describe())

# Prepare features untuk training
features = ['shell_travel_time', 'distance', 'angle', 'enemy_speed', 'offset_x']
target = 'estimated_speed_fraction'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
print("\n=== Training Random Forest Regressor ===")
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

print(f"\nTrain MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"Train R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred_test):.4f}")

# Feature importance
print("\nFeature Importance:")
for feature, importance in zip(features, rf_model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Predict speed fraction untuk semua data
df['speed_fraction'] = rf_model.predict(df[features])
df['speed_fraction'] = df['speed_fraction'].clip(0.1, 1.0)  # Batasi range

# Hitung actual speed
df['actual_speed'] = df['enemy_speed'] * df['speed_fraction']

# Kategorisasi speed
def categorize_speed(fraction):
    if fraction >= 0.875:
        return 'full_speed'
    elif fraction >= 0.625:
        return '3/4_speed'
    elif fraction >= 0.375:
        return '1/2_speed'
    else:
        return '1/4_speed'

df['speed_category'] = df['speed_fraction'].apply(categorize_speed)

# Tampilkan statistik
print("\n=== Hasil Koreksi ===")
print("\nDistribusi Speed Fraction:")
print(df['speed_fraction'].describe())
print("\nDistribusi Speed Category:")
print(df['speed_category'].value_counts())

# Simpan hasil
output_file = 'data_tembakan_corrected.csv'
df.to_csv(output_file, index=False)
print(f"\n✓ Data yang sudah dikoreksi disimpan ke: {output_file}")

# Simpan model
model_file = 'speed_correction_model.pkl'
joblib.dump(rf_model, model_file)
print(f"✓ Model disimpan ke: {model_file}")

# Tampilkan contoh hasil
print("\n=== Contoh Hasil Koreksi ===")
sample_cols = ['distance', 'angle', 'enemy_speed', 'actual_speed', 
               'speed_fraction', 'speed_category', 'offset_x']
print(df[sample_cols].head(10).to_string())

print("\n=== Selesai! ===")
print(f"File output: {output_file}")
print("Gunakan file ini untuk script prediksi offset_x")
