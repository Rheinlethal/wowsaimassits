import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

"""
Script 1 (Bagian Training): Training Model dari Data yang Sudah Dikoreksi
File ini untuk ONE-TIME training saja
Setelah model di-generate, gunakan Script 2 untuk prediksi
"""

# Load data yang sudah dikoreksi
print("=" * 60)
print("TRAINING MODEL PREDIKSI OFFSET X")
print("=" * 60)
print("\nLoading corrected data...")

try:
    df = pd.read_csv('data_tembakan_corrected.csv')
except FileNotFoundError:
    print("❌ Error: File 'data_tembakan_corrected.csv' tidak ditemukan!")
    print("Jalankan Script 1 (01_speed_correction.py) terlebih dahulu")
    exit()

print(f"Total data: {len(df)} rows")
print("\nContoh data:")
print(df.head())

# Feature engineering
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

df['angle_rad'] = np.deg2rad(df['angle'])
df['sin_angle'] = np.sin(df['angle_rad'])
df['cos_angle'] = np.cos(df['angle_rad'])
df['speed_time_product'] = df['actual_speed'] * df['shell_travel_time']
df['distance_normalized'] = df['distance'] / 1000

# Features untuk prediksi offset_x
features = ['distance', 'angle', 'sin_angle', 'cos_angle', 
            'shell_travel_time', 'actual_speed', 'speed_time_product',
            'distance_normalized']
target = 'offset_x'

X = df[features]
y = df[target]

print(f"Features: {features}")
print(f"Target: {target}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train Random Forest model
print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST REGRESSOR")
print("=" * 60)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Training model...")
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

print("\n" + "=" * 60)
print("PERFORMA MODEL")
print("=" * 60)
print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):.4f} units")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.4f} units")
print(f"Train R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred_test):.4f}")

# Feature importance
print("\nFeature Importance:")
feature_importance = sorted(zip(features, rf_model.feature_importances_), 
                           key=lambda x: x[1], reverse=True)
for feature, importance in feature_importance:
    print(f"  {feature}: {importance:.4f}")

# Simpan model
model_file = 'offset_prediction_model.pkl'
joblib.dump(rf_model, model_file)
print(f"\n✓ MODEL DISIMPAN: {model_file}")

# Test prediksi
print("\n" + "=" * 60)
print("TEST PREDIKSI")
print("=" * 60)

test_case = {
    'distance': 10000,
    'angle': 45,
    'shell_travel_time': 8.5,
    'enemy_max_speed': 30
}

print(f"Test Case:")
print(f"  Distance          : {test_case['distance']} m")
print(f"  Angle             : {test_case['angle']}°")
print(f"  Shell Travel Time : {test_case['shell_travel_time']} s")
print(f"  Enemy Max Speed   : {test_case['enemy_max_speed']} knots")

# Prediksi untuk berbagai speed
speed_configs = {
    'full_speed (100%)': 1.0,
    '3/4_speed (75%)': 0.75,
    '1/2_speed (50%)': 0.5,
    '1/4_speed (25%)': 0.25
}

print(f"\nPrediksi Offset X:")
print("-" * 60)

for speed_name, fraction in speed_configs.items():
    actual_speed = test_case['enemy_max_speed'] * fraction
    
    angle_rad = np.deg2rad(test_case['angle'])
    sin_angle = np.sin(angle_rad)
    cos_angle = np.cos(angle_rad)
    speed_time_product = actual_speed * test_case['shell_travel_time']
    distance_normalized = test_case['distance'] / 1000
    
    input_data = pd.DataFrame({
        'distance': [test_case['distance']],
        'angle': [test_case['angle']],
        'sin_angle': [sin_angle],
        'cos_angle': [cos_angle],
        'shell_travel_time': [test_case['shell_travel_time']],
        'actual_speed': [actual_speed],
        'speed_time_product': [speed_time_product],
        'distance_normalized': [distance_normalized]
    })
    
    offset = rf_model.predict(input_data)[0]
    print(f"{speed_name:20s}: {offset:8.2f} units")

print("\n" + "=" * 60)
print("TRAINING SELESAI!")
print("=" * 60)
print(f"\n✓ Model tersimpan: {model_file}")
print("\nSelanjutnya:")
print("Gunakan Script 2 (02_predict_offset.py) untuk prediksi real-time")
