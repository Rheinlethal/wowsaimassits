import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

"""
Script untuk prediksi offset_x berdasarkan data yang sudah dikoreksi
Input: data_tembakan_corrected.csv
Output: Model yang bisa prediksi offset_x untuk berbagai speed
"""

# Load data yang sudah dikoreksi
print("Loading corrected data...")
df = pd.read_csv('data_tembakan_corrected.csv')

print(f"Total data: {len(df)} rows")
print("\nContoh data:")
print(df.head())

# Prepare features untuk training
# Features: distance, angle, shell_travel_time, actual_speed
# Target: offset_x
features = ['distance', 'angle', 'shell_travel_time', 'actual_speed']
target = 'offset_x'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
print("\n=== Training Random Forest Regressor untuk Offset Prediction ===")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
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

# Simpan model
model_file = 'offset_prediction_model.pkl'
joblib.dump(rf_model, model_file)
print(f"\n✓ Model disimpan ke: {model_file}")

# Fungsi untuk prediksi offset dengan berbagai speed
def predict_offset_multi_speed(distance, angle, shell_travel_time, enemy_max_speed):
    """
    Prediksi offset_x untuk berbagai speed fraction
    
    Parameters:
    - distance: jarak ke musuh
    - angle: sudut antara aim dan arah laju musuh
    - shell_travel_time: waktu tempuh peluru
    - enemy_max_speed: max speed musuh
    
    Returns:
    - Dictionary dengan offset untuk berbagai speed
    """
    speed_fractions = {
        'full_speed': 1.0,
        '3/4_speed': 0.75,
        '1/2_speed': 0.5,
        '1/4_speed': 0.25
    }
    
    results = {}
    for speed_name, fraction in speed_fractions.items():
        actual_speed = enemy_max_speed * fraction
        
        # Prepare input
        input_data = pd.DataFrame({
            'distance': [distance],
            'angle': [angle],
            'shell_travel_time': [shell_travel_time],
            'actual_speed': [actual_speed]
        })
        
        # Predict
        offset = rf_model.predict(input_data)[0]
        results[speed_name] = offset
    
    return results

# Test prediksi dengan contoh data
print("\n=== Test Prediksi ===")
test_cases = [
    {'distance': 10000, 'angle': 45, 'shell_travel_time': 8.5, 'enemy_max_speed': 30},
    {'distance': 15000, 'angle': 60, 'shell_travel_time': 12.0, 'enemy_max_speed': 35},
    {'distance': 8000, 'angle': 30, 'shell_travel_time': 6.0, 'enemy_max_speed': 28},
]

for i, test in enumerate(test_cases, 1):
    print(f"\nTest Case {i}:")
    print(f"Distance: {test['distance']}m")
    print(f"Angle: {test['angle']}°")
    print(f"Shell Travel Time: {test['shell_travel_time']}s")
    print(f"Enemy Max Speed: {test['enemy_max_speed']} knots")
    print("\nPrediksi Offset X:")
    
    results = predict_offset_multi_speed(
        test['distance'],
        test['angle'],
        test['shell_travel_time'],
        test['enemy_max_speed']
    )
    
    for speed_type, offset in results.items():
        print(f"  {speed_type}: {offset:.2f} units")

# Simpan fungsi prediksi sebagai script terpisah
print("\n=== Membuat Script Prediksi Standalone ===")

standalone_script = '''import pandas as pd
import joblib

# Load model
model = joblib.load('offset_prediction_model.pkl')

def predict_offset(distance, angle, shell_travel_time, enemy_max_speed):
    """
    Prediksi offset_x untuk berbagai speed
    
    Parameters:
    - distance: jarak ke musuh (meter)
    - angle: sudut antara aim dan arah laju musuh (derajat)
    - shell_travel_time: waktu tempuh peluru (detik)
    - enemy_max_speed: max speed musuh (knots)
    
    Returns:
    - Dictionary dengan offset untuk full, 3/4, 1/2, 1/4 speed
    """
    speed_fractions = {
        'full_speed': 1.0,
        '3/4_speed': 0.75,
        '1/2_speed': 0.5,
        '1/4_speed': 0.25
    }
    
    results = {}
    for speed_name, fraction in speed_fractions.items():
        actual_speed = enemy_max_speed * fraction
        
        input_data = pd.DataFrame({
            'distance': [distance],
            'angle': [angle],
            'shell_travel_time': [shell_travel_time],
            'actual_speed': [actual_speed]
        })
        
        offset = model.predict(input_data)[0]
        results[speed_name] = offset
    
    return results

# Contoh penggunaan
if __name__ == "__main__":
    print("=== World of Warships Aim Assist ===")
    print("Masukkan data target:\\n")
    
    distance = float(input("Distance (meter): "))
    angle = float(input("Angle (derajat): "))
    shell_travel_time = float(input("Shell Travel Time (detik): "))
    enemy_max_speed = float(input("Enemy Max Speed (knots): "))
    
    print("\\n=== Hasil Prediksi Offset X ===")
    results = predict_offset(distance, angle, shell_travel_time, enemy_max_speed)
    
    for speed_type, offset in results.items():
        print(f"{speed_type}: {offset:.2f} units")
'''

with open('predict_offset.py', 'w') as f:
    f.write(standalone_script)

print("✓ Script standalone disimpan ke: predict_offset.py")

print("\n=== Selesai! ===")
print("\nCara menggunakan:")
print("1. Jalankan script ini untuk training model")
print("2. Gunakan 'predict_offset.py' untuk prediksi real-time")
print("3. Atau import fungsi predict_offset_multi_speed() ke script lain")
