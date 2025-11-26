import pandas as pd
import numpy as np

"""
Script untuk generate contoh data CSV
Jalankan script ini jika kamu belum punya file data_tembakan.csv
"""

# Generate sample data
np.random.seed(42)
n_samples = 500

data = {
    'shell_travel_time': np.random.uniform(3, 15, n_samples),
    'distance': np.random.uniform(5000, 20000, n_samples),
    'angle': np.random.uniform(0, 90, n_samples),
    'enemy_speed': np.random.uniform(20, 40, n_samples),
}

# Generate offset_x berdasarkan formula yang masuk akal
# offset_x dipengaruhi oleh: actual_speed, shell_travel_time, angle
df = pd.DataFrame(data)

# Simulasi speed fraction (musuh kadang full speed, kadang tidak)
speed_fractions = np.random.choice([0.25, 0.5, 0.75, 1.0], n_samples, 
                                   p=[0.1, 0.2, 0.3, 0.4])
actual_speeds = df['enemy_speed'] * speed_fractions

# Hitung offset_x (simplified physics-like formula)
angle_rad = np.deg2rad(df['angle'])
df['offset_x'] = (actual_speeds * 
                  df['shell_travel_time'] * 
                  np.sin(angle_rad) * 
                  (df['distance'] / 10000) *
                  np.random.uniform(0.95, 1.05, n_samples))  # noise

# Round untuk realism
df['shell_travel_time'] = df['shell_travel_time'].round(2)
df['distance'] = df['distance'].round(0)
df['angle'] = df['angle'].round(1)
df['enemy_speed'] = df['enemy_speed'].round(1)
df['offset_x'] = df['offset_x'].round(2)

# Save to CSV
df.to_csv('data_tembakan.csv', index=False)

print("✓ Sample data generated: data_tembakan.csv")
print(f"Total samples: {len(df)}")
print("\nContoh data:")
print(df.head(10))
print("\nStatistik:")
print(df.describe())
