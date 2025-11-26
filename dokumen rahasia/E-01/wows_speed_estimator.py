import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

"""
Skrip untuk mengestimasi actual_speed dari enemy_speed (max_speed)
menggunakan Random Forest Regression
"""

def estimate_actual_speed(input_csv, output_csv):
    """
    Mengestimasi actual speed dan generate CSV baru dengan kolom actual_speed
    
    Parameters:
    - input_csv: path ke file CSV input
    - output_csv: path ke file CSV output yang sudah dikoreksi
    """
    
    print("=" * 60)
    print("WORLD OF WARSHIPS - ACTUAL SPEED ESTIMATOR")
    print("=" * 60)
    
    # Load data
    print(f"\n[1] Loading data dari: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"    Total data: {len(df)} baris")
    print(f"    Kolom: {list(df.columns)}")
    
    # Tampilkan statistik awal
    print("\n[2] Statistik Data Awal:")
    print(df.describe())
    
    # Feature engineering: buat fitur tambahan untuk estimasi speed
    print("\n[3] Feature Engineering...")
    
    # Hitung expected_offset berdasarkan asumsi fisika sederhana
    # offset_x seharusnya proporsional dengan: speed * time * sin(angle)
    df['angle_rad'] = np.radians(df['angle'])
    df['expected_movement'] = df['enemy_speed'] * df['shell_travel_time'] * np.abs(np.sin(df['angle_rad']))
    
    # Ratio antara expected dan actual offset (indikator speed ratio)
    # Hindari division by zero
    df['offset_ratio'] = np.where(
        df['expected_movement'] > 0,
        df['offset_x'] / df['expected_movement'],
        0
    )
    
    # Bersihkan outlier ekstrem pada offset_ratio
    df = df[(df['offset_ratio'] >= 0) & (df['offset_ratio'] <= 2)]
    
    print(f"    Data setelah cleaning: {len(df)} baris")
    
    # Fitur untuk prediksi speed_ratio
    feature_cols = ['distance', 'angle', 'shell_travel_time', 'enemy_speed', 
                    'offset_x', 'expected_movement']
    X = df[feature_cols]
    
    # Target: speed_ratio (actual_speed / enemy_speed)
    # Clip antara 0-1 karena actual speed tidak bisa lebih dari max speed
    y = df['offset_ratio'].clip(0, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n[4] Training Random Forest Regressor...")
    print(f"    Training data: {len(X_train)} baris")
    print(f"    Test data: {len(X_test)} baris")
    
    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Prediksi speed ratio
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Evaluasi
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n[5] Evaluasi Model:")
    print(f"    Training MAE: {train_mae:.4f}")
    print(f"    Test MAE: {test_mae:.4f}")
    print(f"    Training R²: {train_r2:.4f}")
    print(f"    Test R²: {test_r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n[6] Feature Importance:")
    print(feature_importance.to_string(index=False))
    
    # Prediksi untuk semua data
    print(f"\n[7] Menghitung actual_speed untuk semua data...")
    df_full = pd.read_csv(input_csv)
    df_full['angle_rad'] = np.radians(df_full['angle'])
    df_full['expected_movement'] = df_full['enemy_speed'] * df_full['shell_travel_time'] * np.abs(np.sin(df_full['angle_rad']))
    
    X_full = df_full[feature_cols]
    speed_ratio_pred = rf_model.predict(X_full)
    
    # Clip speed_ratio antara 0-1
    speed_ratio_pred = np.clip(speed_ratio_pred, 0, 1)
    
    # Hitung actual_speed
    df_full['speed_ratio'] = speed_ratio_pred
    df_full['actual_speed'] = df_full['enemy_speed'] * speed_ratio_pred
    
    # Drop kolom temporary
    df_full = df_full.drop(['angle_rad', 'expected_movement'], axis=1)
    
    # Save ke CSV
    df_full.to_csv(output_csv, index=False)
    print(f"\n[8] Data berhasil disimpan ke: {output_csv}")
    print(f"    Kolom baru: speed_ratio, actual_speed")
    
    # Statistik hasil
    print(f"\n[9] Statistik Hasil:")
    print(f"    Mean speed_ratio: {df_full['speed_ratio'].mean():.3f}")
    print(f"    Median speed_ratio: {df_full['speed_ratio'].median():.3f}")
    print(f"    Min speed_ratio: {df_full['speed_ratio'].min():.3f}")
    print(f"    Max speed_ratio: {df_full['speed_ratio'].max():.3f}")
    
    # Visualisasi (opsional)
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.scatter(y_test, y_pred_test, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual Speed Ratio')
        plt.ylabel('Predicted Speed Ratio')
        plt.title('Prediction vs Actual')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.hist(df_full['speed_ratio'], bins=50, edgecolor='black')
        plt.xlabel('Speed Ratio')
        plt.ylabel('Frequency')
        plt.title('Distribution of Speed Ratio')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        plot_filename = output_csv.replace('.csv', '_analysis.png')
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"\n[10] Plot disimpan ke: {plot_filename}")
        plt.close()
    except Exception as e:
        print(f"\n[10] Warning: Tidak bisa membuat plot ({e})")
    
    print("\n" + "=" * 60)
    print("SELESAI!")
    print("=" * 60)
    
    return df_full


if __name__ == "__main__":
    # Contoh penggunaan
    INPUT_FILE = "data_tembakan.csv"  # Ganti dengan nama file CSV kamu
    OUTPUT_FILE = "data_tembakan_corrected.csv"
    
    print("\nCatatan:")
    print("- File input harus punya kolom: shell_travel_time, distance, angle, enemy_speed, offset_x")
    print("- File output akan punya kolom tambahan: speed_ratio, actual_speed")
    print()
    
    try:
        result_df = estimate_actual_speed(INPUT_FILE, OUTPUT_FILE)
        print(f"\nPreview 10 baris pertama:")
        print(result_df[['distance', 'angle', 'enemy_speed', 'actual_speed', 'offset_x']].head(10))
    except FileNotFoundError:
        print(f"\nERROR: File '{INPUT_FILE}' tidak ditemukan!")
        print("Silakan buat file CSV dengan format yang benar atau ubah INPUT_FILE")
    except Exception as e:
        print(f"\nERROR: {e}")
