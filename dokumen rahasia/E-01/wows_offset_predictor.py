import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

"""
Skrip untuk training model prediksi offset_x
menggunakan actual_speed yang sudah dikoreksi
"""

class OffsetPredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        
    def train(self, csv_file):
        """
        Train model untuk prediksi offset_x
        
        Parameters:
        - csv_file: path ke CSV yang sudah ada actual_speed
        """
        
        print("=" * 60)
        print("WORLD OF WARSHIPS - OFFSET PREDICTOR TRAINING")
        print("=" * 60)
        
        # Load data
        print(f"\n[1] Loading data dari: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"    Total data: {len(df)} baris")
        print(f"    Kolom: {list(df.columns)}")
        
        # Cek apakah ada kolom actual_speed
        if 'actual_speed' not in df.columns:
            raise ValueError("CSV harus punya kolom 'actual_speed'! Jalankan skrip 1 dulu.")
        
        # Statistik
        print("\n[2] Statistik Data:")
        print(df[['distance', 'angle', 'shell_travel_time', 'actual_speed', 'offset_x']].describe())
        
        # Feature engineering
        print("\n[3] Feature Engineering...")
        df['angle_rad'] = np.radians(df['angle'])
        df['angle_sin'] = np.sin(df['angle_rad'])
        df['angle_cos'] = np.cos(df['angle_rad'])
        df['angle_abs_sin'] = np.abs(df['angle_sin'])
        
        # Interaksi fitur yang relevan untuk ballistik
        df['speed_time'] = df['actual_speed'] * df['shell_travel_time']
        df['speed_time_sin'] = df['speed_time'] * df['angle_abs_sin']
        df['distance_time'] = df['distance'] / (df['shell_travel_time'] + 1e-6)
        
        # Fitur untuk prediksi
        self.feature_cols = [
            'distance', 
            'angle',
            'shell_travel_time', 
            'actual_speed',
            'angle_sin',
            'angle_cos',
            'angle_abs_sin',
            'speed_time',
            'speed_time_sin',
            'distance_time'
        ]
        
        X = df[self.feature_cols]
        y = df['offset_x']
        
        # Bersihkan NaN dan inf
        mask = ~(X.isnull().any(axis=1) | np.isinf(X).any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"    Data valid: {len(X)} baris")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n[4] Training Random Forest Regressor...")
        print(f"    Training data: {len(X_train)} baris")
        print(f"    Test data: {len(X_test)} baris")
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train, y_train)
        
        # Prediksi
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Evaluasi
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\n[5] Evaluasi Model:")
        print(f"    Training MAE: {train_mae:.4f} satuan offset")
        print(f"    Test MAE: {test_mae:.4f} satuan offset")
        print(f"    Training RMSE: {train_rmse:.4f}")
        print(f"    Test RMSE: {test_rmse:.4f}")
        print(f"    Training R²: {train_r2:.4f}")
        print(f"    Test R²: {test_r2:.4f}")
        
        # Cross validation
        print(f"\n[6] Cross Validation (5-fold)...")
        cv_scores = cross_val_score(self.model, X, y, cv=5, 
                                    scoring='neg_mean_absolute_error', n_jobs=-1)
        print(f"    CV MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n[7] Feature Importance:")
        print(feature_importance.to_string(index=False))
        
        # Visualisasi
        self._create_plots(y_train, y_pred_train, y_test, y_pred_test, feature_importance)
        
        print("\n" + "=" * 60)
        
        return self.model
    
    def _create_plots(self, y_train, y_pred_train, y_test, y_pred_test, feature_importance):
        """Buat visualisasi hasil training"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Prediction vs Actual (Train)
            axes[0, 0].scatter(y_train, y_pred_train, alpha=0.3, s=10)
            axes[0, 0].plot([y_train.min(), y_train.max()], 
                           [y_train.min(), y_train.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual offset_x')
            axes[0, 0].set_ylabel('Predicted offset_x')
            axes[0, 0].set_title('Training: Prediction vs Actual')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Prediction vs Actual (Test)
            axes[0, 1].scatter(y_test, y_pred_test, alpha=0.5, s=10, color='orange')
            axes[0, 1].plot([y_test.min(), y_test.max()], 
                           [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0, 1].set_xlabel('Actual offset_x')
            axes[0, 1].set_ylabel('Predicted offset_x')
            axes[0, 1].set_title('Test: Prediction vs Actual')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Residuals
            residuals_test = y_test - y_pred_test
            axes[1, 0].scatter(y_pred_test, residuals_test, alpha=0.5, s=10, color='green')
            axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[1, 0].set_xlabel('Predicted offset_x')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].set_title('Residual Plot (Test)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Feature Importance
            top_features = feature_importance.head(10)
            axes[1, 1].barh(range(len(top_features)), top_features['importance'])
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features['feature'])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].invert_yaxis()
            
            plt.tight_layout()
            plt.savefig('offset_model_analysis.png', dpi=150, bbox_inches='tight')
            print("\n[8] Plot disimpan ke: offset_model_analysis.png")
            plt.close()
        except Exception as e:
            print(f"\n[8] Warning: Tidak bisa membuat plot ({e})")
    
    def save_model(self, filename='offset_model.pkl'):
        """Save model ke file"""
        if self.model is None:
            raise ValueError("Model belum di-train!")
        
        model_data = {
            'model': self.model,
            'feature_cols': self.feature_cols
        }
        joblib.dump(model_data, filename)
        print(f"\n[9] Model disimpan ke: {filename}")
    
    def load_model(self, filename='offset_model.pkl'):
        """Load model dari file"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.feature_cols = model_data['feature_cols']
        print(f"Model berhasil di-load dari: {filename}")
    
    def predict(self, distance, angle, shell_travel_time, actual_speed):
        """
        Prediksi offset_x dari input parameter
        
        Parameters:
        - distance: jarak ke musuh
        - angle: sudut dalam derajat
        - shell_travel_time: waktu tempuh peluru
        - actual_speed: kecepatan actual musuh
        
        Returns:
        - offset_x: prediksi offset dalam satuan binocular
        """
        if self.model is None:
            raise ValueError("Model belum di-train atau di-load!")
        
        # Feature engineering (sama seperti saat training)
        angle_rad = np.radians(angle)
        angle_sin = np.sin(angle_rad)
        angle_cos = np.cos(angle_rad)
        angle_abs_sin = np.abs(angle_sin)
        speed_time = actual_speed * shell_travel_time
        speed_time_sin = speed_time * angle_abs_sin
        distance_time = distance / (shell_travel_time + 1e-6)
        
        # Buat input array
        input_data = pd.DataFrame({
            'distance': [distance],
            'angle': [angle],
            'shell_travel_time': [shell_travel_time],
            'actual_speed': [actual_speed],
            'angle_sin': [angle_sin],
            'angle_cos': [angle_cos],
            'angle_abs_sin': [angle_abs_sin],
            'speed_time': [speed_time],
            'speed_time_sin': [speed_time_sin],
            'distance_time': [distance_time]
        })
        
        # Prediksi
        offset_pred = self.model.predict(input_data)[0]
        
        return offset_pred
    
    def predict_batch(self, input_csv, output_csv):
        """
        Prediksi offset_x untuk batch data dari CSV
        
        Parameters:
        - input_csv: CSV dengan kolom distance, angle, shell_travel_time, actual_speed
        - output_csv: output CSV dengan kolom predicted_offset_x
        """
        if self.model is None:
            raise ValueError("Model belum di-train atau di-load!")
        
        print(f"\nBatch prediction dari: {input_csv}")
        df = pd.read_csv(input_csv)
        
        predictions = []
        for _, row in df.iterrows():
            pred = self.predict(
                row['distance'],
                row['angle'],
                row['shell_travel_time'],
                row['actual_speed']
            )
            predictions.append(pred)
        
        df['predicted_offset_x'] = predictions
        df.to_csv(output_csv, index=False)
        print(f"Hasil disimpan ke: {output_csv}")
        
        return df


def demo_usage():
    """Demo penggunaan model"""
    print("\n" + "=" * 60)
    print("DEMO: Prediksi Offset")
    print("=" * 60)
    
    predictor = OffsetPredictor()
    predictor.load_model('offset_model.pkl')
    
    # Contoh prediksi
    examples = [
        {"distance": 10000, "angle": 45, "shell_travel_time": 8.5, "actual_speed": 25},
        {"distance": 15000, "angle": 90, "shell_travel_time": 12.0, "actual_speed": 30},
        {"distance": 8000, "angle": 30, "shell_travel_time": 6.0, "actual_speed": 20},
    ]
    
    print("\nContoh Prediksi:")
    print("-" * 60)
    for i, ex in enumerate(examples, 1):
        offset = predictor.predict(**ex)
        print(f"\nExample {i}:")
        print(f"  Distance: {ex['distance']}m")
        print(f"  Angle: {ex['angle']}°")
        print(f"  Shell Travel Time: {ex['shell_travel_time']}s")
        print(f"  Actual Speed: {ex['actual_speed']} knots")
        print(f"  → Predicted Offset: {offset:.2f} satuan")


if __name__ == "__main__":
    import sys
    
    # File input (hasil dari skrip 1)
    INPUT_FILE = "data_tembakan_corrected.csv"
    MODEL_FILE = "offset_model.pkl"
    
    print("\nMODE PILIHAN:")
    print("1. Training model baru")
    print("2. Load model dan demo prediksi")
    print()
    
    mode = input("Pilih mode (1/2): ").strip()
    
    if mode == "1":
        # Training mode
        try:
            predictor = OffsetPredictor()
            predictor.train(INPUT_FILE)
            predictor.save_model(MODEL_FILE)
            
            print("\n" + "=" * 60)
            print("TRAINING SELESAI!")
            print("=" * 60)
            print(f"\nModel disimpan di: {MODEL_FILE}")
            print("Gunakan mode 2 untuk testing prediksi.")
            
        except FileNotFoundError:
            print(f"\nERROR: File '{INPUT_FILE}' tidak ditemukan!")
            print("Jalankan skrip 1 (estimate_actual_speed.py) terlebih dahulu!")
        except Exception as e:
            print(f"\nERROR: {e}")
    
    elif mode == "2":
        # Demo mode
        try:
            demo_usage()
        except FileNotFoundError:
            print(f"\nERROR: Model file '{MODEL_FILE}' tidak ditemukan!")
            print("Jalankan mode 1 untuk training dulu!")
        except Exception as e:
            print(f"\nERROR: {e}")
    
    else:
        print("Mode tidak valid!")
