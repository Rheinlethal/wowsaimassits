import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class WoWSAimAssist:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.speed_ratio = 0.85  # Initial guess: actual speed = 85% of max speed
        
    def load_data(self, csv_path):
        """Load data dari CSV"""
        print(f"Loading data dari {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"Data loaded: {len(self.df)} baris")
        print(f"\nKolom: {list(self.df.columns)}")
        print(f"\nSample data:\n{self.df.head()}")
        return self.df
    
    def estimate_speed_ratio(self):
        """Estimasi ratio actual_speed terhadap max_speed menggunakan optimization"""
        print("\n" + "="*60)
        print("ESTIMASI ACTUAL SPEED RATIO")
        print("="*60)
        
        def objective(ratio):
            """
            Fungsi objektif: cari ratio yang membuat perhitungan paling konsisten
            Logika: perpendicular_distance = actual_speed * time * sin(angle)
            offset_x seharusnya proporsional dengan perpendicular_distance / distance
            """
            actual_speeds = self.df['enemy_speed'] * ratio[0]
            
            # Hitung perpendicular movement (movement tegak lurus line of sight)
            # angle dalam derajat, konversi ke radian
            angles_rad = np.radians(self.df['angle'])
            perp_movement = actual_speeds * self.df['shell_travel_time'] * np.sin(angles_rad)
            
            # Angular offset (dalam radian atau unit angular)
            angular_offset = perp_movement / self.df['distance']
            
            # Cari korelasi antara angular_offset dengan offset_x
            # Semakin tinggi korelasi, semakin baik ratio-nya
            correlation = np.corrcoef(angular_offset, self.df['offset_x'])[0, 1]
            
            # Return negative correlation (karena kita minimize)
            return -abs(correlation)
        
        # Optimize untuk cari ratio terbaik (range 0.5 - 1.0)
        result = minimize(objective, x0=[0.85], bounds=[(0.5, 1.0)], method='L-BFGS-B')
        
        self.speed_ratio = result.x[0]
        print(f"\nOptimal speed ratio: {self.speed_ratio:.3f}")
        print(f"Artinya: actual_speed = {self.speed_ratio:.1%} × max_speed")
        
        # Hitung actual speeds
        self.df['actual_speed'] = self.df['enemy_speed'] * self.speed_ratio
        
        return self.speed_ratio
    
    def create_features(self):
        """Buat features untuk training"""
        print("\n" + "="*60)
        print("MEMBUAT FEATURES")
        print("="*60)
        
        # Konversi angle ke radian
        self.df['angle_rad'] = np.radians(self.df['angle'])
        
        # Perpendicular movement (movement tegak lurus)
        self.df['perp_movement'] = (self.df['actual_speed'] * 
                                     self.df['shell_travel_time'] * 
                                     np.sin(self.df['angle_rad']))
        
        # Longitudinal movement (movement searah)
        self.df['long_movement'] = (self.df['actual_speed'] * 
                                     self.df['shell_travel_time'] * 
                                     np.cos(self.df['angle_rad']))
        
        # Angular offset (sudut offset dalam radian)
        self.df['angular_offset'] = self.df['perp_movement'] / self.df['distance']
        
        # Features tambahan
        self.df['speed_time'] = self.df['actual_speed'] * self.df['shell_travel_time']
        self.df['sin_angle'] = np.sin(self.df['angle_rad'])
        self.df['cos_angle'] = np.cos(self.df['angle_rad'])
        
        print("\nFeatures created:")
        print(self.df[['actual_speed', 'perp_movement', 'angular_offset', 'offset_x']].describe())
        
    def train_model(self):
        """Train model untuk prediksi offset_x"""
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        # Features untuk training
        feature_cols = ['distance', 'angle', 'shell_travel_time', 'actual_speed',
                       'perp_movement', 'angular_offset', 'sin_angle', 'cos_angle']
        
        X = self.df[feature_cols]
        y = self.df['offset_x']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("\nTraining Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"\nHasil Training:")
        print(f"  Train MAE: {train_mae:.3f} satuan offset")
        print(f"  Test MAE:  {test_mae:.3f} satuan offset")
        print(f"  Train R²:  {train_r2:.3f}")
        print(f"  Test R²:   {test_r2:.3f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance:")
        print(importance.to_string(index=False))
        
        return test_mae, test_r2
    
    def generate_corrected_csv(self, output_path):
        """Generate CSV dengan actual_speed yang sudah dikoreksi"""
        print("\n" + "="*60)
        print("GENERATE CORRECTED CSV")
        print("="*60)
        
        # Buat dataframe baru dengan kolom yang diperlukan
        corrected_df = pd.DataFrame({
            'shell_travel_time': self.df['shell_travel_time'],
            'distance': self.df['distance'],
            'angle': self.df['angle'],
            'enemy_speed_max': self.df['enemy_speed'],  # Max speed asli
            'enemy_speed_actual': self.df['actual_speed'],  # Actual speed hasil estimasi
            'offset_x': self.df['offset_x']
        })
        
        corrected_df.to_csv(output_path, index=False)
        print(f"\nCorrected CSV saved to: {output_path}")
        print(f"Jumlah baris: {len(corrected_df)}")
        print(f"\nSample data:")
        print(corrected_df.head(10))
        
        return corrected_df
    
    def save_model(self, model_path='wows_aim_model.pkl', scaler_path='wows_scaler.pkl'):
        """Save trained model dan scaler"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.speed_ratio, 'wows_speed_ratio.pkl')
        print(f"\nModel saved:")
        print(f"  - {model_path}")
        print(f"  - {scaler_path}")
        print(f"  - wows_speed_ratio.pkl")
    
    def predict_offset(self, distance, angle, shell_travel_time, enemy_max_speed):
        """Prediksi offset_x untuk input baru"""
        # Hitung actual speed
        actual_speed = enemy_max_speed * self.speed_ratio
        
        # Hitung features
        angle_rad = np.radians(angle)
        perp_movement = actual_speed * shell_travel_time * np.sin(angle_rad)
        long_movement = actual_speed * shell_travel_time * np.cos(angle_rad)
        angular_offset = perp_movement / distance
        speed_time = actual_speed * shell_travel_time
        sin_angle = np.sin(angle_rad)
        cos_angle = np.cos(angle_rad)
        
        # Buat feature array
        features = np.array([[distance, angle, shell_travel_time, actual_speed,
                             perp_movement, angular_offset, sin_angle, cos_angle]])
        
        # Scale dan predict
        features_scaled = self.scaler.transform(features)
        offset_pred = self.model.predict(features_scaled)[0]
        
        return offset_pred, actual_speed


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print("="*60)
    print("WORLD OF WARSHIPS AIM ASSIST TRAINER")
    print("="*60)
    
    # Initialize
    aim_assist = WoWSAimAssist()
    
    # Load data
    input_csv = 'wows_data.csv'  # Ganti dengan nama file CSV kamu
    try:
        aim_assist.load_data(input_csv)
    except FileNotFoundError:
        print(f"\nERROR: File {input_csv} tidak ditemukan!")
        print("Membuat sample data untuk testing...")
        
        # Buat sample data
        sample_data = {
            'shell_travel_time': [8.65, 4.6, 8.54, 7.2, 5.8, 9.1, 6.3, 4.9, 7.8, 8.2],
            'distance': [13.55, 8.38, 13.35, 11.2, 9.5, 14.1, 10.2, 8.1, 12.3, 12.9],
            'angle': [74, 86, 16, 45, 60, 30, 70, 80, 25, 50],
            'enemy_speed': [21.0, 30.0, 23.0, 25.0, 28.0, 20.0, 26.0, 32.0, 22.0, 24.0],
            'offset_x': [3.5, 2.5, 1.8, 2.8, 3.2, 2.1, 3.6, 2.9, 2.0, 3.0]
        }
        pd.DataFrame(sample_data).to_csv(input_csv, index=False)
        aim_assist.load_data(input_csv)
    
    # Estimasi speed ratio
    aim_assist.estimate_speed_ratio()
    
    # Create features
    aim_assist.create_features()
    
    # Train model
    aim_assist.train_model()
    
    # Generate corrected CSV
    output_csv = 'wows_data_corrected.csv'
    aim_assist.generate_corrected_csv(output_csv)
    
    # Save model
    aim_assist.save_model()
    
    # Test prediction
    print("\n" + "="*60)
    print("TEST PREDICTION")
    print("="*60)
    print("\nContoh prediksi:")
    test_cases = [
        (13.55, 74, 8.65, 21.0),
        (8.38, 86, 4.6, 30.0),
        (13.35, 16, 8.54, 23.0),
    ]
    
    for dist, ang, time, speed in test_cases:
        offset, actual = aim_assist.predict_offset(dist, ang, time, speed)
        print(f"\nInput: dist={dist:.2f}, angle={ang}°, time={time:.2f}s, max_speed={speed:.1f}")
        print(f"  → Actual speed: {actual:.2f} knots")
        print(f"  → Predicted offset_x: {offset:.2f} satuan")
    
    print("\n" + "="*60)
    print("SELESAI!")
    print("="*60)
    print(f"\nFile output:")
    print(f"  1. {output_csv} - Data dengan actual speed")
    print(f"  2. wows_aim_model.pkl - Model terlatih")
    print(f"  3. wows_scaler.pkl - Scaler untuk normalisasi")
    print(f"  4. wows_speed_ratio.pkl - Speed ratio optimal")
    print("\nCara pakai model untuk prediksi:")
    print("  from wows_aim_assistant import WoWSAimAssist")
    print("  aim = WoWSAimAssist()")
    print("  aim.load_model()")
    print("  offset = aim.predict_offset(distance, angle, time, max_speed)")


if __name__ == "__main__":
    main()