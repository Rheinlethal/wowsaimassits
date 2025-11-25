import numpy as np
import joblib
import pandas as pd
import os

class WoWSAimPredictor:
    """
    Class untuk pakai model yang sudah di-train
    Gunakan ini di script cheat/overlay kamu
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.speed_ratio = None
        
    def load_model(self, 
                   model_path='wows_aim_model.pkl',
                   scaler_path='wows_scaler.pkl',
                   ratio_path='wows_speed_ratio.pkl'):
        """Load model yang sudah di-train"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.speed_ratio = joblib.load(ratio_path)
            print(f"✓ Model loaded successfully!")
            print(f"✓ Speed ratio: {self.speed_ratio:.3f} (actual speed = {self.speed_ratio:.1%} × max speed)")
            return True
        except FileNotFoundError as e:
            print(f"✗ Error: Model files tidak ditemukan!")
            print(f"  Pastikan file berikut ada di folder yang sama:")
            print(f"  - {model_path}")
            print(f"  - {scaler_path}")
            print(f"  - {ratio_path}")
            print(f"\n  Jalankan dulu 'wows_aim_assistant.py' untuk training model!")
            return False
        
    def predict(self, distance, angle, shell_travel_time, enemy_max_speed):
        """
        Prediksi offset_x untuk aim assist
        
        Parameters:
        -----------
        distance : float
            Jarak ke musuh (km)
        angle : float
            Sudut antara aim dan arah gerak musuh (derajat, 0-90)
        shell_travel_time : float
            Waktu tempuh peluru (detik)
        enemy_max_speed : float
            Max speed musuh (knots)
            
        Returns:
        --------
        offset_x : float
            Offset dalam satuan garis binocular
        actual_speed : float
            Estimasi actual speed musuh
        """
        # Hitung actual speed
        actual_speed = enemy_max_speed * self.speed_ratio
        
        # Konversi angle ke radian
        angle_rad = np.radians(angle)
        
        # Hitung features
        sin_angle = np.sin(angle_rad)
        cos_angle = np.cos(angle_rad)
        perp_movement = actual_speed * shell_travel_time * sin_angle
        long_movement = actual_speed * shell_travel_time * cos_angle
        angular_offset = perp_movement / distance
        speed_time = actual_speed * shell_travel_time
        
        # Buat feature array (harus sesuai urutan saat training)
        features = np.array([[
            distance,           # 0
            angle,              # 1
            shell_travel_time,  # 2
            actual_speed,       # 3
            perp_movement,      # 4
            angular_offset,     # 5
            sin_angle,          # 6
            cos_angle           # 7
        ]])
        
        # Scale dan predict
        features_scaled = self.scaler.transform(features)
        offset_x = self.model.predict(features_scaled)[0]
        
        return offset_x, actual_speed


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def get_float_input(prompt, min_val=None, max_val=None):
    """Helper function untuk input float dengan validasi"""
    while True:
        try:
            value = float(input(prompt))
            if min_val is not None and value < min_val:
                print(f"  ✗ Nilai harus >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"  ✗ Nilai harus <= {max_val}")
                continue
            return value
        except ValueError:
            print("  ✗ Input harus berupa angka!")
        except KeyboardInterrupt:
            print("\n\nProgram dihentikan.")
            exit(0)


def interactive_mode():
    """Mode interaktif untuk input manual"""
    print("\n" + "="*70)
    print(" "*15 + "🎯 WORLD OF WARSHIPS AIM PREDICTOR 🎯")
    print("="*70)
    
    # Initialize predictor
    predictor = WoWSAimPredictor()
    
    # Load model
    print("\n📦 Loading model...")
    if not predictor.load_model():
        return
    
    print("\n" + "="*70)
    print("MODE: INTERACTIVE INPUT")
    print("="*70)
    print("\nKetik 'q' atau tekan Ctrl+C untuk keluar\n")
    
    while True:
        try:
            print("-" * 70)
            print("Input data tembakan:")
            print("-" * 70)
            
            # Get inputs
            distance = get_float_input(
                "Distance (km, jarak ke musuh)      : ",
                min_val=0.1, max_val=30.0
            )
            
            angle = get_float_input(
                "Angle (°, sudut 0-90)              : ",
                min_val=0, max_val=90
            )
            
            shell_travel_time = get_float_input(
                "Shell travel time (detik)          : ",
                min_val=0.1, max_val=30.0
            )
            
            enemy_max_speed = get_float_input(
                "Enemy max speed (knots)            : ",
                min_val=0, max_val=50
            )
            
            # Predict
            offset_x, actual_speed = predictor.predict(
                distance, angle, shell_travel_time, enemy_max_speed
            )
            
            # Display results
            print("\n" + "="*70)
            print("🎯 HASIL PREDIKSI")
            print("="*70)
            print(f"\n📊 Input Summary:")
            print(f"   • Distance          : {distance:.2f} km")
            print(f"   • Angle             : {angle:.1f}°")
            print(f"   • Shell travel time : {shell_travel_time:.2f} s")
            print(f"   • Enemy max speed   : {enemy_max_speed:.1f} knots")
            
            print(f"\n🧮 Calculation:")
            print(f"   • Actual speed (est): {actual_speed:.2f} knots ({actual_speed/enemy_max_speed*100:.0f}% of max)")
            
            print(f"\n🎯 AIM LEAD:")
            print(f"   ╔{'═'*66}╗")
            print(f"   ║  OFFSET X: {offset_x:+.2f} satuan garis binocular{' '*(66-len(f'OFFSET X: {offset_x:+.2f} satuan garis binocular'))}║")
            print(f"   ╚{'═'*66}╝")
            
            # Direction hint
            if abs(offset_x) < 0.3:
                direction = "🎯 AIM LANGSUNG (hampir center, offset minimal)"
                color = "green"
            elif offset_x > 0:
                direction = f"➡️  GESER KANAN {abs(offset_x):.2f} garis"
                color = "yellow"
            else:
                direction = f"⬅️  GESER KIRI {abs(offset_x):.2f} garis"
                color = "yellow"
            
            print(f"\n   {direction}")
            
            # Confidence indicator
            if angle < 10:
                print(f"\n   ⚠️  Warning: Angle sangat kecil ({angle}°), musuh hampir bow-on/stern")
                print(f"      Lead mungkin kurang akurat untuk angle ekstrim")
            elif angle > 85:
                print(f"\n   ℹ️  Info: Angle besar ({angle}°), musuh broadside penuh")
                print(f"      Lead maksimal dibutuhkan")
            
            print("\n" + "="*70)
            
            # Continue or exit
            print("\n")
            continue_input = input("Prediksi lagi? (Enter = ya, q = keluar): ").strip().lower()
            if continue_input == 'q':
                print("\n👋 Terima kasih! Good luck hunting! 🚢💥")
                break
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Program dihentikan. Good luck! 🚢💥")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("Coba lagi...\n")


def batch_mode_from_file():
    """Mode batch: baca dari file CSV"""
    print("\n" + "="*70)
    print(" "*15 + "📁 BATCH MODE - READ FROM CSV")
    print("="*70)
    
    # Initialize predictor
    predictor = WoWSAimPredictor()
    
    # Load model
    print("\n📦 Loading model...")
    if not predictor.load_model():
        return
    
    # Input file
    input_file = input("\nMasukkan nama file CSV (contoh: test_data.csv): ").strip()
    
    try:
        # Read CSV
        df = pd.read_csv(input_file)
        print(f"\n✓ File loaded: {len(df)} baris data")
        
        # Check columns
        required_cols = ['distance', 'angle', 'shell_travel_time', 'enemy_speed_max']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"\n✗ Error: Kolom tidak lengkap!")
            print(f"  Kolom yang diperlukan: {required_cols}")
            print(f"  Kolom yang hilang: {missing_cols}")
            return
        
        # Predict
        print("\n🎯 Predicting...")
        predictions = []
        actual_speeds = []
        
        for _, row in df.iterrows():
            offset, actual = predictor.predict(
                row['distance'],
                row['angle'],
                row['shell_travel_time'],
                row['enemy_speed_max']
            )
            predictions.append(offset)
            actual_speeds.append(actual)
        
        df['predicted_offset_x'] = predictions
        df['actual_speed'] = actual_speeds
        
        # Save results
        output_file = input_file.replace('.csv', '_predictions.csv')
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ Predictions saved to: {output_file}")
        print(f"\nSample results:")
        print(df.head(10).to_string(index=False))
        
    except FileNotFoundError:
        print(f"\n✗ Error: File '{input_file}' tidak ditemukan!")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def main_menu():
    """Main menu"""
    while True:
        print("\n" + "="*70)
        print(" "*10 + "🎯 WORLD OF WARSHIPS AIM PREDICTOR 🎯")
        print("="*70)
        print("\nPilih mode:")
        print("  1. Interactive Mode  (input manual satu-satu)")
        print("  2. Batch Mode        (baca dari file CSV)")
        print("  3. Exit")
        print("-" * 70)
        
        choice = input("\nPilihan (1/2/3): ").strip()
        
        if choice == '1':
            interactive_mode()
        elif choice == '2':
            batch_mode_from_file()
        elif choice == '3':
            print("\n👋 Terima kasih! Good luck hunting! 🚢💥\n")
            break
        else:
            print("\n✗ Pilihan tidak valid! Pilih 1, 2, atau 3.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    main_menu()