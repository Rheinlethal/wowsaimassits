import pandas as pd
import numpy as np
import joblib
import sys

"""
Script 2: Prediksi Offset X (User Input)
Gunakan script ini untuk prediksi real-time saat main game

Cara pakai:
1. Pastikan model sudah di-training (jalankan 01_train_model.py)
2. Jalankan script ini
3. Input data target
4. Dapatkan prediksi offset untuk 4 speed
"""

# Load model
print("=" * 60)
print("WORLD OF WARSHIPS - AIM ASSIST")
print("=" * 60)

try:
    print("\nLoading model...")
    model = joblib.load('offset_prediction_model.pkl')
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model file tidak ditemukan!")
    print("\nPastikan file 'offset_prediction_model.pkl' ada di folder ini")
    print("Jika belum ada, jalankan training dulu:")
    print("  python 01_speed_correction.py")
    print("  python 01_train_model.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

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
    speed_configs = {
        'full_speed (100%)': 1.0,
        '3/4_speed (75%)': 0.75,
        '1/2_speed (50%)': 0.5,
        '1/4_speed (25%)': 0.25
    }
    
    results = {}
    
    for speed_name, fraction in speed_configs.items():
        # Hitung actual speed
        actual_speed = enemy_max_speed * fraction
        
        # Feature engineering (sama seperti saat training)
        angle_rad = np.deg2rad(angle)
        sin_angle = np.sin(angle_rad)
        cos_angle = np.cos(angle_rad)
        speed_time_product = actual_speed * shell_travel_time
        distance_normalized = distance / 1000
        
        # Prepare input data
        input_data = pd.DataFrame({
            'distance': [distance],
            'angle': [angle],
            'sin_angle': [sin_angle],
            'cos_angle': [cos_angle],
            'shell_travel_time': [shell_travel_time],
            'actual_speed': [actual_speed],
            'speed_time_product': [speed_time_product],
            'distance_normalized': [distance_normalized]
        })
        
        # Predict
        offset = model.predict(input_data)[0]
        results[speed_name] = round(offset, 2)
    
    return results

def display_results(distance, angle, shell_travel_time, enemy_max_speed, results):
    """Tampilkan hasil prediksi dengan format yang rapi"""
    print("\n" + "=" * 60)
    print("INPUT DATA")
    print("=" * 60)
    print(f"Distance          : {distance:,.0f} m")
    print(f"Angle             : {angle}°")
    print(f"Shell Travel Time : {shell_travel_time} s")
    print(f"Enemy Max Speed   : {enemy_max_speed} knots")
    
    print("\n" + "=" * 60)
    print("HASIL PREDIKSI OFFSET X")
    print("=" * 60)
    
    for speed_type, offset in results.items():
        print(f"{speed_type:20s}: {offset:8.2f} units")
    
    print("=" * 60)

def validate_input(value, name, min_val=None, max_val=None):
    """Validasi input user"""
    try:
        val = float(value)
        if min_val is not None and val < min_val:
            print(f"⚠️  Warning: {name} terlalu kecil (min: {min_val})")
        if max_val is not None and val > max_val:
            print(f"⚠️  Warning: {name} terlalu besar (max: {max_val})")
        return val
    except ValueError:
        raise ValueError(f"{name} harus berupa angka!")

# Main program
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("INPUT DATA TARGET")
    print("=" * 60)
    
    try:
        # Input dari user
        print("\nMasukkan data target:\n")
        
        distance_input = input("Distance (meter)         : ")
        distance = validate_input(distance_input, "Distance", min_val=1000, max_val=25000)
        
        angle_input = input("Angle (derajat)          : ")
        angle = validate_input(angle_input, "Angle", min_val=0, max_val=90)
        
        shell_time_input = input("Shell Travel Time (detik): ")
        shell_travel_time = validate_input(shell_time_input, "Shell Travel Time", 
                                          min_val=0.5, max_val=30)
        
        enemy_speed_input = input("Enemy Max Speed (knots)  : ")
        enemy_max_speed = validate_input(enemy_speed_input, "Enemy Max Speed", 
                                         min_val=10, max_val=50)
        
        # Prediksi
        results = predict_offset(distance, angle, shell_travel_time, enemy_max_speed)
        
        # Tampilkan hasil
        display_results(distance, angle, shell_travel_time, enemy_max_speed, results)
        
        # Tanya apakah mau prediksi lagi
        print("\nPrediksi lagi? (y/n): ", end="")
        again = input().lower()
        
        while again == 'y':
            print("\n" + "=" * 60)
            print("INPUT DATA TARGET")
            print("=" * 60)
            print("\nMasukkan data target:\n")
            
            distance_input = input("Distance (meter)         : ")
            distance = validate_input(distance_input, "Distance", min_val=1000, max_val=25000)
            
            angle_input = input("Angle (derajat)          : ")
            angle = validate_input(angle_input, "Angle", min_val=0, max_val=90)
            
            shell_time_input = input("Shell Travel Time (detik): ")
            shell_travel_time = validate_input(shell_time_input, "Shell Travel Time", 
                                              min_val=0.5, max_val=30)
            
            enemy_speed_input = input("Enemy Max Speed (knots)  : ")
            enemy_max_speed = validate_input(enemy_speed_input, "Enemy Max Speed", 
                                             min_val=10, max_val=50)
            
            results = predict_offset(distance, angle, shell_travel_time, enemy_max_speed)
            display_results(distance, angle, shell_travel_time, enemy_max_speed, results)
            
            print("\nPrediksi lagi? (y/n): ", end="")
            again = input().lower()
        
        print("\n✓ Terima kasih! Good luck, captain! ⚓")
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n✓ Program dihentikan. Good luck, captain! ⚓")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
