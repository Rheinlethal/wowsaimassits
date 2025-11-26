# World of Warships Aim Assist

Project untuk memprediksi offset aim di World of Warships menggunakan Random Forest Regression.

## 📋 File-file dalam Project

### Training (One-time):
1. **generate_sample_data.py** - Generate contoh data (jika belum punya)
2. **01_speed_correction.py** - Koreksi nilai enemy_speed → actual_speed
3. **01_train_model.py** - Training model dari data yang sudah dikoreksi

### Prediction (Real-time):
4. **02_predict_offset.py** - **Script utama untuk prediksi saat main game**

## 🎯 Konsep Project

### Masalah:
- Dataset punya `enemy_speed` yang tidak akurat (biasanya max speed)
- Perlu koreksi ke `actual_speed` yang sebenarnya
- User perlu prediksi offset untuk berbagai kecepatan musuh

### Solusi:
```
┌──────────────────────┐
│ Dataset Original     │
│ (enemy_speed salah)  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Script 1: Koreksi    │ ← ONE-TIME TRAINING
│ enemy_speed          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Dataset Corrected    │
│ (actual_speed benar) │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Script 1b: Training  │ ← ONE-TIME TRAINING
│ Model RF             │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Model Tersimpan      │
│ (.pkl)               │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Script 2: Prediksi   │ ← PAKAI SAAT MAIN
│ User Input → Output  │
│ offset 4 speed       │
└──────────────────────┘
```

## 🚀 Setup Awal (One-Time)

### Step 1: Siapkan Data
Buat file `data_tembakan.csv`:
```csv
shell_travel_time,distance,angle,enemy_speed,offset_x
8.5,10000,45.0,30.0,125.50
12.0,15000,60.0,35.0,180.25
6.0,8000,30.0,28.0,95.30
```

**Keterangan:**
- `shell_travel_time`: Waktu peluru sampai target (detik)
- `distance`: Jarak ke musuh (meter)
- `angle`: Sudut aim vs arah musuh (derajat, 0-90°)
- `enemy_speed`: Max speed musuh dari stat ship (knots)
- `offset_x`: Offset aktual di binocular (satuan garis)

**Tips:**
- Minimal 200-500 data
- Musuh harus jalan **lurus**
- Catat dengan akurat

**Jika belum punya data:**
```bash
python generate_sample_data.py
```

### Step 2: Koreksi Enemy Speed
```bash
python 01_speed_correction.py
```

Output: `data_tembakan_corrected.csv` (dengan `actual_speed`)

### Step 3: Training Model
```bash
python 01_train_model.py
```

Output: `offset_prediction_model.pkl` (model siap pakai)

**✓ Setup selesai! Sekarang siap untuk prediksi real-time**

## 🎮 Cara Pakai Saat Main Game

### Script 2: Prediksi Real-Time

```bash
python 02_predict_offset.py
```

### Workflow In-Game:

1. **Lock target musuh** (auto lock)

2. **Catat data target:**
   ```
   Distance          : 10000 (dari rangefinder)
   Angle             : 45 (estimasi sudut aim vs arah musuh)
   Shell Travel Time : 8.5 (dari UI game)
   Enemy Max Speed   : 30 (dari stat ship musuh)
   ```

3. **Input ke script:**
   ```
   Distance (meter)         : 10000
   Angle (derajat)          : 45
   Shell Travel Time (detik): 8.5
   Enemy Max Speed (knots)  : 30
   ```

4. **Dapatkan output:**
   ```
   ============================================================
   HASIL PREDIKSI OFFSET X
   ============================================================
   full_speed (100%)   :   125.50 units
   3/4_speed (75%)     :    94.12 units
   1/2_speed (50%)     :    62.75 units
   1/4_speed (25%)     :    31.37 units
   ============================================================
   ```

5. **Pilih offset sesuai kecepatan musuh:**
   - Musuh full throttle → gunakan `full_speed (100%)`
   - Musuh 3/4 throttle → gunakan `3/4_speed (75%)`
   - Musuh half throttle → gunakan `1/2_speed (50%)`
   - Musuh slow/turning → gunakan `1/4_speed (25%)`

6. **Adjust aim di binocular:**
   - Geser crosshair sebesar nilai offset
   - Positif = geser ke arah pergerakan musuh
   - Negatif = geser berlawanan arah

7. **Fire!** 🎯

### Tips Aiming:
- Musuh belok? **Jangan tembak**, tunggu jalan lurus
- Musuh jauh (15km+)? Double-check offset
- Lihat throttle musuh untuk estimasi speed
- Practice makes perfect!

## 📊 Kenapa 4 Speed?

User tidak tahu exact speed musuh, jadi diberikan 4 pilihan:

| Speed Type | Fraction | Actual Speed | Keterangan |
|------------|----------|--------------|------------|
| full_speed | 100% | 30.0 knots | Full throttle |
| 3/4_speed | 75% | 22.5 knots | 3/4 throttle |
| 1/2_speed | 50% | 15.0 knots | Half throttle |
| 1/4_speed | 25% | 7.5 knots | Slow/turning |

Model akan prediksi offset untuk semua 4 kondisi, user tinggal pilih yang sesuai.

## 📈 Meningkatkan Akurasi

### 1. Tambah Data Training
Semakin banyak data, semakin akurat model:
- Target: 500+ samples
- Variasi jarak: 5km - 20km
- Variasi sudut: 0° - 90°
- Variasi kecepatan musuh

### 2. Re-train Model
Setelah tambah data baru:
```bash
python 01_speed_correction.py
python 01_train_model.py
```

Model baru akan otomatis menggantikan yang lama.

### 3. Quality Control Data
- ✅ Musuh jalan lurus
- ✅ Enemy_speed dari stat yang benar
- ✅ Catat offset dengan akurat
- ❌ Skip data saat musuh belok
- ❌ Skip data saat kondisi ekstrem

## ⚙️ Requirements

```bash
pip install pandas numpy scikit-learn joblib
```

Versi:
- Python 3.8+
- pandas 1.3+
- numpy 1.21+
- scikit-learn 1.0+
- joblib 1.0+

## 🔧 Troubleshooting

### Q: Model tidak ditemukan?
**A:** 
```bash
# Jalankan training dulu:
python 01_speed_correction.py
python 01_train_model.py
```

### Q: Prediksi tidak akurat?
**A:**
- Cek data training minimal 200 samples
- Cek kualitas data (musuh jalan lurus?)
- Re-train dengan data lebih banyak

### Q: Error saat input?
**A:**
- Input harus angka (gunakan titik untuk desimal)
- Cek range: distance (1000-25000), angle (0-90), dll
- Jangan input teks atau karakter khusus

### Q: Offset terlalu besar/kecil?
**A:**
- Cek satuan offset_x di dataset konsisten
- Cek shell_travel_time akurat
- Mungkin perlu kalibrasi ulang satuan binocular

## 📁 File Structure

```
project/
├── data_tembakan.csv              # Data original
├── data_tembakan_corrected.csv    # Data setelah koreksi
├── offset_prediction_model.pkl    # Model trained
│
├── generate_sample_data.py        # Generate contoh data
├── 01_speed_correction.py         # Koreksi speed
├── 01_train_model.py              # Training model
└── 02_predict_offset.py           # Prediksi real-time ⭐
```

## 🔍 Technical Details

### Script 1: Speed Correction
- Input: `data_tembakan.csv` (enemy_speed mungkin salah)
- Process: Random Forest koreksi ke actual_speed
- Output: `data_tembakan_corrected.csv`

### Script 1b: Model Training
- Input: `data_tembakan_corrected.csv`
- Process: Random Forest learn pattern offset_x
- Output: `offset_prediction_model.pkl`

### Script 2: Prediction
- Input: User input (distance, angle, shell_travel_time, enemy_max_speed)
- Process: Model predict dengan 4 speed fraction (1.0, 0.75, 0.5, 0.25)
- Output: Offset_x untuk setiap speed

### Features Used:
- distance, angle, sin_angle, cos_angle
- shell_travel_time, actual_speed
- speed_time_product, distance_normalized

## 💡 Advanced Usage

### Import ke Script Lain
```python
from predict_offset import predict_offset

# Prediksi
results = predict_offset(
    distance=10000,
    angle=45,
    shell_travel_time=8.5,
    enemy_max_speed=30
)

print(results)
# Output:
# {
#     'full_speed (100%)': 125.50,
#     '3/4_speed (75%)': 94.12,
#     '1/2_speed (50%)': 62.75,
#     '1/4_speed (25%)': 31.37
# }
```

### Batch Prediction
```python
targets = [
    (10000, 45, 8.5, 30),
    (15000, 60, 12.0, 35),
    (8000, 30, 6.0, 28)
]

for dist, ang, stt, spd in targets:
    results = predict_offset(dist, ang, stt, spd)
    print(f"Target {dist}m: {results['full_speed (100%)']}")
```

## ⚠️ Disclaimer

- Tool ini untuk **tujuan edukasi** dan analisis data
- **Tidak** meng-hook atau memodifikasi game
- Perhitungan berbasis **machine learning**, bukan cheat
- Akurasi tergantung **kualitas data training**
- Gunakan dengan bijak dan ikuti ToS game

## 📝 Summary

1. **Setup (one-time):**
   ```bash
   python 01_speed_correction.py
   python 01_train_model.py
   ```

2. **Saat main game (real-time):**
   ```bash
   python 02_predict_offset.py
   ```

3. **Input data target → Dapatkan offset untuk 4 speed → Aim & Fire!**

---

**Good luck dan fair seas, captain! ⚓🎯**
