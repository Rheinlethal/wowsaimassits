# World of Warships Aim Assist

Project untuk memprediksi offset aim di World of Warships menggunakan Random Forest Regression.

## 📋 File-file dalam Project

1. **generate_sample_data.py** - Generate contoh data (jika belum punya)
2. **01_speed_correction.py** - Koreksi enemy_speed (max → actual speed)
3. **02_offset_prediction.py** - Training model prediksi offset_x
4. **predict_offset.py** - Script standalone untuk prediksi real-time

## 🚀 Cara Penggunaan

### Step 1: Siapkan Data
Pastikan kamu punya file `data_tembakan.csv` dengan format:
```csv
shell_travel_time,distance,angle,enemy_speed,offset_x
8.5,10000,45.0,30.0,125.50
12.0,15000,60.0,35.0,180.25
```

**Keterangan kolom:**
- `shell_travel_time`: Waktu tempuh peluru (detik)
- `distance`: Jarak ke musuh (meter)
- `angle`: Sudut antara aim dan arah laju musuh (derajat)
- `enemy_speed`: Max speed musuh (knots)
- `offset_x`: Offset di binocular (satuan garis)

**Jika belum punya data:**
```bash
python generate_sample_data.py
```

### Step 2: Koreksi Enemy Speed
Jalankan script untuk mengoreksi enemy_speed dari max speed ke actual speed:

```bash
python 01_speed_correction.py
```

**Output:**
- `data_tembakan_corrected.csv` - Data dengan actual_speed
- `speed_correction_model.pkl` - Model untuk koreksi speed

### Step 3: Training Model Prediksi Offset
Jalankan script untuk training model prediksi offset_x:

```bash
python 02_offset_prediction.py
```

**Output:**
- `offset_prediction_model.pkl` - Model untuk prediksi offset
- `predict_offset.py` - Script standalone

### Step 4: Gunakan untuk Prediksi
Ada 2 cara:

#### A. Interactive Mode
```bash
python predict_offset.py
```
Kemudian input data target secara interaktif.

#### B. Import ke Script Lain
```python
from predict_offset import predict_offset

# Prediksi offset
results = predict_offset(
    distance=10000,        # meter
    angle=45,              # derajat
    shell_travel_time=8.5, # detik
    enemy_max_speed=30     # knots
)

print(results)
# Output:
# {
#     'full_speed': 125.50,
#     '3/4_speed': 94.12,
#     '1/2_speed': 62.75,
#     '1/4_speed': 31.37
# }
```

## 📊 Output Prediksi

Setiap prediksi akan memberikan offset untuk 4 kecepatan:
- **full_speed** (100%): Musuh jalan full speed
- **3/4_speed** (75%): Musuh jalan 3/4 speed
- **1/2_speed** (50%): Musuh jalan setengah speed
- **1/4_speed** (25%): Musuh jalan pelan

## 🎯 Cara Pakai di Game

1. Lock target musuh (auto lock)
2. Catat: distance, angle, shell_travel_time, enemy_max_speed
3. Jalankan prediksi
4. Sesuaikan aim sesuai offset yang diprediksi:
   - Lihat kecepatan musuh (penuh/setengah/dll)
   - Geser aim sebanyak offset yang sesuai
   - Fire!

## 📈 Meningkatkan Akurasi

Untuk hasil lebih akurat:
1. **Kumpulkan lebih banyak data** (500+ samples recommended)
2. **Data yang konsisten**: Musuh jalan lurus, tidak belok-belok
3. **Variasi situasi**: Berbagai jarak, sudut, dan kecepatan
4. **Re-train model** setelah ada data baru

## ⚙️ Requirements

```bash
pip install pandas numpy scikit-learn joblib
```

## 🔧 Troubleshooting

**Q: Model tidak akurat?**
- Pastikan data cukup banyak (min 200 samples)
- Cek apakah enemy_speed konsisten (max speed yang benar)
- Training ulang dengan parameter yang berbeda

**Q: Error saat load model?**
- Pastikan file .pkl ada di folder yang sama
- Pastikan versi scikit-learn sama

**Q: Hasil offset terlalu besar/kecil?**
- Cek satuan offset_x di data (harus konsisten)
- Normalisasi ulang jika perlu

## ⚠️ Disclaimer

Tool ini untuk tujuan edukasi dan analisis data. Gunakan dengan bijak dan ikuti terms of service game.

## 📝 Notes

- Model menggunakan Random Forest Regression
- Tidak meng-hook atau memodifikasi game
- Perhitungan berdasarkan data historis, bukan hukum fisika
- Akurasi tergantung kualitas dan kuantitas data training
