import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# === CONFIG ===
INPUT_FILE = "excel datasets tembakan_new_data_DATA_.csv"
OUTPUT_FILE = "tembakan_with_speed_group.csv"
N_CLUSTERS = 3   # 3 cluster sesuai permintaan

# === LOAD DATA ===
df = pd.read_csv(INPUT_FILE)

# Pastikan kolom yang diperlukan ada
required_cols = ["shell_travel_time", "enemy_speed_max", "offset_x"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan dalam dataset!")

# === FITUR UNTUK CLUSTERING ===
X = df[["shell_travel_time", "enemy_speed_max", "offset_x"]]

# === NORMALISASI ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === KMEANS ===
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X_scaled)

# === TAMBAHKAN KE DATAFRAME ===
df["speed_group"] = clusters

# === SAVE ===
df.to_csv(OUTPUT_FILE, index=False)
print(f"Done! File disimpan sebagai: {OUTPUT_FILE}")
print(df.head())
