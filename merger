import pandas as pd

# list file CSV yang mau digabung
files = [
    "ship_data1.csv",
    "ship_data2.csv",
    "ship_data3.csv",
    "ship_data4.csv",
    "ship_data5.csv",
]

dfs = [pd.read_csv(f) for f in files]  # baca semua CSV
merged = pd.concat(dfs, ignore_index=True)  # gabung semua

merged.to_csv("ship_datasets.csv", index=False)  # simpan hasil
print("Selesai! Semua file digabung ke merged.csv")
