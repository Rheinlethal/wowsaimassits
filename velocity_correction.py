import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("data velocity_Sheet1_DATA_.csv")

# Convert range_value (km) -> meter
dist_m = df["range_value"].values * 1000

# Extract time dan tooltip velocity
times = df["time"].values
tooltips = df["velocity"].values

# Hitung effective velocity = distance / time
v_effs = dist_m / times

# Hitung scale ratio = v_eff / tooltip_velocity
scales = v_effs / tooltips

print("v_eff samples (m/s):")
print(v_effs)
print("\nMedian v_eff:", np.median(v_effs))

print("\nscale samples:")
print(scales)
print("\nMedian scale:", np.median(scales))

# Fungsi untuk koreksi velocity berdasarkan median scale
median_scale = np.median(scales)

def corrected_velocity(tooltip_v):
    return tooltip_v * median_scale

print("\nCorrected velocity for tooltip 1 sample:")
print(corrected_velocity(tooltips[2]))
