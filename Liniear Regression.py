import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# ============================
# 1. BACA CSV
# ============================

df = pd.read_csv("data velocity_Sheet1_DATA_.csv")

# ============================
# 2. AMBIL KOLOM FITUR & TARGET
# ============================

x = df[['range_value','velocity']].to_numpy()
z = df['time'].to_numpy()

# ============================
# 3. TRAIN MODEL
# ============================

model = LinearRegression()
model.fit(x,z)

# ============================
# 4. TAMPILIN RUMUS
# ============================

print("=== HASIL RUMUS ===")
print("Coefficient=", model.coef_)
print("Intercept=", model.intercept_)
