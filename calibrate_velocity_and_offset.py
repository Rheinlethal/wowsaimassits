# calibrate_velocity_and_offset.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.interpolate import UnivariateSpline
import math

CSV = "data velocity_Sheet1_DATA_.csv"  # ganti path kalau perlu

# ---------- 1) Load and preprocess ----------
df = pd.read_csv(CSV)
# expected columns: velocity, range_value, time
# ensure numeric
df = df[['velocity','range_value','time']].dropna().astype(float)

# convert range_value (km) -> meters
df['distance_m'] = df['range_value'] * 1000.0

# compute effective velocity from observed time
df['v_eff'] = df['distance_m'] / df['time']

# optional: add simple features
df['inv_time'] = 1.0 / df['time']  # sometimes useful
df['tooltip'] = df['velocity']

print("Loaded samples:", len(df))

# ---------- 2) Quick EDA print ----------
print("\nSome v_eff samples:")
print(df[['distance_m','time','tooltip','v_eff']].head())

# ---------- 3) Modeling strategy ----------
# OPTION A: Predict v_eff from tooltip and distance (direct)
X = df[['distance_m','tooltip']].values
y = df['v_eff'].values

# train/test split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear baseline
lr = LinearRegression().fit(Xtr, ytr)
y_pred_lr = lr.predict(Xte)
print("\nLinearReg (v_eff) R2:", r2_score(yte, y_pred_lr), "MAE:", mean_absolute_error(yte, y_pred_lr))
print("Coefficients:", lr.coef_, "Intercept:", lr.intercept_)

# Ridge (regularized)
ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
y_pred_ridge = ridge.predict(Xte)
print("Ridge (v_eff) R2:", r2_score(yte, y_pred_ridge), "MAE:", mean_absolute_error(yte, y_pred_ridge))

# Random Forest (non-linear)
rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=8)
rf.fit(Xtr, ytr)
y_pred_rf = rf.predict(Xte)
print("RandomForest (v_eff) R2:", r2_score(yte, y_pred_rf), "MAE:", mean_absolute_error(yte, y_pred_rf))

# cross-validate best model quickly (optional)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
print("RF cv R2 mean:", cv_scores.mean())

# ---------- 4) Alternative: predict time directly ----------
X_time = df[['distance_m','tooltip']].values
y_time = df['time'].values
lr_time = LinearRegression().fit(X_time, y_time)
print("\nLinearReg (time) R2:", r2_score(y_time, lr_time.predict(X_time)), "MAE:", mean_absolute_error(y_time, lr_time.predict(X_time)))

# ---------- 5) Choose model and create helper functions ----------
# pick model: prefer rf if it significantly outperforms linear; else use linear for simplicity
# We'll pick rf if its R2 > linearR2 + 0.02, otherwise linear
r2_lr = r2_score(yte, y_pred_lr)
r2_rf = r2_score(yte, y_pred_rf)
model = rf if (r2_rf > r2_lr + 0.02) else lr
print("\nSelected model:", "RandomForest" if model is rf else "LinearRegression")

def predict_v_eff(tooltip_velocity, range_km):
    d_m = range_km * 1000.0
    Xq = np.array([[d_m, float(tooltip_velocity)]])
    return float(model.predict(Xq)[0])

def predict_time_from_model(tooltip_velocity, range_km):
    v = predict_v_eff(tooltip_velocity, range_km)
    return (range_km*1000.0) / v

def aim_offset(distance_km, tooltip_velocity, enemy_speed_mps, angle_deg):
    # Compute predicted flight time using model
    t = predict_time_from_model(tooltip_velocity, distance_km)
    angle_rad = math.radians(angle_deg)
    offset_m = enemy_speed_mps * math.cos(angle_rad) * t
    return offset_m, t

# ---------- 6) Demonstration on all dataset rows ----------
df['v_eff_pred'] = df.apply(lambda r: predict_v_eff(r['tooltip'], r['range_value']), axis=1)
df['time_pred'] = df['distance_m'] / df['v_eff_pred']

print("\nSample comparisons (observed vs predicted):")
print(df[['distance_m','time','time_pred','v_eff','v_eff_pred']].head(10))

# Save predictions if needed
df.to_csv("velocity_with_predictions.csv", index=False)
print("\nSaved predictions to velocity_with_predictions.csv")

# ---------- 7) Usage example ----------
# Example: compute offset for a target at 14.85 km with tooltip 823 and assumed enemy_speed 25 m/s and angle 90 deg
example = {'range_km':14.85, 'tooltip':823, 'enemy_speed_mps':25.0, 'angle_deg':90}
vpred = predict_v_eff(example['tooltip'], example['range_km'])
tpred = predict_time_from_model(example['tooltip'], example['range_km'])
offset, tcalc = aim_offset(example['range_km'], example['tooltip'], example['enemy_speed_mps'], example['angle_deg'])
print("\nExample prediction:")
print("predicted_v_eff:", vpred)
print("predicted_time:", tpred)
print("predicted_offset_m:", offset)
