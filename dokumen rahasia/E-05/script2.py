import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("output_speed.csv")

X = df[["distance", "angle", "shell_travel_time", "actual_speed"]]
y = df["offset_x"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=300, max_depth=12)
model.fit(X_scaled, y)

joblib.dump(model, "offset_model.pkl")
joblib.dump(scaler, "offset_scaler.pkl")

print("Training done! Model saved.")
