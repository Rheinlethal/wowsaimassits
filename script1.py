import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# ===== LOAD DATASET =====
df = pd.read_csv("raw_data.csv")

# ===== SPLIT FITUR & TARGET =====
X = df[["shell_travel_time", "distance", "angle", "max_enemy_speed"]]
y = df["offset_x"]

# ===== SPLIT TRAIN & TEST =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== TRAIN MODEL =====
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# ===== SAVE MODEL =====
joblib.dump(model, "offset_model.pkl")

print("Model udah kelar dilatih & disimpen jadi offset_model.pkl ✨")
