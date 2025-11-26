import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# ===== TRAIN MODEL (LINEAR REGRESSION) =====
model = LinearRegression()
model.fit(X_train, y_train)

# ===== SAVE MODEL =====
joblib.dump(model, "offset_model_linear.pkl")
print(model.coef_)
print(model.intercept_)

print("Model disimpan")
