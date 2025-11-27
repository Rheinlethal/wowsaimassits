import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

print("Loading data...")
df = pd.read_csv("data_tembak.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

df = df.dropna()
print(f"After dropping NA: {df.shape}")

print("Feature engineering...")
df['angle_rad'] = np.deg2rad(df['angle'])
df['sin_a'] = np.sin(df['angle_rad'])
df['cos_a'] = np.cos(df['angle_rad'])
df['dist_time'] = df['distance'] * df['shell_travel_time']
df['time_sin'] = df['shell_travel_time'] * df['sin_a']
df['time_cos'] = df['shell_travel_time'] * df['cos_a']

features = ['shell_travel_time', 'distance', 'angle', 'angle_rad', 'sin_a', 'cos_a', 'dist_time', 'time_sin', 'time_cos']
X = df[features]
y = df['offset_x']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest model...")
rf = RandomForestRegressor(random_state=42)
params_rf = {
    'n_estimators': [100, 300],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 3, 5]
}
gs_rf = GridSearchCV(rf, params_rf, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
gs_rf.fit(X_train, y_train)

print(f"Best RF params: {gs_rf.best_params_}")

best_model = gs_rf.best_estimator_
pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"Test MAE={mae:.3f}, RMSE={rmse:.3f}")

print("Saving model...")
joblib.dump(best_model, "offset_predictor.pkl")
print("Model saved successfully to offset_predictor.pkl")
