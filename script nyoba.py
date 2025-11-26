# contoh_regresi_offset.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# 1) Load data
df = pd.read_csv("excel datasets tembakan_new_data_nambah actual speed_.csv")

# 2) Basic cleaning: drop NA, optionally remove outliers
df = df.dropna()

# 3) Feature engineering
# convert angle (assumed degrees) to radians; if angle is already radian, skip conversion
df['angle_rad'] = np.deg2rad(df['angle'])
df['sin_a'] = np.sin(df['angle_rad'])
df['cos_a'] = np.cos(df['angle_rad'])
# interaction features
df['dist_time'] = df['distance'] * df['shell_travel_time']
df['time_sin'] = df['shell_travel_time'] * df['sin_a']
df['time_cos'] = df['shell_travel_time'] * df['cos_a']

# Choose features and target
features = ['shell_travel_time','distance','angle','angle_rad','sin_a','cos_a','dist_time','time_sin','time_cos']
X = df[features]
y = df['offset_x']

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5) Pipeline + Ridge (baseline)
pipe_ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])
params_ridge = {
    'ridge__alpha': [0.1, 1.0, 10.0, 50.0]
}
gs_ridge = GridSearchCV(pipe_ridge, params_ridge, cv=5, scoring='neg_mean_absolute_error')
gs_ridge.fit(X_train, y_train)
print("Best Ridge params:", gs_ridge.best_params_)

# 6) Random Forest (nonlinear)
rf = RandomForestRegressor(random_state=42)
params_rf = {
    'n_estimators': [100, 300],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 3, 5]
}
gs_rf = GridSearchCV(rf, params_rf, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
gs_rf.fit(X_train, y_train)
print("Best RF params:", gs_rf.best_params_)

# 7) Evaluate both on test set
models = {
    'ridge': gs_ridge.best_estimator_,
    'rf': gs_rf.best_estimator_
}
for name, m in models.items():
    m.fit(X_train, y_train)  # ensure trained on full training fold
    pred = m.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name} MAE={mae:.3f}, RMSE={rmse:.3f}")

# Choose best model (example: rf)
best_model = models['rf'] if (mean_absolute_error(y_test, models['rf'].predict(X_test))
                              < mean_absolute_error(y_test, models['ridge'].predict(X_test))) else models['ridge']

# 8) Save model
joblib.dump(best_model, "offset_predictor.pkl")

# 9) Example of using model at runtime
def predict_offset(sample):
    # sample: dict with keys shell_travel_time,distance,angle
    s = pd.DataFrame([sample])
    s['angle_rad'] = np.deg2rad(s['angle'])
    s['sin_a'] = np.sin(s['angle_rad'])
    s['cos_a'] = np.cos(s['angle_rad'])
    s['dist_time'] = s['distance'] * s['shell_travel_time']
    s['time_sin'] = s['shell_travel_time'] * s['sin_a']
    s['time_cos'] = s['shell_travel_time'] * s['cos_a']
    Xs = s[features]
    pred = best_model.predict(Xs)[0]
    # Round/clamp to integer binocular ticks (if required)
    pred_rounded = int(round(pred))
    # optionally clamp within observed range
    min_o, max_o = int(df['offset_x'].min()), int(df['offset_x'].max())
    pred_rounded = max(min(pred_rounded, max_o), min_o)
    return pred_rounded

# Example usage:
# print(predict_offset({'shell_travel_time':0.45,'distance':120,'angle':15}))
