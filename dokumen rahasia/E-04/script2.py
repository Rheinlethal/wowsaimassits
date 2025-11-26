import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load corrected CSV
df = pd.read_csv('excel datasets tembakan_new_data_DATA_.csv')

# Features and target
X = df[['shell_travel_time', 'distance', 'angle', 'actual_speed']]
y = df['offset_x']

# Train model
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'offset_model.pkl')
