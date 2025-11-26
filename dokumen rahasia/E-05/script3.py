import joblib
import numpy as np

model = joblib.load("offset_model.pkl")
scaler = joblib.load("offset_scaler.pkl")

def predict_offset(distance, angle, shell_time, speed):
    X = np.array([[distance, angle, shell_time, speed]])
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)[0]

if __name__ == "__main__":
    distance = float(input("distance: "))
    angle = float(input("angle: "))
    shell = float(input("shell_travel_time: "))
    enemy_speed = float(input("enemy max speed: "))

    speeds = {
        "full": enemy_speed,
        "3/4": enemy_speed * 0.75,
        "1/2": enemy_speed * 0.5,
        "1/4": enemy_speed * 0.25
    }

    print("\nPrediksi offset_x:")
    for k, v in speeds.items():
        off = predict_offset(distance, angle, shell, v)
        print(f"{k} speed → offset_x = {off:.3f}")
