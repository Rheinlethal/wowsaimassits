import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load('offset_model.pkl')

def predict_all_speeds(distance, angle, shell_time, max_speed):
    speeds = [max_speed, max_speed * 0.75, max_speed * 0.5, max_speed * 0.25]
    results = []

    for s in speeds:
        X = pd.DataFrame([{
            'shell_travel_time': shell_time,
            'distance': distance,
            'angle': angle,
            'actual_speed': s
        }])
        pred = model.predict(X)[0]
        results.append((s, pred))

    return results

if __name__ == '__main__':
    distance = float(input('Distance: '))
    angle = float(input('Angle: '))
    shell_time = float(input('Shell Travel Time: '))
    max_speed = float(input('Enemy Max Speed: '))

    results = predict_all_speeds(distance, angle, shell_time, max_speed)

    print("\nPredicted offset_x:")
    for spd, off in results:
        print(f"Speed {spd:.2f}: offset_x = {off:.3f}")
