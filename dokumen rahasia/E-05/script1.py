import pandas as pd
import numpy as np

class Kalman1D:
    def __init__(self, process_var=1e-3, meas_var=1e-1):
        self.x = 0.0          # state (speed)
        self.P = 1.0          # covariance
        self.Q = process_var  # process noise
        self.R = meas_var     # measurement noise
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.initialized = True

        # prediction
        self.P = self.P + self.Q

        # Kalman gain
        K = self.P / (self.P + self.R)

        # correction
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P

        return self.x


def estimate_actual_speed(df):
    kf = Kalman1D()

    df["actual_speed"] = 0.0

    for i in range(1, len(df)):
        dt = df.loc[i, "shell_travel_time"]
        if dt <= 0:
            continue

        # estimate speed from offset_x change
        dx = df.loc[i, "offset_x"] - df.loc[i - 1, "offset_x"]
        meas_speed = abs(dx) / dt

        # limit by max_enemy_speed
        meas_speed = min(meas_speed, df.loc[i, "max_enemy_speed"])

        filtered_speed = kf.update(meas_speed)
        filtered_speed = min(filtered_speed, df.loc[i, "max_enemy_speed"])

        df.loc[i, "actual_speed"] = filtered_speed

    return df


if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "output_speed.csv"

    df = pd.read_csv(input_file)
    df = estimate_actual_speed(df)
    df.to_csv(output_file, index=False)

    print("Done! File saved:", output_file)
