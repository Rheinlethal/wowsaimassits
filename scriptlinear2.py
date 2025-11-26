import joblib

# ===== LOAD MODEL =====
model = joblib.load("offset_model_linear.pkl")

print("Masukin data musuh:")

shell_time = float(input("shell_travel_time : "))
distance = float(input("distance          : "))
angle = float(input("angle             : "))
enemy_speed = float(input("max_enemy_speed   : "))

# ===== BIKIN SAMPLE =====
sample = [[shell_time, distance, angle, enemy_speed]]

# ===== PREDIKSI =====
pred = model.predict(sample)[0]

print("\nPrediksi offset_x dari model Linear Regression:")
print(pred)
