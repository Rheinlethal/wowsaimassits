# Coefficient (coef): [0.30738513 0.17150275 0.04636734 0.00121224]
# Intercept: -8.441058642326574

Coefficient=[ 0.30738513, 0.17150275, 0.04636734, 0.00121224]
Intercept= -8.441058642326574

distance=20
enemy_speed=21
angle=0
shell_velocity=880
maximum_dispersion=210

lead=(Coefficient[0]*distance)+(Coefficient[1]*enemy_speed)+(Coefficient[2]*angle)+(Coefficient[3]*shell_velocity)+(Intercept)
print(lead)