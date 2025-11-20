# Coefficient (coef): [ 0.30536723  0.13579032  0.03708899 -0.01169925 -0.01805536]
# Intercept: 6.960388575949178

Coefficient=[ 0.30536723,  0.13579032,  0.03708899, -0.01169925, -0.01805536]
Intercept= 6.960388575949178

distance=20
enemy_speed=0
angle=0
shell_velocity=880
maximum_dispersion=210

lead=(Coefficient[0]*distance)+(Coefficient[1]*enemy_speed)+(Coefficient[2]*angle)+(Coefficient[3]*shell_velocity)+(Coefficient[4]*maximum_dispersion)+(Intercept)
print(lead)