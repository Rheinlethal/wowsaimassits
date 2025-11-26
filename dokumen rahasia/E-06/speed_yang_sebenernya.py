# speed = (distance × offset_x) / (shell_travel_time × sin(angle))

import math

distance = 13.55
offset_x = 3.5
shell_travel_time = 8.65
angle = 74

speed = (distance * offset_x) / (shell_travel_time * math.sin(math.radians(angle)))
print(speed)