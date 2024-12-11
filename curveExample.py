from trackModel import two_step_curve
from matplotlib import pyplot as plt

# x and y coordinates of the start of the curve
curve_start_x = -5
curve_start_y = 5

# the angle the car is traveling when entering the curve
curve_start_angle = 4 # degrees

# the x and y coordinates of the end of the curve
curve_end_x = 5
curve_end_y = -5

# the angle the car is traveling when exiting the curve
curve_end_angle = -135 # degrees

# creating the curve object
curve = two_step_curve(curve_start_x, curve_start_y, curve_start_angle, curve_end_x, curve_end_y, curve_end_angle, 0)

# this method plots the curve on matplotlib
curve.plot_curve()

plt.axis('equal') # set axis equal
plt.show()