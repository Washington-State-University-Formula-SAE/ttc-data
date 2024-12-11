from matplotlib import pyplot as plt
import numpy as np

points_x = [0, 12.6, 29.8, 33.4, 22.7, 23.8, 14.2, -1.3, -8.9, -12.9, -8.5, -22.2, -25.234, -18.875]
points_y = [50, 30.3, 26.4, 20.6, 10.3, -3.5, -14.6, -12.1, -17.8, -4.6, 9.1, 18.7, 23.98, 40.1]

plt.plot(points_x, points_y, linestyle='none', marker='o')
plt.show