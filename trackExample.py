from matplotlib import pyplot as plt
import numpy as np
import trackModel

points_x = [0, 246.3, 600.2, 668.7, 454.3, 476.5, 284.2, -26, -178.2, -260.9, -170, -444.4, -504.68, -370]
points_y = [1000, 606.7, 48.9, 412, 206.1, -70, -292.3, -242.5, -348.9, -92.4, 182.2, 374.7, 460.4, 823.4]

#plt.plot(points_x, points_y, linestyle='none', marker='o')

track = trackModel.track(points_x, points_y)
#track.plot_track()
track.plot_sim('single_point')


#plt.axis('equal')
plt.show()