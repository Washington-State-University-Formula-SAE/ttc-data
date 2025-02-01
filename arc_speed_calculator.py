import numpy as np


###################################################################################################################
## This Function accepts an input of arc radius, maximum acceleration of tire, dx, and arc length or arc degree. ##
## It outputs the maximum speed at which the car may travel through that arc as an individual value or list.     ##
###################################################################################################################

def arc_speed_max(radius, a, dx = 0, distance = 0, degree = False):
    
    max_speed = np.sqrt(a*radius)   # Calculating maximum speed
    
    if dx == 0 and distance == 0:    # if distance of arc not included, will print individual value
        return max_speed
    elif dx != 0 and distance != 0:   # if distance of arc (as a degree or arc length) included, will print a list of max speeds
        if degree == False: 
            n_degree = int(distance/dx)
            return np.full((1,n_degree), max_speed)
        elif degree == True:
            n_arc_length = int(distance*radius/dx)
            return np.full((1,n_arc_length), max_speed)


### Examples ###

ex1 = arc_speed_max(5, 1.2)
print(f'\n\nThe maximum speed for an arc without dx or distance known is: {ex1} \n')

ex2 = arc_speed_max(38, 1.1, dx=2, distance=5)
print(f'The maximum speed for an arc where dx and distance are known is: {ex2} \n')

ex3 = arc_speed_max(38, 1.1, dx=2, distance=0.3, degree=True)
print(f'The maximum speed for an arc where dx and distance are known, and distance is in degrees is: {ex3} \n\n')
