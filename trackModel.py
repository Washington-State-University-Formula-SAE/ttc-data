# The following program generates a curve consisting of two segments, each segment being of a specific radius and arclength
# The curve has specified starting and ending locations and angles.
# The first segment of the curve is reffered to as curve A, and the second as curve B.
# 
# For a curve to be generated, seven inputs are required.
#   - The first two of these inputs are the x and y coordinates of the curve
#   - The next input is the angle (in degrees) at which the curve starts (zero degrees points to the right, increasing this angle 
#     rotates it ccw)
#   - The fourth and fifth inputs respectively are the x and y coordinates where the curve must finish
#   - The sixth input is the angle the car is moving at when exiting the curve

import numpy as np
from matplotlib import pyplot as plt
import sympy as sym
import lapsim

plt_resolution = 3

# quadratic formula equation, returns 2 outputs
quad_form = lambda A, B, C : ((-B + (B**2 - 4*A*C)**0.5)/(2*A),
                              (-B - (B**2 - 4*A*C)**0.5)/(2*A))

transform_vect = np.vectorize(lambda s, v, t : s * v + t)

class two_step_curve():

    def __init__ (self, Ax, Ay, A_theta, Bx, By, B_theta, scaler):
        # defining the starting positition of the curve
        self.Ax = Ax
        self.Ay = Ay
        # defines the direction (in radians) of the center of curve A from its starting position
        self.A_theta = np.deg2rad(A_theta + 270)
        # defines the ending position of the curve
        self.Bx = Bx
        self.By = By
        # defines the direction (in radians) of the center of curve B from its ending position
        self.B_theta = np.deg2rad(B_theta + 90)

        self.A_B_ratio = 2**scaler

        # finds the 
        self.find_curve()


    def find_curve(self):
        # Unit vectors that definine the center of curves A and B from the start/end of the curve
        self.UAx = np.cos(self.A_theta) * self.A_B_ratio
        self.UAy = np.sin(self.A_theta) * self.A_B_ratio
        self.UBx = np.cos(self.B_theta)
        self.UBy = np.sin(self.B_theta)

        a = (self.UAx - self.UBx)**2 + (self.UAy - self.UBy)**2 - (1 + self.A_B_ratio)**2
        b = 2 * ((self.UAx - self.UBx)*(self.Ax - self.Bx) + (self.UAy - self.UBy)*(self.Ay - self.By))
        c = (self.Ax - self.Bx)**2 + (self.Ay - self.By)**2

        r1, r2 = quad_form(a, b, c)

        length_A1, length_B1, arc_angle_A1, arc_angle_B1 = self.get_arc_length(r1)
        length_A2, length_B2, arc_angle_A2, arc_angle_B2 = self.get_arc_length(r2)

        if length_A1 + length_B1 < length_A2 + length_B2:
            self.arc_length_A = length_A1
            self.arc_length_B = length_B1
            self.arc_angle_A = arc_angle_A1
            self.arc_angle_B = arc_angle_B1
            self.radius = r1
            self.radius_A = abs(r1 * self.A_B_ratio)
            self.radius_B = abs(r1)
        else:
            self.arc_length_A = length_A2
            self.arc_length_B = length_B2
            self.arc_angle_A = arc_angle_A2
            self.arc_angle_B = arc_angle_B2
            self.radius = r2
            self.radius_A = abs(r2 * self.A_B_ratio)
            self.radius_B = abs(r2)

    
    def get_arc_length(self, r):
        # Defines x and y coords of the point which curve A rotates around
        Ac_x = self.Ax + self.UAx * r
        Ac_y = self.Ay + self.UAy * r
        # Defines x and y coords of the point which curve B rotates around
        Bc_x = self.Bx + self.UBx * r
        Bc_y = self.By + self.UBy * r

        # Defines the starting angle of Curve A
        if r < 0:
            curve_A_start = self.A_theta
        else:
            curve_A_start = self.A_theta - np.pi

        # Defining the ending angle of Curve A
        curve_A_end = np.arccos((Bc_x - Ac_x) / abs(r + r * self.A_B_ratio))
        if Bc_y < Ac_y:
            curve_A_end = 2 * np.pi - curve_A_end

        # Finding the arclength of Curve A using the start and ending angle
        if r > 0: # If r1 is positive, the curve moves clockwise
            arc_angle_A = (curve_A_start - curve_A_end) % (2*np.pi)
        else: # If r1 is negative, the curve moves counterclockwise
            arc_angle_A = (curve_A_end - curve_A_start) % (2*np.pi)
        
        # Finding arc length of curve A
        arc_length_A = abs(arc_angle_A * r * self.A_B_ratio)

        # Defining the starting angle of Curve B
        curve_B_start = np.arccos((Ac_x - Bc_x) / abs(r + r * self.A_B_ratio))
        if Ac_y < Bc_y:
            curve_B_start = 2 * np.pi - curve_B_start
        
         # Defines the ending angle of Curve B
        if r < 0:
            curve_B_end = self.B_theta
        else:
            curve_B_end = self.B_theta - np.pi

        # Finding the arclength of Curve B using the start and ending angle
        if r < 0: # If r1 is positive, the curve moves counterclockwise
            arc_angle_B = (curve_B_start - curve_B_end) % (2*np.pi)
        else: # If r1 is negative, the curve moves clockwise
            arc_angle_B = (curve_B_end - curve_B_start) % (2*np.pi)
        
        # Finding arc length of curve B
        arc_length_B = abs(arc_angle_B * r)

        return arc_length_A, arc_length_B, arc_angle_A, arc_angle_B
    

    def plot_curve(self):
        if self.radius > 0:
            direction = 'cw'
        else:
            direction = 'ccw'
        
        # Defines x and y coords of the point which curve A rotates around
        Ac_x = self.Ax + self.UAx * self.radius
        Ac_y = self.Ay + self.UAy * self.radius
        
        # Defines the starting angle of Curve A
        if self.radius < 0:
            curve_A_start = self.A_theta
        else:
            curve_A_start = self.A_theta - np.pi
        
        self.plot_arc(Ac_x, Ac_y, curve_A_start, self.arc_angle_A, abs(self.radius) * self.A_B_ratio, direction)


        # Defines x and y coords of the point which curve B rotates around
        Bc_x = self.Bx + self.UBx * self.radius
        Bc_y = self.By + self.UBy * self.radius

        # Defines the ending angle of Curve B
        if self.radius < 0:
            curve_B_end = self.B_theta
        else:
            curve_B_end = self.B_theta - np.pi
        
        self.plot_arc(Bc_x, Bc_y, curve_B_end, self.arc_angle_B, abs(self.radius), direction)
        

    def plot_arc(self, x_center, y_center, start_angle, arc_angle, r, direction = 'ccw'):
        if direction == 'ccw':
            theta = np.linspace(start_angle, start_angle+arc_angle, int(arc_angle * r) * plt_resolution)
        elif direction == 'cw':
            theta = np.linspace(start_angle-arc_angle, start_angle, int(arc_angle * r) * plt_resolution)
        
        x = []
        y = []
        for i in theta:
            x.append(x_center + r * np.cos(i))
            y.append(y_center + r * np.sin(i))
        
        plt.plot(x, y)










class track():

    def __init__ (self, x, y):
        self.points_x = x
        self.points_y = y
        self.points = len(x)-1
        self.new_track()

    def new_track(self):
        self.angles = []

        for i in range(len(self.points_x)):
            j = i - 1
            if j < 0:
                j = self.points
            start_angle = np.arccos((self.points_x[i] - self.points_x[j]) / ((self.points_x[i] - self.points_x[j])**2 + (self.points_y[i] - self.points_y[j])**2)**0.5)
            if self.points_y[j] > self.points_y[i]:
                start_angle = 2*np.pi - start_angle
            start_angle = np.rad2deg(start_angle)
            self.angles.append(start_angle)
        
        self.track_segments = []
        for i in range(len(self.angles)-1):
            self.track_segments.append(two_step_curve(self.points_x[i], self.points_y[i], self.angles[i], self.points_x[i+1], self.points_y[i+1], self.angles[i+1], 0))
        self.track_segments.append(two_step_curve(self.points_x[self.points], self.points_y[self.points], self.angles[self.points], self.points_x[0], self.points_y[0], self.angles[0], 0))

    def plot_track(self):
        for i in self.track_segments:
            i.plot_curve()
    
    def run_sim(self, sim_type):
        arc_lengths = []
        arc_radii = []
        for i in self.track_segments:
            arc_lengths.append(i.arc_length_A)
            arc_lengths.append(i.arc_length_B)
            arc_radii.append(i.radius_A)
            arc_radii.append(i.radius_B)

        match sim_type:
            case 'single_point':
                return lapsim.single_point.run(np.array(arc_lengths), np.array(arc_radii))

    def plot_sim(self, sim_type):
        v, s = self.run_sim(sim_type)
        plt.plot(v, s)
    
    '''def new_track(self):
        angle_sums = []
        for i in range(self.points - 1):
            angle_sums.append(np.arccos((self.points_x[i + 1] - self.points_x[i])/((self.points_x[i + 1] - self.points_x[i])**2 + (self.points_y[i + 1] - self.points_y[i])**2)**0.5) * 2)
            if self.points_y[i] > self.points_y[i + 1]:
                angle_sums[i] = 2*np.pi - angle_sums[i]
        
        angle_sums.append(np.arccos((self.points_x[0] - self.points_x[self.points-1])/((self.points_x[0] - self.points_x[self.points-1])**2 + (self.points_y[0] - self.points_y[self.points-1])**2)**0.5) * 2)
        if self.points_y[self.points-1] > self.points_y[0]:
            angle_sums[self.points-1] = 2*np.pi - angle_sums[self.points-1]
        
        angle_sums = np.array(angle_sums)
        
        angle_matrix = []
        for i in range(self.points-1):
            angle_matrix.append(np.linspace(0, 0, self.points))
            angle_matrix[i][i] = 1
            angle_matrix[i][i+1] = 1
            #angle_matrix[i][self.points] = angle_sums[i]
        
        angle_matrix.append(np.linspace(0, 0, self.points))
        angle_matrix[self.points-1][self.points-1] = 1
        angle_matrix[self.points-1][0] = 1
        #angle_matrix[self.points-1][self.points] = angle_sums[self.points-1]
        
        #angle_matrix = np.matrix(angle_matrix)

        print(angle_sums)

        print(np.linalg.solve(angle_matrix, angle_sums))

        self.angles = []'''

        #print(angle_matrix)



        #for i in angle:
        #    print(i)
        #print(angle)