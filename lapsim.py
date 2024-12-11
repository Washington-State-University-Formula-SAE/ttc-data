import numpy as np
from matplotlib import pyplot as plt
pi = np.pi

# total weight of car (minus driver) (lbm)
w_car = 569
# weight of driver (lbm)
w_driver = 130
# weight bias, if less than 0.5, then the rear of the car will have more weight, if more than 0.5, then the front will have more weight
w_bias = 0.507
# length of wheelbase (in)
l = 60
# vertical center of gravity (in)
h = 15





# Run one: How quickly can the car start and stop along a straight line?

# Track components

# Determining the shape of the track

# Each element in the t_rad list corresponds to the radius of that element (in)
class single_point:
    def run(t_len_tot, t_rad):
        
        # Finding total length of track
        track = np.sum(t_len_tot)

        # discretizing track
        n = 500
        dx = track/n

        # nodespace
        nds = np.linspace(0,track,int(n+1))

        # Determining maximum lateral acceleration for every turn
        t_vel = np.sqrt(a*t_rad)

        # List showing radius at every node. Used to calculate maximum tangential acceleration
        nd_rad = np.zeros(int(n+1))

        nd_rad[0:int(np.ceil(t_len_tot[0]/dx))] = t_rad[0]
        nd_rad[int(np.ceil(t_len_tot[0]/dx)):int(np.ceil(np.sum(t_len_tot[0:2])/dx))] = t_rad[1]
        nd_rad[int(np.ceil(np.sum(t_len_tot[0:2])/dx)):int(np.ceil(np.sum(t_len_tot[0:3])/dx))] = t_rad[2]
        nd_rad[int(np.ceil((np.sum(t_len_tot[0:3]))/dx)):] = t_rad[3]

        # Determine the speed if the car accelerated for the entire length of the traffic, starting from 0 mph at node 0
        v1 = np.zeros(int(n+1))

        v1[0:int(np.ceil(t_len_tot[0]/dx))] = t_vel[0]
        v1[int(np.ceil(t_len_tot[0]/dx)):int(np.ceil(np.sum(t_len_tot[0:2])/dx))] = t_vel[1]
        v1[int(np.ceil(np.sum(t_len_tot[0:2])/dx)):int(np.ceil(np.sum(t_len_tot[0:3])/dx))] = t_vel[2]
        v1[int(np.ceil((np.sum(t_len_tot[0:3]))/dx)):] = t_vel[3]
        v1[0] = 0

        for i in np.arange(n):
            # Below section determines maximum longitudinal acceleration (a_tan) by selecting whichever is lower, engine accel. limit or tire grip limit as explained in word doc.
            a_tan_tire = np.sqrt(abs(a**2 - ((v1[i]**4)/(nd_rad[i]**2))))
            a_tan_engne = a_array[int(round(v1[int(i)]/17.6))]
            if a_tan_tire > a_tan_engne:
                a_tan = a_tan_engne
            else:
                a_tan = a_tan_tire
            if (np.sqrt(v1[int(i)]**2 + 2*a_tan*dx) < v1[int(i+1)]) or (v1[int(i+1)] == 0.):
                v1[int(i+1)] = np.sqrt(v1[int(i)]**2 + 2*a_tan*dx)

        # Determine the speed if the car deaccelerated for the entire length of the traffic, ending at 0 mph at node n
        v2 = np.zeros(int(n+1))
        v2[0:int(np.ceil(t_len_tot[0]/dx))] = t_vel[0]
        v2[int(np.ceil(t_len_tot[0]/dx)):int(np.ceil(np.sum(t_len_tot[0:2])/dx))] = t_vel[1]
        v2[int(np.ceil(np.sum(t_len_tot[0:2])/dx)):int(np.ceil(np.sum(t_len_tot[0:3])/dx))] = t_vel[2]
        v2[int(np.ceil((np.sum(t_len_tot[0:3]))/dx)):] = t_vel[3]

        for i in np.arange(1,n+1):
            a_tan = np.sqrt(abs(a**2 - ((v2[-i]**4)/(nd_rad[-i]**2))))
            if (np.sqrt(v2[int(-i)]**2 + 2*a_tan*dx) < v2[int(-i-1)]) or (v2[int(-i-1)] == 0.):
                v2[int(-i-1)] = np.sqrt(v2[int(-i)]**2 + 2*a_tan*dx)


        # Determine which value of the two above lists is lowest. This list is the theoretical velocity at each node to satisfy the stated assumptions
        v3 = np.zeros(int(n+1))
        for i in np.arange(int(n+1)):
            if v1[i] < v2 [i]:
                v3[i] = (v1[int(i)])
            else:
                v3[i] = (v2[int(i)])

        # Determining the total time it takes to travel the track by rewriting the equation v1 = v0 + a*t
        t = 0
        for i in np.arange(len(v2)-1):
            t += np.abs(v3[i+1]-v3[i])/a
    
        
        return nds, v4
