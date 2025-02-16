import numpy as np
from matplotlib import pyplot as plt
import pickle
pi = np.pi

# Importing Car Model from car_model using PICKLE. Make sure your own file path goes into "pkl_fl_pth"
pkl_fl_pth_car = 'C:/Users/maxwe/Downloads/FSAE/2023-2024 Car/Repo/car_model.pkl'
pkl_fl_pth_eng = 'C:/Users/maxwe/Downloads/FSAE/2023-2024 Car/Repo/engine_data.pkl'

with open(pkl_fl_pth_car, 'rb') as f:
    car_model = pickle.load(f)

# total weight of car (minus driver) (lbm)
W_T = car_model['W_T']
# weight bias, if less than 0.5, then the rear of the car will have more weight, if more than 0.5, then the front will have more weight
w_bias = car_model['W_bias']
# length of wheelbase (in)
l = car_model['l']
# vertical center of gravity (in)
h = car_model['h']
# tire grip limit (G's)
a = car_model['tire_a']
# tire grip limit (in/s^2)
a_ins = a*32.2*12

# Importing Engine Data from engine_data using PICKLE
with open(pkl_fl_pth_eng, 'rb') as f:
    engine_data = pickle.load(f)

a_array = engine_data['a_array']
vel_array = engine_data['vel_array']


# Run one: How quickly can the car start and stop along a straight line?

# Track components

# Determining the shape of the track

# Each element in the t_rad list corresponds to the radius of that element (in)
class single_point:
    def __init__ (self, t_len_tot, t_rad):
        self.t_len_tot = np.array(t_len_tot)
        self.t_rad = np.array(t_rad)

    def run(self):
        
        # Finding total length of track
        track = np.sum(self.t_len_tot)

        # discretizing track
        dx = 2.5
        n = int(track/dx)
        # nodespace
        nds = np.linspace(0,track,int(n+1))

        # Determining maximum lateral acceleration for every turn
        self.t_vel = np.sqrt(a*self.t_rad)

       # List showing radius at every node. Used to calculate maximum tangential acceleration
        self.nd_rad = np.zeros(int(n+1))

        # Each line sets the maximum velocity for each 
        self.arc_beginning_node = []
        for i in np.arange(len(self.t_len_tot)):
            self.nd_rad[int(np.ceil(np.sum(self.t_len_tot[0:i])/dx)):int(np.ceil(np.sum(self.t_len_tot[0:i+1])/dx))] = self.t_rad[i]
            self.arc_beginning_node.append(int(np.ceil(np.sum(self.t_len_tot[0:i])/dx)))
        self.arc_beginning_node.append(n+1)

        self.t_rad[-1] = self.t_rad[-2]

        # Determine the speed if the car accelerated for the entire length of the traffic, starting from 0 mph at node 0
        v1 = np.zeros(int(n+1))

        for i in np.arange(len(self.t_len_tot)):
            v1[int(np.ceil(np.sum(self.t_len_tot[0:i])/dx)):int(np.ceil(np.sum(self.t_len_tot[0:i+1])/dx))] = self.t_vel[i]
        v1[0] = 0
        v1[-1] = v1[-2]

        for i in np.arange(n):
            # Below section determines maximum longitudinal acceleration (a_tan) by selecting whichever is lower, engine accel. limit or tire grip limit as explained in word doc.
            a_tan_tire = np.sqrt(abs(a**2 - ((v1[i]**4)/(self.nd_rad[i]**2))))
            a_tan_engne = a_array[int(round(v1[int(i)]/17.6))]
            if a_tan_tire > a_tan_engne:
                a_tan = a_tan_engne
            else:
                a_tan = a_tan_tire
            if (np.sqrt(v1[int(i)]**2 + 2*a_tan*dx) < v1[int(i+1)]) or (v1[int(i+1)] == 0.):
                v1[int(i+1)] = np.sqrt(v1[int(i)]**2 + 2*a_tan*dx)

        # Determine the speed if the car deaccelerated for the entire length of the traffic, ending at 0 mph at node n
        v2 = np.zeros(int(n+1))
        for i in np.arange(len(self.t_len_tot)):
            v2[int(np.ceil(np.sum(self.t_len_tot[0:i])/dx)):int(np.ceil(np.sum(self.t_len_tot[0:i+1])/dx))] = self.t_vel[i]
        v2[-1] = v2[-2]

        for i in np.arange(n,-1,-1):
            a_tan = np.sqrt(abs(a**2 - ((v2[i]**4)/(self.nd_rad[i]**2))))
            if (np.sqrt(v2[int(i)]**2 + 2*a_tan*dx) < v2[int(i-1)]) or (v2[int(i-1)] == 0.):
                v2[int(i-1)] = np.sqrt(v2[int(i)]**2 + 2*a_tan*dx)


        # Determine which value of the two above lists is lowest. This list is the theoretical velocity at each node to satisfy the stated assumptions
        v3 = np.zeros(int(n+1))
        for i in np.arange(int(n+1)):
            if v1[i] < v2 [i]:
                v3[i] = (v1[int(i)])
            else:
                v3[i] = (v2[int(i)])

        # Determining the total time it takes to travel the track by rewriting the equation x = v * t as t = x /v
        t = 0
        for i in np.arange(1, len(v2)-1):
            t += dx/v3[i]
        
        self.dx = dx
        self.n = n
        self.nds = nds
        self.v3 = v3
        self.v2 = v2
        self.v1 = v1
        self.t = t

        #plt.plot(nds, v3)
        #plt.show()
        
        return nds, v3, t
    

    def arcEvaluator(self, starting_arc, ending_arc, new_t_len, new_t_rad):
        dx = self.dx

        new_nd_rad = np.zeros(int(np.sum(new_t_len)/dx))
        for i in np.arange(len(new_t_len)):
            new_nd_rad[int(np.ceil(np.sum(new_t_len[0:i])/dx)):int(np.ceil(np.sum(new_t_len[0:i+1])/dx))] = new_t_rad[i]

        starting_node = self.arc_beginning_node[starting_arc]
        ending_node = self.arc_beginning_node[ending_arc]
        new_ending_node = starting_node + len(new_nd_rad)

        n = self.n + len(new_nd_rad) + starting_node - ending_node
        track = np.sum(self.t_len_tot)
        nds = np.linspace(0,track,int(n+1))

        t_vel = self.t_vel
        t_len_tot = self.t_len_tot
        t_rad = self.t_rad

        t_vel[starting_arc:ending_arc] = np.sqrt(a*np.array(new_t_rad))
        t_len_tot[starting_arc:ending_arc] = new_t_len
        t_rad[starting_arc:ending_arc] = new_t_rad

        nd_rad = np.zeros(int(n+1))
        nd_rad[0:starting_node] = self.nd_rad[0:starting_node]
        nd_rad[starting_node:new_ending_node] = new_nd_rad
        nd_rad[new_ending_node:n] = self.nd_rad[ending_node:self.n]
        
        v1 = np.zeros(int(n+1))
        v1[0:starting_node] = self.v1[0:starting_node]
        v1[new_ending_node:n] = self.v1[ending_node:self.n]

        old_v1 = np.array(v1)
        old_v1[starting_node:new_ending_node] = np.inf

        v2 = np.zeros(int(n+1))
        v2[0:starting_node] = self.v2[0:starting_node]
        v2[new_ending_node:n] = self.v2[ending_node:self.n]

        old_v2 = v2
        old_v2[starting_node:new_ending_node] = np.inf

        nd_vel = np.zeros(n+1)
        for i in np.arange(0, len(t_rad)):
            nd_vel[int(np.ceil(np.sum(t_len_tot[0:i])/dx)):int(np.ceil(np.sum(t_len_tot[0:i+1])/dx))] = t_vel[i]
        nd_vel[0] = 1
        nd_vel[-1] = nd_vel[-2]

        for i in np.arange(starting_node-1, n):
            # Below section determines maximum longitudinal acceleration (a_tan) by selecting whichever is lower, engine accel. limit or tire grip limit as explained in word doc.
            a_tan_tire = np.sqrt(abs(a**2 - ((v1[i]**4)/(nd_rad[i]**2))))
            a_tan_engne = a_array[int(round(v1[int(i)]/17.6))]
            if a_tan_tire > a_tan_engne:
                a_tan = a_tan_engne
            else:
                a_tan = a_tan_tire
            if (np.sqrt(v1[int(i)]**2 + 2*a_tan*dx) < nd_vel[int(i+1)]) or (nd_vel[int(i+1)] == 0.):
                v1[int(i+1)] = np.sqrt(v1[int(i)]**2 + 2*a_tan*dx)
            else:
                v1[int(i+1)] = nd_vel[int(i+1)]
            if v1[i] > old_v2[i] and old_v1[i] > old_v2[i] and i > new_ending_node:
                break

        # Determine the speed if the car deaccelerated for the entire length of the traffic, ending at 0 mph at node n
        for i in np.arange(new_ending_node, -1, -1):
            a_tan = np.sqrt(abs(a**2 - ((v2[i]**4)/(nd_rad[i]**2))))
            if (np.sqrt(v2[int(i)]**2 + 2*a_tan*dx) < nd_vel[int(i-1)]) or (nd_vel[int(i-1)] == 0.):
                v2[int(i-1)] = np.sqrt(v2[int(i)]**2 + 2*a_tan*dx)
            else:
                v2[int(i-1)] = nd_vel[int(i-1)]
            if v2[i] > old_v1[i] and old_v2[i] > old_v1[i] and i < starting_node:
                break

        # Determine which value of the two above lists is lowest. This list is the theoretical velocity at each node to satisfy the stated assumptions
        v3 = np.zeros(int(n+1))
        for i in np.arange(int(n+1)):
            if v1[i] < v2 [i]:
                v3[i] = (v1[int(i)])
            else:
                v3[i] = (v2[int(i)])

        # Determining the total time it takes to travel the track by rewriting the equation x = v * t as t = x /v
        t = 0
        for i in np.arange(1, len(v3)-2):
            if v3[i] == 0:
                v3[i] = (v3[i-1] + v3[i+1])/2
            t += dx/v3[i]

        return self.t - t
    
    
        
        
