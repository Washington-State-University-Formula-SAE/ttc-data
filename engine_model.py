import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from IPython.display import display



# Importing Engine Data
cbf650r_engine_data_file_path = "C:/Users/maxwe/Downloads/FSAE/2023-2024 Car/Repo/engine_data.xlsx"
cbf650r_engine_data = pd.read_excel(cbf650r_engine_data_file_path)

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
# setting pi as a number
pi = 3.14159
# tire grip limit (G's)
a = 1.2
# tire grip limit (in/s^2)
a_ins = a*32.2*12


# This function takes in two lists (x & y), and simplifies the data into simple points
def ave_cln(x,y):

# Averaging Process
    x_ave = list(np.round(x))
    y_ave = []
    for i in np.arange(len(x_ave)):
        j = x_ave[i]
        if j+1 in x_ave:
            y_ave.append(np.average(y[x_ave.index(j):x_ave.index(j+1)-1]))
        else:
            y_ave.append(np.average(y[x_ave.index(j):]))

    # Cleaning Process
    x_ave_cln = []
    y_ave_cln = []
    for i in np.arange(len(x_ave)-3): # The last 3 data points aren't included because they break the simulator
            if x_ave[i] not in x_ave_cln:
                x_ave_cln.append(x_ave[i])
            if y_ave[i] not in y_ave_cln:
                y_ave_cln.append(y_ave[i])

    return x_ave_cln, y_ave_cln


# Gear 1
g1_vel = list(np.round(cbf650r_engine_data['V1']))
g1_f = list(cbf650r_engine_data['F1'])

# Gear 2
g2_vel = list(np.round(cbf650r_engine_data['V2']))
g2_f = list(cbf650r_engine_data['F2'])

# Gear 3
g3_vel = list(np.round(cbf650r_engine_data['V3']))
g3_f = list(cbf650r_engine_data['F3'])

# Gear 4
g4_vel = list(np.round(cbf650r_engine_data['V4']))
g4_f = list(cbf650r_engine_data['F4'])

# Gear 5
g5_vel = list(np.round(cbf650r_engine_data['V5']))
g5_f = list(cbf650r_engine_data['F5'])

# Gear 6
g6_vel = list(np.round(cbf650r_engine_data['V6']))
g6_f = list(cbf650r_engine_data['F6'])

# Defining new gear force lists that are averaged

g1_vel, g1_f = ave_cln(g1_vel, g1_f)
g2_vel, g2_f = ave_cln(g2_vel, g2_f)
g3_vel, g3_f = ave_cln(g3_vel, g3_f)
g4_vel, g4_f = ave_cln(g4_vel, g4_f)
g5_vel, g5_f = ave_cln(g5_vel, g5_f)
g6_vel, g6_f = ave_cln(g6_vel, g6_f)

# Dataframe that organizes averaged force values for each gear at a given velocity

# Data for data frame
g_f_d = {'Gear 1' : pd.Series(g1_f, index=g1_vel),
        'Gear 2' : pd.Series(g2_f, index=g2_vel),
        'Gear 3' : pd.Series(g3_f, index=g3_vel),
        'Gear 4' : pd.Series(g4_f, index=g4_vel),
        'Gear 5' : pd.Series(g5_f, index=g5_vel),
        'Gear 6' : pd.Series(g6_f, index=g6_vel),}
g_f_df = pd.DataFrame(data=g_f_d, index=np.arange(int(max(g6_vel)))) # Creating Data Frame
g_f_df = g_f_df.fillna(0) # replacing na values with 0's so the max value can be selected
g_f_array = g_f_df.to_numpy() # converting to a numpy array

# array with all the the max force value at a given velocity
f_array = np.zeros((int(max(g6_vel))))
for i in np.arange(len(g_f_array)):
    f_array[i]= max(g_f_array[i,:]) # getting rid of any force values that aren't the highest

# divides force by car mass to get longitudinal acceleration potential for a given velociy
a_array = f_array/(w_car+w_driver) # acceleration in G's

# replaces any value over the tire limit with the tire limit. 
# Values of zero are also converted the lowest RPM force value available for 1st gear.
for i in np.arange(len(a_array)):
    if a_array[i] > a:
        a_array[i] = a
    if a_array[i] == 0:
        a_array[i] = g1_f[0]/(w_car+w_driver)

# accel values converted from G's to in/s^2
a_array = a_array*32.17*12

# Ensuring car can't accelerate past max speed:
a_array = list(a_array)

a_array += [0,0,0,0,0]

# Defining the velocity array
vel_array = np.arange(len(a_array))

# Plots the acceleration potential
plt.plot(vel_array, a_array)
plt.title('Acceleration Potential vs. speed')
plt.xlabel('Speed of Car (mph)')
plt.ylabel('Acceleration Potential (in/s^2) ')
plt.show()
        
        
# Define a sample dictionary
data = {'vel_array': vel_array,
        'a_array': a_array,}

print(len(vel_array), len(a_array))

# Pickling the dictionary
with open('C:/Users/maxwe/Downloads/FSAE/2023-2024 Car/Repo/engine_data.pkl', 'wb') as f:
    pickle.dump(data, f)