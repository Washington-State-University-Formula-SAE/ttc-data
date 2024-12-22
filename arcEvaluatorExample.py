import lapsim
import numpy as np
import matplotlib.pyplot as plt

arcLen = np.array([200, 50, 180, 300])
arcRad = np.array([50, 20, 60, 100])

a = lapsim.arcEvaluator()

aAnswer = a.run(100, 8, 1, 1.2, arcLen, arcRad) # entr_spd, ext_spd, dx, a, t_len, t_rad

print(f'Time to travel = {aAnswer[0]}s')
plt.plot(aAnswer[1], aAnswer[2])
plt.show()