import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = "RT_data.pkl"

dataframe = pd.read_pickle(filename)
currents = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.55, 0.60, 0.65, 0.70, 0.75]
one = dataframe.loc[dataframe["current"] == 0.00]


R_S = "Sample Resistance, R_S (Ohmns)"
R_S_E = "Error in R_S, alpha_R_S (Ohms)"
A = "Angle, theta (degrees)"
A_E = "Error in theta, alpha_theta (degrees)"


fig = plt.figure()
f, ax = plt.subplots(2, sharex = True)

num_plots = 16
plt.gca().set_prop_cycle('color',plt.cm.YlGnBu(np.linspace(0,1,30)))

for i in currents:
    slice = dataframe.loc[dataframe["current"] == i]
    if (i <= 8):
        ax[0].plot(slice[A], slice[R_S], label = '%s'%i)
    if (i >= 9):
        ax[1].plot(slice[A], slice[R_S], label = '%s'%i)


#chartBox = ax[1].get_position()
#ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.9, chartBox.height])

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True, ncol=1)
#plt.show()
plt.savefig("figure")
