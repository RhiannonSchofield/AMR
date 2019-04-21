import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = "LN_data.pkl"

dataframe = pd.read_pickle(filename)
currents = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
one = dataframe.loc[dataframe["current"] == 0.00]


R_S = "Sample Resistance, R_S (Ohmns)"
R_S_E = "Error in R_S, alpha_R_S (Ohms)"
A = "Angle, theta (degrees)"
A_E = "Error in theta, alpha_theta (degrees)"


fig = plt.figure()
ax = plt.subplot(111)

num_plots = 16
plt.gca().set_prop_cycle('color',plt.cm.winter(np.linspace(0,1,num_plots)))

for i in currents:
    slice = dataframe.loc[dataframe["current"] == i]
    ax.plot(slice[A], slice[R_S], label = '%s'%i, linewidth = 5)

chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.9, chartBox.height])

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True, ncol=1)
plt.show()
plt.savefig("LNfigure1")
