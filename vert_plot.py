import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = "Vert_RT_data.pkl"

dataframe = pd.read_pickle(filename)
currents = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]


R_S = "Sample Resistance, R_S (Ohmns)"
R_S_E = "Error in R_S, alpha_R_S (Ohms)"
A = "Angle, theta (degrees)"
A_E = "Error in theta, alpha_theta (degrees)"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "11"

fig, ax = plt.subplots(1, 1)

num_plots = 7
ax.set_prop_cycle('color',plt.cm.winter(np.linspace(0,1,num_plots)))
#ax2.set_prop_cycle('color',plt.cm.winter(np.linspace(0,1,num_plots)))
#ax1.set_ylim(0.8, 1.4)
#ax2.set_ylim(0.8, 1.4)

for i in currents:
    slice = dataframe.loc[dataframe["current"] == i]
    ax.plot(slice[A], slice[R_S], label = ('%.2f'%i +'A'), linewidth = 1)

#for i in currents:
#    slice = dataframe2.loc[dataframe2["current"] == i]
#    ax2.plot(slice[A], slice[R_S], label = ('%.2f'%i + 'A'), linewidth = 1)

fig.text(0.5, 0.04, "Angle (degrees)", ha='center')
fig.text(0.04, 0.5, "Sample Resistance (ohms)", va='center', rotation='vertical')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True, ncol=1, prop = {'size':8})
#plt.legend(loc='upper left', bbox_to_anchor=(-1.05, 1), prop={'size':8})
#plt.subplots_adjust(wspace=0.05, hspace=0)
#plt.tight_layout(pad = 7)
#plt.show()
plt.savefig("vert_plot", dpi = 1000)
