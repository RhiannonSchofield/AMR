import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

filename1 = "LN_data.pkl"
filename2 = "RT_data.pkl"

dataframe1 = pd.read_pickle(filename1)
dataframe2 = pd.read_pickle(filename2)
currents = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


R_S = "Sample Resistance, R_S (Ohmns)"
R_S_E = "Error in R_S, alpha_R_S (Ohms)"
A = "Angle, theta (degrees)"
A_E = "Error in theta, alpha_theta (degrees)"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "16"

num_plots = 20
#ax1.set_prop_cycle('color',plt.cm.winter(np.linspace(0,1,num_plots)))
#ax2.set_prop_cycle('color',plt.cm.winter(np.linspace(0,1,num_plots)))
#ax1.set_ylim(0.8, 1.4)
#ax2.set_ylim(0.8, 1.4)


for i in currents:
    slice = dataframe1.loc[dataframe1["current"] == i]
    plt.plot(slice[A], slice[R_S], label = ('%.2f'%i +'A'), linewidth = 1, color = 'red')

for i in currents:
    slice = dataframe2.loc[dataframe2["current"] == i]
    plt.plot(slice[A], slice[R_S], label = ('%.2f'%i + 'A'), linewidth = 1, color = 'blue')

plt.ylabel("Sample Resistance (Ohms)")
plt.xlabel("Angle (Degrees)")

red_patch = mpatches.Patch(color='red', label='Liquid Nitrogen')
blue_patch = mpatches.Patch(color='blue', label='Room Temperature')
plt.legend(handles=[red_patch, blue_patch], loc='center left', bbox_to_anchor=(0, 0.55), prop = {'size':20})
plt.xticks(np.arange(-90, 420, 90))
#fig.text(0.5, 0.04, "Angle (degrees)", ha='center')
#fig.text(0.04, 0.5, "Sample Resistance (ohms)", va='center', rotation='vertical')
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True, ncol=1, prop = {'size':8})
#plt.legend(loc='upper left', bbox_to_anchor=(-1.05, 1), prop={'size':8})
#plt.subplots_adjust(wspace=0.05, hspace=0)
#plt.tight_layout(pad = 7)
#plt.show()
plt.savefig("LNRTfig", dpi = 1000)
