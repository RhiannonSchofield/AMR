import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.odr.odrpack as odrpack

filename = "RT_data.pkl"

dataframe = pd.read_pickle(filename)
#currents = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
currents = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

R_S = "Sample Resistance, R_S (Ohmns)"
R_S_E = "Error in R_S, alpha_R_S (Ohms)"
A = "Angle, theta (degrees)"
A_E = "Error in theta, alpha_theta (degrees)"


fig = plt.figure()
ax = plt.subplot(111)

num_plots = 7
#plt.gca().set_prop_cycle('color',plt.cm.YlGnBu(np.linspace(0,1,30)))

def f(params, x):
    return params[0] + (params[1] - params[0])*np.cos(x)**2

for i in currents:
    slice = dataframe.loc[dataframe["current"] == i]
    xs = slice[A]
    ys = slice[R_S]
    sx = slice[A_E]
    sy = slice[R_S_E]
    print(i)
    beta=[1., 1.]
    linear = odrpack.Model(f)
    mydata = odrpack.RealData(xs, ys, sx=sx, sy=sy)

    for _ in range(10):
        myodr = odrpack.ODR(mydata, linear, beta)
        myoutput = myodr.run()
        beta = myoutput.beta
    print beta
    #offset = f(beta, 180)
    offset = beta[0]
    ys = ys - offset
    print offset
    ax.plot(xs, ys, label = '%s'%i)

chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.9, chartBox.height])

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True, ncol=1)
#plt.show()
plt.savefig("RT_adj_fig_2")
