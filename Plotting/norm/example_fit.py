import sys

import numpy as np
import scipy.odr.odrpack as odrpack
import pandas as pd
import matplotlib.pyplot as plt

CURRENT = 0.40

infile = sys.argv[1]
outfile = sys.argv[2]

dataframe = pd.read_pickle(infile)

def f(params, x):
    rho_per, rho_par, shift = params
    return rho_per + (rho_par - rho_per) * np.cos(x - shift)**2

slice = dataframe.loc[dataframe.current == CURRENT]
R_S = "Sample Resistance, R_S (Ohmns)"
R_S_E = "Error in R_S, alpha_R_S (Ohms)"
A = "Angle, theta (degrees)"
A_E = "Error in theta, alpha_theta (degrees)"
xs = slice[A] * np.pi / 180
ys = slice[R_S]
sx = slice[A_E] * np.pi / 180
sy = slice[R_S_E]
beta0 = [0., 2., 0.]
linear = odrpack.Model(f)
mydata = odrpack.RealData(xs, ys, sx=sx, sy=sy)
myodr = odrpack.ODR(mydata, linear, beta0, maxit=100)
myoutput = myodr.run()

fig, ax = plt.subplots()

ax.scatter(xs, ys, marker="+")
ax.plot(xs, f(myoutput.beta, xs))

fig.savefig(outfile)
