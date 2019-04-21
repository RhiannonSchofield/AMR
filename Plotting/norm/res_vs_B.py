import sys

import numpy as np
import scipy.odr.odrpack as odrpack
import pandas as pd
import matplotlib.pyplot as plt

infile = sys.argv[1]
outfile = sys.argv[2]

df = pd.read_pickle(infile)

def f(params, x):
    rho_per, rho_par, shift = params
    return rho_per + (rho_par - rho_per) * np.cos(x - shift)**2

outxs = []
outys1 = []
outys2 = []

for current in df.current.unique():
    slice = df.loc[df.current == current]
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

    outxs.append(current)
    outys1.append(myoutput.beta[0])
    outys2.append(myoutput.beta[1])

fig, ax = plt.subplots()

ax.scatter(outxs, outys1, marker="+", color="red", label="rho_per")
ax.scatter(outxs, outys2, marker="+", color="blue", label="rho_par")

fig.legend()

fig.savefig(outfile)
