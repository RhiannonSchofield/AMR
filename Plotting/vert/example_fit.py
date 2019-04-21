import sys

import numpy as np
import scipy.odr.odrpack as odrpack
import pandas as pd
import matplotlib.pyplot as plt

CURRENT = 0.35

infile = sys.argv[1]
outfile = sys.argv[2]

df = pd.read_pickle(infile)

def f(params, x):
    a, b, c, d = params
    return a + b * np.tan(c*x - d)

slice = df.loc[df.current == CURRENT]
R_S = "Sample Resistance, R_S (Ohmns)"
R_S_E = "Error in R_S, alpha_R_S (Ohms)"
A = "Angle, theta (degrees)"
A_E = "Error in theta, alpha_theta (degrees)"
xs = slice[A] * np.pi / 180
ys = slice[R_S]
sx = slice[A_E] * np.pi / 180
sy = slice[R_S_E]
beta0 = [1.0, -1.0, 1.0, 2.0]
linear = odrpack.Model(f)
mydata = odrpack.RealData(xs, ys, sx=sx, sy=sy)
myodr = odrpack.ODR(mydata, linear, beta0, maxit=100)
myoutput = myodr.run()

fig, ax = plt.subplots()

ax.scatter(xs, ys, marker="+")
fitxs = np.linspace(min(xs), max(xs), 100)
fitys = f(myoutput.beta, fitxs)
ax.plot(fitxs, fitys)

fig.savefig(outfile)
