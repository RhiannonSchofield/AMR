import scipy as sp
import numpy as np
import pandas as pd
import scipy.odr.odrpack as odrpack
filename1 = "LN_data.pkl"
filename2 = "RT_data.pkl"

dataframe1 = pd.read_pickle(filename1)
dataframe2 = pd.read_pickle(filename2)
currents = [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

R_S = "Sample Resistance, R_S (Ohmns)"
R_S_E = "Error in R_S, alpha_R_S (Ohms)"
A = "Angle, theta (degrees)"
A_E = "Error in theta, alpha_theta (degrees)"

slice = dataframe1.loc[dataframe1["current"] == 0.05]
x = slice[A]
y = slice[R_S]
sx = slice[A_E]
sy = slice[R_S_E]

print(x, y, sx, sy)

def f(params, x):
    return params[0]*np.sin(params[1]*x + params[2]) + params[3]

linear = odrpack.Model(f)
mydata = odrpack.RealData(x, y, sx=sx, sy=sy)
myodr = odrpack.ODR(mydata, linear, beta0=[0., 2., 3., 1.])
myoutput = myodr.run()
myoutput.pprint()
#print(myoutput.beta)
