import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

infile = sys.argv[1] 
outfile = sys.argv[2]

df = pd.read_pickle(infile)

R_S = "Sample Resistance, R_S (Ohmns)"
R_S_E = "Error in R_S, alpha_R_S (Ohms)"
A = "Angle, theta (degrees)"
A_E = "Error in theta, alpha_theta (degrees)"

fig, ax = plt.subplots()

for current in df.current.unique():
    slice = df.loc[df.current == current]
    ax.plot(slice[A], slice[R_S], linewidth=1, color="red")

ax.set_xlabel("Angle (Degrees)")
ax.set_ylabel("Sample Resistance (Ohms)")

plt.savefig(outfile)
