import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

infile1 = sys.argv[1] 
infile2 = sys.argv[2] 
outfile = sys.argv[3]

df1 = pd.read_pickle(infile1)
df2 = pd.read_pickle(infile2)

R_S = "Sample Resistance, R_S (Ohmns)"
R_S_E = "Error in R_S, alpha_R_S (Ohms)"
A = "Angle, theta (degrees)"
A_E = "Error in theta, alpha_theta (degrees)"

fig, ax = plt.subplots()

for df, color in [(df1, "red"), (df2, "blue")]:
    for current in df.current.unique():
        slice = df.loc[df.current == current]
        ax.plot(slice[A], slice[R_S], linewidth=1, color=color)

ax.set_xlabel("Angle (Degrees)")
ax.set_ylabel("Sample Resistance (Ohms)")

plt.savefig(outfile)
