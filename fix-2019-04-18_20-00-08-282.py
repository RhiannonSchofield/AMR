import pandas as pd

FILENAME = "Vert_RT_data.pkl"

def fix(index):
    before = df.iloc[index-1]["Angle, theta (degrees)"]
    if index < 27:
        after = df.iloc[index+1]["Angle, theta (degrees)"]
    else:
        after = 0

    if index == 0:
        before = -after
    if index == 27:
        after = 2*df.iloc[index]["Angle, theta (degrees)"] - before

    df.iloc[index]["Error in theta, alpha_theta (degrees)"] = after - before

df = pd.read_pickle(FILENAME)

slice = df.loc[df["current"] == 0.45]
slice.iloc[26]["Angle, theta (degrees)"] = 360

nans = df.loc[df["Error in theta, alpha_theta (degrees)"].isnull()]
indices = nans.index
print nans
for index in indices:
    fix(index)
print slice.loc[slice["Angle, theta (degrees)"]]

df.to_pickle(FILENAME)
