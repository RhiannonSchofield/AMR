import pandas as pd

PICKLE_FILE = "data.pkl"

# read data
data = pd.read_pickle(PICKLE_FILE)

LN = data["Liquid Nitrogen"]
RT = data["Room Temperature"]

LN_data = LN["Data"]
RT_data = RT["Data"]

LN_cali = LN["Calibration"]
RT_cali = RT["Calibration"]

# save datasets to folders
datasets = [LN_data, RT_data, LN_cali, RT_cali]
folders = ["LN_data/", "RT_data/", "LN_cali/", "RT_cali/"]

for dataset, folder in zip(datasets, folders):
    for temp, dataframe in dataset.items(): 
        filepath = "{}/{}.pkl".format(folder, temp)
        dataframe.to_pickle(filepath)

