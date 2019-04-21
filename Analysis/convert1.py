import pandas as pd
import pickle

PICKLE_FILE = "datas_vert.pkl"

# read data
data = pickle.load(open(PICKLE_FILE, "rb"))
#print data
#exit()
#print data.keys()
#exit()
#LN = data["Liquid Nitrogen"]
RT = data["Room Temperature"]

#LN_data = LN["Data"]
RT_data = RT["Data"]

#LN_cali = LN["Calibration"]
RT_cali = RT["Calibration"]

# save datasets to folders
datasets = [RT_data, RT_cali]
folders = ["RT_data/", "RT_cali/"]

for dataset, folder in zip(datasets, folders):
    for temp, dataframe in dataset.items():
        filepath = "{}/{}.pkl".format(folder, temp)
        dataframe.to_pickle(filepath)
