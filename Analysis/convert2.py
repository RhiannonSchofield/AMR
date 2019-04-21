import pandas as pd

CURRENTS = {
    "I_0p15A.pkl":  0.15,
    "I_0p25A.pkl":  0.25,
    "I_0p35A.pkl":  0.35,
    "I_0p45A.pkl":  0.45,
    "I_0p55A.pkl":  0.55,
    "I_0p65A.pkl":  0.65,
    "I_0p75A.pkl":  0.75
}

FOLDERS = ["RT_data/", "RT_cali/"]
OUTFILES = ["Vert_RT_data.pkl", "Vert_RT_cali.pkl"]

for folder, outfile in zip(FOLDERS, OUTFILES):
    outdata = []
    for infile, current in CURRENTS.items():
        dataframe = pd.read_pickle(folder + infile)

        # add 'current' column
        dataframe["current"] = current

        outdata.append(dataframe)

    # concatenate dataframes and save to single file
    out = pd.concat(outdata, ignore_index=True)
    out.to_pickle(outfile)
