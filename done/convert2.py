import pandas as pd

CURRENTS = {
    "I_0p0A.pkl":   0.00,
    "I_0p05A.pkl":  0.05,
    "I_0p1A.pkl":   0.10,
    "I_0p15A.pkl":  0.15,
    "I_0p2A.pkl":   0.20,
    "I_0p25A.pkl":  0.25,
    "I_0p3A.pkl":   0.30,
    "I_0p35A.pkl":  0.35,
    "I_0p4A.pkl":   0.40,
    "I_0p45A.pkl":  0.45,
    "I_0p5A.pkl":   0.50,
    "I_0p55A.pkl":  0.55,
    "I_0p6A.pkl":   0.60,
    "I_0p65A.pkl":  0.65,
    "I_0p7A.pkl":   0.70,
    "I_0p75A.pkl":  0.75
}

FOLDERS = ["LN_data/", "RT_data/", "LN_cali/", "RT_cali/"]
OUTFILES = ["LN_data.pkl", "RT_data.pkl", "LN_cali.pkl", "RT_cali.pkl"]

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

