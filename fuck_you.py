import pandas as pd

excel_file = 'data_vert.xlsx'
data = pd.read_excel(excel_file)
data.to_pickle("vert.pkl")
