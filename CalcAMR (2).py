from __future__ import print_function, division
import AMR_funcs as amr

folder_path = 'C:\\Users\\rhian\\Documents\\Uni\\Labs\\AMR\\Data\\'
save_path = folder_path
init_col_names = ['Time, t (s)', 'Hall Probe Voltage, V_H (V)', 'Hall Probe Current, I_H (A)',
                  'Thermometer Voltage, V_T (V)', 'Thermometer Current, I_T (A)', 'Sample Voltage, V_S (V)',
                  'Sample Current, I_S (A)']
new_cols = ['Hall Probe Voltage, V_H (V)', 'Error in V_H, alpha_V_H (V)', 'Hall Probe Current, I_H (A)',
            'Error in I_H, alpha_I_H (A)', 'Hall Probe Resistance, R_H (Ohms)', 'Error in R_H, alpha_R_H (Ohms)',
            'Thermometer Voltage, V_T (V)', 'Error in V_T, alpha_V_T (V)', 'Thermometer Current, I_T (A)',
            'Error in I_T, alpha_I_T (A)', 'Thermometer Resistance, R_T (Ohms)', 'Error in R_T, alpha_R_T (Ohms)',
            'Sample Voltage, V_S (V)', 'Error in V_S, alpha_V_S (V)', 'Sample Current, I_S (A)',
            'Error in I_S, alpha_I_S (A)', 'Sample Resistance, R_S (Ohmns)', 'Error in R_S, alpha_R_S (Ohms)']
pts_per_itr = 20

dir_DF = amr.read_data(folder_path)  # Reads data files
uniq_I, uniq_angs, cal_file_inds = amr.check_data(dir_DF)  # Checks all read data
all_data, new_data = amr.import_data(dir_DF, folder_path, init_col_names, pts_per_itr, new_cols)  # Import data
# Separate room temperature and liquid nitrogen temperatures as well as any respective calibration data
data_dict = amr.separate_data(dir_DF, new_data, uniq_I, uniq_angs, cal_file_inds)
data_dict = amr.hp_cal(data_dict)  # Calibrates angles using hall_probe data
#amr.diagnostics_plot1(data_dict, dir_DF, uniq_I, uniq_angs, cal_file_inds)  # Produces basic plots to check initial data
data_dict_DF = amr.write_data(save_path, data_dict, new_cols, uniq_I)  # Writes data to file

data_dict_DF.to_pickle("data.pkl")
print(data_dict_DF)
