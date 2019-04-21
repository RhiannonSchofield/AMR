from __future__ import print_function, division
import numpy as np
import os
import re
import pandas as pd
import traceback
import matplotlib.pyplot as plt

def read_data(folder_path):
    """Reads all data from the target folder and filenames"""
    amr_dir = os.listdir(folder_path)
    current_i = np.zeros([len(amr_dir)])
    itr_i = np.zeros([len(amr_dir)])
    LN_i = np.zeros([len(amr_dir)])

    # Pull current and angle number from filenames
    for i in np.arange(0, len(amr_dir)):
        re_current_i = re.findall('_(\d*[p]\d*)_', amr_dir[i])
        rep_re_I = re.sub('[p]', '.', re_current_i[0])
        current_i[i] = float(rep_re_I)

        re_anglenum_i = re.findall('[\dG]_(.*)\.', amr_dir[i])
        rep_re_itr_1 = re.sub('p', '.', re_anglenum_i[0])
        rep_re_itr_2 = re.sub('(HPMIN)', '99', rep_re_itr_1)
        rep_re_itr_3 = re.sub('(HPMAX)', '100', rep_re_itr_2)
        itr_i[i] = float(rep_re_itr_3)

        if re.search("(LN)", amr_dir[i]):
            LN_i[i] = 1

    # Store file names in DF and sort by current then angle number
    dir_data = {'Filename': amr_dir, 'Current, I (A)': current_i, 'Angle Number': itr_i, 'Liquid Nitrogen': LN_i}
    dir_DF = pd.DataFrame(data=dir_data)
    dir_DF = dir_DF.sort_values(['Liquid Nitrogen', 'Current, I (A)', 'Angle Number'])
    dir_DF = dir_DF.reset_index(drop=True)

    return dir_DF


def check_data(dir_DF):
    """Checks all expected data is there and raise an exception if it isn't (ignores maximum and minimum calibrations"""
    cal_file_inds = np.any(np.stack([dir_DF['Angle Number'].values == 99, dir_DF['Angle Number'].values == 100], axis=1), axis=1)
    if np.any(cal_file_inds):
        num_cal_files = sum(cal_file_inds)
        print("%d calibration files detected" % num_cal_files)
        dir_DF = dir_DF[~cal_file_inds]

    uniq_I, uniq_I_counts = np.unique(dir_DF['Current, I (A)'].values, return_counts=True)
    uniq_angs, uniq_angs_counts = np.unique(dir_DF['Angle Number'].values, return_counts=True)

    if np.ptp(uniq_angs_counts) != 0 or np.ptp(uniq_I_counts) != 0:  # Raises error if ranges of counts are not 0
        raise Exception("Missing data")

    return uniq_I, uniq_angs, cal_file_inds

def import_data(dir_DF, folder_path, init_col_names, pts_per_itr, new_cols):
    """Imports the data into a 4D array and averages over repeat measurements, as well as maximum/minimum calibration
    files"""
    all_data = np.zeros([pts_per_itr, len(init_col_names), len(dir_DF)])  # Initialise 4D data array

    # Import data into 4D array
    for i in np.arange(0, len(dir_DF)):
        temp_DF = pd.read_table(folder_path + dir_DF['Filename'][i], header=None, names=init_col_names)
        try:
            all_data[:, :, i] = temp_DF.values
        except Exception:
            traceback.print_exc()
            print(dir_DF['Filename'][i])  # Print filename causing the problem
    print('Data Imported')

    # Average over iterations
    new_data = np.zeros([len(new_cols), len(dir_DF)])
    new_data[[0, 2, 6, 8, 12, 14], :] = np.mean(all_data[:, 1:, :], axis=0)  # Calculate averages (doesn't operate on
    # time column, hence the 1:)
    new_data[[1, 3, 7, 9, 13, 15], :] = np.std(all_data[:, 1:, :], axis=0, ddof=1)  # Calculate std devs on averages
    new_data[[4, 10, 16], :] = new_data[[0, 6, 12], :] / new_data[[2, 8, 14], :]  # Calculate resistances
    # Calculate stddevs on resistances
    new_data[[5, 11, 17], :] = \
        new_data[[4, 10, 16], :] * np.sqrt((new_data[[1, 7, 13], :] / new_data[[0, 6, 12], :]) ** 2 +
              (new_data[[3, 9, 15], :] / new_data[[2, 8, 14], :]) ** 2)
    print('Averages Calculated')

    return all_data, new_data


def separate_data(dir_DF, new_data, uniq_I, uniq_angs, cal_file_inds):
    """Separates data into room temperature, Liquid nitrogen temperature, and their respective max/min calibrations"""
    LN_inds = dir_DF["Liquid Nitrogen"].values == 1
    temp_RT = new_data[:, ~LN_inds & ~cal_file_inds]
    temp_LN = new_data[:, LN_inds & ~cal_file_inds]
    c = 0

    RT_data = np.zeros([len(new_data), len(uniq_I), len(uniq_angs)])
    LN_data = np.zeros([len(new_data), len(uniq_I), len(uniq_angs)])
    RT_cal_data = np.zeros([len(new_data), len(uniq_I), 2])
    LN_cal_data = np.zeros([len(new_data), len(uniq_I), 2])
    for i in np.arange(0, len(uniq_I)):
        for j in np.arange(0, len(uniq_angs)):
            if temp_LN.size != 0:
                LN_data[:, i, j] = temp_LN[:, c]
            if temp_RT.size != 0:
                RT_data[:, i, j] = temp_RT[:, c]
            c = c + 1

        # Find calibration indices for LN at this current (and store data)
        LN_is_cal = np.all(np.vstack([dir_DF["Current, I (A)"].values == uniq_I[i], cal_file_inds, LN_inds]), axis=0)
        if np.sum(LN_is_cal) == 2:
            LN_cal_data[:, i, :] = new_data[:, LN_is_cal]

        # Find calibration indices for RT at this current (and store data)
        RT_is_cal = np.all(np.vstack([dir_DF["Current, I (A)"].values == uniq_I[i], cal_file_inds, ~LN_inds]),
                           axis=0)
        if np.sum(RT_is_cal) == 2:
            RT_cal_data[:, i, :] = new_data[:, RT_is_cal]

    data_dict = {'Liquid Nitrogen': {'Data': LN_data, 'Calibration': LN_cal_data},
                 'Room Temperature': {'Data': RT_data, 'Calibration': RT_cal_data}}

    # Delete entries where no data is present
    labels = data_dict.keys()
    for k in np.arange(0, len(data_dict)):
        if np.sum(data_dict[labels[k]]['Data']) == 0:
            del data_dict[labels[k]]

    return data_dict

def hp_cal(data_dict):
    """Calculates angle data based on hall probe values"""
    labels = data_dict.keys()

    for i in np.arange(0, len(data_dict)):
        temp = labels[i]
        hall_data = data_dict[temp]['Data'][4, :, :]
        if hall_data.all() == np.zeros(hall_data.shape).all():
            continue  # Exits loop if no data for a particular run is present
        hall_errors = data_dict[temp]['Data'][5, :, :]
        hall_cal = data_dict[temp]['Calibration'][4, :, :]
        hall_calerr = data_dict[temp]['Calibration'][5, :, :]

        # Replace bad/missing calibration values with max/min from dataset (error is distance to nearest point)
        no_min_inds = np.any(np.stack([np.min(hall_data, axis=1) <
                                       hall_cal[:, 0], hall_cal[:, 0] == 0], axis=1), axis=1)
        no_max_inds = np.any(np.stack([np.max(hall_data, axis=1) >
                                       hall_cal[:, 1], hall_cal[:, 1] == 0], axis=1), axis=1)
        if sum(no_min_inds):
            hall_cal[no_min_inds, 0] = np.min(hall_data[no_min_inds, :], axis=1)
            min_diff = abs(hall_data[no_min_inds, :] - np.min(hall_data[no_min_inds, :], axis=1).reshape(-1, 1))
            hall_calerr[no_min_inds, 0] = np.sort(min_diff, axis=1)[:, 1]

        if sum(no_max_inds):
            hall_cal[no_max_inds, 1] = np.max(hall_data[no_max_inds, :], axis=1)
            max_diff = abs(hall_data[no_max_inds, :] - np.max(hall_data[no_max_inds, :], axis=1).reshape(-1, 1))
            hall_calerr[no_max_inds, 1] = np.sort(max_diff, axis=1)[:, 1]

        # Calculate angles with errors
        A = (hall_cal[:, 0] - hall_cal[:, 1]).reshape(-1, 1) / 2  # parameter in model of y = Acosx + B
        alpha_A = np.sqrt(np.sum(hall_calerr**2, axis=1)).reshape(-1, 1) / 2
        B = (hall_cal[:, 0] + hall_cal[:, 1]).reshape(-1, 1) / 2
        alpha_B = alpha_A
        cos_angs = np.round((hall_data - B) / A, 15)
        alpha_cos_angs = abs(cos_angs) * np.sqrt(
            (hall_errors ** 2 + alpha_B ** 2) / (hall_data - B) ** 2 + (alpha_A / A) ** 2)
        angs = np.rad2deg(np.arccos(cos_angs))
        angs = remove_degen(angs)  # Removes angle degeneracy
        cos_angs[abs(cos_angs) == 1] = np.nan  # Remove calibration values (used for calibration, so not valid data
        alpha_angs = np.rad2deg(alpha_cos_angs / np.sqrt(1 - cos_angs**2))  # Error

        # Store angle data
        data_dict[temp]['Data'] = np.concatenate((data_dict[temp]['Data'], angs.reshape((1,) + angs.shape),
                                                  alpha_angs.reshape((1,) + alpha_angs.shape)))

    return data_dict


def remove_degen(angs):
    """Removes the degeneracy caused by arccos whilst preserving the order of the data"""
    max_mins = np.diff(np.sign(np.diff(angs)))  # Finds where there are sign changes in the differences between points
    max_mins_pad = np.pad(max_mins, ((0, 0), (1, 1)), 'edge') + \
                   np.pad(max_mins, ((0, 0), (2, 0)), 'constant', constant_values=(0, 0)) + \
                   np.pad(max_mins[:, 1:], ((0, 0), (0, 3)), 'constant', constant_values=(0, 0)) + \
                   np.pad(max_mins[:, :-1], ((0, 0), (3, 0)), 'constant', constant_values=(0, 0)) + \
                   np.pad(max_mins, ((0, 0), (0, 2)), 'constant', constant_values=(0, 0))  # Pad the array so it it the
    # same size as the original, but also includes points either side of the turning point to iterate over

    max_loc_pos = max_mins_pad == -2  # Find possible indices of maxima
    min_loc_pos = max_mins_pad == 2  # Find possible indices of minima
    temp_sum = np.sum(min_loc_pos, axis=1) == 3
    min_loc_pos[temp_sum, :] = np.pad(min_loc_pos[temp_sum, 1:], ((0, 0), (0, 1)), 'constant', constant_values=(0, 0)) + \
                  np.pad(min_loc_pos[temp_sum, :-1], ((0, 0), (1, 0)), 'constant', constant_values=(0, 0))
    new_min_loc_pos = np.zeros(min_loc_pos.shape, dtype=bool)
    new_min_loc_pos[:, :] = min_loc_pos[:, :]

    # Insert possible missing minima
    min_loc_sum = np.sum(new_min_loc_pos, axis=1)
    for i in np.arange(0, len(min_loc_sum)):
        if min_loc_sum[i] != 10:
            min_temp_inds = np.where(new_min_loc_pos[i, :])
            if min_temp_inds[0].size == 0:
                new_min_loc_pos[i, :5] = True
                new_min_loc_pos[i, -5:] = True
            elif np.average(min_temp_inds[0]) > new_min_loc_pos.size / (2*len(new_min_loc_pos)):
                new_min_loc_pos[i, :5] = True
            else:
                new_min_loc_pos[i, -5:] = True

    # Find actual minima
    diff_std_min = np.zeros([int(np.sum(new_min_loc_pos)/5), 4])
    min_angs = angs[new_min_loc_pos].reshape(-1, 5)  # Create permanent array of potential minimum angles
    temp_angs = np.zeros(min_angs.shape)  # Initialise array of temporary angles
    no_change_std = np.std(np.diff(min_angs, axis=1), axis=1)
    for i in np.arange(1, 5):
        temp_angs[:, :] = min_angs[:, :]  # Copy values of minimum angles
        temp_angs[:, i:] = - min_angs[:, i:]
        temp_diff = np.diff(temp_angs, axis=1)
        diff_std_min[:, i - 1] = np.std(temp_diff, axis=1)

    # Correct values before minima (inc minima)
    min_min_dev = np.isin(diff_std_min, np.min(diff_std_min, axis=1))  # Find minimum std dev
    double_sum_min = np.sum(min_min_dev, axis=1) == 2
    if sum(double_sum_min):
        first_true_min = np.where(min_min_dev[double_sum_min, :])
        min_min_dev[double_sum_min, first_true_min[1][0:-1:2]] = False  # Remove double trues for non-calibrated sets
    keep_no_change = no_change_std < np.min(diff_std_min, axis=1)
    orig_mins = np.isin(min_angs, angs[min_loc_pos].reshape(-1, 5)).all(axis=1)
    override = np.stack([keep_no_change, ~orig_mins], axis=1).all(axis=1)
    min_min_dev[override, :] = False
    pad_min_dev = np.pad(min_min_dev, ((0, 0), (1, 0)), 'constant', constant_values=0)  # Now same size as min_angs
    min_loc = np.isin(angs, min_angs[pad_min_dev])  # Returns boolean array of min locations
    min_inds = np.where(min_loc)  # Returns indices of minima as two tuples
    for i in np.arange(0, len(min_inds[0])):
        if min_inds[1][i] < new_min_loc_pos.size / (2*len(new_min_loc_pos)):
            angs[min_inds[0][i], :min_inds[1][i]] = - angs[min_inds[0][i], :min_inds[1][i]]
        elif min_inds[1][i] > new_min_loc_pos.size / (2*len(new_min_loc_pos)):
            angs[min_inds[0][i], min_inds[1][i]:] = - angs[min_inds[0][i], min_inds[1][i]:]

    # Find actual maxima
    diff_std_max = np.zeros([len(angs), 4])
    import sys
    try:
        max_angs = angs[max_loc_pos].reshape(-1, 5)
    except ValueError:
        print (max_loc_pos)
        print (max_mins_pad)
        print (max_mins)
        sys.exit()
        #max_angs = angs[max_loc_pos].reshape(-1, 5)  # Create permanent array of potential maximum angles
    temp_angs = np.zeros(max_angs.shape)  # Initialise array of temporary angles
    for i in np.arange(1, 5):
        temp_angs[:, :] = max_angs[:, :]  # Copy values of maximum angles
        temp_angs[:, i:] = 360 - max_angs[:, i:]
        temp_diff = np.diff(temp_angs, axis=1)
        diff_std_max[:, i - 1] = np.std(temp_diff, axis=1)

    # Correct values after maxima (inc maxima)
    min_max_dev = np.isin(diff_std_max, np.min(diff_std_max, axis=1))  # Find minimum std dev
    double_sum_max = np.sum(min_max_dev, axis=1) == 2
    if sum(double_sum_max):
        first_true_max = np.where(min_max_dev[double_sum_max, :])
        min_max_dev[double_sum_max, first_true_max[1][0:-1:2]] = False  # Remove double trues for non-calibrated sets
    pad_max_dev = np.pad(min_max_dev, ((0, 0), (1, 0)), 'constant', constant_values=0)  # Now same size as max_angs
    max_loc = np.isin(angs, max_angs[pad_max_dev])  # Returns boolean array of max locations
    max_inds = np.where(max_loc)  # Returns indices of maxima as two tuples
    for i in np.arange(0, len(max_inds[0])):
        angs[max_inds[0][i], max_inds[1][i]:] = 360 - angs[max_inds[0][i], max_inds[1][i]:]

    return angs


def diagnostics_plot1(data_dict, dir_DF, uniq_I, uniq_angs, cal_file_inds):
    """Produces diagnostic plots to check data calibrations and fix any problems"""
    labels = data_dict.keys()
    for i in np.arange(0, len(data_dict)):
        temp = labels[i]
        data = data_dict[temp]['Data']

        # Only plot non-calibration data for each temperature
        if temp == 'Liquid Nitrogen':
            temp_DF = dir_DF[np.all(np.vstack([dir_DF['Liquid Nitrogen'], ~cal_file_inds]), axis=0)]
        elif temp == 'Room Temperature':
            temp_DF = dir_DF[np.all(np.vstack([dir_DF['Liquid Nitrogen'] == 0, ~cal_file_inds]), axis=0)]

        title_array = temp_DF['Filename'].reshape(len(uniq_I), len(uniq_angs))  # array of titles for plot

        for j in np.arange(0, int(data.shape[1])):
            data_j = data[:, j, :]
            title = re.findall('(.*)_\d[.]lvm', title_array[j, 0])[0]  # Remove angle number and .lvm extension

            fig = plt.figure()
            ax1 = plt.subplot(311)
            plt.title(title)
            plt.ylabel('R_S')
            ax2 = plt.subplot(312)
            plt.ylabel('R_T')
            ax3 = plt.subplot(313)
            plt.ylabel('R_H')
            plt.xlabel('Angle')

            scatter1 = ax1.scatter(data_j[18, :], data_j[16, :])
            scatter2 = ax2.scatter(data_j[18, :], data_j[10, :])
            scatter3 = ax3.scatter(data_j[18, :], data_j[4, :])

            # Shows angle number (to make sure degeneracy manipulation hasn't reordered the data)
            for k in np.arange(0, len(data_j[16, :])):
                ax1.annotate(k + 1, (data_j[18, k], data_j[16, k]))

            plt.show()

def write_data(save_path, data_dict, col_names, uniq_I):
    """Writes data to an excel spreadsheet and returns the data in a readable format as a dictionary of dataframes"""
    # Initialise workbook for data
    wb_name = raw_input("Please enter a filename to store data: ")
    if re.search('xlsx', wb_name) == None:  # Add .xlsx suffix if it isn't already there
        wb_rmov = re.sub('\..*', '', wb_name)
        wb_name = wb_rmov + '.xlsx'

    writer = pd.ExcelWriter(save_path + wb_name)
    workbook = writer.book

    # Create empty dataframe from data dictionary
    temp_i = data_dict.keys()
    data_dict_DF = {'Room Temperature': {'Data': {}, 'Calibration': {}}, 'Liquid Nitrogen': {'Data': {}, 'Calibration': {}}}

    # Copy columns and append angles to non calibration set
    new_col_names = [''] * len(col_names)
    new_col_names[:] = col_names[:]
    new_col_names.extend(['Angle, theta (degrees)', 'Error in theta, alpha_theta (degrees)'])

    for i in np.arange(0, len(data_dict)):
        temp = temp_i[i]
        for j in np.arange(0, len(uniq_I)):
            # Shorter temperature labels for sheetnames
            if temp == 'Liquid Nitrogen':
                short_temp = 'LN_'
            else:
                short_temp = 'RT_'

            current_label = 'I_' + re.sub('\.', 'p', str(uniq_I[j])) + 'A'  # Label for dataframe
            temp_DF = pd.DataFrame(data_dict[temp]['Data'][:, j, :].T, index=None,
                                     columns=new_col_names, copy=True)  # Convert to dataframe format for each current
            sheet_name = short_temp + current_label
            temp_DF.to_excel(writer, sheet_name, index=False)  # Store data
            data_dict_DF[temp]['Data'][current_label] = temp_DF

            cal_DF = pd.DataFrame(data_dict[temp]['Calibration'][:, j, :].T, index=None,
                                   columns=col_names, copy=True)  # Convert to dataframe format for each current
            cal_sheet_name = 'Cal_' + short_temp + current_label
            cal_DF.to_excel(writer, cal_sheet_name, index=False)  # Store data
            data_dict_DF[temp]['Calibration'][current_label] = cal_DF

            # Adjust column width for data
            column_no_all = temp_DF.shape[1]
            for a in np.arange(0, column_no_all, 1, dtype=int):
                worksheet = writer.sheets[sheet_name]
                worksheet.set_column(a, a, len(new_col_names[a]))

            # Adjust column width for calibration
            column_no_all = cal_DF.shape[1]
            for a in np.arange(0, column_no_all, 1, dtype=int):
                worksheet = writer.sheets[cal_sheet_name]
                worksheet.set_column(a, a, len(col_names[a]))

    writer.save()  # Save data
    print('Data saved to ' + save_path + wb_name)

    return data_dict_DF
