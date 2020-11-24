"""
2020.09.30
Lena Armstrong
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    Random Forest algorithm to correlate pre-ictal functional connectivity matrices with feature matrix
    created from patient structural adjacency matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:
    1.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input: file directory, array of sub_IDs, array of HUP_IDs, array of random atlases, array of standard atlases,
list of permutations, array of starting ictal times, array of ending ictal times, feature matrix, and the Functional
connectivity matrices

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output: Random Forest predictions of FC based on SC

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pickle

import numpy as np
import pandas as pd


def FC_SC_random_forest(file_directory, sub_ID_array, HUP_ID, random_atlases, standard_atlases, perm_list, start_ictal,
                        end_ictal, features, FC_list):

    # Get electrode localization
    electrode_localization_by_atlas_file_paths = []
    for s in sub_ID_array:
        for ra in random_atlases:
            for p in range(len(perm_list)):
                if p < 9:
                    file = '{0}sub-{1}/electrode_localization/electrode_localization_by_atlas/sub-{2}_electrode_' \
                           'coordinates_mni_{3}_Perm000{4}.csv'.format(
                        file_directory, s, s, ra, perm_list[p])
                else:
                    file = '{0}sub-{1}/electrode_localization/electrode_localization_by_atlas/sub-{2}_electrode_' \
                           'coordinates_mni_{3}_Perm00{4}.csv'.format(
                        file_directory, s, s, ra, perm_list[p])
                electrode_localization_by_atlas_file_paths.append(file)
        for sa in standard_atlases:
            file = '{0}sub-{1}/electrode_localization/electrode_localization_by_atlas/sub-{2}_electrode_coordinates_' \
                   'mni_{3}.csv'.format(file_directory, s, s, sa)
            electrode_localization_by_atlas_file_paths.append(file)

    # Creates labels, which are functional connectivity matrices (ictal)
    FC_file_path_array = []
    for x in range(len(standard_atlases)):
        for s in range(len(sub_ID_array)):
            file = '{0}sub-{1}/connectivity_matrices/functional/eeg/sub-{2}_{3}_{4}_{5}_functionalConnectivity.pickle'\
                .format(file_directory, sub_ID_array[s], sub_ID_array[s], HUP_ID[s], start_ictal[s], end_ictal[s])
            FC_file_path_array.append(file)
    print(len(FC_file_path_array))

    # Get functional connectivity data in pickle file format
    for FC_file_path in FC_file_path_array:
        with open(FC_file_path, 'rb') as f: broadband, alphatheta, beta, lowgamma, highgamma, \
                                    electrode_row_and_column_names, order_of_matrices_in_pickle_file = pickle.load(f)
        FC_list.append([broadband, alphatheta, beta, lowgamma, highgamma])

    '''
    Get Pre-ictal
    Preictal
    FC: Tpi x NChan x NChan
    
    Ictal
    FC: Ti x NChan x Nchan
    
    Take average of
    Preictal: NChan x Nchan
    
    
    Take average of
    Ictal: NChan x Nchan
    
    --> Subtract
    Ictal - Preictal: Nchan x Nchan
    '''

    # Get electrode localization by atlas csv file data. From get_electrode_localization.py
    electrode_localization_by_atlas = []
    for electrode_localization_by_atlas_file in electrode_localization_by_atlas_file_paths:
        electrode_localization_by_atlas.append(pd.read_csv(electrode_localization_by_atlas_file))

    # Remove electrodes in "electrode localization" not found in Functional Connectivity matrices
    electrode_localization_names = []
    for i in range(len(electrode_localization_by_atlas)):
        electrode_localization_names.append(np.array(electrode_localization_by_atlas[i]['electrode_name']))
        electrode_localization_by_atlas[i] = electrode_localization_by_atlas[i][
            np.in1d(electrode_localization_names[i], electrode_row_and_column_names)]

    # Remove electrodes in the Functional Connectivity matrices not found in "electrode localization"
    for x in range(len(electrode_localization_names)):
        not_in_functional_connectivity = np.in1d(electrode_row_and_column_names, electrode_localization_names)
        for i in range(len(FC_list)):
            for j in range(len(FC_list[i])):
                FC_list[i][j] = FC_list[i][j][not_in_functional_connectivity, :, :]
                FC_list[i][j] = FC_list[i][j][:, not_in_functional_connectivity, :]

    # Fisher z-transform of functional connectivity data. This is to take means of correlations and do correlations to
    # the structural connectivity
    for i in range(len(FC_list)):
        for j in range(len(FC_list[i])):
            FC_list[i][j] = np.arctanh(FC_list[i][j])

    # Remove structural ROIs not in electrode_localization ROIs
    electrode_ROIs = []
    structural_index = []
    for i in range(len(electrode_localization_by_atlas)):
        electrode_ROIs.append(np.unique(np.array(electrode_localization_by_atlas[i].iloc[:, 4])))
        electrode_ROIs[i] = electrode_ROIs[i][~(electrode_ROIs[i] == 0)]  # remove region 0
        structural_index.append(np.array(electrode_ROIs[i] - 1))  # subtract 1 because of python's zero indexing
    for j in range(len(features)):
        for x in range(len(features[j])):
            features[j][x] = features[j][x][structural_index[j], :]
            features[j][x] = features[j][x][:, structural_index[j]]

    # Taking average functional connectivity for those electrodes in same atlas regions
    ROIs = []
    for i in range(len(FC_list)):
        print('length FC_List', len(FC_list))
        for j in range(len(FC_list[i])):
            print('length i FC_List', len(FC_list[i]))
            ROIs.append(np.array(electrode_localization_by_atlas[i].iloc[:, 4]))
            for r in range(len(electrode_ROIs[i])):
                index_logical = (ROIs[i] == electrode_ROIs[i][r])
                index_first = np.where(index_logical)[0][0]
                index_second_to_end = np.where(index_logical)[0][1:]
                mean = np.mean(FC_list[i][j][index_logical, :, :], axis=0)
                # Fill in with mean.
                FC_list[i][j][index_first, :, :] = mean
                FC_list[i][j][:, index_first, :] = mean
                # delete the other rows and columns belonging to same region.
                FC_list[i][j] = np.delete(FC_list[i][j], index_second_to_end, axis=0)
                FC_list[i][j] = np.delete(FC_list[i][j], index_second_to_end, axis=1)
                # keeping track of which electrode labels correspond to which rows and columns
                ROIs[i] = np.delete(ROIs[i], index_second_to_end, axis=0)
            # remove electrodes in the ROI labels as zero
            index_logical = (ROIs[i] == 0)
            index = np.where(index_logical)[0]
            FC_list[i][j] = np.delete(FC_list[i][j], index, axis=0)
            FC_list[i][j] = np.delete(FC_list[i][j], index, axis=1)
            ROIs[i] = np.delete(ROIs, index, axis=0)

    features_2D = []
    labels_1D = []

    # break down feature matrix 2D - 1D FC matrix !!!
    for i in range(len(features)):
        all_depth = []
        for col in range(len(features[i])):
            for j in range(len(features[i])):  # Depth of 9
                for x in range(len(features[i][j])):
                    all_depth.append(features[i][col][x])
                features_2D.append(all_depth)

    print("length features: ", len(features_2D))
    # print(features_2D)

    for i in range(len(FC_list[0])):
        labels_1D.append(FC_list[0][j]) # only use broadband

    print("length labels: ", len(labels_1D))
    # print(labels_1D)

    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features_2D, labels_1D,
                                                                                test_size=0.25, random_state=42)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)