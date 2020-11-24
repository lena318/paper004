"""
2020.09.30
Lena Armstrong
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    Script for Random Forest algorithm to correlate pre-ictal functional connectivity matrices with feature matrix
    created from patient structural adjacency matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:
    1. Creates file directory
    2. Creates array of sub_IDs, array of HUP_IDs, array of random atlases, array of standard atlases,
    list of permutations, array of starting ictal times, array of ending ictal times
    3. Creates a feature matrix for each file
    4. Created an emtpy FC list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input: file directory, array of sub_IDs, array of HUP_IDs, array of random atlases, array of standard atlases,
list of permutations, array of starting ictal times, array of ending ictal times

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output: Random Forest predictions of FC based on SC

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import random_forest
from create_feature_matrix import create_feature_matrix

# 7 patients, 24 atlases (11 random atlases with 30 permutations & 13 standard atlases)
# File stats: 2,401 files - 2,310 random atlases (330 per patient) & 91 standard atlases (13 per patient)
# 343 files per patient for SC
# 23 files for FC
# 7,889 correlations between FC & SC
# Each structural connectivity matrix gets a label of a functional connectivity matrix,
# which is done multiple times if there were multiple seizures

# Patients
sub_ID_array = ['RID0278', 'RID0278', 'RID0278', 'RID0278', 'RID0278']
# 'RID0194',
# 'RID0320', 'RID0320', 'RID0320',
# 'RID0309', 'RID0309',
# 'RID0440', 'RID0440', 'RID0440', 'RID0440', 'RID0440',
# 'RID0502', 'RID0502', 'RID0502',
# 'RID0536', 'RID0536', 'RID0536', 'RID0536']
HUP_ID = ['HUP138_phaseII', 'HUP138_phaseII', 'HUP138_phaseII', 'HUP138_phaseII', 'HUP138_phaseII']
# 'HUP134_phaseII_D02'
# 'HUP140_phaseII_D02', 'HUP140_phaseII_D02', 'HUP140_phaseII_D02',
# 'HUP151_phaseII', 'HUP151_phaseII',
# 'HUP172_phaseII', 'HUP172_phaseII', 'HUP172_phaseII', 'HUP172_phaseII', 'HUP172_phaseII',
# 'HUP182_phaseII', 'HUP182_phaseII', 'HUP182_phaseII',
# 'HUP195_phaseII_D01', 'HUP195_phaseII_D01', 'HUP195_phaseII_D01', 'HUP195_phaseII_D01']

# Atlases
random_atlases = []  # 'RA_N0100', 'RA_N0200']
# 'RA_N0010', 'RA_N0030', 'RA_N0050', 'RA_N0075', 'RA_N0300', 'RA_N0400', 'RA_N0500', 'RA_N1000', 'RA_N2000'
standard_atlases = ['aal_res-1x1x1', 'desikan_res-1x1x1', 'DK_res-1x1x1', 'JHU_aal_combined_res-1x1x1']
'''
                    'AAL600', 'CPAC200_res-1x1x1', 'JHU_res-1x1x1',
                    'Schaefer2018_100Parcels_17Networks_order_FSLMNI152_1mm',
                    'Schaefer2018_200Parcels_17Networks_order_FSLMNI152_1mm',
                    'Schaefer2018_300Parcels_17Networks_order_FSLMNI152_1mm',
                    'Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm',
                    'Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_1mm', 
                    'Talairach_res-1x1x1'
'''

# Permutations
perm_list = list(range(1, 31))

# Pre-ictal
start_preictal = []
start_preictal.append(248432340000) # RID0278
start_preictal.append(338848220000) # RID0278
start_preictal.append(415933490000) # RID0278
start_preictal.append(429398830000) # RID0278
start_preictal.append(458393300000) # RID0278

end_preictal = []
end_preictal.append(248525740000) # RID0278
end_preictal.append(339008330000) # RID0278
end_preictal.append(416023190000) # RID0278
end_preictal.append(429498590000) # RID0278
end_preictal.append(458504560000) # RID0278

# Connectivity start time ictal
start_ictal = []
# start_ictal.append(179302933433)  # RID0194

start_ictal.append(248525740000)  # RID0278
start_ictal.append(339008330000)  # RID0278
start_ictal.append(416023190000)  # RID0278
start_ictal.append(429498590000)  # RID0278
start_ictal.append(458504560000)  # RID0278

# start_ictal.append(331979921071)  # RID0320
# start_ictal.append(340528040025)  # RID0320
# start_ictal.append(344573548935)  # RID0320

# start_ictal.append(494776000000)  # RID0309
# start_ictal.append(530011424682)  # RID0309

# start_ictal.append(402704260829)  # RID0440
# start_ictal.append(408697930000)  # RID0440
# start_ictal.append(586990000000)  # RID0440
# start_ictal.append(664976000000)  # RID0440
# start_ictal.append(692879000000)  # RID0440

# tart_ictal.append(401068946176)  # RID0502
# start_ictal.append(405830095852)  # RID0502
# start_ictal.append(707691579507)  # RID0502

# start_ictal.append(84729094008)  # RID0536
# start_ictal.append(164694572385)  # RID0536
# start_ictal.append(250710930770)  # RID0536
# start_ictal.append(286819539584)  # RID0536

# Connectivity end time ictal
end_ictal = []
#end_ictal.append(179381931054)  # RID0194

end_ictal.append(248619140000)  # RID0278
end_ictal.append(339168440000)  # RID0278
end_ictal.append(416112890000)  # RID0278
end_ictal.append(429598350000)  # RID0278
end_ictal.append(458615820000)  # RID0278

#end_ictal.append(332335094986)  # RID0320
#end_ictal.append(340813068619)  # RID0320
#end_ictal.append(344880383072)  # RID0320

#end_ictal.append(494850000000)  # RID0309
#end_ictal.append(530084532533)  # RID0309

#end_ictal.append(402756680000)  # RID0440
#end_ictal.append(408758390000)  # RID0440
#end_ictal.append(587052288685)  # RID0440
#end_ictal.append(665027219061)  # RID0440
#end_ictal.append(692937000000)  # RID0440

#end_ictal.append(401639357332)  # RID0502
#end_ictal.append(405873843036)  # RID0502
#end_ictal.append(707789139213)  # RID0502

#end_ictal.append(84848666031)  # RID0536
#end_ictal.append(164762187960)  # RID0536
#end_ictal.append(250847964650)  # RID0536
#end_ictal.append(286885777902)  # RID0536

file_directory1 = '/Users/larmstrong2020/mount/DATA/Human_Data/BIDS_processed/'

# Creates features, which are structural connectivity matrices
features = []
for s in sub_ID_array:
    for ra in random_atlases:
        for p in range(len(perm_list)):
            if p < 9:
                file = '{0}sub-{1}/connectivity_matrices/structural/{2}/sub-{3}_ses-preop3T_dwi-eddyMotionB0Corrected' \
                       '.nii.gz.trk.gz.{4}_Perm000{5}.count.pass.connectivity.mat'.format(file_directory1, s, ra, s, ra,
                                                                                          perm_list[p])
            else:
                file = '{0}sub-{1}/connectivity_matrices/structural/{2}/sub-{3}_ses-preop3T_dwi-eddyMotionB0Corrected' \
                       '.nii.gz.trk.gz.{4}_Perm00{5}.count.pass.connectivity.mat'.format(file_directory1, s, ra, s, ra,
                                                                                         perm_list[p])
            features.append(create_feature_matrix(file))
    for sa in standard_atlases:
        file = '{0}sub-{1}/connectivity_matrices/structural/{2}/sub-{3}_ses-preop3T_dwi-eddyMotionB0Corrected.nii.gz.' \
               'trk.gz.{4}.count.pass.connectivity.mat'.format(file_directory1, s, sa, s, sa)
        features.append(create_feature_matrix(file))

FC_list = []

random_forest.FC_SC_random_forest(file_directory1, sub_ID_array, HUP_ID, [], standard_atlases, [], start_ictal,
                                  end_ictal, features, FC_list)
