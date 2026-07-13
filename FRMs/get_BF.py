import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

valid_FRM_HUs_file = 'valid_FRMs.pckl'
f = open(valid_FRM_HUs_file, 'rb')
obj = pickle.load(f)
f.close()

datasets = ['NS3', 'NS3_PEG', 'NS2_include_single']

min_F = 500
max_F = 23000
steps_per_octave = 8
n_steps = int(np.floor(steps_per_octave*np.log2(max_F/min_F)))
frequencies = min_F*(2**(np.arange(n_steps + 1)/steps_per_octave))

exclude_FRMs = False
set_low_to_0 = True
BF_method = 'peak'

BFs = {}

for dataset_ID in datasets:
    dataset_FRM_file = dataset_ID + '_FRMs.pckl'
    f = open(dataset_FRM_file, 'rb')
    dataset_FRM_obj = pickle.load(f)
    f.close()
    
    dataset_FRMs = dataset_FRM_obj[0]
    FRM_params = dataset_FRM_obj[1]
    
    dataset_valid_HUs = obj[dataset_ID]
    
    n_lambdas = np.size(dataset_FRMs, 0)
    lambdas = dataset_FRM_obj[2]
    lambdas = [str(lamb) for lamb in lambdas]
    
    dataset_BFs = dict.fromkeys(lambdas)
    
    for lamb in range(n_lambdas):
        lamb_value = lambdas[lamb]

        if exclude_FRMs:
            lamb_valid_HUs = dataset_valid_HUs[lamb_value]
        else:
            lamb_valid_HUs = np.arange(0, np.size(dataset_FRMs, 1), 1) # keep all HUs in analysis
        
        lamb_FRMs = dataset_FRMs[lamb]
        valid_FRMs = lamb_FRMs[lamb_valid_HUs, :, :]
        
        lamb_BFs = np.zeros((len(lamb_valid_HUs)))
        
        for HU in range(len(lamb_valid_HUs)):
            HU_FRM = valid_FRMs[HU]
            if set_low_to_0:
                HU_FRM[HU_FRM < np.nanmean(HU_FRM)] = 0
            else:
                HU_FRM[HU_FRM < 1e-2] = 0
            
            if BF_method == 'peak':
                if len(np.argwhere(HU_FRM == np.max(HU_FRM))) > 0:
                    max_idx = np.argmax(HU_FRM)
                    row, col = np.unravel_index(max_idx, HU_FRM.shape)
                    
                    lamb_BFs[HU] = frequencies[int(row)]
                else:
                    lamb_BFs[HU] = np.nan
                    
            elif BF_method == 'BW10':
                dB_threshold = np.min(np.where(HU_FRM > 0)[1])
                dB_BW10 = dB_threshold + 1
                
                FRM_BW10 = HU_FRM[:, dB_BW10]
                lamb_BFs[HU] = np.sum(frequencies*FRM_BW10)/np.sum(FRM_BW10)
            
        dataset_BFs[lamb_value] = lamb_BFs
        
    BFs[dataset_ID] = dataset_BFs

BF_params = {'exclude': exclude_FRMs, 'set_low_FRM_0': set_low_to_0, 'method': BF_method, 'FRM_params': FRM_params}

file_num = 0
file_name = 'valid_BFs_'
while True:
    if os.path.isfile(file_name + str(file_num) + '.pckl'):
        file_num += 1
    else:
        break
file_name = file_name + str(file_num) + '.pckl'

f = open(file_name, 'wb')
pickle.dump([BFs, BF_params], f)
f.close()