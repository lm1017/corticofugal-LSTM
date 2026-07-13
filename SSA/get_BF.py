import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

datasets = ['NS3_PEG']
model_ID = 'pop_f_LSTM'

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
    dataset_FRM_file = 'FRMs/' + dataset_ID + '_' + model_ID + '_FRMs_' + '.pckl'
    f = open(dataset_FRM_file, 'rb')
    dataset_FRM_obj = pickle.load(f)
    f.close()
    
    dataset_FRMs = dataset_FRM_obj[0]
    FRM_params = dataset_FRM_obj[1]

    n_lambdas = np.size(dataset_FRMs, 0)

    lambdas = dataset_FRM_obj[2]
    lambdas = [str(lamb) for lamb in lambdas]
    
    dataset_BFs = dict.fromkeys(lambdas)
    
    for lamb in range(n_lambdas):
        lamb_value = lambdas[lamb]

        if exclude_FRMs:
            break
        else:
            lamb_valid_us = np.arange(0, np.size(dataset_FRMs, 1), 1) # keep all HUs in analysis
        
        lamb_FRMs = dataset_FRMs[lamb]
        valid_FRMs = lamb_FRMs[lamb_valid_us, :, :]
        
        lamb_BFs = np.zeros((len(lamb_valid_us)))
        
        for u in range(len(lamb_valid_us)):
            u_FRM = valid_FRMs[u]
            if set_low_to_0:
                u_FRM[u_FRM < np.nanmean(u_FRM)] = 0
            else:
                u_FRM[u_FRM < 1e-2] = 0
            
            if BF_method == 'peak':
                if len(np.argwhere(u_FRM == np.max(u_FRM))) > 0:
                    max_idx = np.argmax(u_FRM)
                    row, col = np.unravel_index(max_idx, u_FRM.shape)
                    
                    lamb_BFs[u] = frequencies[int(row)]
                else:
                    lamb_BFs[u] = np.nan
                    
            elif BF_method == 'BW10':
                dB_threshold = np.min(np.where(u_FRM > 0)[1])
                dB_BW10 = dB_threshold + 1
                
                FRM_BW10 = u_FRM[:, dB_BW10]
                lamb_BFs[u] = np.sum(frequencies*FRM_BW10)/np.sum(FRM_BW10)
            
        dataset_BFs[lamb_value] = lamb_BFs
        
    BFs[dataset_ID] = dataset_BFs

BF_params = {'exclude': exclude_FRMs, 'set_low_FRM_0': set_low_to_0, 'method': BF_method, 'FRM_params': FRM_params}

file_num = 0
file_name = 'BFs/' + dataset_ID + '_' + model_ID + '_BFs_'
while True:
    if os.path.isfile(file_name + str(file_num) + '.pckl'):
        file_num += 1
    else:
        break
file_name = file_name + str(file_num) + '.pckl'

f = open(file_name, 'wb')
pickle.dump([BFs, BF_params], f)
f.close()