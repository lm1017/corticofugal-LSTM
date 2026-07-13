import numpy as np
import pickle
import matplotlib.pyplot as plt

dataset_ID = 'NS3'
model_ID = 'pop_f_LSTM'

protocol = 'oddball'
start = 'silence'
BF_stims = True
norm = 0

save_results = False

if BF_stims:
    suffix = 0
    
    SSA_resp_file = 'BF_resps/' + start + '_start/Norm_' + str(norm) + '/' + dataset_ID + '_' + 'cond_2_resps_' + \
        model_ID + '.pckl'
    f = open(SSA_resp_file, 'rb')
    all_resps = pickle.load(f)
    f.close()
    
    SSA_stim_file = 'BF_stims/' + start + '_start/' + dataset_ID + '_SSA_protocols_stims_' + model_ID + '.pckl'
    f = open(SSA_stim_file, 'rb')
    all_stims = pickle.load(f)
    f.close()
    
    BF_file = 'BFs/' + dataset_ID + '_' + model_ID + '_BFs_' + str(suffix) + '.pckl'
    f = open(BF_file, 'rb')
    BF_results = pickle.load(f)
    f.close()
    
    BFs = BF_results[0][dataset_ID]
    BF_params = BF_results[1]
    
    data_dir = 'Results/'
    
    fit_file = data_dir + 'final_fit_' + model_ID + '_' + dataset_ID + '.pckl'
    f = open(fit_file, 'rb')
    obj = pickle.load(f)
    f.close()
    
    n_neurons = len(obj[1])
    model_meta_info = obj[2]
    
    best_lambda = model_meta_info['best_lambda'] # best overall lambda (mode of all best lambdas for each neuron)
    neuron_best_lambdas = model_meta_info['neuron_best_lambdas'] # best lambda for each neuron
    unique_best_lambdas = np.unique(neuron_best_lambdas)
    n_F = model_meta_info['n_F'] # number of frequency channels
    n_epochs = model_meta_info['epochs'] # number of training epochs
    
    all_SI = np.zeros((n_neurons))
    
    for nID in range(n_neurons):
        n_bl = neuron_best_lambdas[nID]
        n_BF = BFs[str(n_bl)][nID]
        n_resps = all_resps[n_BF][str(n_bl)]
        
        d_f0, d_f1, s_f0, s_f1 = [0, 0, 0, 0]
        d_f0_ctr, d_f1_ctr, s_f0_ctr, s_f1_ctr = [0, 0, 0, 0]
        
        for block in n_resps:
            block_resps = n_resps[block]
            f_seq = all_stims[n_BF]['f_seq'][block]
            f_seq = f_seq - 1
            
            standard_f = int(block[-1])
            dev_f = int(not(standard_f))
            
            for tone in range(len(f_seq)):
                tone_f = f_seq[tone]
                if tone_f == standard_f:
                    locals()['s_f' + str(standard_f)] += block_resps[nID, tone]
                    locals()['s_f' + str(standard_f) + '_ctr'] += 1
                else:
                    locals()['d_f' + str(dev_f)] += block_resps[nID, tone]
                    locals()['d_f' + str(dev_f) + '_ctr'] += 1
        
        d_f0 = d_f0/d_f0_ctr
        d_f1 = d_f1/d_f1_ctr
        s_f0 = s_f0/s_f0_ctr
        s_f1 = s_f1/s_f1_ctr
        
        n_SI = (d_f0 + d_f1 - (s_f0 + s_f1))/(d_f0 + d_f1 + s_f0 + s_f1)
        all_SI[nID] = n_SI
        
    if save_results:
        f, ax = plt.subplots()
        ax.hist(all_SI)
        ax.set_xlabel('SI')
        ax.set_title('SI - ' + dataset_ID + ', ' + model_ID)
        
        file_name = 'BF_SI/' + start + '_start/Norm_' + str(norm) + '/' + dataset_ID + '_cond_2_SI_' + model_ID + '.pckl'
        f = open(file_name, 'wb')
        pickle.dump(all_SI, f)
        f.close()
    
else:
    SSA_resp_file = 'Resps/' + start + '_start/' + dataset_ID + '_' + protocol + '_resps_' + model_ID + '.pckl'
    f = open(SSA_resp_file, 'rb')
    all_resps = pickle.load(f)
    f.close()
    
    SSA_stim_file = 'Stims/' + start + '_start/' + dataset_ID + '_SSA_protocols_stims_' + model_ID + '.pckl'
    f = open(SSA_stim_file, 'rb')
    obj = pickle.load(f)
    f.close()
    
    oddball_protocol = obj[0]
    switch_oddball_protocol = obj[1]
    resp_curve_protocol = obj[2]
    
    if protocol == 'switch_oddball' or protocol == 'resp_curve':
        f_seq = locals()[protocol + '_protocol']['f_seq']
        f_seq = f_seq - 1
    
    all_SI = dict.fromkeys(all_resps)

    for lamb in all_resps:
        lamb_resps = all_resps[lamb]
        
        if protocol == 'oddball':
            n_neurons = np.size(lamb_resps['cond 0']['block 0'], 0)
            lamb_SI = dict.fromkeys(lamb_resps)
            for cond in lamb_resps:
                cond_resps = lamb_resps[cond]
                
                d_f0 = np.zeros((n_neurons))
                d_f1 = np.zeros((n_neurons))
                s_f0 = np.zeros((n_neurons))
                s_f1 = np.zeros((n_neurons))
                
                d_f0_ctr = 0
                d_f1_ctr = 0
                s_f0_ctr = 0
                s_f1_ctr = 0
                
                for block in cond_resps:
                    if block == 'block 2':
                        break
                    
                    block_resps = cond_resps[block]
                    f_seq = locals()[protocol + '_protocol']['f_seq'][cond][block]
                    f_seq = f_seq - 1
                    
                    standard_f = int(block[-1])
                    dev_f = int(not(standard_f))
                    
                    for tone in range(len(f_seq)):
                        tone_f = f_seq[tone]
                        if tone_f == standard_f:
                            locals()['s_f' + str(standard_f)] += block_resps[:, tone]
                            locals()['s_f' + str(standard_f) + '_ctr'] += 1
                        else:
                            locals()['d_f' + str(dev_f)] += block_resps[:, tone]
                            locals()['d_f' + str(dev_f) + '_ctr'] += 1
                            
                d_f0 = d_f0/d_f0_ctr
                d_f1 = d_f1/d_f1_ctr
                s_f0 = s_f0/s_f0_ctr
                s_f1 = s_f1/s_f1_ctr
                
                cond_SI = (d_f0 + d_f1 - (s_f0 + s_f1))/(d_f0 + d_f1 + s_f0 + s_f1)
                lamb_SI[cond] = cond_SI
                
                f, ax = plt.subplots()
                ax.hist(cond_SI)
                ax.set_xlabel('SI')
                ax.set_title('SI - ' + dataset_ID + ', ' + model_ID + ', lambda ' + lamb + ', ' + protocol + ', ' + cond)

            all_SI[lamb] = lamb_SI
                
        elif protocol == 'switch_oddball':
            n_neurons = np.size(lamb_resps, 0)
            block_len = len(f_seq)
            f_switch_len = int(block_len/2)
            
            d_f0 = np.zeros((n_neurons))
            d_f1 = np.zeros((n_neurons))
            s_f0 = np.zeros((n_neurons))
            s_f1 = np.zeros((n_neurons))
            
            d_f0_ctr = 0
            d_f1_ctr = 0
            s_f0_ctr = 0
            s_f1_ctr = 0
            
            for tone in range(np.size(lamb_resps, 1)):
                block_idx = tone // block_len
                within_block_tone_idx = tone % block_len
                
                if within_block_tone_idx < f_switch_len:
                    standard_f = 0
                    dev_f = 1
                else:
                    standard_f = 1
                    dev_f = 0
                    
                tone_f = f_seq[within_block_tone_idx]
                
                if tone_f == standard_f:
                    locals()['s_f' + str(standard_f)] += lamb_resps[:, tone]
                    locals()['s_f' + str(standard_f) + '_ctr'] += 1
                else:
                    locals()['d_f' + str(dev_f)] += lamb_resps[:, tone]
                    locals()['d_f' + str(dev_f) + '_ctr'] += 1
                
            d_f0 = d_f0/d_f0_ctr
            d_f1 = d_f1/d_f1_ctr
            s_f0 = s_f0/s_f0_ctr
            s_f1 = s_f1/s_f1_ctr
            
            SI = (d_f0 + d_f1 - (s_f0 + s_f1))/(d_f0 + d_f1 + s_f0 + s_f1)
            
            f, ax = plt.subplots()
            ax.hist(SI)
            ax.set_xlabel('SI')
            ax.set_title('SI - ' + dataset_ID + ', ' + model_ID + ', lambda ' + lamb + ', ' + protocol)
            
            all_SI[lamb] = SI
    
    if save_results:
        file_name = 'SI/' + start + '_start/' + dataset_ID + '_' + protocol + '_' + model_ID + '.pckl'
        f = open(file_name, 'wb')
        pickle.dump(all_SI, f)
        f.close()