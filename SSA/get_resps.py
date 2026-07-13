import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

import all_models

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_ID = 'NS3'
model_ID = 'pop_f_LSTM'

protocol = 'oddball'
start = 'silence'
BF_stims = True
norm = 0

if BF_stims:
    suffix = 0
    
    SSA_stim_file = 'BF_stims/' + start + '_start/' + dataset_ID + '_SSA_protocols_stims_' + model_ID + '.pckl'
    f = open(SSA_stim_file, 'rb')
    protocol = pickle.load(f)
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
    
    # choose fitted parameters file corresponding to the hyperparameters (frequency channels and training epochs) used
    model_params_file = data_dir + 'fit_results_' + model_ID + '_' + dataset_ID +'.pckl'
    f = open(model_params_file, 'rb')
    model_fit_results = pickle.load(f)
    f.close()
    
    n_HUs = model_meta_info['hidden_size']
    n_lambdas = len(unique_best_lambdas)
    
    model_params = model_meta_info['all_model_params']
    
    resps = dict.fromkeys(protocol)
    for cf in protocol:
        stims = protocol[cf]['stims']
        
        if norm: # if normalising stimuli
            mean_stim = np.mean([np.mean(stims[block]) for block in stims])
            
            concat_stims = np.zeros((np.size(stims['block 0'], 0), np.size(stims['block 0'], 1), 
                                     len(stims)*np.size(stims['block 0'], 2)))
            block_ctr = 0
            for block in stims:
                block_stims = stims[block]
                concat_stims[:, :, block_ctr*np.size(stims['block 0'], 2):(block_ctr + 1)*np.size(stims['block 0'], 2)] = \
                    block_stims
                
                block_ctr += 1
            std_stim = np.std(concat_stims, ddof=1)
        
        win = protocol[cf]['win']
        win[win != -70] = True
        win[win == -70] = False
        win = np.nanmean(win, 0)
        win_diff = win[1:] - win[:-1]
        win_start_idxs = np.where(win_diff == 1)[0]
        if start == 'sound':
            win_start_idxs = np.insert(win_start_idxs, 0, 0)
        win_end_idxs = np.where(win_diff == -1)[0]
        
        f_seq = protocol[cf]['f_seq']
        n_stims = len(f_seq['block 0'])
        
        if n_stims != len(win_start_idxs):
            raise ValueError("Incorrect number of tones")
        
        cf_resps = dict.fromkeys([str(lamb) for lamb in unique_best_lambdas])
        for lamb_idx in range(len(unique_best_lambdas)):
            print('lambda ' + str(lamb_idx))
            lamb = unique_best_lambdas[lamb_idx]
            lamb_model_fit = model_fit_results[0][0][lamb][0]

            if model_ID == 'pop_f_LSTM':
                dummy_resps = torch.zeros(n_neurons, np.size(stims['block 0'], -1))
            else:
                dummy_resps = torch.zeros(n_neurons, np.size(stims['block 0'], -1) - model_meta_info['padding_bins'])
    
            model_params['stim_size'] = np.size(stims['block 0'], -1)
            model_params['resp_size'] = dummy_resps.size(-1)
            model = getattr(all_models, model_ID)(model_params) # define and initialise model from model_ID
            model.to(device) # if available, "upload" model to GPU
            model.reload(lamb_model_fit)

            lamb_resps = dict.fromkeys(stims)
            for block in range(len(stims)):
                stim_block = torch.tensor(stims['block ' + str(block)]).to(torch.float32).to(device)
                if norm:
                    stim_block = (stim_block - mean_stim)/std_stim

                f_seq_block = f_seq['block ' + str(block)]
                
                out, _ = model.forward_loop(stim_block, torch.zeros(n_neurons, stim_block.size(-1)), 'val')
                out = out.detach().cpu().numpy()
                
                block_resps = np.zeros((n_neurons, n_stims))
                for win_idx in range(n_stims):
                    win_resps = out[win_start_idxs[win_idx]:win_end_idxs[win_idx]]
                    block_resps[:, win_idx] = np.nanmean(win_resps, 0)
            
                lamb_resps['block ' + str(block)] = block_resps
            cf_resps[str(lamb)] = lamb_resps
        resps[cf] = cf_resps
        
    file_name = 'BF_resps/' + start + '_start/' + 'Norm_' + str(norm) + '/' +  dataset_ID + '_' + 'cond_2_resps_' + \
        model_ID + '.pckl'
    f = open(file_name, 'wb')
    pickle.dump(resps, f)
    f.close()
    
else:
    SSA_stim_file = 'Stims/' + start + '_start/' + dataset_ID + '_SSA_protocols_stims_' + model_ID + '.pckl'
    f = open(SSA_stim_file, 'rb')
    obj = pickle.load(f)
    f.close()
    
    oddball_protocol = obj[0]
    switch_oddball_protocol = obj[1]
    resp_curve_protocol = obj[2]
    
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
    
    # choose fitted parameters file corresponding to the hyperparameters (frequency channels and training epochs) used
    model_params_file = data_dir + 'fit_results_' + model_ID + '_' + dataset_ID + '.pckl'
    f = open(model_params_file, 'rb')
    model_fit_results = pickle.load(f)
    f.close()
    
    n_HUs = model_meta_info['hidden_size']
    n_lambdas = len(unique_best_lambdas)
    
    model_params = model_meta_info['all_model_params']
    
    stims = locals()[protocol + '_protocol']['stims']
    
    win = locals()[protocol + '_protocol']['win']
    win[win != -70] = True
    win[win == -70] = False
    win = np.nanmean(win, 0)
    win_diff = win[1:] - win[:-1]
    win_start_idxs = np.where(win_diff == 1)[0]
    if start == 'sound':
        win_start_idxs = np.insert(win_start_idxs, 0, 0)
    win_end_idxs = np.where(win_diff == -1)[0]
    
    f_seq = locals()[protocol + '_protocol']['f_seq']
    
    if protocol == 'oddball':
        n_stims = len(f_seq['cond 0']['block 0'])
    elif protocol == 'switch_oddball':
        f_seq = np.tile(f_seq, 20)
        n_stims = len(f_seq)
    elif protocol == 'resp_curve':
        n_stims = len(f_seq)
    
    if n_stims != len(win_start_idxs):
        raise ValueError("Incorrect number of tones")
    
    if protocol == 'oddball':
        resps = dict.fromkeys([str(lamb) for lamb in unique_best_lambdas])
        for lamb_idx in range(len(unique_best_lambdas)):
            print('lambda ' + str(lamb_idx))
            lamb = unique_best_lambdas[lamb_idx]
            lamb_model_fit = model_fit_results[0][0][lamb][0]
            
            if model_ID == 'pop_f_LSTM':
                dummy_resps = torch.zeros(n_neurons, np.size(stims['cond 0']['block 0'], -1))
            else:
                dummy_resps = torch.zeros(n_neurons, np.size(stims['cond 0']['block 0'], -1) - model_meta_info['padding_bins'])
    
            model_params['stim_size'] = np.size(stims['cond 0']['block 0'], -1)
            model_params['resp_size'] = dummy_resps.size(-1)
            model = getattr(all_models, model_ID)(model_params) # define and initialise model from model_ID
            model.to(device) # if available, "upload" model to GPU
            model.reload(lamb_model_fit)
            
            lamb_resps = dict.fromkeys(locals()[protocol + '_protocol']['stims'])
            for cond in range(len(stims)):
                print('cond ' + str(cond))
                stim_blocks = stims['cond ' + str(cond)]
                f_seq_blocks = f_seq['cond ' + str(cond)]
                
                cond_resps = dict.fromkeys(stim_blocks)
                for block in range(len(stim_blocks)):
                    stim_block = torch.tensor(stim_blocks['block ' + str(block)]).to(torch.float32).to(device)
                    f_seq_block = f_seq_blocks['block ' + str(block)]
                    
                    out, _ = model.forward_loop(stim_block, torch.zeros(n_neurons, stim_block.size(-1)), 'val')
                    out = out.detach().cpu().numpy()
                    
                    block_resps = np.zeros((n_neurons, n_stims))
                    for win_idx in range(n_stims):
                        win_resps = out[win_start_idxs[win_idx]:win_end_idxs[win_idx]]
                        block_resps[:, win_idx] = np.nanmean(win_resps, 0)
                
                    cond_resps['block ' + str(block)] = block_resps
                lamb_resps['cond ' + str(cond)] = cond_resps
            resps[str(lamb)] = lamb_resps
    
    elif protocol == 'switch_oddball' or protocol == 'resp_curve':
        resps = dict.fromkeys([str(lamb) for lamb in unique_best_lambdas])
        stims = torch.tensor(stims).to(torch.float32).to(device)
        
        if model_ID == 'pop_f_LSTM':
            dummy_resps = torch.zeros(n_neurons, stims.size(-1))
        else:
            dummy_resps = torch.zeros(n_neurons, stims.size(-1) - model_meta_info['padding_bins'])
            
        for lamb_idx in range(len(unique_best_lambdas)):
            print('lambda ' + str(lamb_idx))
            lamb = unique_best_lambdas[lamb_idx]
            lamb_model_fit = model_fit_results[0][0][lamb][0]
            
            model_params['stim_size'] = stims.size(-1)
            model_params['resp_size'] = dummy_resps.size(-1)
            model = getattr(all_models, model_ID)(model_params) # define and initialise model from model_ID
            model.to(device) # if available, "upload" model to GPU
            model.reload(lamb_model_fit)
    
            out, _ = model.forward_loop(stims, dummy_resps, 'val')
            out = out.detach().cpu().numpy()
            
            resps[str(lamb)] = np.zeros((n_neurons, n_stims))
            for win_idx in range(n_stims):
                win_resps = out[win_start_idxs[win_idx]:win_end_idxs[win_idx]]
                resps[str(lamb)][:, win_idx] = np.nanmean(win_resps, 0)
    
    file_name = 'Resps/' + start + '_start/' + dataset_ID + '_' + protocol + '_resps_' + model_ID + '.pckl'
    f = open(file_name, 'wb')
    pickle.dump(resps, f)
    f.close()