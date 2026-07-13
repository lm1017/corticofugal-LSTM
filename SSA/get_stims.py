import os
import torch
import numpy as np
import pickle

from SSA_utils import get_oddball, get_switch_oddball, get_resp_curve, get_oddball_cond_2, get_cochleagram
from tensorize_mod import tensorize_mod
# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_ID = 'NS3'
model_ID = 'pop_f_LSTM'

use_mode = False
start = 'silence'
BF_stims = True

data_dir = 'Results/'

fit_file = data_dir + 'final_fit_' + model_ID + '_' + dataset_ID + '.pckl'
f = open(fit_file, 'rb')
obj = pickle.load(f)
f.close()

model_meta_info = obj[2]

best_lambda = model_meta_info['best_lambda'] # best overall lambda (mode of all best lambdas for each neuron)
neuron_best_lambdas = model_meta_info['neuron_best_lambdas'] # best lambda for each neuron
unique_best_lambdas = np.unique(neuron_best_lambdas)
n_F = model_meta_info['n_F'] # number of frequency channels
n_epochs = model_meta_info['epochs'] # number of training epochs
n_neurons = len(neuron_best_lambdas)

# choose fitted parameters file corresponding to the hyperparameters (frequency channels and training epochs) used
model_params_file = data_dir + 'fit_results_' + model_ID + '_' + dataset_ID +'.pckl'
f = open(model_params_file, 'rb')
model_fit_results = pickle.load(f)
f.close()

n_HUs = model_meta_info['hidden_size']
n_lambdas = len(unique_best_lambdas)

if BF_stims:
    # Cochleagram parameters
    bin_size = model_meta_info['bin_size'] # change bin size to 1 ms to simulate experimental conditions
    padding_bins = model_meta_info['padding_bins']
    zero_padding = model_meta_info['zero_padding']
    n_h = model_meta_info['n_h']
    min_F = 400 # changed because 500 is the BF of many neurons
    max_F = 22000
    f_axis = np.logspace(np.log10(min_F), np.log10(max_F), n_F) # logarithmically spaced frequency axis
    
    # SSA stimulus parameters
    dB = 60
    fs = 195000 # sampling frequency (Hz)
    dur = 230 # tone duration (ms)
    rise_dur = 10 # rise ramp duration (ms)
    fall_dur = 10 # fall ramp duration (ms)
    num_samples = dur/1000*195000 + 1
    isi = 506 # inter-stimulus interval duration (ms)
    win_dur = 330 # response window duration (ms)
    
    oddball_params = {3: {'df': 0.10, 'prob': 90}}
    num_blocks = 2
    num_tones = 400
    
    file_name = 'BF_stims/f_seq.pckl'
    if os.path.isfile(file_name):
        f = open(file_name, 'rb')
        blocks_f_seq = pickle.load(f)
        f.close()
    else:
        blocks_f_seq = {}
        for block in range(num_blocks):
            if block == 0:
                f1_prob = oddball_params[3]['prob']/100
            elif block == 1:
                f1_prob = (100 - oddball_params[3]['prob'])/100
                
            f2_prob = 1 - f1_prob
            f_sequence = np.random.choice([1, 2], num_tones, p=[f1_prob, f2_prob])
            blocks_f_seq[block] = f_sequence

        f = open(file_name, 'wb')
        pickle.dump(blocks_f_seq, f)
        f.close()
    
    BF_file = 'BFs/' + dataset_ID + '_' + model_ID + '_BFs.pckl'
    f = open(BF_file, 'rb')
    BF_results = pickle.load(f)
    f.close()
    
    BFs = BF_results[0][dataset_ID]
    BF_params = BF_results[1]
    
    n_BFs = np.zeros((n_neurons))
    for nID in range(n_neurons):
        n_lamb = neuron_best_lambdas[nID]
        n_BFs[nID] = BFs[str(n_lamb)][nID]

    unique_BFs = np.unique(n_BFs)
    BF_stims = {}
    for fID in range(len(unique_BFs)):
        print(fID)
        cf = unique_BFs[fID]
        
        cond_2_stims = get_oddball_cond_2(cf, fs, dB, dur/1000, rise_dur/1000, fall_dur/1000, isi/1000, win_dur/1000, 
                                          oddball_params, blocks_f_seq, num_blocks, num_tones, start, fix_f_seq=True)

        # Oddball cochleagrams
        oddball_coch = dict.fromkeys(cond_2_stims)
        oddball_f_seq = dict.fromkeys(cond_2_stims)
        
        for block in cond_2_stims:
            stim = cond_2_stims[block][0]
            X_ft, _ = get_cochleagram(stim, fs, bin_size, n_F, min_F, max_F, 'spec_power', padding_bins, zero_padding)
            oddball_coch[block] = tensorize_mod(X_ft, n_h)
            
            wins = cond_2_stims[block][2]
            X_ft, _ = get_cochleagram(wins, fs, bin_size, n_F, min_F, max_F, 'spec_power', padding_bins, zero_padding=False)
            oddball_win_coch = X_ft
            
            f_seq = cond_2_stims[block][1]
            oddball_f_seq[block] = f_seq

        bf_oddball = {'stims': oddball_coch, 'win': oddball_win_coch, 'f_seq': oddball_f_seq}
        BF_stims[cf] = bf_oddball

    file_name = 'BF_stims/' + start + '_start/' + dataset_ID + '_SSA_protocols_stims_' + model_ID + '.pckl'
    f = open(file_name, 'wb')
    pickle.dump(BF_stims, f)
    f.close()
    
else:
    cf = 17234.3
    dB = 40
    fs = 195000 # sampling frequency (Hz)
    dur = 230 # tone duration (ms)
    rise_dur = 10 # rise ramp duration (ms)
    fall_dur = 10 # fall ramp duration (ms)
    num_samples = dur/1000*195000 + 1
    isi = 506 # inter-stimulus interval duration (ms)
    win_dur = 330 # response window duration (ms)
    
    # Oddball design params
    num_conds = 4
    oddball_params = {}
    
    for cond in range(num_conds):
        if cond == 0 or cond == 1:
            cond_df = 0.37
        elif cond == 2:
            cond_df = 0.10
        elif cond == 3:
            cond_df = 0.04
            
        if cond == 0:
            cond_prob = 70
        else:
            cond_prob = 90
            
        oddball_params[cond+1] = {'df': cond_df, 'prob': cond_prob}
    
    num_blocks = 3
    num_tones = 400
    
    oddball_stims = get_oddball(cf, fs, dB, dur/1000, rise_dur/1000, fall_dur/1000, isi/1000, win_dur/1000, num_conds, 
                                oddball_params, num_blocks, num_tones, start)
    
    # Switching oddball design params
    switch_oddball_params = {'df': 0.37, 'prob': 80}
    num_blocks = 20
    num_tones = 40
    switch_oddball_stims = get_switch_oddball(cf, fs, dB, dur/1000, rise_dur/1000, fall_dur/1000, isi/1000, win_dur/1000,
                                              switch_oddball_params, num_blocks, num_tones, start)
    
    # Response-curve design params
    num_freqs = 20
    f_range_oct = 0.97
    num_reps = 10
    
    response_curve_stims = get_resp_curve(cf, fs, dB, dur/1000, rise_dur/1000, fall_dur/1000, isi/1000, win_dur/1000, num_freqs, 
                                          f_range_oct, num_reps, start)
    
    bin_size = model_meta_info['bin_size'] # change bin size to 1 ms to simulate experimental conditions
    padding_bins = model_meta_info['padding_bins']
    zero_padding = model_meta_info['zero_padding']
    n_h = model_meta_info['n_h']
    min_F = 500
    max_F = 22000
    f_axis = np.logspace(np.log10(min_F), np.log10(max_F), n_F) # logarithmically spaced frequency axis
    
    # Oddball cochleagrams
    oddball_coch = dict.fromkeys(oddball_stims)
    oddball_f_seq = dict.fromkeys(oddball_stims)
    for cond in range(num_conds):
        oddball_coch['cond ' + str(cond)] = dict.fromkeys(oddball_stims['cond ' + str(cond)])
        oddball_f_seq['cond ' + str(cond)] = dict.fromkeys(oddball_stims['cond ' + str(cond)])
    
        for block in range(len(oddball_stims['cond ' + str(cond)])):
            stim = oddball_stims['cond ' + str(cond)]['block ' + str(block)][0]
            X_ft, _ = get_cochleagram(stim, fs, bin_size, n_F, min_F, max_F, 'spec_power', padding_bins, zero_padding)
            oddball_coch['cond ' + str(cond)]['block ' + str(block)] = tensorize_mod(X_ft, n_h)
            
            wins = oddball_stims['cond ' + str(cond)]['block ' + str(block)][2]
            X_ft, _ = get_cochleagram(wins, fs, bin_size, n_F, min_F, max_F, 'spec_power', padding_bins, zero_padding=False)
            oddball_win_coch = X_ft
            
            f_seq = oddball_stims['cond ' + str(cond)]['block ' + str(block)][1]
            oddball_f_seq['cond ' + str(cond)]['block ' + str(block)] = f_seq
    
    # Switching oddball cochleagrams
    X_ft, _ = get_cochleagram(switch_oddball_stims[0], fs, bin_size, n_F, min_F, max_F, 'spec_power', padding_bins, zero_padding)
    switch_odd_coch = tensorize_mod(X_ft, n_h)
    
    X_ft, _ = get_cochleagram(switch_oddball_stims[2], fs, bin_size, n_F, min_F, max_F, 'spec_power', padding_bins, 
                              zero_padding=False)
    switch_odd_win_coch = X_ft
    
    switch_odd_f_seq = switch_oddball_stims[1]
    
    # Response curve cochleagrams
    X_ft, _ = get_cochleagram(response_curve_stims[0], fs, bin_size, n_F, min_F, max_F, 'spec_power', padding_bins, zero_padding)
    resp_curve_coch = tensorize_mod(X_ft, n_h)
    
    X_ft, _ = get_cochleagram(response_curve_stims[2], fs, bin_size, n_F, min_F, max_F, 'spec_power', padding_bins, 
                              zero_padding=False)
    resp_curve_win_coch = X_ft
    
    resp_curve_f_seq = response_curve_stims[1]
    
    oddball = {'stims': oddball_coch, 'win': oddball_win_coch, 'f_seq': oddball_f_seq}
    switch_oddball = {'stims': switch_odd_coch, 'win': switch_odd_win_coch, 'f_seq': switch_odd_f_seq}
    resp_curve = {'stims': resp_curve_coch, 'win': resp_curve_win_coch, 'f_seq': resp_curve_f_seq}
    
    file_name = 'Stims/' + start + '_start/' + dataset_ID + '_SSA_protocols_stims_' + model_ID + '.pckl'
    f = open(file_name, 'wb')
    pickle.dump([oddball, switch_oddball, resp_curve], f)
    f.close()
