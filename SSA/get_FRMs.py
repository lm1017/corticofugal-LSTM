import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

from FRM_utils import sine_with_cosine_ramp, get_cochleagram, save_FRMs_pdf
from tensorize_mod import tensorize_mod
import all_models

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_ID = 'NS3_PEG'
model_ID = 'pop_f_LSTM'

separate_silence = False
use_whole_tone = False
resp_diff_mode = 'square'

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

n_lambdas = len(unique_best_lambdas)

stim_intensity = np.arange(0, 80, 10)

min_F = 500
max_F = 23000
steps_per_octave = 8
n_steps = int(np.floor(steps_per_octave*np.log2(max_F/min_F)))
frequencies = min_F*(2**(np.arange(n_steps + 1)/steps_per_octave))

    
fs = 195000 # sampling frequency (Hz)
dur = 50 # tone duration (ms)
rise_dur = 5 # rise ramp duration (ms)
fall_dur = 0.5 # fall ramp duration (ms)
num_samples = dur/1000*195000 + 1

if separate_silence:
    tone_array = np.zeros((len(frequencies), len(stim_intensity), int(num_samples)))
else:
    tone_array = np.zeros((len(frequencies), len(stim_intensity), 3*int(num_samples)))
freq = frequencies[0]
db = stim_intensity[2]

for f_idx in range(len(frequencies)):
    freq = frequencies[f_idx]
    
    for dB_idx in range(len(stim_intensity)):
        dB = stim_intensity[dB_idx]
        
        if separate_silence:
            tone_array[f_idx, dB_idx] = sine_with_cosine_ramp(freq, dur/1000, fs, rise_dur/1000, fall_dur/1000, dB)
        else:
            tone = sine_with_cosine_ramp(freq, dur/1000, fs, rise_dur/1000, fall_dur/1000, dB)        
            silence = np.tile(np.zeros_like(tone), 2) # 100 ms of silence before each tone
    
            tone_array[f_idx, dB_idx] = np.concatenate((silence, tone))

bin_size = model_meta_info['bin_size'] # change bin size to 1 ms to simulate experimental conditions
padding_bins = model_meta_info['padding_bins']
zero_padding = model_meta_info['zero_padding']
n_h = model_meta_info['n_h']
f_axis = np.logspace(np.log10(min_F), np.log10(max_F), n_F) # logarithmically spaced frequency axis

if separate_silence:
    tone_cochleagram_array = np.zeros((len(frequencies), len(stim_intensity), n_F, n_h, round(dur/bin_size)-1))
else:
    if 'CNN' in model_ID:
        tone_cochleagram_array = np.zeros((len(frequencies), len(stim_intensity), n_F, n_h, 
                                           3*round(dur/bin_size) + padding_bins))
    else:
        tone_cochleagram_array = np.zeros((len(frequencies), len(stim_intensity), n_F, n_h, 3*round(dur/bin_size)))
for f_idx in range(len(frequencies)):
    for dB_idx in range(len(stim_intensity)):
        tone = tone_array[f_idx, dB_idx]
        
        X_ft, t_axis = get_cochleagram(tone, fs, bin_size, n_F, min_F, max_F, 'spec_power', padding_bins, zero_padding)

        tone_cochleagram_array[f_idx, dB_idx] = tensorize_mod(X_ft, n_h)

tone_cochleagram_array = torch.tensor(tone_cochleagram_array).to(torch.float32).to(device)

if separate_silence:
    silence = np.zeros_like(tone)
    silence_cochleagram, _ = get_cochleagram(silence, fs, bin_size, n_F, min_F, max_F, 'spec_power', padding_bins, zero_padding)
    silence_cochleagram = tensorize_mod(silence_cochleagram, n_h)
    silence_cochleagram = torch.tensor(silence_cochleagram).to(torch.float32).to(device)

model_params = model_meta_info['all_model_params']

if separate_silence:
    silence_win_left = 0 # no silence window needed if silence is separate
    silence_win_right = 0
else:
    silence_win_left = 12 # 50 ms from start of silence
    silence_win_right = 24 # end of silence (100 ms)

if use_whole_tone:
    win_left = silence_win_right
    win_right = None
else:
    win_left = silence_win_right + 2
    win_right = silence_win_right + 7

dsr = np.zeros((len(unique_best_lambdas), n_neurons, len(frequencies), len(stim_intensity)))

for lamb_idx in range(len(unique_best_lambdas)):
    lamb = unique_best_lambdas[lamb_idx]
    lamb_model_fit = model_fit_results[0][0][lamb][0]
    
    model_params['stim_size'] = np.size(tone_cochleagram_array, -1)
    model = getattr(all_models, model_ID)(model_params) # define and initialise model from model_ID
    model.to(device) # if available, "upload" model to GPU
    model.reload(lamb_model_fit)
    
    if separate_silence:
        silence_traces = model.np_forward_loop(silence_cochleagram)
        silence_o_traces = silence_traces['h']
        silence_o_win = silence_o_traces[:, win_left:win_right] # same window as tone
        
    for f_idx in range(len(frequencies)):
        for dB_idx in range(len(stim_intensity)):
            tone_cochleagram = tone_cochleagram_array[f_idx, dB_idx]
            
            traces = model.np_forward_loop(tone_cochleagram)
            o_traces = traces['o']
            
            if 'CNN' in model_ID:
                o_traces = o_traces[0]

            if not separate_silence:
                silence_o_win = o_traces[:, silence_win_left:silence_win_right]
            tone_o_win = o_traces[:, win_left:win_right]
            
            if use_whole_tone:
                dsr[lamb_idx, :, f_idx, dB_idx] = np.nanmean(tone_o_win - silence_o_win, 1) # driven spike rate
            else:
                silence_mean = np.expand_dims(np.nanmean(silence_o_win, 1), 1)
                if resp_diff_mode == 'square':
                    dsr[lamb_idx, :, f_idx, dB_idx] = np.nanmean(np.square(tone_o_win - silence_mean), 1)
                elif resp_diff_mode == 'abs':
                    dsr[lamb_idx, :, f_idx, dB_idx] = np.nanmean(np.abs(tone_o_win - silence_mean), 1)

max_sr = np.nanmax(dsr, (2, 3))
#max_sr = np.ones_like(dsr)
FRMs = np.zeros_like(dsr)
for lamb_idx in range(np.size(FRMs, 0)):
    for nID in range(np.size(FRMs, 1)):
        FRMs[lamb_idx, nID] = dsr[lamb_idx, nID]/max_sr[lamb_idx, nID]
  
save_FRMs_pdf(FRMs, frequencies, stim_intensity, dataset_ID, unique_best_lambdas)

FRM_params = {'separate_silence': separate_silence, 'whole_tone': use_whole_tone, 'spike_rate_mode': resp_diff_mode}

file_num = 0
file_name = 'FRMs/' + dataset_ID + '_' + model_ID + '_FRMs_'
while True:
    if os.path.isfile(file_name + str(file_num) + '.pckl'):
        file_num += 1
    else:
        break
file_name = file_name + str(file_num) + '.pckl'

f = open(file_name, 'wb')
pickle.dump([FRMs, FRM_params, unique_best_lambdas], f)
f.close()