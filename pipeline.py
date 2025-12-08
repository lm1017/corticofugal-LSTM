import time

t_track = time.time() # start timer

import build_datasets

import pickle
import torch
import torch.nn as nn
import numpy as np

from util_funcs import train_val_test_split, build_grid_NS2_include_single_NS3_concat
import train_val_test_loops

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cudnn_benchmark = True
torch.backends.cudnn.benchmark = cudnn_benchmark

# boolean variable, controls use of Automatic Mixed Precision (AMP); should enable faster training but no effect found here
amp_type = False
early_stop = False # boolean variable, controls use of early stopping

# model types: convolutional or not - determines whether input stimuli must be tensorised or not
model_types = {'not_conv': ['STRF', 'LN', 'pop_NRF', 'pop_LRU', 'pop_RNN', 'pop_MGU', 'pop_GRU', 'pop_LSTM', 'pop_subLSTM', 
                            'pop_f_LSTM', 'pop_rc_LSTM', 'pop_gatesLSTM', 'pop_gatesGRU', 'pop_gatessubLSTM', 
                            'pop_gatesfLSTM', 'pop_directrcLSTM', 'pop_fix_subLSTM', 'pop_t_out_subLSTM'],
               'conv': ['LN_conv', 'single_CNN', 'pop_STRF', 'pop_STRF_noFC', 'pop_LN', 'pop_LN_noFC', 'oneD_CNN', 
                        'oneD_x2_CNN', 'twoD_CNN', 'pop_conv_GRU']}

# architecture types: single or population
pop_and_single_models = {'single': ['STRF', 'LN', 'LN_conv', 'single_CNN'], 
                         'pop': ['pop_STRF', 'pop_STRF_noFC', 'pop_LN_noFC', 'pop_NRF', 'pop_LRU', 'pop_RNN', 'pop_MGU', 
                                 'pop_GRU', 'pop_LSTM', 'pop_subLSTM', 'pop_f_LSTM', 'pop_rc_LSTM', 
                                 'pop_gatesLSTM', 'pop_gatesGRU', 'pop_gatessubLSTM', 'pop_gatesfLSTM', 'pop_directrcLSTM', 
                                 'pop_fix_subLSTM', 'pop_t_out_subLSTM', 'pop_LN', 'pop_conv_GRU', 'oneD_CNN', 'oneD_x2_CNN', 
                                 'twoD_CNN']}

dataset_ID = 'NS3_PEG'
reduced = False
cochleagram_type = 'spec_power'
type_data = 'torch' # data type, either NumPy or PyTorch
model_ID = 'pop_f_LSTM'

# assign type and architecture (see above) to model
for key in model_types:
    if model_ID in model_types[key]:
        model_type = key
        
for key in pop_and_single_models:
    if model_ID in pop_and_single_models[key]:
        pop_or_single = key

# cochleagram parameters
bin_size = 4 # time discretisation
n_F = 8 # number of frequency channels

amended_f_range = False

if amended_f_range:
    min_F = 200 # minimum channel centre frequency
    max_F = 20000 # maximum channel centre frequency
else:
    min_F = 500 # minimum channel centre frequency
    max_F = 22000 # maximum channel centre frequency

build_dataset_fcn = 'build_dataset_' + dataset_ID
build_dataset = getattr(build_datasets, build_dataset_fcn) # use building function for the chosen dataset

n_h = 8 # history steps for cochleagram tensorisation
padding_bins = n_h-1 # number of bins for zero-padding - usually equal to tensorisation history - 1, but not for conv models
if model_type == 'conv':
    n_h = 1 # cochleagrams are not tensorised when using convolutional models

params = {'cochleagram_type': cochleagram_type, 'bin_size': bin_size, 'n_F': n_F, 'min_F': min_F, 'max_F': max_F, 
          'padding_bins': padding_bins, 'device': device}

# determine if stimuli must be zero-padded (for padding_bins bins)
# this enables to predict whole responses - instead of not predicting first n_h (tensorisation) bins
zero_padding = True

# build dataset with specified parameters
stim_dataset, resp_dataset, params = build_dataset(params, zero_padding)
# stim_dataset contains a list with two NumPy arrays, one with multi-repeat cochleagrams, and one with single-repeat 
# cochleagrams (dimensions -> (f, t, stimuli))

n_neurons = params['n_neurons']
n_stimuli = params['n_stimuli']
n_repeats = params['n_repeats']

bin_size = params['bin_size']
n_F = params['n_F']
min_F = params['min_F']
max_F = params['max_F']
t_axis = params['t_axis']

params['n_h'] = n_h

clip_length = 0 # length (in ms) to clip from the beginning of stimuli and responses
clip_bins = int(np.round(clip_length/bin_size)) # number of time bins corresponding to clip_length

shift_bins = 0 # number of bins to shift the responses by to account for neuronal latency

multi_rep_cochleagrams = stim_dataset[0] # multi-repeat cochleagrams
single_rep_cochleagrams = stim_dataset[1] # single-repeat cochleagrams

# clip given number of bins from start of cochleagram - if zero-padded, keep the padding_bins bins at the start
if zero_padding:
    multi_rep_cochleagrams = np.concatenate((multi_rep_cochleagrams[:, :padding_bins, :], 
                                             multi_rep_cochleagrams[:, padding_bins+clip_bins:, :]), axis=1)
    single_rep_cochleagrams = np.concatenate((single_rep_cochleagrams[:, :padding_bins, :], 
                                              single_rep_cochleagrams[:, padding_bins+clip_bins:, :]), axis=1)
else:
    multi_rep_cochleagrams = multi_rep_cochleagrams[:, clip_bins:, :]
    single_rep_cochleagrams = single_rep_cochleagrams[:, clip_bins:, :]

# split into train, validation, and test sets
folds, test_stimuli, cv_stimuli, validation_stimuli, training_stimuli, stimuli = \
    train_val_test_split(dataset_ID, reduced, single_rep_cochleagrams)


if dataset_ID == 'NS2_include_single' or dataset_ID == 'NS3':
    folds = 1
    stimuli['validation_stimuli'] = [[6, 3, 13, 7, 17, 9, 5]]
    stimuli['training_stimuli'] = [[0, 1, 4, 16, 10, 12, 15] + list(range(np.size(single_rep_cochleagrams, 2)))]
elif dataset_ID == 'NS3_PEG': # chnage indices to use the same stimulus sets in NS3 and NS3_PEG
    folds = 1
    stimuli['validation_stimuli'] = [[7, 16, 9, 11, 0, 3, 15]]
    stimuli['training_stimuli'] = [[1, 4, 12, 2, 5, 6, 13] + list(range(np.size(single_rep_cochleagrams, 2)))]

# mean and std of CV set - if zero-padded, exclude padding_bins from the calculation
if zero_padding:
    # CV set made up of CV multi-repeat stimuli and all single-repeat stimuli
    mean_cv = np.mean([np.mean(single_rep_cochleagrams[:, padding_bins:, :][:]), 
                       np.mean(multi_rep_cochleagrams[:, padding_bins:, cv_stimuli[:14]][:])])
    std_cv = np.std(np.concatenate((multi_rep_cochleagrams[:, :, cv_stimuli[:14]], single_rep_cochleagrams), axis=2)
                    [:, padding_bins:, :][:], ddof=1)
else:
    mean_cv = np.mean([np.mean(single_rep_cochleagrams[:]), \
                       np.mean(multi_rep_cochleagrams[:, :, cv_stimuli[:14]][:])])
    std_cv = np.std(np.concatenate((multi_rep_cochleagrams[:, :, cv_stimuli[:14]], single_rep_cochleagrams), 
                                   axis=2)[:], ddof=1)

# normalisation parameter: determines whether dataset is normalised to zero-mean and unit variance over all stimuli 
# in a set or stimulus by stimulus
norm = 'all'

# get datasets: training stimuli, training responses (PSTHs), validation stimuli, validation responses (PSTHs), 
# multi-repeat spike times, single-repeat spike times
train_stim_dataset, train_resp_dataset, val_stim_dataset, val_resp_dataset, cv_stim_dataset, cv_resp_dataset, \
    test_stim_dataset, test_resp_dataset, multi_rep_resps, single_rep_resps = \
        build_grid_NS2_include_single_NS3_concat(single_rep_cochleagrams, multi_rep_cochleagrams, n_neurons, 
                                                 resp_dataset, stimuli, params, norm, mean_cv, std_cv, clip_bins, 
                                                 shift_bins, type_data, zero_padding, padding_bins, t_axis)

# Uncomment to save datasets
'''f = open(dataset_ID + '_t' + str(bin_size) + '_f' + str(n_F) + '_h' + str(n_h) + '_concat.pckl', 'wb')
pickle.dump([train_stim_dataset, train_resp_dataset, val_stim_dataset, val_resp_dataset, cv_stim_dataset, 
             cv_resp_dataset, test_stim_dataset, test_resp_dataset, multi_rep_resps, single_rep_resps], f)
f.close()'''

tune_stim_dataset = val_stim_dataset # tuning stimuli, same as validation stimuli
tune_resp_dataset = val_resp_dataset # tuning responses, same as validation responses

# training/validation stimuli and responses: used to compute CCnorm and loss on training data to compare to tuning 
# values; only training multi-repeats are considered, this is necessary to be able to compute CCnorm
train_val_stim_dataset = train_stim_dataset[:, :, :, 
                                            :int(train_stim_dataset.size(3)/len(stimuli['training_stimuli'][0]))*7]
train_val_resp_dataset = train_resp_dataset[:, :, 
                                            :int(train_resp_dataset.size(2)/len(stimuli['training_stimuli'][0]))*7]

# for practical reasons, redefine resp_dataset and stim_dataset to only include multi-repeat stimuli and responses
resp_dataset = multi_rep_resps
stim_dataset = multi_rep_cochleagrams

# L1 regularisation strength
lambda_sequence = np.array([1e-3, 2e-4, 1.17e-4, 6.84e-5, 4e-5, 2.34e-5, 1.37e-5, 8e-6, 4.68e-6, 2.74e-6, 1.6e-6, 9.36e-7, 
                            5.41e-7, 3.2e-7, 6.4e-8, 1.28e-8, 2.56e-9, 5.12e-10])

split = 1 # split stims and resps in chunks, each of which is this fraction of the original one

# stimulus and response lengths are not necessarily the same - e.g., if padding_bins is not equal to his_steps-1
stim_size = int(train_stim_dataset.size(3)/len(stimuli['training_stimuli'][0])*split) # stimulus length (time bins)
resp_size = int(train_resp_dataset.size(2)/len(stimuli['training_stimuli'][0])*split) # response length (time bins)
batch_data_num = int(1/split) # number of stimuli/responses in one minibatch
# stimulus minibatch size in time bins
stim_batch_size = int(batch_data_num*(train_stim_dataset.size(3)/len(stimuli['training_stimuli'][0])*split))
# response minibatch size in time bins
resp_batch_size = int(batch_data_num*(train_resp_dataset.size(2)/len(stimuli['training_stimuli'][0])*split))

# for practical reasons, redefine stimulus indices in datasets to only include multi-repeat stimuli and responses
if dataset_ID == 'NS2_include_single' or dataset_ID == 'NS3':
    training_stimuli = [[0, 1, 4, 16, 10, 12, 15]]
    validation_stimuli = [[6, 3, 13, 7, 17, 9, 5]]
elif dataset_ID == 'NS3_PEG':
    training_stimuli = [[1, 4, 12, 2, 5, 6, 13]]
    validation_stimuli = [[7, 16, 9, 11, 0, 3, 15]]

# Model hyperparameters
num_epochs = 100 # number of training epochs
learning_rate = 1e-3 # optimisation learning rate
adaptive_lr = False # controls whether learning rate changes dependent on tuning loss

criterion_type = 'MSE' # loss function type
loss_fcn = nn.MSELoss() # loss function
optim = 'NAdam' # optimiser type

regul_type = 'L1_weights' # regularisation type
out_nl_type = 'sigmoid' # output nonlinearity type for model units
# controls whether gradient is set to 0 or None upon resetting gradients of all optimised model parameters
grad_set_to_none = True
grad_clipping = True # controls use of gradient clipping

# model parameters
model_params = {'input_size': n_F*n_h, 'hidden_size': 1, 'output_size': n_neurons, 'learning_rate': learning_rate,
                'initialisation': 'default', 'dropout' : False, 'stim_size': stim_size, 'resp_size': resp_size, 
                'his_steps': n_h, 'split': split, 't_gate_memory': 'hidden'}

# additional parameters for convolutional models
conv_model_params = {'gru_hidden_size': 120, 'n_F': n_F, 'n_out': n_neurons, 'n_HU_1': 120, 'n_filters_1': 100, 
                     'n_filters_2': 10, 'n_filters_3': 10, 'filter_len_1': 61, 'filter_len_2': 25, 'filter_area': (3, 21)}

if model_type == 'conv':
    model_params.update(conv_model_params) # if model is of the convolutional type, add additional parameters

neuron_val_performance = np.zeros((n_neurons)) # array to hold CCnorm on validation data for each neuron
neuron_train_performance = np.zeros((n_neurons)) # array to hold CCnorm on training data for each neuron

# full dataset
dataset = {'train_stim': train_stim_dataset, 'train_resp': train_resp_dataset, 'tune_stim': tune_stim_dataset, 
           'tune_resp': tune_resp_dataset, 'val_stim': val_stim_dataset, 'val_resp': val_resp_dataset,
           'train_val_stim': train_val_stim_dataset, 'train_val_resp': train_val_resp_dataset}

# population or single-neuron model fitting
fitting_loop = getattr(train_val_test_loops, pop_or_single + '_cross_val_loop')

# model fitting
crossval_ccnorms, model_fits, loss_lr_curves = fitting_loop(dataset, lambda_sequence, model_ID, model_params, loss_fcn, 
                                                            optim, num_epochs, stim_batch_size, resp_batch_size, 
                                                            learning_rate, early_stop, grad_set_to_none, grad_clipping, 
                                                            adaptive_lr, validation_stimuli, training_stimuli, resp_dataset, 
                                                            stim_dataset, n_h, bin_size, norm, mean_cv, std_cv, clip_bins, 
                                                            shift_bins, zero_padding, padding_bins, model_type, t_axis,
                                                            split)

for nID in range(n_neurons):
    # CCnorm for each neuron: average CCnorm over folds, and pick best average CCnorm across all lambdas
    neuron_val_performance[nID] = np.nanmax(np.nanmean(crossval_ccnorms['val'][nID], 0))
    neuron_train_performance[nID] = np.nanmax(np.nanmean(crossval_ccnorms['train'][nID], 0))

elapsed = time.time() - t_track # stop timer

# meta info of model fit
meta_info = {'cochleagram' : cochleagram_type, 'n_F' : n_F, 'clip_bins' : clip_bins, 'shift_bins': shift_bins, 'n_h' : n_h, 
             'norm' : norm, 'lambdas' : lambda_sequence, 'hidden_size' : model_params['hidden_size'], 'epochs' : num_epochs, 
             'lr' : learning_rate, 'stim_batch_size' : stim_batch_size, 'resp_batch_size' : resp_batch_size, 
             'batch_data_num' : batch_data_num, 'out_nl' : out_nl_type, 'loss' : criterion_type, 'optimiser' : optim, 
             'regul' : regul_type, 'adaptive_lr' : adaptive_lr, 'time' : elapsed, 'cudnn_benchmark' : cudnn_benchmark, 
             'early_stopping' : early_stop, 'init_network' : model_params['initialisation'], 'AMP' : amp_type, 
             'grad_set_to_none' : grad_set_to_none, 'grad_clipping' : grad_clipping, 'dropout' : model_params['dropout'], 
             'bin_size': bin_size, 'zero_padding': zero_padding, 'padding_bins': padding_bins, 'reduced': reduced, 
             'type_data': type_data, 'model_type': model_type, 'pop_or_single': pop_or_single, 
             'all_model_params': model_params, 'time_axis': t_axis, 'split': split, 'amended_f_range': amended_f_range}
    
noise_ratios = params['noise_ratios']

# average training and validation CCnorms over all neurons
avg_all_train_cc = np.nanmean(neuron_train_performance)
avg_all_val_cc = np.nanmean(neuron_val_performance)

# save fit results, fitted model parameters, and model meta info to pickle files
save_fit = False
if save_fit is True:
    f = open('fit_ccnorms_' + model_ID + '_' + dataset_ID + '.pckl', 'wb')
    pickle.dump([neuron_train_performance, crossval_ccnorms['train'], neuron_val_performance, crossval_ccnorms['val']], f)
    f.close()

    f = open('fit_results_' + model_ID + '_' + dataset_ID + '.pckl', 'wb')
    pickle.dump([model_fits, meta_info], f)
    f.close()