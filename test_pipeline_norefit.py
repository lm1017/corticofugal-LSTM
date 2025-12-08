import time

t_track = time.time() # start timer

import build_datasets

import pickle
import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats

from util_funcs import train_val_test_split, build_grid_NS2_include_single_NS3_concat
from train_val_test_loops import val_loop

import all_models

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_ID = 'NS3_PEG'
model_ID = 'pop_f_LSTM'
n_F = 8 # number of frequency channels
n_epochs = 5
split = 1

amended_f_range = False

filename = 'fit_results_' + model_ID + '_' + dataset_ID + '.pckl'

f = open(filename, 'rb')
model_fit_results = pickle.load(f)
f.close()

model_fitted_params = model_fit_results[0]
model_meta_info = model_fit_results[1]

# boolean variable, controls use of cudnn benchmarking
cudnn_benchmark = model_meta_info['cudnn_benchmark']
torch.backends.cudnn.benchmark = cudnn_benchmark

# boolean variable, controls use of Automatic Mixed Precision (AMP)
amp_type = model_meta_info['AMP']
early_stop = model_meta_info['early_stopping'] # boolean variable, controls use of early stopping

reduced = model_meta_info['reduced']
cochleagram_type = model_meta_info['cochleagram']
type_data = model_meta_info['type_data'] # data type, either NumPy or PyTorch

bin_size = model_meta_info['bin_size'] # time discretisation

if amended_f_range:
    min_F = 200 # minimum channel centre frequency
    max_F = 20000 # maximum channel centre frequency
else:
    min_F = 500 # minimum channel centre frequency
    max_F = 22000 # maximum channel centre frequency

# assign type and architecture (see above) to model
model_type = model_meta_info['model_type']
pop_or_single = model_meta_info['pop_or_single']

build_dataset_fcn = 'build_dataset_' + dataset_ID
build_dataset = getattr(build_datasets, build_dataset_fcn) # use building function for the chosen dataset

n_h = model_meta_info['n_h'] # history steps for cochleagram tensorisation
# number of bins for zero-padding - usually equal to tensorisation history - 1, but not for conv models
padding_bins = model_meta_info['padding_bins']

params = {'cochleagram_type': cochleagram_type, 'bin_size': bin_size, 'n_F': n_F, 'min_F': min_F, 'max_F': max_F, 
          'padding_bins': padding_bins, 'device': device}

# determine if stimuli must be zero-padded (for padding_bins bins)
# this enables to predict whole responses - instead of not predicting first n_h (tensorisation) bins
zero_padding = model_meta_info['zero_padding']

# build dataset with specified parameters
stim_dataset, resp_dataset, params = build_dataset(params, zero_padding)
# for NS1, NS2: stim_dataset contains all cochleagrams in a NumPy array (dimensions -> (f, t, stimuli));
# for NS2_include_single, NS3: stim_dataset contains a list with two NumPy arrays, one with multi-repeat cochleagrams, and
# one with single-repeat cochleagrams (dimensions -> (f, t, stimuli))

n_neurons = params['n_neurons']
n_stimuli = params['n_stimuli']
n_repeats = params['n_repeats']

bin_size = params['bin_size']
n_F = params['n_F']
min_F = params['min_F']
max_F = params['max_F']
t_axis = params['t_axis']

params['n_h'] = n_h

clip_bins = model_meta_info['clip_bins'] # number of time bins corresponding to clip_length
shift_bins = model_meta_info['shift_bins'] # number of bins to shift the responses by to account for neuronal latency
    

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
elif dataset_ID == 'NS3_PEG':
    folds = 1
    stimuli['validation_stimuli'] = [[7, 16, 9, 11, 0, 3, 15]]
    stimuli['training_stimuli'] = [[1, 4, 12, 2, 5, 6, 13] + list(range(np.size(single_rep_cochleagrams, 2)))]

test_stimuli = [test_stimuli + stimuli['validation_stimuli'][0]][0]
if dataset_ID == 'NS2_include_single' or dataset_ID == 'NS3':
    cv_stimuli = [0, 1, 4, 16, 10, 12, 15]
elif dataset_ID == 'NS3_PEG':
    cv_stimuli = [1, 4, 12, 2, 5, 6, 13]

# mean and std of CV set - if zero-padded, exclude padding_bins from the calculation
if zero_padding:
    # CV set made up of CV multi-repeat stimuli and all single-repeat stimuli
    mean_cv = np.mean([np.mean(single_rep_cochleagrams[:, padding_bins:, :][:]), 
                       np.mean(multi_rep_cochleagrams[:, padding_bins:, cv_stimuli][:])])
    std_cv = np.std(np.concatenate((multi_rep_cochleagrams[:, :, cv_stimuli], single_rep_cochleagrams), axis=2)
                    [:, padding_bins:, :][:], ddof=1)
else:
    mean_cv = np.mean([np.mean(single_rep_cochleagrams[:]), np.mean(multi_rep_cochleagrams[:, :, cv_stimuli][:])])
    std_cv = np.std(np.concatenate((multi_rep_cochleagrams[:, :, cv_stimuli], single_rep_cochleagrams), axis=2)[:], 
                    ddof=1)

# normalisation parameter: determines whether dataset is normalised to zero-mean and unit variance over all stimuli 
# in a set or stimulus by stimulus
norm = model_meta_info['norm']

# get dataset variables: training stimuli, training responses (PSTHs), validation stimuli, validation responses 
# (PSTHs), multi-repeat spike times, single-repeat spike times
train_stim_dataset, train_resp_dataset, val_stim_dataset, val_resp_dataset, cv_stim_dataset, cv_resp_dataset, \
    test_stim_dataset, test_resp_dataset, multi_rep_resps, single_rep_resps = \
        build_grid_NS2_include_single_NS3_concat(single_rep_cochleagrams, multi_rep_cochleagrams, n_neurons, 
                                                 resp_dataset, stimuli, params, norm, mean_cv, std_cv, clip_bins, 
                                                 shift_bins, type_data, zero_padding, padding_bins, t_axis)

tune_stim_dataset = val_stim_dataset # tuning stimuli, same as validation stimuli
tune_resp_dataset = val_resp_dataset # tuning responses, same as validation responses

# for practical reasons, redefine resp_dataset and stim_dataset to only include multi-repeat stimuli and responses
resp_dataset = multi_rep_resps
stim_dataset = multi_rep_cochleagrams


# CV set corresponds to training set
cv_stim_dataset = torch.squeeze(train_stim_dataset, 0)
cv_resp_dataset = torch.squeeze(train_resp_dataset, 1)

test_stim_dataset = torch.cat((test_stim_dataset, torch.squeeze(val_stim_dataset, 0)), dim=2)
test_resp_dataset = torch.cat((test_resp_dataset, torch.squeeze(val_resp_dataset, 1)), dim=1)

# training/validation stimuli and responses: used to compute CCnorm and loss on training data to compare to tuning 
# values; only training multi-repeats are considered, this is necessary to be able to compute CCnorm
cv_val_stim_dataset = cv_stim_dataset[:, :, :int(train_stim_dataset.size(3)/len(stimuli['training_stimuli'][0]))*7]
cv_val_resp_dataset = cv_resp_dataset[:, :int(train_resp_dataset.size(2)/len(stimuli['training_stimuli'][0]))*7]



filename = 'fit_ccnorms_' + model_ID + '_' + dataset_ID + '.pckl' 

f = open(filename, 'rb')
model_fit_ccnorms = pickle.load(f)
f.close()


train_ccnorms = model_fit_ccnorms[1]
val_ccnorms = model_fit_ccnorms[3]

neuron_best_lambdas = np.zeros((n_neurons), dtype=int)

for nID in range(n_neurons):
    if np.all(np.isnan(np.nanmean(val_ccnorms[nID], 0))):
        neuron_best_lambdas[nID] = 0
    else:
        neuron_best_lambdas[nID] = np.nanargmax(np.nanmean(val_ccnorms[nID], 0))

unique_best_lambdas = np.unique(neuron_best_lambdas) 
best_lambda = stats.mode(neuron_best_lambdas)[0][0]

# L1 regularisation strength
lambda_sequence = model_meta_info['lambdas']

# stimulus and response lengths are not necessarily the same - e.g., if padding_bins is not equal to his_steps-1
stim_batch_size = model_meta_info['stim_batch_size'] # stimulus minibatch size in time bins
resp_batch_size = model_meta_info['resp_batch_size'] # response minibatch size in time bins

# Model hyperparameters
num_epochs = model_meta_info['epochs'] # number of training epochs
learning_rate = model_meta_info['lr'] # optimisation learning rate
adaptive_lr = model_meta_info['adaptive_lr'] # controls whether learning rate changes dependent on tuning loss

criterion_type = model_meta_info['loss'] # loss function type
loss_fcn = nn.MSELoss() # loss function, defined here for practical reasons
optim = model_meta_info['optimiser'] # optimiser type

regul_type = model_meta_info['regul'] # regularisation type
out_nl_type = model_meta_info['out_nl'] # output nonlinearity type for model units
# controls whether gradient is set to 0 or None upon resetting gradients of all optimised model parameters
grad_set_to_none = model_meta_info['grad_set_to_none']
grad_clipping = model_meta_info['grad_clipping'] # controls use of gradient clipping

# model parameters: input layer, hidden layer, and output layer sizes, learning rate, initialisation type, dropout, stimulus
# length, history steps
model_params = model_meta_info['all_model_params']

# full dataset
dataset = {'cv_stim': cv_stim_dataset, 'cv_resp': cv_resp_dataset, 'test_stim': test_stim_dataset, 
           'test_resp': test_resp_dataset, 'cv_val_stim': cv_val_stim_dataset, 'cv_val_resp': cv_val_resp_dataset}

lamb_test_ccnorms = np.zeros((len(unique_best_lambdas), n_neurons))

for lamb in range(len(unique_best_lambdas)):
    model = getattr(all_models, model_ID)(model_params) # define and initialise model from model_ID
    model.to(device)
    model.reload(model_fitted_params[0][unique_best_lambdas[lamb]][0])
    
    test_ccnorms = val_loop(model, loss_fcn, test_stim_dataset, test_resp_dataset, n_neurons, None, None, test_stimuli, 
                            resp_dataset, stim_dataset, model_params['his_steps'], bin_size, norm, mean_cv, std_cv, 
                            clip_bins, shift_bins, zero_padding, padding_bins, model_type, t_axis, split)
    
    lamb_test_ccnorms[lamb] = test_ccnorms

elapsed = time.time() - t_track # stop timer

# meta info of model run
meta_info = {'cochleagram' : cochleagram_type, 'n_F' : n_F, 'clip_bins' : clip_bins, 'shift_bins': shift_bins, 'n_h' : n_h, 
             'norm' : norm, 'lambdas' : lambda_sequence, 'hidden_size' : model_params['hidden_size'], 'epochs' : num_epochs, 
             'lr' : learning_rate, 'stim_batch_size' : stim_batch_size, 'resp_batch_size' : resp_batch_size, 
             'batch_data_num' : model_meta_info['batch_data_num'], 'out_nl' : out_nl_type, 'loss' : criterion_type, 
             'optimiser' : optim, 'regul' : regul_type, 'adaptive_lr' : adaptive_lr, 'time' : elapsed, 
             'cudnn_benchmark' : cudnn_benchmark, 'early_stopping' : early_stop, 
             'init_network' : model_params['initialisation'], 'AMP' : amp_type, 'grad_set_to_none' : grad_set_to_none, 
             'grad_clipping' : grad_clipping, 'dropout' : model_params['dropout'], 'bin_size': bin_size, 
             'zero_padding': zero_padding, 'padding_bins': padding_bins, 'reduced': reduced, 'type_data': type_data, 
             'model_type': model_type, 'pop_or_single': pop_or_single, 'all_model_params': model_params, 'time_axis': t_axis,
             'neuron_best_lambdas': neuron_best_lambdas, 'best_lambda': best_lambda}

bl_test_ccnorms = np.zeros((n_neurons))

for nID in range(n_neurons):
    nID_bl_idx = np.where(unique_best_lambdas == neuron_best_lambdas[nID])[0][0]
    bl_test_ccnorms[nID] = lamb_test_ccnorms[nID_bl_idx, nID]

avg_test_ccnorm = np.nanmean(bl_test_ccnorms)
med_test_ccnorm = np.nanmedian(bl_test_ccnorms)

# save fit results, fitted model parameters, and model meta info to pickle files
save_fit = False
if save_fit is True:
    f = open('final_fit_' + model_ID + '_' + dataset_ID + '.pckl', 'wb')
    pickle.dump([lamb_test_ccnorms, bl_test_ccnorms, meta_info], f)
    f.close()