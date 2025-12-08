#!/usr/bin/env python3
import sys
sys.path.insert(1, '/home/lorenzom/All_code')

import build_datasets
import os

import pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import cochlea_lorenzo
import soundfile as sf
from util_funcs import train_val_test_split, plot_stimuli
from tensorize_mod import tensorize_mod
from itertools import count

import all_models

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
# model types: convolutional or not - determines whether input stimuli must be tensorised or not
model_types = {'not_conv': ['STRF', 'LN', 'pop_NRF', 'pop_LRU', 'pop_RNN', 'pop_MGU', 'pop_GRU', 'pop_LSTM', 'pop_subLSTM', 
                            'pop_f_LSTM', 'pop_rc_LSTM', 'pop_gatesLSTM', 'pop_gatesGRU', 'pop_gatessubLSTM', 
                            'pop_gatesfLSTM', 'pop_directrcLSTM', 'pop_fix_subLSTM', 'pop_t_out_subLSTM'],
               'conv': ['LN_conv', 'single_CNN', 'pop_STRF', 'pop_STRF_noFC', 'pop_LN', 'pop_LN_noFC', 'oneD_CNN', 
                        'oneD_x2_CNN', 'twoD_CNN', 'pop_conv_GRU']}

cochleagram_type = 'spec_power'
type_data = 'torch' # data type, either NumPy or PyTorch
model_ID = 'pop_f_LSTM'

fixed_sets = True

# assign type and architecture (see above) to model
for key in model_types:
    if model_ID in model_types[key]:
        model_type = key

# cochleagram parameters
bin_size = 4 # time discretisation
n_F = 8 # number of frequency channels

amended_f_range = True

if amended_f_range:
    min_F = 20 # minimum channel centre frequency
    max_F = 20000 # maximum channel centre frequency
else:
    min_F = 500 # minimum channel centre frequency
    max_F = 22000 # maximum channel centre frequency
    
f_range = np.logspace(np.log10(min_F), np.log10(max_F), n_F)

if os.path.isfile(str(n_F) + '_f_cochleagrams_t_axes.pckl'):
    cochleagram_file = str(n_F) + '_f_cochleagrams_t_axes.pckl'
    f = open(cochleagram_file, 'rb')
    obj = pickle.load(f)
    f.close()
    
    cochleagrams = obj[0]
    t_axes = obj[1]
    
else:
    sounds_dir = 'Sounds/' # stimulus files directory
    soundFiles = os.listdir(sounds_dir) # list content of directory
    
    # compute cochleagrams and relative time and frequency axes for multi-repeat stimuli
    cochleagram_fcn = 'cochleagram_' + cochleagram_type
    cochleagram = getattr(cochlea_lorenzo, cochleagram_fcn) # use cochleagram function for the chosen type
    
    cochleagrams = []
    t_axes = []
    
    for sID in range(len(soundFiles)):
        print(sID)
        # Read in sounds and their sampling frequencies
        y, fs = sf.read(sounds_dir + soundFiles[sID])
        
        if len(np.shape(y)) > 1:
            y = y[:, -1]
            
        
        # 'log' indicates log spacing of frequency channels
        X_ft, params = cochleagram(y, fs, bin_size, 'NS3', False, 0, spacing='log', freq_info=[min_F, max_F, n_F])
        t = params['spectrogram_t']
        
        #plot_stimuli(y, fs, X_ft, bin_size, n_F, min_F, max_F)
        
        cochleagrams.append(X_ft)
        t_axes.append(t)
        
    f = open(str(n_F) + '_f_cochleagrams_t_axes.pckl', 'wb')
    pickle.dump([cochleagrams, t_axes], f)
    f.close()

n_sounds = len(cochleagrams)
folds = 4

# split into train, validation, and test sets
if fixed_sets:
    idxs_file = 'all_dataset_idxs.pckl'

    f = open(idxs_file, 'rb')
    obj = pickle.load(f)
    f.close()

    trainval_idxs = obj[0]
    test_idxs = obj[1]
    train_idxs = obj[2]
    val_idxs = obj[3]
    
else:
    trainval_dataset_length = int(n_sounds*0.9)
    shuffled_idxs = np.random.permutation(n_sounds)
    trainval_idxs = shuffled_idxs[:trainval_dataset_length]
    test_idxs = shuffled_idxs[trainval_dataset_length:]

    val_dataset_length = int(trainval_dataset_length*0.1)
    train_idxs = []
    val_idxs = []
    for fold in range(folds):
        if fold == 0:
            train_idxs.append(trainval_idxs[(fold+1)*val_dataset_length:])
        else:
            train_idxs.append(np.concatenate((trainval_idxs[:fold*val_dataset_length], trainval_idxs[(fold+1)*val_dataset_length:])))
        val_idxs.append(trainval_idxs[fold*val_dataset_length:(fold+1)*val_dataset_length])

    f = open('all_dataset_idxs.pckl', 'wb')
    pickle.dump([trainval_idxs, test_idxs, train_idxs, val_idxs], f)
    f.close()

# mean and std of CV set - if zero-padded, exclude padding_bins from the calculation
trainval_dataset = [cochleagrams[idx] for idx in trainval_idxs]
mean_cv = np.mean([np.mean(stim) for stim in trainval_dataset])
std_cv = np.std(np.concatenate(trainval_dataset, axis=None).ravel(), ddof=1)
mean_all = mean_cv
std_all = std_cv

pred_window_size = 5
past_window_size = 8

# normalisation parameter: determines whether dataset is normalised to zero-mean and unit variance over all stimuli 
# in a set or stimulus-by-stimulus
norm = 'all'

if os.path.isfile('Cochleagrams/' + str(n_F) + '_f_t_cochleagrams_' + str(past_window_size) + '_' + \
                  str(pred_window_size) + '.pckl'):
    t_cochleagram_file = 'Cochleagrams/' + str(n_F) + '_f_t_cochleagrams_' + str(past_window_size) + '_' + \
        str(pred_window_size) + '.pckl'
    f = open(t_cochleagram_file, 'rb')
    obj = pickle.load(f)
    f.close()
    
    t_cochleagrams = obj[0]
    
else:
    t_cochleagrams = []
    
    for sID in range(n_sounds):
        print(sID)
        if norm == 'all': # if "all", normalise stimuli using whole set (usually, cross-validation set)
            X_ft_normalized = (cochleagrams[sID] - mean_all) / std_all

        X_ft_normalized = torch.tensor(X_ft_normalized, dtype=torch.float32).to(device)
        past_windows = torch.tensor([], dtype=torch.float32).to(device)
        pred_windows = torch.tensor([], dtype=torch.float32).to(device)
        
        for i in range(past_window_size, np.size(X_ft_normalized, 1) - pred_window_size):
            past_win = torch.unsqueeze(X_ft_normalized[:, i-past_window_size:i], 2)
            pred_win = torch.unsqueeze(X_ft_normalized[:, i:i+pred_window_size], 2)
            
            past_windows = torch.cat((past_windows, past_win), 2)
            pred_windows = torch.cat((pred_windows, pred_win), 2)
        
        t_cochleagrams.append([past_windows, pred_windows])
        
    f = open('Cochleagrams/' + str(n_F) + '_f_t_cochleagrams_' + str(past_window_size) + '_' + str(pred_window_size) + \
             '.pckl', 'wb')
    pickle.dump([t_cochleagrams], f)
    f.close()

train_dataset = []
val_dataset = []
for fold in range(folds):
    train_dataset.append([t_cochleagrams[idx] for idx in train_idxs[fold]])
    val_dataset.append([t_cochleagrams[idx] for idx in val_idxs[fold]])
trainval_dataset = [t_cochleagrams[idx] for idx in trainval_idxs]
test_dataset = [t_cochleagrams[idx] for idx in test_idxs]

model_params = {'input_size': n_F*past_window_size, 'hidden_size': 150, 'output_size': n_F*pred_window_size, 
                'dropout': False, 'stim_size': 5, 'initialisation': 'default'}
lambdas = [1.28e-8, 2.56e-9, 5.12e-10, 1.024e-10, 2.048e-11, 4.096e-12]
batch_size = 1
epochs = 250

mode = 'train'

if mode == 'train':
    all_train_loss = []
    all_val_loss = []
    all_model_fits = []
    
    for lamb_idx, lamb in enumerate(lambdas):
        print('lambda ' + str(lamb_idx))
        f_train_loss = []
        f_val_loss = []
        f_model_fits = []
        
        for fold in range(folds):
            print('fold ' + str(fold))
            
            net = getattr(all_models, model_ID)(model_params) # define and initialise model from model_ID
            net.to(device)
            
            f_train_dataset = train_dataset[fold]
            f_val_dataset = val_dataset[fold]
    
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
            
            train_loss_arr = []
            val_loss_arr = []
            
            for epoch in range(epochs):
                train_temp = []
                val_temp = []
                
                n_batches = int(len(f_train_dataset)/batch_size)
                for batch in range(n_batches):
                    clip = f_train_dataset[batch]
                    clip_past_frames = torch.transpose(torch.reshape(clip[0], (n_F*past_window_size, -1)), 1, 0)
                    clip_next_frames = torch.transpose(torch.reshape(clip[1], (n_F*pred_window_size, -1)), 1, 0)
                    optimizer.zero_grad()
                        
                    stage = 'train'
                    
                    if 'NRF' in model_ID:
                        clip_prediction = net(clip_past_frames, stage)
                    else:
                        h_n = torch.zeros((1, net.hidden_size), device=device)
                        
                        if 'LSTM' not in model_ID:
                            clip_prediction, _ = net(clip_past_frames, h_n, stage)
                        else:
                            c_n = torch.zeros((1, net.hidden_size), device=device)
                            clip_prediction, _, _ = net(clip_past_frames, h_n, c_n, stage)
            
                    loss = nn.functional.mse_loss(clip_next_frames, clip_prediction) + net.regul(lamb)
                            
                    loss.backward()
                    optimizer.step()
                    train_temp.append(loss.item())
                   
                # Val loop
                n_batches = int(len(f_val_dataset)/batch_size)
                for batch in range(n_batches):                                          
                    with torch.no_grad():
                        clip = f_val_dataset[batch]
                        clip_past_frames = torch.transpose(torch.reshape(clip[0], (n_F*past_window_size, -1)), 1, 0)
                        clip_next_frames = torch.transpose(torch.reshape(clip[1], (n_F*pred_window_size, -1)), 1, 0)
                        
                        if 'NRF' in model_ID:
                            clip_prediction = net(clip_past_frames, stage)
                        else:
                            h_n = torch.zeros((1, net.hidden_size), device=device)
                            
                            if 'LSTM' not in model_ID:
                                clip_prediction, _ = net(clip_past_frames, h_n, stage)
                            else:
                                c_n = torch.zeros((1, net.hidden_size), device=device)
                                clip_prediction, _, _ = net(clip_past_frames, h_n, c_n, stage)
                                    
                        loss = nn.functional.mse_loss(clip_next_frames, clip_prediction) + net.regul(lamb)
            
                        val_temp.append(loss.item())
                    
                train_loss_arr.append(np.mean(train_temp))
                val_loss_arr.append(np.mean(val_temp))
                
                if epoch % 50 == 0:
                    print('Epoch:', epoch, '\tLoss:', train_loss_arr[-1])
            
            model_fit = net.get_params()
            
            f_train_loss.append(train_loss_arr)
            f_val_loss.append(val_loss_arr) 
            f_model_fits.append(model_fit)
            
        all_train_loss.append(f_train_loss)
        all_val_loss.append(f_val_loss)
        all_model_fits.append(f_model_fits)
    
    meta_info = {'cochleagram' : cochleagram_type, 'n_F' : n_F, 'norm' : norm, 'hidden_size' : model_params['hidden_size'], 
                 'epochs' : epochs, 'lr' : optimizer.state_dict()['param_groups'][0]['lr'], 'lambda': lambdas, 'folds': folds,
                 'init_network' : model_params['initialisation'], 'dropout' : model_params['dropout'], 'bin_size': bin_size, 
                 'type_data': type_data, 'model_type': model_type, 'all_model_params': model_params, 'time_axis': t_axes, 
                 'amended_f_range': amended_f_range, 'past_win': past_window_size, 'pred_win': pred_window_size, 
                 'fixed_sets': fixed_sets}
    
    # save fit results, fitted model parameters, and model meta info to pickle files
    save_fit = False
    if save_fit is True:
        f = open('Fit_files/' + str(n_F) + '_f/' + str(model_params['hidden_size']) + '_HU/all_val_results_' + model_ID + '_' + \
                 str(past_window_size) + '_' + str(pred_window_size) + '_' + str(epochs) + '.pckl', 'wb')
        pickle.dump([all_model_fits, all_train_loss, all_val_loss, meta_info], f)
        f.close()
        
elif mode == 'test':
    HU = 150
    f_chan = 8
    
    fit_file = 'Fit_files/' + str(f_chan) + '_f/' + str(HU) + '_HU/all_val_results_' + model_ID + \
        '_' + str(past_window_size) + '_' + str(pred_window_size) + '_' + str(epochs) + '.pckl'
    f = open(fit_file, 'rb')
    obj = pickle.load(f)
    f.close()
    
    lambdas = obj[3]['lambda']
    n_lambdas = len(lambdas)
    n_folds = obj[3]['folds']
    
    f_train_err = np.zeros((n_lambdas, n_folds))
    f_val_err = np.zeros((n_lambdas, n_folds))
    
    for lamb in range(n_lambdas):
        for fold in range(n_folds):
            f_train_err[lamb, fold] = obj[1][lamb][fold][-1]
            f_val_err[lamb, fold] = obj[2][lamb][fold][-1]
    
    f_mean_train_err = np.mean(f_train_err, 1)
    f_mean_val_err = np.mean(f_val_err, 1)
    
    best_val_error = np.nanmin(f_mean_val_err)
    best_lamb = np.nanargmin(f_mean_val_err)
    
    model_fit = obj[0][best_lamb]
    
    model_params = {'input_size': n_F*past_window_size, 'hidden_size': HU, 'output_size': n_F*pred_window_size, 
                    'dropout': False, 'stim_size': 5, 'initialisation': 'default'}
    
    trainval_loss = []
    test_loss = []

    net = getattr(all_models, model_ID)(model_params) # define and initialise model from model_ID
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        trainval_temp = []
        
        n_batches = int(len(trainval_dataset)/batch_size)
        for batch in range(n_batches):
            clip = trainval_dataset[batch]
            clip_past_frames = torch.transpose(torch.reshape(clip[0], (n_F*past_window_size, -1)), 1, 0)
            clip_next_frames = torch.transpose(torch.reshape(clip[1], (n_F*pred_window_size, -1)), 1, 0)
            optimizer.zero_grad()
                
            stage = 'train'
            
            if 'NRF' in model_ID:
                clip_prediction = net(clip_past_frames, stage)
            else:
                h_n = torch.zeros((1, net.hidden_size), device=device)
                
                if 'LSTM' not in model_ID:
                    clip_prediction, _ = net(clip_past_frames, h_n, stage)
                else:
                    c_n = torch.zeros((1, net.hidden_size), device=device)
                    clip_prediction, _, _ = net(clip_past_frames, h_n, c_n, stage)
    
            loss = nn.functional.mse_loss(clip_next_frames, clip_prediction) + net.regul(lambdas[best_lamb])
                    
            loss.backward()
            optimizer.step()
            trainval_temp.append(loss.item())
           
        # Test loop
        test_temp = []
        
        n_batches = int(len(test_dataset)/batch_size)
        for batch in range(n_batches):                                          
            with torch.no_grad():
                clip = test_dataset[batch]
                clip_past_frames = torch.transpose(torch.reshape(clip[0], (n_F*past_window_size, -1)), 1, 0)
                clip_next_frames = torch.transpose(torch.reshape(clip[1], (n_F*pred_window_size, -1)), 1, 0)
                
                if 'NRF' in model_ID:
                    clip_prediction = net(clip_past_frames, stage)
                else:
                    h_n = torch.zeros((1, net.hidden_size), device=device)
                    
                    if 'LSTM' not in model_ID:
                        clip_prediction, _ = net(clip_past_frames, h_n, stage)
                    else:
                        c_n = torch.zeros((1, net.hidden_size), device=device)
                        clip_prediction, _, _ = net(clip_past_frames, h_n, c_n, stage)
                            
                loss = nn.functional.mse_loss(clip_next_frames, clip_prediction) + net.regul(lambdas[best_lamb])
    
                test_temp.append(loss.item())
            
        trainval_loss.append(np.mean(trainval_temp))
        test_loss.append(np.mean(test_temp))
        
        if epoch % 50 == 0:
            print('Epoch:', epoch, '\tLoss:', trainval_loss[-1])
        
    plt.figure()
    plt.plot(trainval_loss, label='Trainval')
    plt.plot(test_loss, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.title(model_ID)
    plt.show()
    
    model_fit = net.get_params()
    
    meta_info = {'cochleagram' : cochleagram_type, 'n_F' : n_F, 'norm' : norm, 'hidden_size' : model_params['hidden_size'], 
                 'epochs' : epochs, 'lr' : optimizer.state_dict()['param_groups'][0]['lr'], 'lambda': lambdas, 'folds': folds,
                 'init_network' : model_params['initialisation'], 'dropout' : model_params['dropout'], 'bin_size': bin_size, 
                 'type_data': type_data, 'model_type': model_type, 'all_model_params': model_params, 'time_axis': t_axes, 
                 'amended_f_range': amended_f_range, 'past_win': past_window_size, 'pred_win': pred_window_size, 
                 'fixed_sets': fixed_sets}
    
    # save fit results, fitted model parameters, and model meta info to pickle files
    save_fit = False
    if save_fit is True:
        f = open('Fit_files/Test/' + str(f_chan) + '_' + str(model_params['hidden_size']) + '_test_results_' + model_ID + \
                 '_' + str(past_window_size) + '_' + str(pred_window_size) + '_' + str(epochs) + '.pckl', 'wb')
        pickle.dump([model_fit, trainval_loss, test_loss, meta_info], f)
        f.close()
        
elif mode == 'test_no_loop':
    HU = 150
    f_chan = 8
    
    fit_file = 'Fit_files/' + str(f_chan) + '_f/' + str(HU) + '_HU/all_val_results_' + model_ID + \
        '_' + str(past_window_size) + '_' + str(pred_window_size) + '_' + str(epochs) + '.pckl'
    f = open(fit_file, 'rb')
    obj = pickle.load(f)
    f.close()
    
    lambdas = obj[3]['lambda']
    n_lambdas = len(lambdas)
    n_folds = obj[3]['folds']
    
    f_train_err = np.zeros((n_lambdas, n_folds))
    f_val_err = np.zeros((n_lambdas, n_folds))
    
    for lamb in range(n_lambdas):
        for fold in range(n_folds):
            f_train_err[lamb, fold] = obj[1][lamb][fold][-1]
            f_val_err[lamb, fold] = obj[2][lamb][fold][-1]
    
    f_mean_train_err = np.mean(f_train_err, 1)
    f_mean_val_err = np.mean(f_val_err, 1)
    
    best_val_error = np.nanmin(f_mean_val_err)
    best_lamb = np.nanargmin(f_mean_val_err)
    
    model_fit = obj[0][best_lamb]
    
    model_params = {'input_size': n_F*past_window_size, 'hidden_size': HU, 'output_size': n_F*pred_window_size, 
                    'dropout': False, 'stim_size': 5, 'initialisation': 'default'}
    
    trainval_loss = []

    net = getattr(all_models, model_ID)(model_params) # define and initialise model from model_ID
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        print(epoch)
        trainval_temp = []
        
        n_batches = int(len(trainval_dataset)/batch_size)
        for batch in range(n_batches):
            clip = trainval_dataset[batch]
            clip_past_frames = torch.transpose(torch.reshape(clip[0], (n_F*past_window_size, -1)), 1, 0)
            clip_next_frames = torch.transpose(torch.reshape(clip[1], (n_F*pred_window_size, -1)), 1, 0)
            optimizer.zero_grad()
                
            stage = 'train'
            
            if 'NRF' in model_ID:
                clip_prediction = net(clip_past_frames, stage)
            else:
                h_n = torch.zeros((1, net.hidden_size), device=device)
                
                if 'LSTM' not in model_ID:
                    clip_prediction, _ = net(clip_past_frames, h_n, stage)
                else:
                    c_n = torch.zeros((1, net.hidden_size), device=device)
                    clip_prediction, _, _ = net(clip_past_frames, h_n, c_n, stage)
    
            loss = nn.functional.mse_loss(clip_next_frames, clip_prediction) + net.regul(lambdas[best_lamb])
                    
            loss.backward()
            optimizer.step()
            trainval_temp.append(loss.item())
        
        trainval_loss.append(np.mean(trainval_temp))
        
        # Test loop
        test_temp = []
    
    test_loss = []
    n_batches = int(len(test_dataset)/batch_size)
    for batch in range(n_batches):                                          
        with torch.no_grad():
            clip = test_dataset[batch]
            clip_past_frames = torch.transpose(torch.reshape(clip[0], (n_F*past_window_size, -1)), 1, 0)
            clip_next_frames = torch.transpose(torch.reshape(clip[1], (n_F*pred_window_size, -1)), 1, 0)
            
            if 'NRF' in model_ID:
                clip_prediction = net(clip_past_frames, stage)
            else:
                h_n = torch.zeros((1, net.hidden_size), device=device)
                
                if 'LSTM' not in model_ID:
                    clip_prediction, _ = net(clip_past_frames, h_n, stage)
                else:
                    c_n = torch.zeros((1, net.hidden_size), device=device)
                    clip_prediction, _, _ = net(clip_past_frames, h_n, c_n, stage)
                        
            loss = nn.functional.mse_loss(clip_next_frames, clip_prediction) + net.regul(lambdas[best_lamb])

            test_loss.append(loss.item())

    model_fit = net.get_params()
    
    meta_info = {'cochleagram' : cochleagram_type, 'n_F' : n_F, 'norm' : norm, 'hidden_size' : model_params['hidden_size'], 
                 'epochs' : epochs, 'lr' : optimizer.state_dict()['param_groups'][0]['lr'], 'lambda': lambdas, 'folds': folds,
                 'init_network' : model_params['initialisation'], 'dropout' : model_params['dropout'], 'bin_size': bin_size, 
                 'type_data': type_data, 'model_type': model_type, 'all_model_params': model_params, 'time_axis': t_axes, 
                 'amended_f_range': amended_f_range, 'past_win': past_window_size, 'pred_win': pred_window_size, 
                 'fixed_sets': fixed_sets}
    
    # save fit results, fitted model parameters, and model meta info to pickle files
    save_fit = False
    if save_fit is True:
        f = open('Fit_files/Test/' + str(f_chan) + '_' + str(model_params['hidden_size']) + '_test_no_loop_results_' + \
                 model_ID + '_' + str(past_window_size) + '_' + str(pred_window_size) + '_' + str(epochs) + '.pckl', 'wb')
        pickle.dump([model_fit, trainval_loss, test_loss, meta_info], f)
        f.close()
