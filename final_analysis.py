import build_datasets

import pickle
import torch
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt

from util_funcs import train_val_test_split, build_grid_NS2_include_single_NS3_concat

from analysis_funcs import corr_with_stims, corr_with_stims_silence, corr_with_stims_silence_by_stim, \
    corr_with_stims_sound, stims_revcorr, effective_HUs, corr_with_HUs, corr_with_gates, corr_with_t_gate, \
        weight_sparsity, gate_weights, any_CC_norms, corr_with_ON_OFF_stims, multi_analysis_scatter, \
            multi_analysis_stim_only, rcLSTM_weight_share, rcLSTM_weighted_output, count_params, calc_se_median
from plot_funcs import plot_corr_stims, plot_corr_stims_silence_or_sound, plot_gate_kernels, plot_corr_HUs, \
    plot_corr_gates, plot_weight_sparsity, plot_gate_weights, plot_repetitiveness, comparisons, plot_neurons, \
        plot_loss_curves, depth_sw_analysis, depth_sw_sess_analysis, depth_cell_analysis, depth_t_gate_analysis, \
            depth_gates_analysis, depth_gatesmodels_analysis, depth_silence_analysis, depth_ON_OFF_analysis, \
                depth_rcLSTM_analysis

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_set_type = 'Val_+_test'

datasets = ['NS3_PEG']

models = ['pop_LN', 'pop_NRF', 'pop_RNN', 'pop_MGU', 'pop_GRU', 'pop_LSTM', 'pop_subLSTM', 'pop_f_LSTM', 'pop_rc_LSTM', 
          'oneD_CNN', 'oneD_x2_CNN', 'twoD_CNN']
models = ['pop_f_LSTM']

reduced = False

nr_dict = {}
results_dir = 'Results/'

results_dict = {}

bl_av_results_dict = {} 
bl_med_results_dict = {}

fits_dict = {}
built_data = {}

best_lambdas = {}

corr_stims = {}
corr_stims_silence = {}
corr_stims_sound = {}
corr_ON_OFF_stims = {}

gate_kernels = {} 

eff_HUs = {}

corr_HUs = {}
corr_gates = {}

sparse_out_weights = {}

gates_weights = {}

param_count = {}
se_median = {}

depths = {}
depth_axis = {}
CCnorm_depth_diff = {}
CCnorm_depth_diff_se = {}

for dataset_ID in datasets:
    print(dataset_ID)
    
    models_results_dict = {}
    
    models_bl_av_results_dict = {}
    models_bl_med_results_dict = {}
    
    models_fits_dict = {}
    models_built_data = {}
    
    models_best_lambdas = {}
    
    models_corr_stims = {}
    models_corr_stims_silence = {}
    models_corr_stims_sound = {}
    models_corr_ON_OFF_stims = {}
    
    models_any_ccnorms = {}
    
    models_gate_kernels = {}
    
    models_eff_HUs = {}
    
    models_corr_HUs = {}
    models_corr_gates = {}
    
    models_sparse_out_weights = {}
    
    models_gates_weights = {}
    rcLSTM_weights = {}
    
    models_param_count = {}
    models_se_median = {}
    
    for model_ID in models:
        print(model_ID)
        
        build_dataset_fcn = 'build_dataset_' + dataset_ID
        build_dataset = getattr(build_datasets, build_dataset_fcn)
        
        fit_file = results_dir + 'final_fit_' + model_ID + '_' + dataset_ID + '.pckl'
        f = open(fit_file, 'rb')
        obj = pickle.load(f)
        f.close()
        
        models_bl_av_results_dict[model_ID] = [np.nanmean(obj[1]), np.nanstd(obj[1], ddof=1)/np.sqrt(len(obj[1]))]
        models_bl_med_results_dict[model_ID] = [np.nanmedian(obj[1])]
        
        model_meta_info = obj[2]
        
        best_lambda = model_meta_info['best_lambda'] # best overall lambda (mode of all best lambdas for each neuron)
        neuron_best_lambdas = model_meta_info['neuron_best_lambdas'] # best lambda for each neuron
        models_best_lambdas[model_ID] = neuron_best_lambdas
        
        # index of overall best lambda out of the unique best lambdas
        best_lamb_idx = list(np.unique(neuron_best_lambdas)).index(best_lambda)
        
        models_results_dict[model_ID] = [obj[0], obj[1]]
        
        n_F = model_meta_info['n_F'] # number of frequency channels
        n_epochs = model_meta_info['epochs'] # number of training epochs
        
        model_params_file = results_dir + 'fit_results_' + model_ID + '_' + dataset_ID + '.pckl'
        f = open(model_params_file, 'rb')
        model_fit_results = pickle.load(f)
        f.close()
        
        models_fits_dict[model_ID] = [model_fit_results[0], model_meta_info]
        
        cochleagram_type = model_meta_info['cochleagram']
        type_data = model_meta_info['type_data'] # data type, either NumPy or PyTorch
        
        model_type = model_meta_info['model_type'] # model type: convolutional or nonconvolutional
        pop_or_single = model_meta_info['pop_or_single'] # population or single model
        
        # cochleagram parameters
        bin_size = model_meta_info['bin_size'] # time discretisation
        min_F = 500 # minimum channel centre frequency
        max_F = 22000 # maximum channel centre frequency
        
        n_h = model_meta_info['n_h'] # history steps for cochleagram tensorisation
        # number of bins for zero-padding - usually equal to tensorisation history - 1, but not for conv models
        padding_bins = model_meta_info['padding_bins']

        params = {'cochleagram_type': cochleagram_type, 'bin_size': bin_size, 'n_F': n_F, 'min_F': min_F, 'max_F': max_F, 
                  'padding_bins': padding_bins, 'device': device}
        
        # determine if stimuli must be zero-padded (for padding_bins bins)
        # this enables to predict whole responses - instead of not predicting first n_h (tensorisation) bins
        # which datasets are padded has been determined post-hoc, comparing results between the two conditions
        zero_padding = model_meta_info['zero_padding']
        
        # build dataset with specified parameters
        stim_dataset, resp_dataset, params = build_dataset(params, zero_padding)
        # stim_dataset contains a list with two NumPy arrays, one with multi-repeat cochleagrams, and one with single-repeat 
        # cochleagrams (dimensions -> (f, t, stimuli))
        
        n_neurons = params['n_neurons']
        n_stimuli = params['n_stimuli']
        t_axis = params['t_axis'] # time axis used in cochleagram computation
        params['n_h'] = n_h
        
        noise_ratios = params['noise_ratios']
            
        nr_dict[dataset_ID] = np.array(noise_ratios)
        
        #####################################################################################################################
        clip_bins = model_meta_info['clip_bins'] # number of time bins to clip from the start of sounds and responses
        shift_bins = model_meta_info['shift_bins'] # number of bins to shift responses by to account for neuronal latency
        
        
        multi_rep_cochleagrams = stim_dataset[0] # multi-repeat cochleagrams
        single_rep_cochleagrams = stim_dataset[1] # single-repeat cochleagrams
        
        # clip given number of bins from start of cochleagrams - if zero-padded, keep padding bins at the start
        if zero_padding:
            multi_rep_cochleagrams = np.concatenate((multi_rep_cochleagrams[:, :padding_bins, :], 
                                                     multi_rep_cochleagrams[:, padding_bins+clip_bins:, :]), axis=1)
            single_rep_cochleagrams = np.concatenate((single_rep_cochleagrams[:, :padding_bins, :], 
                                                      single_rep_cochleagrams[:, padding_bins+clip_bins:, :]), 
                                                     axis=1)
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
        
        # mean and std of CV set - if zero-padded, exclude padding_bins from the calculation
        if zero_padding:
            # CV set made up of CV multi-repeat stimuli and all single-repeat stimuli
            mean_cv = np.mean([np.mean(single_rep_cochleagrams[:, padding_bins:, :][:]), 
                               np.mean(multi_rep_cochleagrams[:, padding_bins:, cv_stimuli][:])])
            std_cv = np.std(np.concatenate((multi_rep_cochleagrams[:, :, cv_stimuli], single_rep_cochleagrams), 
                                           axis=2)[:, padding_bins:, :][:], ddof=1)
        else:
            mean_cv = np.mean([np.mean(single_rep_cochleagrams[:]), 
                               np.mean(multi_rep_cochleagrams[:, :, cv_stimuli][:])])
            std_cv = np.std(np.concatenate((multi_rep_cochleagrams[:, :, cv_stimuli], single_rep_cochleagrams), 
                                           axis=2)[:], ddof=1)
        
        # normalisation parameter: determines whether dataset is normalised to zero-mean and unit variance over all 
        # stimuli in a set or stimulus-by-stimulus
        norm = model_meta_info['norm']
        
        # get dataset variables: training stimuli, training responses (PSTHs), validation stimuli, validation 
        # responses (PSTHs), multi-repeat spike times, single-repeat spike times
        train_stim_dataset, train_resp_dataset, val_stim_dataset, val_resp_dataset, cv_stim_dataset, \
            cv_resp_dataset, test_stim_dataset, test_resp_dataset, multi_rep_resps, single_rep_resps = \
                build_grid_NS2_include_single_NS3_concat(single_rep_cochleagrams, multi_rep_cochleagrams, 
                                                         n_neurons, resp_dataset, stimuli, params, norm, mean_cv, 
                                                         std_cv, clip_bins, shift_bins, type_data, zero_padding, 
                                                         padding_bins, t_axis)
        
        built_data_dict = {'train_stim': train_stim_dataset, 'train_resp': train_resp_dataset, 'val_stim': val_stim_dataset, 
                           'val_resp': val_resp_dataset, 'cv_stim': cv_stim_dataset, 'cv_resp': cv_resp_dataset,
                           'test_stim': test_stim_dataset, 'test_resp': test_resp_dataset, 'multi_stim': stim_dataset,
                           'multi_resp': resp_dataset}
        models_built_data[model_ID] = [built_data_dict, stimuli, folds, [mean_cv, std_cv], params]
        
        # correlation between stimuli and responses
        '''models_corr_stims[model_ID] = corr_with_stims('pearson', 'average_stim', n_neurons, stim_dataset, test_resp_dataset, 
                                                      stimuli['test_stimuli'], zero_padding, padding_bins, n_F, 1)'''
     
        # correlation between silence-triggered stimuli and responses
        # uncomment for SDRM measured on the whole dataset, including single-repeat data
        '''stim_size = int(test_stim_dataset.size(-1)/len(stimuli['test_stimuli'][0]))
        resp_size = int(test_resp_dataset.size(-1)/len(stimuli['test_stimuli'][0]))
        all_resp_dataset = torch.cat((test_resp_dataset, train_resp_dataset), dim=2)
        all_stimuli = [stimuli['test_stimuli'][0] + stimuli['training_stimuli'][0][:7] + list(np.arange(18, 299))]
        stim_dataset = np.concatenate((multi_rep_cochleagrams, single_rep_cochleagrams), axis=2)
        models_corr_stims_silence[model_ID] = corr_with_stims_silence('pearson', 'average_stim', n_neurons, stim_dataset, 
                                                                      all_resp_dataset, all_stimuli, 
                                                                      models_fits_dict[model_ID], zero_padding, padding_bins, 
                                                                      n_F, 1)'''
        
        # uncomment for SDRM measured on the whole multi-repeat dataset (including 7 sounds used in training)
        '''stim_size = int(test_stim_dataset.size(-1)/len(stimuli['test_stimuli'][0]))
        resp_size = int(test_resp_dataset.size(-1)/len(stimuli['test_stimuli'][0]))
        all_multi_stim_dataset = torch.cat((test_stim_dataset, train_stim_dataset[:, :, :, :7*stim_size]), dim=3)
        all_multi_resp_dataset = torch.cat((test_resp_dataset, train_resp_dataset[:, :, :7*resp_size]), dim=2)
        all_multi_stimuli = [stimuli['test_stimuli'][0] + stimuli['training_stimuli'][0][:7]]
        models_corr_stims_silence[model_ID] = corr_with_stims_silence('pearson', 'average_stim', n_neurons, stim_dataset, 
                                                                      all_multi_resp_dataset, all_multi_stimuli, 
                                                                      models_fits_dict[model_ID], zero_padding, padding_bins, 
                                                                      n_F, 1)'''
        
        # uncomment for SDRM measured on the test set only (either val+test or test only) 
        '''models_corr_stims_silence[model_ID] = corr_with_stims_silence('pearson', 'average_stim', n_neurons, stim_dataset, 
                                                                      test_resp_dataset, stimuli['test_stimuli'], 
                                                                      models_fits_dict[model_ID], zero_padding, padding_bins, 
                                                                      n_F, 1)'''
        
        # uncomment for SDRM measured only on the stimuli with the n highest SDRMs when measured separately        
        '''stim_size = int(test_stim_dataset.size(-1)/len(stimuli['test_stimuli'][0]))
        resp_size = int(test_resp_dataset.size(-1)/len(stimuli['test_stimuli'][0]))
        all_multi_stim_dataset = torch.cat((test_stim_dataset, train_stim_dataset[:, :, :, :7*stim_size]), dim=3)
        all_multi_resp_dataset = torch.cat((test_resp_dataset, train_resp_dataset[:, :, :7*resp_size]), dim=2)
        all_multi_stimuli = [stimuli['test_stimuli'][0] + stimuli['training_stimuli'][0][:7]]
        n = 4
        models_corr_stims_silence[model_ID], stims_used = corr_with_stims_silence_by_stim('pearson', 'average_CC', n, 
                                                                                          n_neurons, stim_dataset, 
                                                                                          all_multi_resp_dataset, 
                                                                                          all_multi_stimuli, zero_padding, 
                                                                                          padding_bins, n_F, 1)
        
        CCnorms = any_CC_norms('together', 'CC_norm', all_multi_stim_dataset, all_multi_resp_dataset, stim_dataset, 
                               resp_dataset, stims_used, models_fits_dict[model_ID], models_built_data, all_multi_stimuli, 
                               models_best_lambdas, n_neurons, model_ID, 0)
        models_any_ccnorms[model_ID] = [[], [], CCnorms] # adding two empty lists for compatibility with other functions'''
        
        # CCnorms calculated on any chosen sounds (stims_used)
        '''stim_size = int(test_stim_dataset.size(-1)/len(stimuli['test_stimuli'][0]))
        resp_size = int(test_resp_dataset.size(-1)/len(stimuli['test_stimuli'][0]))
        all_multi_stim_dataset = torch.cat((test_stim_dataset, train_stim_dataset[:, :, :, :7*stim_size]), dim=3)
        all_multi_resp_dataset = torch.cat((test_resp_dataset, train_resp_dataset[:, :, :7*resp_size]), dim=2)
        all_multi_stimuli = [stimuli['test_stimuli'][0] + stimuli['training_stimuli'][0][:7]]
        stims_used = all_multi_stimuli[0]
        CCnorms = any_CC_norms('separate', 'CC_norm', all_multi_stim_dataset, all_multi_resp_dataset, stim_dataset, 
                               resp_dataset, stims_used, models_fits_dict[model_ID], models_built_data, all_multi_stimuli, 
                               models_best_lambdas, n_neurons, model_ID, 0)
        models_any_ccnorms[model_ID] = [[], [], CCnorms] # adding two empty lists for compatibility with other functions'''
        
        # CCraws calculated on full dataset (can also separate each stimulus and neuron)
        '''stim_size = int(test_stim_dataset.size(-1)/len(stimuli['test_stimuli'][0]))
        resp_size = int(test_resp_dataset.size(-1)/len(stimuli['test_stimuli'][0]))
        all_stim_dataset = torch.cat((test_stim_dataset, train_stim_dataset), dim=3)
        all_resp_dataset = torch.cat((test_resp_dataset, train_resp_dataset), dim=2)
        if dataset_ID == 'NS3' or dataset_ID == 'NS3_PEG':
            all_stimuli = [stimuli['test_stimuli'][0] + stimuli['training_stimuli'][0][:7] + list(np.arange(18, 591))]
        if dataset_ID == 'NS2_include_single':
            all_stimuli = [stimuli['test_stimuli'][0] + stimuli['training_stimuli'][0][:7] + list(np.arange(18, 299))]
        stim_dataset = np.concatenate((multi_rep_cochleagrams, single_rep_cochleagrams), axis=2)
        full_resp_dataset = []
        for nID in range(n_neurons):
            full_resp_dataset.append(multi_rep_resps[nID] + single_rep_resps[nID])
        stims_used = all_stimuli[0]
        CCraws = any_CC_norms('separate', 'CC_raw', all_stim_dataset, all_resp_dataset, stim_dataset, full_resp_dataset, 
                               stims_used, models_fits_dict[model_ID], models_built_data, all_stimuli, models_best_lambdas, 
                               n_neurons, model_ID, 0)
        models_any_ccnorms[model_ID] = [[], [], CCraws]'''
        
        # correlation between sound-triggered stimuli and responses
        '''models_corr_stims_sound[model_ID] = corr_with_stims_sound('pearson', 'average_stim', n_neurons, stim_dataset, 
                                                                  test_resp_dataset, stimuli['test_stimuli'], zero_padding, 
                                                                  padding_bins, n_F, 1)'''
        
        # reverse correlation on stimuli
        '''models_gate_kernels[model_ID] = stims_revcorr(n_neurons, test_stim_dataset, test_resp_dataset, 
                                                      stimuli['test_stimuli'], models_fits_dict[model_ID], model_ID, 1)'''
        
        # effective hidden units
        '''models_eff_HUs[model_ID] = effective_HUs(n_neurons, test_stim_dataset, models_fits_dict[model_ID], model_ID, 1)'''

        # correlation between model predictions and hidden unit output
        '''models_corr_HUs[model_ID] = corr_with_HUs('c', n_neurons, test_stim_dataset, test_resp_dataset, 
                                                  models_fits_dict[model_ID], model_ID, 1)'''
        
        # correlation between model predictions and gates' activity (with either CCraw or CCnorm)
        '''stim_size = int(test_stim_dataset.size(-1)/len(stimuli['test_stimuli'][0]))
        resp_size = int(test_resp_dataset.size(-1)/len(stimuli['test_stimuli'][0]))
        all_multi_stim_dataset = torch.cat((test_stim_dataset, train_stim_dataset[:, :, :, :7*stim_size]), dim=3)
        all_multi_resp_dataset = torch.cat((test_resp_dataset, train_resp_dataset[:, :, :7*resp_size]), dim=2)
        all_multi_stimuli = [stimuli['test_stimuli'][0] + stimuli['training_stimuli'][0][:7]]
        models_corr_gates[model_ID] = corr_with_gates('CCraw', n_neurons, all_multi_stim_dataset, all_multi_resp_dataset, 
                                                      stim_dataset, resp_dataset, all_multi_stimuli, 
                                                      models_fits_dict[model_ID], models_built_data, model_ID, 1, 0)'''
        
        '''models_corr_gates[model_ID] = corr_with_gates('CCnorm', n_neurons, test_stim_dataset, test_resp_dataset, 
                                                      stim_dataset, resp_dataset, stimuli['test_stimuli'], 
                                                      models_fits_dict[model_ID], models_built_data, model_ID, 1, 0)'''
    
        # correlation between PSTHs and input gate (f-LSTM) activity
        '''models_corr_gates[model_ID] = corr_with_t_gate('CCraw', n_neurons, test_stim_dataset, test_resp_dataset, 
                                                       stim_dataset, resp_dataset, stimuli['test_stimuli'], 
                                                       models_fits_dict[model_ID], models_built_data, model_ID, 1, 0)'''
        
        # output weight sparsity
        '''models_sparse_out_weights[model_ID] = weight_sparsity(n_neurons, models_fits_dict[model_ID], model_ID, 1)'''
    
        # distribution of weights coming from gates in "gates" models
        '''models_gates_weights[model_ID] = gate_weights(n_neurons, val_stim_dataset, val_resp_dataset, 
                                                      models_fits_dict[model_ID], model_ID, folds, models_best_lambdas)'''
        
        # sudden changes correlation
        '''models_corr_ON_OFF_stims[model_ID] = corr_with_ON_OFF_stims('pearson', 'average_stim', n_neurons, stim_dataset, 
                                                                    test_resp_dataset, stimuli['test_stimuli'], 
                                                                    models_fits_dict[model_ID], zero_padding, 
                                                                    padding_bins, n_F, 1)'''
    
        # direct-rcLSTM weights from added fc layer vs from subLSTM layer
        '''rcLSTM_weights[model_ID] = rcLSTM_weight_share(n_neurons, val_stim_dataset, val_resp_dataset, 
                                                       models_fits_dict[model_ID], model_ID, folds, models_best_lambdas)'''
        
        # direct-rcLSTM weights from added fc layer vs from subLSTM layer
        '''rcLSTM_weights[model_ID] = rcLSTM_weighted_output(n_neurons, test_stim_dataset, test_resp_dataset, 
                                                          models_fits_dict[model_ID], model_ID, folds, models_best_lambdas)'''
    
        # uncomment to plot example neurons for all 18 multi-repeat stimuli
        '''stim_size = int(test_stim_dataset.size(-1)/len(stimuli['test_stimuli'][0]))
        resp_size = int(test_resp_dataset.size(-1)/len(stimuli['test_stimuli'][0]))
        models_built_data[model_ID][0]['test_stim'] = \
            torch.cat((test_stim_dataset, train_stim_dataset[:, :, :, :7*stim_size]), dim=3)
        models_built_data[model_ID][0]['test_resp'] = \
            torch.cat((test_resp_dataset, train_resp_dataset[:, :, :7*resp_size]), dim=2)
        models_built_data[model_ID][1]['test_stimuli'] = [stimuli['test_stimuli'][0] + stimuli['training_stimuli'][0][:7]]'''
        
        # get parameter counts for each model
        '''models_param_count[model_ID] = count_params(models_fits_dict[model_ID])'''
        
        # get variance (standard error) of median
        '''models_se_median[model_ID] = calc_se_median(models_results_dict[model_ID])'''
    
    results_dict[dataset_ID] = models_results_dict
    
    bl_av_results_dict[dataset_ID] = models_bl_av_results_dict
    bl_med_results_dict[dataset_ID] = models_bl_med_results_dict
    
    fits_dict[dataset_ID] = models_fits_dict
    built_data[dataset_ID] = models_built_data
    
    best_lambdas[dataset_ID] = models_best_lambdas
    
    corr_stims[dataset_ID] = models_corr_stims
    '''neurons_list = plot_corr_stims('CCnorm_diff', models_corr_stims, models_results_dict, noise_ratios, dataset_ID, models)'''
    
    corr_stims_silence[dataset_ID] = models_corr_stims_silence
    '''neurons_list, models_neuron_list = plot_corr_stims_silence_or_sound('CCnorm_diff', models_corr_stims_silence, 
                                                                        models_results_dict, noise_ratios, dataset_ID, 
                                                                        models)'''
    
    '''neurons_list, models_neuron_list = plot_corr_stims_silence_or_sound('CCnorm_diff', models_corr_stims_silence, 
                                                                        models_any_ccnorms, noise_ratios, dataset_ID, 
                                                                        models)'''
    
    corr_stims_sound[dataset_ID] = models_corr_stims_sound
    '''neurons_list, models_neuron_list = plot_corr_stims_silence_or_sound('CCnorm_diff', models_corr_stims_sound, 
                                                                        models_results_dict, noise_ratios, dataset_ID, 
                                                                        models)'''
    
    gate_kernels[dataset_ID] = models_gate_kernels
    '''plot_gate_kernels('CCnorm_diff', models_gate_kernels, models_eff_HUs, models_results_dict, models_fits_dict, 
                      best_lambdas, noise_ratios, dataset_ID, models)'''
    
    eff_HUs[dataset_ID] = models_eff_HUs
    
    corr_HUs[dataset_ID] = models_corr_HUs
    '''models_neuron_list = plot_corr_HUs('CCnorm_diff', models_corr_HUs, models_eff_HUs, models_results_dict, models_fits_dict, 
                                       n_neurons, best_lambdas, noise_ratios, dataset_ID, models)'''
    
    corr_gates[dataset_ID] = models_corr_gates
    '''models_neuron_list = plot_corr_gates('CCnorm_diff', models_corr_gates, models_eff_HUs, models_results_dict, 
                                         models_fits_dict, n_neurons, best_lambdas, noise_ratios, dataset_ID, models)'''
    
    sparse_out_weights[dataset_ID] = models_sparse_out_weights
    '''plot_weight_sparsity('CCnorm_diff', models_sparse_out_weights, models_eff_HUs, models_results_dict, models_fits_dict, 
                         n_neurons, best_lambdas, noise_ratios, dataset_ID, models)'''
    
    gates_weights[dataset_ID] = models_gates_weights
    '''plot_gate_weights('CCnorm_diff', models_gates_weights, models_eff_HUs, models_results_dict, models_fits_dict, n_neurons, 
                      best_lambdas, noise_ratios, dataset_ID, models)'''
    
    '''plot_repetitiveness('CCnorm_diff', models_any_ccnorms, stims_used, stim_dataset, padding_bins, n_neurons, dataset_ID, 
                        models)'''
    
    corr_ON_OFF_stims[dataset_ID] = models_corr_ON_OFF_stims
    '''neurons_list, models_neuron_list = plot_corr_stims_silence_or_sound('CCnorm_diff', models_corr_ON_OFF_stims, 
                                                                        models_results_dict, noise_ratios, dataset_ID,
                                                                        models)'''
    
    # Multi-analysis with all data
    '''CC_diff_dict, CC_diff_pvalues, reg_model = multi_analysis_scatter('CC_diff', models_any_ccnorms, stims_used, 
                                                                      stim_dataset, padding_bins, zero_padding, n_neurons, 
                                                                      dataset_ID, models)'''
    
    # Just analysis measures, no correlation with CCnorm/CCraw
    '''sil_dur, sud_cha, av_amp = multi_analysis_stim_only(stim_dataset, stims_used, padding_bins, zero_padding, t_axis)'''
    
    # Depth analyses
    '''neuron_depths, depth_ax, depth_CCnorm_diffs, depth_CCnorm_diffs_se, depth_CCnorms, layers = \
        depth_sw_analysis('CCnorm_diff', models_results_dict, models_built_data, dataset_ID, models)
    depths[dataset_ID] = neuron_depths
    depth_axis[dataset_ID] = depth_ax
    CCnorm_depth_diff[dataset_ID] = depth_CCnorm_diffs
    CCnorm_depth_diff_se[dataset_ID] = depth_CCnorm_diffs_se'''
    
    '''depth_sw_sess_analysis('CCnorm_diff', models_results_dict, models_built_data, dataset_ID, models)'''
    
    '''depth_cell_analysis('CCnorm_diff', 'Eff', models_corr_HUs, models_eff_HUs, models_fits_dict, models_built_data, 
                           best_lambdas, dataset_ID, models)'''
    
    '''depth_t_gate_analysis('CCnorm_diff', models_corr_gates, models_fits_dict, models_built_data, best_lambdas, dataset_ID, 
                             models)'''
    
    '''depth_gates_analysis('CCnorm_diff', 'Eff', models_corr_gates, models_eff_HUs, models_fits_dict, models_built_data, 
                            best_lambdas, dataset_ID, models)'''
    
    '''depth_gatesmodels_analysis('CCnorm_diff', 'All', models_gates_weights, models_eff_HUs, models_fits_dict, 
                            models_built_data, best_lambdas, dataset_ID, models)'''
    
    '''depth_silence_analysis(models_corr_stims_silence, models_built_data, dataset_ID, models)'''
    
    '''depth_ON_OFF_analysis(models_corr_ON_OFF_stims, models_built_data, dataset_ID, models)'''
    
    '''depth_rcLSTM_analysis(rcLSTM_weights, models_fits_dict, models_built_data, best_lambdas, dataset_ID, models)'''
    
    # Models' parameter counts
    '''param_count[dataset_ID] = models_param_count'''
    
    # Standard error of median calculation
    '''se_median[dataset_ID] = models_se_median'''

# Neuron by neuron comparisons
'''m_i_n_dict, l_i_n_dict, ttest_dict, ranktest_dict, nr_corr_dict, prop_better_dict = comparisons(datasets, models, 
                                                                                                results_dict, 
                                                                                                bl_av_results_dict, nr_dict)

# Plot most improved neurons
models_neuron_list = m_i_n_dict
plot_neurons(datasets, models, models_neuron_list, fits_dict, results_dict, built_data, best_lambdas, device)'''

'''plot_loss_curves(datasets, models, fits_dict)'''

# Statistical comparisons of model performance
'''ttest_dict = {}
ranktest_dict = {}

for dataset in datasets:
    models_m_i_n_dict = {}
    models_ttest_dict = {}
    models_ranktest_dict = {}
    models_nr_corr_dict = {}
    models_prop_better_dict = {}
    
    for model_i in models:
        for model_j in models:
            test_ccnorms_1 = np.array(results_dict[dataset][model_i][2])
            test_ccnorms_2 = np.array(results_dict[dataset][model_j][2])
            
            models_ttest_dict[model_i + ' vs ' + model_j] = [scipy.stats.ttest_rel(test_ccnorms_1, test_ccnorms_2)[0], \
                                                             scipy.stats.ttest_rel(test_ccnorms_1, test_ccnorms_2)[1]]
            if model_i != model_j:
                models_ranktest_dict[model_i + ' vs ' + model_j] = [scipy.stats.wilcoxon(test_ccnorms_1, test_ccnorms_2)[0], \
                                                                    scipy.stats.wilcoxon(test_ccnorms_1, test_ccnorms_2)[1]]
            
    ttest_dict[dataset] = models_ttest_dict
    ranktest_dict[dataset] = models_ranktest_dict'''

# Depth histogram and distribution
'''bin_edges_1 = np.arange(-1200, 900, 100)
bin_edges_2 = np.arange(-700, 700, 100)

color_rgba1 = list(matplotlib.colors.to_rgba('#C20078'))
color_rgba1[3] = 0.5
color_rgba1 = tuple(color_rgba1)
color_rgba2 = list(matplotlib.colors.to_rgba('#054907'))
color_rgba2[3] = 0.5
color_rgba2 = tuple(color_rgba2)

f, ax = plt.subplots()
hist1 = ax.hist(depths['NS3'], bin_edges_1, histtype='step', edgecolor='#C20078', label='NS3', linewidth=3, fill=True,
                facecolor=color_rgba1)
hist2 = ax.hist(depths['NS2_include_single'], bin_edges_2, histtype='step', edgecolor='#054907', label='NS2_include_single',
                linewidth=3, fill=True, facecolor=color_rgba2)
#ax.set_xlabel('depth')
#ax.legend()
ax.spines[['top', 'right']].set_visible(False)

ax.set_xticks([-1000, -500, 0, 500])
ax.xaxis.set_tick_params(labelcolor='none')
ax.yaxis.set_tick_params(labelcolor='none')'''

'''NS3_percent_count = hist1[0]/191*100
NS2_percent_count = hist2[0]/72*100

NS3_bins = (hist1[1][:-1] + hist1[1][1:])/2
NS2_bins = (hist2[1][:-1] + hist2[1][1:])/2

f, ax = plt.subplots()
ax.plot(NS3_bins, NS3_percent_count, linewidth=3, color='#C20078')
ax.fill_between(NS3_bins, NS3_percent_count, alpha=0.2, color='#C20078')
ax.plot(NS2_bins, NS2_percent_count, linewidth=3, color='#054907')
ax.fill_between(NS2_bins, NS2_percent_count, alpha=0.2, color='#054907')
#ax.set_xlabel('depth')
ax.spines[['top', 'right']].set_visible(False)

ax.set_xticks([-1000, -500, 0, 500])
#ax.xaxis.set_tick_params(labelcolor='none')
#ax.yaxis.set_tick_params(labelcolor='none')'''

'''NS3_depth_axis = depth_axis['NS3']['pop_t_subLSTM vs twoD_CNN']
CCnorm_NS3 = CCnorm_depth_diff['NS3']['pop_t_subLSTM vs twoD_CNN']
CCnorm_se_NS3 = CCnorm_depth_diff_se['NS3']['pop_t_subLSTM vs twoD_CNN']

NS2_depth_axis = depth_axis['NS2_include_single']['pop_t_subLSTM vs twoD_CNN']
CCnorm_NS2 = CCnorm_depth_diff['NS2_include_single']['pop_t_subLSTM vs twoD_CNN']
CCnorm_se_NS2 = CCnorm_depth_diff_se['NS2_include_single']['pop_t_subLSTM vs twoD_CNN']

f, ax = plt.subplots()
ax.plot(NS3_depth_axis, CCnorm_NS3, linewidth=3, label='NS3', color='#C20078')
ax.fill_between(NS3_depth_axis, CCnorm_NS3-CCnorm_se_NS3, CCnorm_NS3+CCnorm_se_NS3, color='#C20078', alpha=0.2)
ax.plot(NS2_depth_axis, CCnorm_NS2, linewidth=3, label='NS2', color='#054907')
ax.fill_between(NS2_depth_axis, CCnorm_NS2-CCnorm_se_NS2, CCnorm_NS2+CCnorm_se_NS2, color='#054907', alpha=0.2)
#ax.legend()

ax.set_yticks([-0.1, -0.05, 0, 0.05])
ax.set_xticks(np.arange(-600, 600, 200))
ax.spines[['right', 'top']].set_visible(False)
#ax.xaxis.set_tick_params(labelcolor='none')
#ax.yaxis.set_tick_params(labelcolor='none')'''

# Difference depth t-tests
'''ccnorm_diffs = {'13': [], '44': [], '56': []}

for nID in range(len(layers)):
    ccnorm_diffs[layers[nID]].append(depth_CCnorms[models[0]][nID] - depth_CCnorms[models[1]][nID])
            
ccnorm_diffs_1 = {'234': ccnorm_diffs['13'] + ccnorm_diffs['44'], '56': ccnorm_diffs['56']}

diff_ttest_1 = scipy.stats.ttest_ind(ccnorm_diffs_1['56'], ccnorm_diffs_1['234'])[:]
diff_ranktest_1 = scipy.stats.mannwhitneyu(ccnorm_diffs_1['56'], ccnorm_diffs_1['234'])[:]'''