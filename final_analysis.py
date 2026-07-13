import build_datasets

import pickle
import torch
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt

from util_funcs import train_val_test_split, build_grid_NS2_include_single_NS3_concat

from analysis_funcs import any_CC_norms, multi_analysis_scatter, multi_analysis_stim_only, count_params, \
    compare_lowpass_LN, HU_tonotopy_FRMs, depth_sw_stats, param_match_compare, count_rcLSTM_params, t_corr_stim_resp
from plot_funcs import comparisons, plot_neurons, depth_sw_analysis, spont_FR_analysis, plot_param_count, plot_t_corr

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

param_count = {}

t_corr_resps = {}
t_corr_stims = {}

depths = {}
depth_axis = {}
CCnorm_depth_diff = {}
CCnorm_depth_diff_se = {}

sws = {}
sw_axis = {}
CCnorm_sw_diff = {}
CCnorm_sw_diff_se = {}

param_match_ttests = {}
param_match_ranktests = {}

for dataset_ID in datasets:
    print(dataset_ID)
    
    models_results_dict = {}
    
    models_bl_av_results_dict = {}
    models_bl_med_results_dict = {}
    
    models_fits_dict = {}
    models_built_data = {}
    
    models_best_lambdas = {}
    
    models_param_count = {}
    
    models_t_corr_resps = {}
    models_t_corr_stims = {}
    
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
        
        
        '''CCnorms = any_CC_norms('together', 'CC_norm', all_multi_stim_dataset, all_multi_resp_dataset, stim_dataset, 
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
        
        # get parameter counts for each model
        '''models_param_count[model_ID] = count_params(models_fits_dict[model_ID])
        # stats comparing f-LSTM to parameter-matched CNNs
        if model_ID == 'pop_f_LSTM' or (model_ID == 'twoD_CNN' and dataset_ID == 'NS3') or \
            (model_ID == 'oneD_CNN' and dataset_ID == 'NS3_PEG') or \
                (model_ID == 'twoD_CNN' and dataset_ID == 'NS2_include_single'): 
                    pm_ttest, pm_ranktest, pm_results, pm_num = param_match_compare(models_results_dict, model_ID, 
                                                                                    dataset_ID, nr_dict)'''
        
        # check tonotopic structure of HU projections to input gate in f-LSTM
        '''HU_tonotopy_FRMs(models_fits_dict, model_ID, dataset_ID, f_range, True, params, True, False, 2, False)'''
        
        # correlation of f-LSTM input gate with stimuli and PSTHs
        all_stim_dataset = torch.cat((test_stim_dataset, train_stim_dataset), dim=3)
        all_resp_dataset = torch.cat((test_resp_dataset, train_resp_dataset), dim=2)
        '''models_t_corr_resps[model_ID], models_t_corr_stims[model_ID] = t_corr_stim_resp(n_neurons, all_stim_dataset, 
                                                                                        all_resp_dataset,
                                                                                        models_fits_dict[model_ID], 
                                                                                        models_best_lambdas, model_ID, 1, 0,
                                                                                        dataset_ID, av_f=False, 
                                                                                        pre_act=False, all_f=True, 
                                                                                        stim_abs=False, only_active=True)'''
        

    results_dict[dataset_ID] = models_results_dict
    
    bl_av_results_dict[dataset_ID] = models_bl_av_results_dict
    bl_med_results_dict[dataset_ID] = models_bl_med_results_dict
    
    fits_dict[dataset_ID] = models_fits_dict
    built_data[dataset_ID] = models_built_data
    
    best_lambdas[dataset_ID] = models_best_lambdas
    
    # Multi-analysis with all data
    '''CC_diff_dict, CC_diff_pvalues, reg_model = multi_analysis_scatter('CC_diff', models_any_ccnorms, stims_used, 
                                                                      stim_dataset, padding_bins, zero_padding, n_neurons, 
                                                                      dataset_ID, models)'''
    
    # Just analysis measures, no correlation with CCnorm/CCraw
    '''sil_dur, sud_cha, av_amp = multi_analysis_stim_only(stim_dataset, stims_used, padding_bins, zero_padding, t_axis)'''
    
    # Depth and sw analyses
    '''neuron_depths, depth_ax, depth_CCnorm_diffs, depth_CCnorm_diffs_se, neuron_sws, sw_ax, sw_CCnorm_diffs, \
        sw_CCnorm_diffs_se, depth_sw_CCnorms, layers = \
            depth_sw_analysis('CCnorm_diff', models_results_dict, models_built_data, dataset_ID, models, plot_sw=True)
    depths[dataset_ID] = neuron_depths
    depth_axis[dataset_ID] = depth_ax
    CCnorm_depth_diff[dataset_ID] = depth_CCnorm_diffs
    CCnorm_depth_diff_se[dataset_ID] = depth_CCnorm_diffs_se
    sws[dataset_ID] = neuron_sws
    sw_axis[dataset_ID] = sw_ax
    CCnorm_sw_diff[dataset_ID] = sw_CCnorm_diffs
    CCnorm_sw_diff_se[dataset_ID] = sw_CCnorm_diffs_se
    
    # Model vs other model depth and SW t-tests
    depth_diff_ttest, sw_diff_ttest = depth_sw_stats(models, layers, sws, depth_sw_CCnorms, dataset_ID, sw_stats=True)'''
    
    # Parameter count analyses
    '''models_results_dict.update(pm_results)
    models_param_count.update(pm_num)
    param_count[dataset_ID] = models_param_count
    if dataset_ID == 'NS3' or dataset_ID == 'NS2_include_single':
        rc_LSTM_param_count, rc_LSTM_performance = count_rcLSTM_params(dataset_ID)
    else:
        rc_LSTM_param_count = []
        rc_LSTM_performance = []
    
    plot_param_count(models_param_count, rc_LSTM_param_count, models_results_dict, rc_LSTM_performance, dataset_ID)
    
    param_match_ttests[dataset_ID] = pm_ttest
    param_match_ranktests[dataset_ID] = pm_ranktest'''
    
    # Spontaneous firing rate analysis
    '''spont_FR_analysis(models_results_dict, dataset_ID, params)'''
    
    # Lowpass-LN analysis
    '''lowpass_ranktest = compare_lowpass_LN(models_results_dict, dataset_ID, params)'''
    
    # Correlation of f-LSTM input gate with stims and resps
    t_corr_resps[dataset_ID] = models_t_corr_resps
    t_corr_stims[dataset_ID] = models_t_corr_stims
    '''plot_t_corr(models_t_corr_resps, models_t_corr_stims, n_neurons, best_lambdas, dataset_ID, models, pre_act=False, 
                all_f=True)'''

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
