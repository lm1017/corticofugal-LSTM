import numpy as np
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy
import all_models
import itertools
import pickle

from util_funcs import extents, redblue, build_grid_NS2_include_single_NS3_concat

def plot_corr_stims(ccnorms_change_measure, models_ccs_with_stim, models_results_dict, noise_ratios, dataset_ID, models):
    '''
    Parameters
    ----------
    ccnorms_change_measure : measure of choice for CCnorm change between two models: either "CCnorm_diff" (difference)
    or CCnorm_percent_change (percentage change)
    models_ccs_with_stim : dictionary of negative correlation of stimuli, for each model, with model predictions and 
    recorded PSTH, and p-values for the correlations with the recorded PSTH
    models_results_dict : dictionary of CCnorm arrays for each model (training, training for all folds and lambdas, 
    validation, validation for all folds and lambdas)
    noise_ratios : list of noise ratios of neurons
    dataset_ID : dataset name
    models : list of models to be compared and plotted

    Returns
    -------
    neurons_list : list of neurons based on given condition
    '''
    
    # define p-value-based colourmap - black indicates non-significance, red indicates significance
    pvalues_cmap = mcolors.ListedColormap(['black', 'red'])
    
    # Boolean variable, controls whether 3D plots of correlation with stimuli, CCnorm change, and noise ratios are generated
    threeD_plot = False
    # Boolean variable, controls whether plots of correlation colour-coded by noise ratio are generated
    nr_colour_plot = False
    
    # plot negative correlation of PSTHs with stimuli for each neuron against its noise ratio 
    plt.figure()
    plt.scatter(noise_ratios, models_ccs_with_stim[models[0]][0])
    plt.xlabel('Noise ratio')
    plt.ylabel('CC with stims')
    plt.title(dataset_ID + ' - response correlation with stimuli vs NR')
    
    # uncomment for plots (both separate and together, for all models) of negative correlation of model prediction with 
    # stimuli against noise ratios and validation CCnorm
    f, ax1 = plt.subplots()
    ax1.set_title(dataset_ID + ' - all models - response correlation with stimuli vs CCnorm')
    
    f, ax4 = plt.subplots()
    ax4.scatter(noise_ratios, models_ccs_with_stim[models[0]][0], label='data')
    ax4.set_title(dataset_ID + ' - all models - response correlation with stimuli vs NR')
    
    f, ax5 = plt.subplots()
    ax5.set_title(dataset_ID + ' - all models - average response correlation with stimuli vs CCnorm')
    
    for model_ID in models:
        ax1.scatter(models_results_dict[model_ID][2], models_ccs_with_stim[model_ID][0], label=model_ID)
        ax1.set_xlabel('CCnorm')
        ax1.set_ylabel('CC with stims')
        
        f, ax2 = plt.subplots()
        ax2.scatter(models_results_dict[model_ID][2], models_ccs_with_stim[model_ID][0])
        ax2.set_xlabel('CCnorm')
        ax2.set_ylabel('CC with stims')
        ax2.set_title(dataset_ID + ' - ' + model_ID + ' - response correlation with stimuli vs CCnorm')
        
        f, ax3 = plt.subplots()
        ax3.scatter(noise_ratios, models_ccs_with_stim[model_ID][0])
        ax3.set_xlabel('Noise ratio')
        ax3.set_ylabel('CC with stims')
        ax3.set_title(dataset_ID + ' - ' + model_ID + ' - response correlation with stimuli vs NR')
        
        ax4.scatter(noise_ratios, models_ccs_with_stim[model_ID][0])
        ax4.set_xlabel('Noise ratio')
        ax4.set_ylabel('CC with stims')
        
        '''ax5.scatter(models_av_results_dict[model_ID][0], np.mean(models_ccs_with_stim[model_ID][1]), label=model_ID)
        ax5.set_xlabel('average CCnorm')
        ax5.set_ylabel('average CC with stims')'''
        
    ax1.legend()
    ax4.legend()
    ax5.legend()
    
    # list of neurons whose correlation with stimuli is > 0.3
    neurons_list = np.nonzero(models_ccs_with_stim[models[0]][0] > 0.3)[0]
    
    for i_model_ID in models: # loop over models
        for j_model_ID in models: # loop over models
            # difference in CCnorm between two models, for each neuron
            ccnorms_diff = models_results_dict[i_model_ID][2] - models_results_dict[j_model_ID][2]
            
            # percentage change in CCnorm between two models, for each neuron
            ccnorms_percent_change = (models_results_dict[i_model_ID][2] - \
                                      models_results_dict[j_model_ID][2])/models_results_dict[j_model_ID][2]*100
            
            # choose CCnorm change measure 
            if ccnorms_change_measure == 'CCnorm_diff':
                ccnorms_change = ccnorms_diff
            elif ccnorms_change_measure == 'CCnorm_percent_change':
                ccnorms_change = ccnorms_percent_change
            
            # linear regression fit of CCnorm change and correlation with stimuli
            X2 = sm.add_constant(models_ccs_with_stim[i_model_ID][0]) # add bias to independent variable
            est = sm.OLS(ccnorms_change, X2) # Ordinary Least Squares fit
            est2 = est.fit()
            a = est2.params[0] # bias
            b = est2.params[1] # slope
            # x-sequence for plotting line of best fit
            xseq = np.linspace(-np.max(np.abs(models_ccs_with_stim[i_model_ID][0]))-0.05, 
                               np.max(np.abs(models_ccs_with_stim[i_model_ID][0]))+0.05, num=100)
            
            # scatter plot of CCnorm change against correlation with stimuli for each neuron, with line of best fit
            f1, ax1 = plt.subplots()
            ax1.scatter(models_ccs_with_stim[i_model_ID][0], ccnorms_change, c=models_ccs_with_stim[i_model_ID][1], 
                        cmap=pvalues_cmap)
            ax1.set_xlabel('CC with stims')
            ax1.set_ylabel(ccnorms_change_measure)
            ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID + '; p = ' + str(est2.pvalues[1]))
            ax1.plot(xseq, a+b*xseq, color='b')
            ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            
            if nr_colour_plot:
                color = [item/40 for item in noise_ratios]
                
                # scatter plot of CCnorm change against correlation with stimuli for each neuron, colour-coded according to
                # noise ratio, with line of best fit
                f2, ax2 = plt.subplots(layout='constrained')
                ax2.scatter(models_ccs_with_stim[i_model_ID][0], ccnorms_percent_change, c=color, 
                            cmap=mpl.colormaps['Blues'], edgecolors='black')
                f2.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 40), cmap='Blues'), ax=ax2, 
                             orientation='vertical', label='noise ratio')
                ax2.set_xlabel('CC with stims')
                ax2.set_ylabel('CCnorm % change')
                ax2.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID + '; p = ' + str(est2.pvalues[1]))
            
            if threeD_plot:
                # 3D scatter plot of CCnorm change against correlation with stimuli and noise ratio for each neuron
                f3 = plt.figure()
                ax3 = f3.add_subplot(projection='3d')
                ax3.scatter(models_ccs_with_stim[i_model_ID][0], ccnorms_percent_change, noise_ratios)
                ax3.set_xlabel('CC with stims')
                ax3.set_ylabel('CCnorm % change')
                ax3.set_zlabel('Noise ratio')
                ax3.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID + '; p = ' + str(est2.pvalues[1]))
                
    return neurons_list

#############################################################################################################################

def plot_corr_stims_silence_or_sound(ccnorms_change_measure, models_ccs_with_stim, models_results_dict, noise_ratios, 
                                     dataset_ID, models):
    '''
    Parameters
    ----------
    ccnorms_change_measure : measure of choice for CCnorm change between two models: either "CCnorm_diff" (difference)
    or CCnorm_percent_change (percentage change)
    models_ccs_with_stim : dictionary of correlation of silence- or sound-triggered stimuli, for each model, with model 
    predictions and recorded PSTH, and p-values for the correlations with the recorded PSTH
    models_results_dict : dictionary of CCnorm arrays for each model -> if analysing validation results, arrays are: 
        training, training for all folds and lambdas, validation, validation for all folds and lambdas; if analysing test 
        results, arrays are: test for all best lambdas, test of mode model, test of best lambda for each neuron
    noise_ratios : list of noise ratios of neurons
    dataset_ID : dataset name
    models : list of models to be compared and plotted

    Returns
    -------
    neurons_list : list of neurons based on given condition
    models_neurons_list : list of neurons for each model comparison, based on given condition
    '''
    
    n_neurons = np.size(models_ccs_with_stim[models[0]][0])
    
    # define p-value-based colourmap - black indicates non-significance, red indicates significance
    pvalues_cmap = mcolors.ListedColormap(['black', 'red'])

    # plot correlation of PSTHs with silence- or sound-triggered stimuli for each neuron against its noise ratio 
    plt.figure()
    plt.scatter(noise_ratios, models_ccs_with_stim[models[0]][0], c=models_ccs_with_stim[models[0]][1], cmap=pvalues_cmap)
    plt.xlabel('Noise ratio')
    plt.ylabel('CC with stims silence/sound duration')
    plt.title(dataset_ID + ' - response correlation with stimulus silence/sound duration vs NR')
    
    models_neurons_list = {} # dictionary to hold list of neurons for each model comparison
    # list of neurons whose correlation with silence- or sound-triggered stimuli is > 0.3
    neurons_list = np.nonzero(models_ccs_with_stim[models[0]][0] > 0.3)[0]
    # list of neurons whose correlation with silence- or sound-triggered stimuli is < 0.2
    #neurons_list = np.nonzero(models_ccs_with_stim[models[0]][0] < -0.2)[0]
    
    for i_model_ID in models: # loop over models
        # uncomment to plot histograms of correlations between PSTHs and silence- or sound-triggered stimuli, for each model
        '''plt.figure()
        plt.hist(models_ccs_with_stim[i_model_ID][0], bins=50)
        plt.title(i_model_ID)'''
        
        for j_model_ID in models: # loop over models
            # difference in CCnorm between two models, for each neuron
            ccnorms_diff = models_results_dict[i_model_ID][2] - models_results_dict[j_model_ID][2]
            
            # percentage change in CCnorm between two models, for each neuron
            ccnorms_percent_change = (models_results_dict[i_model_ID][2] - \
                                      models_results_dict[j_model_ID][2])/models_results_dict[j_model_ID][2]*100
            
            # choose CCnorm change measure 
            if ccnorms_change_measure == 'CCnorm_diff':
                ccnorms_change = ccnorms_diff
            elif ccnorms_change_measure == 'CCnorm_percent_change':
                ccnorms_change = ccnorms_percent_change
            
            nan_idxs = np.argwhere(np.isnan(ccnorms_change))
            ccnorms_change_new = np.delete(ccnorms_change, nan_idxs)
            ccs_with_stim = np.delete(models_ccs_with_stim[i_model_ID][0], nan_idxs)
            # linear regression fit of CCnorm change and correlation with silence- or sound-triggered stimuli
            X2 = sm.add_constant(ccs_with_stim) # add bias to independent variable
            est = sm.OLS(ccnorms_change_new, X2) # Ordinary Least Squares fit
            est2 = est.fit()
            a = est2.params[0] # bias
            b = est2.params[1] # slope
            # x-sequence for plotting line of best fit
            xseq = np.linspace(-np.max(np.abs(models_ccs_with_stim[i_model_ID][0]))-0.05, 
                               np.max(np.abs(models_ccs_with_stim[i_model_ID][0]))+0.05, num=100)
            
            # scatter plot of CCnorm change against correlation with silence- or sound-triggered stimuli for each neuron, 
            # with line of best fit
            scatter_colour = np.delete(models_ccs_with_stim[i_model_ID][1], nan_idxs)
            f1, ax1 = plt.subplots()
            ax1.scatter(ccs_with_stim, ccnorms_change_new, c=scatter_colour, cmap=pvalues_cmap)
            ax1.set_xlabel('CC with stims')
            ax1.set_ylabel(ccnorms_change_measure)
            ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID + '; p = ' + str(est2.pvalues[1]))
            ax1.plot(xseq, a+b*xseq, color='b')
            ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            
            models_neurons_list[i_model_ID + ' vs ' + j_model_ID] = []
            for nID in range(n_neurons): # loop over neurons
                if ccnorms_change[nID] > 0.25 and models_ccs_with_stim[i_model_ID][0][nID] < 0.2:
                    # list of neurons based on above condition
                    models_neurons_list[i_model_ID + ' vs ' + j_model_ID].append(nID)
            
    return neurons_list, models_neurons_list

#############################################################################################################################

def plot_gate_kernels(ccnorms_change_measure, models_gate_kernels, models_eff_HUs, models_results_dict, models_fits_dict, 
                      best_lambdas, noise_ratios, dataset_ID, models):
    '''
    Parameters
    ----------
    ccnorms_change_measure : measure of choice for CCnorm change between two models: either "CCnorm_diff" (difference)
    or CCnorm_percent_change (percentage change)
    models_gate_kernels : dictionary of reverse correlation kernels of gate activity and stimuli; for each model, it
    contains kernels for all folds, lambdas, and gates (where they exist); for each gate, there are two kernels: one for 
    gating activity increases and one for decreases
    models_eff_HUs : dictionary of lists containing, for each model, a list of effective hidden units for each neuron, for 
    each fold and lambda (dimensions -> (folds, lambdas, neurons))
    models_results_dict : dictionary of CCnorm arrays for each model (training, training for all folds and lambdas, 
    validation, validation for all folds and lambdas)
    models_fits_dict : dictionary of fitted parameters and hyperparameters for each model, fold, and lambda
    best_lambdas : dictionary of best lambdas of each neuron, for each dataset and model
    noise_ratios : list of noise ratios of neurons
    dataset_ID : dataset name
    models : list of models to be compared and plotted

    Returns
    -------
    None.
    '''
    
    # minimum and maximum of frequency range
    f_min = 500
    f_max = 22000
    
    redblue_cmap = redblue(256) # define colourmap
    
    for i_model_ID in models: # loop over models
        n_F = models_fits_dict[i_model_ID][1]['n_F'] # number of cochleagram frequency channels
        n_h = models_fits_dict[i_model_ID][1]['n_h'] # number of history steps of cochleagram tensorisation
        bin_size = models_fits_dict[i_model_ID][1]['bin_size'] # PSTH and cochleagram time bin size
        
        f_range = np.logspace(np.log10(f_min), np.log10(f_max), n_F) # frequency axis
        latency = np.linspace(-(n_h+1)*bin_size, -bin_size, num=n_h) # latency (history) axis
        
        for j_model_ID in models: # loop over models
            # difference in CCnorm between two models, for each neuron
            ccnorms_diff = models_results_dict[i_model_ID][2] - models_results_dict[j_model_ID][2] # 
            
            # percentage change in CCnorm between two models, for each neuron
            ccnorms_percent_change = (models_results_dict[i_model_ID][2] - \
                                      models_results_dict[j_model_ID][2])/models_results_dict[j_model_ID][2]*100
            
            # choose CCnorm change measure
            if ccnorms_change_measure == 'CCnorm_diff':
                ccnorms_change = ccnorms_diff
            elif ccnorms_change_measure == 'CCnorm_percent_change':
                ccnorms_change = ccnorms_percent_change
            
            most_improved = np.argpartition(ccnorms_change, -2)[-2:] # two most improved neurons

            for nID in most_improved: # loop over most improved neurons
                neuron_lambda = best_lambdas[dataset_ID][i_model_ID][nID] # best lambda for neuron, for i_model
                # effective HUs of neuron, for i_model and best lambda
                most_improved_eff_HUs = models_eff_HUs[i_model_ID][0][neuron_lambda][nID] 

                # kernels for specific model and lambda
                model_lambda_kernels = models_gate_kernels[i_model_ID][0][neuron_lambda]

                for gate_type in model_lambda_kernels: # loop over gates
                    for eff_HU in [most_improved_eff_HUs[-1]]: # loop over effective HUs of neuron
                        # loop over kernel types: increase or decrease
                        for kernel_type in model_lambda_kernels[gate_type][eff_HU]:
                            kernel = model_lambda_kernels[gate_type][eff_HU][kernel_type]
                            
                            # plot kernel as image; rows are frequencies, columns are history time bins
                            f1, ax1 = plt.subplots()
                            ax1.imshow(kernel, cmap=redblue_cmap, aspect='auto', interpolation='none', 
                                       extent=extents(latency) + extents(f_range), origin='lower')
                            ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID + '; neuron ' + str(nID) + 
                                          ', HU ' + str(eff_HU) + ', gate ' + gate_type + ', type ' + kernel_type[-1])
                            
#############################################################################################################################

def plot_corr_HUs(ccnorms_change_measure, models_corr_HUs, models_eff_HUs, models_results_dict, models_fits_dict, n_neurons,
                  best_lambdas, noise_ratios, dataset_ID, models):
    '''
    Parameters
    ----------
    ccnorms_change_measure : measure of choice for CCnorm change between two models: either "CCnorm_diff" (difference)
    or CCnorm_percent_change (percentage change)
    models_corr_HUs : dictionary of correlations, for each model, between model hidden units and a) pre-activation model 
    output, b) model output, c) recorded PSTH; for each neuron and each fold and lambda (dimensions -> (neurons, folds, 
    lambdas, HUs))
    models_eff_HUs : dictionary of lists containing, for each model, a list of effective hidden units for each neuron, for 
    each fold and lambda (dimensions -> (folds, lambdas, neurons))
    models_results_dict : dictionary of CCnorm arrays for each model (training, training for all folds and lambdas, 
    validation, validation for all folds and lambdas)
    models_fits_dict : dictionary of fitted parameters and hyperparameters for each model, fold, and lambda
    n_neurons : number of neurons in dataset
    best_lambdas : dictionary of best lambdas of each neuron, for each dataset and model
    noise_ratios : list of noise ratios of neurons
    dataset_ID : dataset name
    models : list of models to be compared and plotted

    Returns
    -------
    models_neurons_list : list of neurons for each model, based on given condition
    '''
    
    # define p-value-based colourmap - black indicates non-significance, red indicates significance
    pvalues_cmap = mcolors.ListedColormap(['black', 'red'])
    
    models_neurons_list = {} # dictionary to hold list of neurons for each model
    
    for i_model_ID in models: # loop over models
        i_model_params = models_fits_dict[i_model_ID][1]['all_model_params'] # hyperparameters of i_model
        i_model_type = models_fits_dict[i_model_ID][1]['model_type']
        if i_model_type == 'not_conv':
            i_hidden_size = i_model_params['hidden_size'] # network hidden size
        elif i_model_type == 'conv':
            i_hidden_size = np.size(models_fits_dict[i_model_ID][0][0][0][0]['w_l1'], 0)
        i_model_best_lambdas = best_lambdas[dataset_ID][i_model_ID] # i_model best lambda for each neuron
        
        # NumPy array of correlations between each neuron's recorded PSTH and each HU in i_model, for all folds and lambdas 
        # (dimensions -> (neurons, folds, lambdas, HUs))
        i_model_corr_HUs = models_corr_HUs[i_model_ID]['actual']
        
        i_HU_corr = np.zeros((n_neurons, i_hidden_size)) # array to hold correlations for specific fold and best lambda
        
        for nID in range(n_neurons): # loop over neurons
            # correlations for specific neuron, fold, and the neuron's best lambda
            i_HU_corr[nID, :] = i_model_corr_HUs[nID, 0, i_model_best_lambdas[nID]]
        
        # kurtosis of correlations for each neuron - measure of sparsity of correlations
        i_HU_corr_kurtosis = scipy.stats.kurtosis(i_HU_corr, axis=1)
        # significance of kurtosis for each neuron - transformed into a Boolean significance array
        kurtosis_sig = scipy.stats.kurtosistest(i_HU_corr, axis=1, alternative='greater')[1] < 0.01
        
        # uncomment to plot histogram of kurtosis values
        '''f1, ax1 = plt.subplots()
        ax1.hist(i_HU_corr_kurtosis, bins=50)
        ax1.set_xlabel('kurtosis - HUs correlation with response')
        ax1.set_title(dataset_ID + ' - ' + i_model_ID)
        ax1.spines[['right', 'top']].set_visible(False)'''
        
        # maximum absolute correlation, over all HUs, for each neuron
        i_HU_corr_abs_max = np.max(np.abs(i_HU_corr), axis=1)
        
        # scatter plot of maximum absolute correlation against correlation kurtosis for each neuron
        f2, ax2 = plt.subplots()
        ax2.scatter(i_HU_corr_kurtosis, i_HU_corr_abs_max, c=kurtosis_sig, cmap=pvalues_cmap)
        ax2.set_xlabel('kurtosis - HUs correlation with response')
        ax2.set_ylabel('max absolute HUs correlation with response')
        ax2.set_title(dataset_ID + ' - ' + i_model_ID)
        ax2.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
        
        # product between correlation kurtosis and maximum absolute correlation - measure of "top-rightedness" of 
        # neurons in the above graph
        i_corr_kurt_prod = np.multiply(i_HU_corr_kurtosis, i_HU_corr_abs_max)
        
        for j_model_ID in models: # loop over models
            # difference in CCnorm between two models, for each neuron
            ccnorms_diff = models_results_dict[i_model_ID][2] - models_results_dict[j_model_ID][2]
            
            # percentage change in CCnorm between two models, for each neuron
            ccnorms_percent_change = (models_results_dict[i_model_ID][2] - \
                                      models_results_dict[j_model_ID][2])/models_results_dict[j_model_ID][2]*100
            
            # choose CCnorm change measure
            if ccnorms_change_measure == 'CCnorm_diff':
                ccnorms_change = ccnorms_diff
            elif ccnorms_change_measure == 'CCnorm_percent_change':
                ccnorms_change = ccnorms_percent_change
            
            j_model_params = models_fits_dict[j_model_ID][1]['all_model_params'] # hyperparameters of j_model
            j_model_type = models_fits_dict[j_model_ID][1]['model_type']
            if j_model_type == 'not_conv':
                j_hidden_size = j_model_params['hidden_size'] # network hidden size
            elif j_model_type == 'conv':
                j_hidden_size = np.size(models_fits_dict[j_model_ID][0][0][0][0]['w_l1'], 0)
            j_model_best_lambdas = best_lambdas[dataset_ID][j_model_ID] # j_model best lambda for each neuron
            
            # NumPy array of correlations between each neuron's recorded PSTH and each HU in j_model, for all folds and 
            # lambdas (dimensions -> (neurons, folds, lambdas, HUs))
            j_model_corr_HUs = models_corr_HUs[j_model_ID]['actual']
            
            j_HU_corr = np.zeros((n_neurons, j_hidden_size)) # array to hold correlations for specific fold and best lambda

            for nID in range(n_neurons): # loop over neurons
                # correlations for specific neuron, fold and the neuron's best lambda
                j_HU_corr[nID, :] = j_model_corr_HUs[nID, 0, j_model_best_lambdas[nID]]
            
            # kurtosis of correlations for each neuron
            j_HU_corr_kurtosis = scipy.stats.kurtosis(j_HU_corr, axis=1)
            # maximum absolute correlation, over all HUs, for each neuron
            j_HU_corr_abs_max = np.max(np.abs(j_HU_corr), axis=1)
            
            # product between correlation kurtosis and maximum absolute correlation
            j_corr_kurt_prod = np.multiply(j_HU_corr_kurtosis, j_HU_corr_abs_max)
            
            # uncomment to plot scatter plots of CCnorm change against top-rightedness difference
            # difference in top-rightedness for each neuron between the two models
            corr_kurt_prod_diff = i_corr_kurt_prod - j_corr_kurt_prod
            
            f3, ax3 = plt.subplots()
            ax3.scatter(corr_kurt_prod_diff, ccnorms_change)
            ax3.set_xlabel('Difference in HUs correlation')
            ax3.set_ylabel(ccnorms_change_measure)
            ax3.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID)
            ax3.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            
        neurons = np.nonzero(i_HU_corr_kurtosis > 2)[0] # list of neurons where correlation kurtosis is > 2 (in i_model)
        HU_max_corr = np.argmax(np.abs(i_HU_corr), axis=1) # HU where correlation is maximum, for each neuron (in i_model)
            
        neurons_list = [neurons, HU_max_corr]
        models_neurons_list[i_model_ID] = neurons_list
        
    return models_neurons_list
            
#############################################################################################################################

def plot_corr_gates(ccnorms_change_measure, models_corr_gates, models_eff_HUs, models_results_dict, models_fits_dict, 
                    n_neurons, best_lambdas, noise_ratios, dataset_ID, models):
    '''
    Parameters
    ----------
    ccnorms_change_measure : measure of choice for CCnorm change between two models: either "CCnorm_diff" (difference)
    or CCnorm_percent_change (percentage change)
    models_corr_gates : list of correlations, for each model, between model gates and recorded PSTH; for each neuron and 
    each fold and lambda (dimensions -> (folds, lambdas, gate types, neurons, HUs))
    models_eff_HUs : dictionary of lists containing, for each model, a list of effective hidden units for each neuron, for 
    each fold and lambda (dimensions -> (folds, lambdas, neurons))
    models_results_dict : dictionary of CCnorm arrays for each model (training, training for all folds and lambdas, 
    validation, validation for all folds and lambdas)
    models_fits_dict : dictionary of fitted parameters and hyperparameters for each model, fold, and lambda
    n_neurons : number of neurons in dataset
    best_lambdas : dictionary of best lambdas of each neuron, for each dataset and model
    noise_ratios : list of noise ratios of neurons
    dataset_ID : dataset name
    models : list of models to be compared and plotted

    Returns
    -------
    models_neurons_list : list of neurons for each model , based on given condition
    '''
    
    # define p-value-based colourmap - black indicates non-significance, red indicates significance
    pvalues_cmap = mcolors.ListedColormap(['black', 'red'])
    
    models_neurons_list = {} # dictionary to hold list of neurons for each model
    
    gated_models_list = ['pop_MGU', 'pop_GRU', 'pop_LSTM', 'pop_subLSTM', 'pop_f_LSTM']
    
    for i_model_ID in models: # loop over models
        if i_model_ID in gated_models_list:
            i_model_params = models_fits_dict[i_model_ID][1]['all_model_params'] # hyperparameters of i_model
            i_hidden_size = i_model_params['hidden_size'] # i_model hidden size
            i_input_size = i_model_params['input_size']
            i_model_best_lambdas = best_lambdas[dataset_ID][i_model_ID] # i_model best lambda for each neuron
            
            # NumPy array of correlations between each neuron's recorded PSTH and each gate in i_model, for all lambdas in 
            # one fold (dimensions -> (neurons, folds, lambdas, HUs))
            i_model_corr_gates = models_corr_gates[i_model_ID][0]
            
            i_corr_gates = {} # dictionary to hold correlations with each gate for each neuron, only for its best lambda
            # dictionary to hold product of kurtosis and maximum absolute correlation for each neuron with each gate (see 
            # below)
            i_gates_corr_kurt_prod = {}
            
            for key in i_model_corr_gates[0]: # loop over gates
                # NumPy array to hold correlations with gate for each neuron, only for its best lambda
                if key == 't':
                    i_gate_corr = np.zeros((n_neurons, i_input_size))
                else:
                    i_gate_corr = np.zeros((n_neurons, i_hidden_size))
                
                for nID in range(n_neurons): # loop over neurons
                    # correlations of each neuron with gate, for its best lambda
                    i_gate_corr[nID, :] = i_model_corr_gates[i_model_best_lambdas[nID]][key][nID]
                
                # kurtosis of correlations for each neuron - measure of sparsity of correlations
                i_gate_corr_kurtosis = scipy.stats.kurtosis(i_gate_corr, axis=1)
                # significance of kurtosis for each neuron - transformed into a Boolean significance array
                kurtosis_sig = scipy.stats.kurtosistest(i_gate_corr, axis=1, alternative='greater')[1] < 0.01
                for nID in range(len(kurtosis_sig)):
                    if i_gate_corr_kurtosis[nID] > 0:
                        kurtosis_sig[nID] = kurtosis_sig[nID]
                    else:
                        if kurtosis_sig[nID] == True:
                            kurtosis_sig[nID] = False
                
                # uncomment to plot histogram of kurtosis values
                '''f1, ax1 = plt.subplots()
                ax1.hist(i_gate_corr_kurtosis, bins=50)
                ax1.set_xlabel('kurtosis - ' + key + ' gate correlation with response')
                ax1.set_title(dataset_ID + ' - ' + i_model_ID)
                ax1.spines[['right', 'top']].set_visible(False)'''
                
                # maximum absolute correlation, over all HUs, for each neuron
                i_gate_corr_abs_max = np.max(np.abs(i_gate_corr), axis=1)
                
                # scatter plot of maximum absolute correlation against correlation kurtosis for each neuron
                f2, ax2 = plt.subplots()
                ax2.scatter(i_gate_corr_kurtosis, i_gate_corr_abs_max, c=kurtosis_sig, cmap=pvalues_cmap)
                ax2.set_xlabel('kurtosis - ' + key + ' gate correlation with response')
                ax2.set_ylabel('max absolute gate correlation with response')
                ax2.set_title(dataset_ID + ' - ' + i_model_ID)
                # set top and right plot axes to invisible (prevents box layout)
                ax2.spines[['right', 'top']].set_visible(False)
                
                i_corr_gates[key] = i_gate_corr # insert NumPy array in dictionary
                # product between correlation kurtosis and maximum absolute correlation - measure of 
                # "top-rightedness" of neurons in the above graph; insert it in dictionary
                i_gates_corr_kurt_prod[key] = np.multiply(i_gate_corr_kurtosis, i_gate_corr_abs_max)
            
        
            # uncomment to plot correlation to all gates together
            '''marker = itertools.cycle(('o', '+', '*'))
            gates_legend = i_model_corr_gates[0].keys()
    
            f2, ax2 = plt.subplots()
            for key in i_model_corr_gates[0]: # loop over gates
                # NumPy array to hold correlations with gate for each neuron, only for its best lambda
                i_gate_corr = np.zeros((n_neurons, i_hidden_size))
                
                for nID in range(n_neurons): # loop over neurons
                    # correlations of each neuron with gate, for its best lambda 
                    i_gate_corr[nID, :] = i_model_corr_gates[i_model_best_lambdas[nID]][key][nID]
                
                # kurtosis of correlations for each neuron - measure of sparsity of correlations
                i_gate_corr_kurtosis = scipy.stats.kurtosis(i_gate_corr, axis=1)
                # significance of kurtosis for each neuron - transformed into a Boolean significance array
                kurtosis_sig = scipy.stats.kurtosistest(i_gate_corr, axis=1, alternative='greater')[1] < 0.01
                
                # maximum absolute correlation, over all HUs, for each neuron
                i_gate_corr_abs_max = np.max(np.abs(i_gate_corr), axis=1)
                
                # scatter plot of maximum absolute correlation against correlation kurtosis for each neuron; high-kurtosis, 
                # high-correlation neurons are interesting as they may be acting like a particular small set of gates
                ax2.scatter(i_gate_corr_kurtosis, i_gate_corr_abs_max, c=kurtosis_sig, cmap=pvalues_cmap, 
                            marker=next(marker))
                ax2.set_xlabel('kurtosis - ' + key + ' gate correlation with response')
                ax2.set_ylabel('max absolute gate correlation with response')
                ax2.set_title(dataset_ID + ' - ' + i_model_ID)
                ax2.spines[['right', 'top']].set_visible(False)
                
                i_corr_gates[key] = i_gate_corr # insert NumPy array in dictionary
                # product between correlation kurtosis and maximum absolute correlation - arbitrary measure of 
                # "top-rightedness" of neurons in the above graph; insert it in dictionary
                i_gates_corr_kurt_prod[key] = np.multiply(i_gate_corr_kurtosis, i_gate_corr_abs_max)'''
            
            #ax2.legend(gates_legend)
            
            for j_model_ID in models: # loop over models
                # difference in CCnorm between two models, for each neuron
                ccnorms_diff = models_results_dict[i_model_ID][2] - models_results_dict[j_model_ID][2]
                
                # percentage change in CCnorm between two models, for each neuron
                ccnorms_percent_change = (models_results_dict[i_model_ID][2] - \
                                          models_results_dict[j_model_ID][2])/models_results_dict[j_model_ID][2]*100
                
                # choose CCnorm change measure
                if ccnorms_change_measure == 'CCnorm_diff':
                    ccnorms_change = ccnorms_diff
                elif ccnorms_change_measure == 'CCnorm_percent_change':
                    ccnorms_change = ccnorms_percent_change
                    
                # uncomment to plot CCnorm change against gate correlation kurtosis (x axis in f2)
                '''for key in i_corr_gates:
                    # linear regression fit of CCnorm change and correlation with silence- or sound-triggered stimuli
                    X2 = sm.add_constant(scipy.stats.kurtosis(i_corr_gates[key], axis=1)) # add bias to independent variable
                    est = sm.OLS(ccnorms_change, X2) # Ordinary Least Squares fit
                    est2 = est.fit()
                    a = est2.params[0] # bias
                    b = est2.params[1] # slope
                    # x-sequence for plotting line of best fit
                    xseq = np.linspace(np.min(scipy.stats.kurtosis(i_corr_gates[key], axis=1))-0.05, 
                                       np.max(scipy.stats.kurtosis(i_corr_gates[key], axis=1))+0.05, num=100)
                    
                    f3, ax3 = plt.subplots()
                    ax3.scatter(scipy.stats.kurtosis(i_corr_gates[key], axis=1), ccnorms_change)
                    ax3.set_xlabel(key + ' gate correlation kurtosis')
                    ax3.set_ylabel(ccnorms_change_measure)
                    ax3.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID + '; p = ' + str(est2.pvalues[1]))
                    ax3.plot(xseq, a+b*xseq, color='b')
                    ax3.spines[['right', 'top']].set_visible(False)
                
                # uncomment to plot scatter plots of CCnorm change against top-rightedness in i_model
                for key in i_gates_corr_kurt_prod:
                    # linear regression fit of CCnorm change and correlation with silence- or sound-triggered stimuli
                    X2 = sm.add_constant(i_gates_corr_kurt_prod[key]) # add bias to independent variable
                    est = sm.OLS(ccnorms_change, X2) # Ordinary Least Squares fit
                    est2 = est.fit()
                    a = est2.params[0] # bias
                    b = est2.params[1] # slope
                    # x-sequence for plotting line of best fit
                    xseq = np.linspace(np.min(i_gates_corr_kurt_prod[key])-0.05, 
                                       np.max(i_gates_corr_kurt_prod[key])+0.05, num=100)
                    
                    f3, ax3 = plt.subplots()
                    ax3.scatter(i_gates_corr_kurt_prod[key], ccnorms_change)
                    ax3.set_xlabel(key + ' gate correlation')
                    ax3.set_ylabel(ccnorms_change_measure)
                    ax3.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID + '; p = ' + str(est2.pvalues[1]))
                    ax3.plot(xseq, a+b*xseq, color='b')
                    ax3.spines[['right', 'top']].set_visible(False)'''
                    
            neurons_list = {}
            for key in i_model_corr_gates[0]: # loop over gates
                # list of neurons where correlation kurtosis is > 2 (in i_model)
                neurons = np.nonzero(scipy.stats.kurtosis(i_corr_gates[key], axis=1) > 2)[0]
                # HU where correlation with gate is maximum, for each neuron (in i_model)
                HU_max_corr = np.argmax(np.abs(i_corr_gates[key]), axis=1)
                
                neurons_list[key] = [neurons, HU_max_corr]
                
            models_neurons_list[i_model_ID] = neurons_list
        
    return models_neurons_list
        
#############################################################################################################################

def plot_weight_sparsity(ccnorms_change_measure, models_sparse_out_weights, models_eff_HUs, models_results_dict, 
                         models_fits_dict, n_neurons, best_lambdas, noise_ratios, dataset_ID, models):
    '''
    Parameters
    ----------
    ccnorms_change_measure : measure of choice for CCnorm change between two models: either "CCnorm_diff" (difference)
    or CCnorm_percent_change (percentage change)
    models_sparse_out_weights : dictionary, for each model, of output weight sparsity measures for each neuron/model output 
    unit, as well as their p-values (dimensions -> (neurons, folds, lambdas))
    models_eff_HUs : dictionary of lists containing, for each model, a list of effective hidden units for each neuron, for 
    each fold and lambda (dimensions -> (folds, lambdas, neurons))
    models_results_dict : dictionary of CCnorm arrays for each model (training, training for all folds and lambdas, 
    validation, validation for all folds and lambdas)
    models_fits_dict : dictionary of fitted parameters and hyperparameters for each model, fold, and lambda
    n_neurons : number of neurons in dataset
    best_lambdas : dictionary of best lambdas of each neuron, for each dataset and model
    noise_ratios : list of noise ratios of neurons
    dataset_ID : dataset name
    models : list of models to be compared and plotted

    Returns
    -------
    models_neurons_list : list of neurons for each model , based on given condition
    '''
    
    for i_model_ID in models: # loop over models
        for j_model_ID in models: # loop over models
            # difference in CCnorm between two models, for each neuron
            ccnorms_diff = models_results_dict[i_model_ID][2] - models_results_dict[j_model_ID][2]
            
            # percentage change in CCnorm between two models, for each neuron
            ccnorms_percent_change = (models_results_dict[i_model_ID][2] - \
                                      models_results_dict[j_model_ID][2])/models_results_dict[j_model_ID][2]*100
            
            # choose CCnorm change measure
            if ccnorms_change_measure == 'CCnorm_diff':
                ccnorms_change = ccnorms_diff
            elif ccnorms_change_measure == 'CCnorm_percent_change':
                ccnorms_change = ccnorms_percent_change
            
            # NumPy array to hold difference in sparsity between two models for each neuron
            sparsity_diff = np.zeros((n_neurons))
            
            for nID in range(n_neurons): # loop over neurons
                i_neuron_lambda = best_lambdas[dataset_ID][i_model_ID][nID] # best lambda for neuron in i_model
                j_neuron_lambda = best_lambdas[dataset_ID][j_model_ID][nID] # best lambda for neuron in j_model
                
                # sparsity for neuron in model_i and model_j and its best lambda in each model
                neuron_sparsity_i = models_sparse_out_weights[i_model_ID]['sparsity'][nID, 0, i_neuron_lambda]
                neuron_sparsity_j = models_sparse_out_weights[j_model_ID]['sparsity'][nID, 0, j_neuron_lambda]
                sparsity_diff[nID] = neuron_sparsity_i - neuron_sparsity_j # sparsity difference
            
            # scatter plot of CCnorm change against output weight sparsity difference for each neuron
            f1, ax1 = plt.subplots()
            ax1.scatter(sparsity_diff, ccnorms_change)
            ax1.set_xlabel('Difference in output weight sparsity')
            ax1.set_ylabel(ccnorms_change_measure)
            ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID)
            ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            
#############################################################################################################################

def plot_gate_weights(ccnorms_change_measure, models_gate_weights, models_eff_HUs, models_results_dict, 
                      models_fits_dict, n_neurons, best_lambdas, noise_ratios, dataset_ID, models):
    '''
    Parameters
    ----------
    ccnorms_change_measure : measure of choice for CCnorm change between two models: either "CCnorm_diff" (difference)
    or CCnorm_percent_change (percentage change)
    models_gate_weights : list, for each model, of gate weight contribution arrays for each neuron/model output 
    unit; the first array contains contributions by gate, whereas the second one contains contributions of all gates
    together; this only contains data for "gates" models
    models_eff_HUs : dictionary of lists containing, for each model, a list of effective hidden units for each neuron, for 
    each fold and lambda (dimensions -> (folds, lambdas, neurons))
    models_results_dict : dictionary of CCnorm arrays for each model (training, training for all folds and lambdas, 
    validation, validation for all folds and lambdas)
    models_fits_dict : dictionary of fitted parameters and hyperparameters for each model, fold, and lambda
    n_neurons : number of neurons in dataset
    best_lambdas : dictionary of best lambdas of each neuron, for each dataset and model
    noise_ratios : list of noise ratios of neurons
    dataset_ID : dataset name
    models : list of models to be compared and plotted

    Returns
    -------
    models_neurons_list : list of neurons for each model , based on given condition
    '''
    
    for i_model_ID in models: # loop over models
        if i_model_ID not in ['pop_gatesGRU', 'pop_gatesLSTM', 'pop_gatessubLSTM']:
            return
        else:
            if i_model_ID == 'pop_gatesGRU':
                gates_names = ['r', 'z']
            elif i_model_ID == 'pop_gatesLSTM':
                gates_names = ['i', 'f', 'o']
            
            j_model_ID = i_model_ID.replace('gates', '')
            
            # difference in CCnorm between two models, for each neuron
            ccnorms_diff = models_results_dict[i_model_ID][2] - models_results_dict[j_model_ID][2]
            
            # percentage change in CCnorm between two models, for each neuron
            ccnorms_percent_change = (models_results_dict[i_model_ID][2] - \
                                      models_results_dict[j_model_ID][2])/models_results_dict[j_model_ID][2]*100
            
            # choose CCnorm change measure
            if ccnorms_change_measure == 'CCnorm_diff':
                ccnorms_change = ccnorms_diff
            elif ccnorms_change_measure == 'CCnorm_percent_change':
                ccnorms_change = ccnorms_percent_change
            
            for gID in range(np.size(models_gate_weights[i_model_ID][0], 1)):
                # linear regression fit of CCnorm change and correlation with silence- or sound-triggered stimuli
                X2 = sm.add_constant(models_gate_weights[i_model_ID][0][:, gID]) # add bias to independent variable
                est = sm.OLS(ccnorms_change, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                b = est2.params[1] # slope
                # x-sequence for plotting line of best fit
                xseq = np.linspace(np.min(models_gate_weights[i_model_ID][0][:, gID])-0.05, 
                                   np.max(models_gate_weights[i_model_ID][0][:, gID])+0.05, num=100)
                
                # scatter plot of CCnorm change against output weight sparsity difference for each neuron
                f1, ax1 = plt.subplots()
                ax1.scatter(models_gate_weights[i_model_ID][0][:, gID], ccnorms_change)
                ax1.set_xlabel(gates_names[gID] + ' gate contribution (%)')
                ax1.set_ylabel(ccnorms_change_measure)
                ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID + '; p = ' + str(est2.pvalues[1]))
                ax1.plot(xseq, a+b*xseq, color='b')
                ax1.spines[['right', 'top']].set_visible(False)
            
            # linear regression fit of CCnorm change and correlation with silence- or sound-triggered stimuli
            X2 = sm.add_constant(models_gate_weights[i_model_ID][1]) # add bias to independent variable
            est = sm.OLS(ccnorms_change, X2) # Ordinary Least Squares fit
            est2 = est.fit()
            a = est2.params[0] # bias
            b = est2.params[1] # slope
            # x-sequence for plotting line of best fit
            xseq = np.linspace(np.min(models_gate_weights[i_model_ID][1])-0.05, 
                               np.max(models_gate_weights[i_model_ID][1])+0.05, num=100)
            
            f2, ax2 = plt.subplots()
            ax2.scatter(models_gate_weights[i_model_ID][1], ccnorms_change)
            ax2.set_xlabel('all gates contribution (%)')
            ax2.set_ylabel(ccnorms_change_measure)
            ax2.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID + '; p = ' + str(est2.pvalues[1]))
            ax2.plot(xseq, a+b*xseq, color='b')
            ax2.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            
#############################################################################################################################

def plot_repetitiveness(ccnorms_change_measure, models_ccnorms, stims_used, stim_dataset, padding_bins, n_neurons, 
                        dataset_ID, models):
    
    autocorr_max = np.zeros((np.size(stim_dataset, 2)))
    autocorr_max_lag = np.zeros((np.size(stim_dataset, 2)))
    
    for sID in range(np.size(stim_dataset, 2)):
        stim = stim_dataset[:, padding_bins:, sID]
        stim_autocorr = scipy.signal.correlate2d(stim, stim, mode='same', boundary='wrap')
        autocorr_trace = stim_autocorr[int((np.size(stim_autocorr, 0)-1)/2)]
        autocorr_trace_norm = autocorr_trace/np.nanmax(autocorr_trace)
        autocorr_1sided = autocorr_trace_norm[np.where(autocorr_trace_norm == 1)[0][0]+1:]
        
        if len(np.where(np.diff(autocorr_1sided)>=0)[0]) > 0:
            autocorr_max[stims_used[sID]] = np.nanmax(autocorr_1sided[np.where(np.diff(autocorr_1sided)>=0)[0] + 1])
            autocorr_max_lag[stims_used[sID]] = np.where(autocorr_1sided == \
                                             np.nanmax(autocorr_1sided[np.where(np.diff(autocorr_1sided)>=0)[0] + 1]))[0][0]
        else:
            autocorr_max[stims_used[sID]] = 0
            autocorr_max_lag[stims_used[sID]] = 0

    for model_ID in models:
        models_ccnorms[model_ID][2] = np.nanmean(models_ccnorms[model_ID][2], 0)
    
    for i_model_ID in models: # loop over models
        for j_model_ID in models: # loop over models
            if i_model_ID != j_model_ID:
                # difference in CCnorm between two models, for each neuron
                ccnorms_diff = models_ccnorms[i_model_ID][2] - models_ccnorms[j_model_ID][2]
                
                # percentage change in CCnorm between two models, for each neuron
                ccnorms_percent_change = \
                    (models_ccnorms[i_model_ID][2] - models_ccnorms[j_model_ID][2])/models_ccnorms[j_model_ID][2]*100

                # choose CCnorm change measure
                if ccnorms_change_measure == 'CCnorm_diff':
                    ccnorms_change = ccnorms_diff
                elif ccnorms_change_measure == 'CCnorm_percent_change':
                    ccnorms_change = ccnorms_percent_change
                
                # linear regression fit of CCnorm change and correlation with silence- or sound-triggered stimuli
                X2 = sm.add_constant(autocorr_max) # add bias to independent variable
                est = sm.OLS(ccnorms_change, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                b = est2.params[1] # slope
                # x-sequence for plotting line of best fit
                xseq = np.linspace(np.min(autocorr_max)-0.05, np.max(autocorr_max)+0.05, num=100)
                
                f1, ax1 = plt.subplots()
                ax1.scatter(autocorr_max, ccnorms_change)
                ax1.set_xlabel('Max autocorrelation')
                ax1.set_ylabel('CCnorm difference')
                ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID + '; p = ' + str(est2.pvalues[1]))
                ax1.set_xlim([0.92, 1])
                ax1.plot(xseq, a+b*xseq, color='b')
                ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
    
                
                # linear regression fit of CCnorm change and correlation with silence- or sound-triggered stimuli
                X2 = sm.add_constant(autocorr_max_lag) # add bias to independent variable
                est = sm.OLS(ccnorms_change, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                b = est2.params[1] # slope
                # x-sequence for plotting line of best fit
                xseq = np.linspace(np.min(autocorr_max_lag)-0.05, np.max(autocorr_max_lag)+0.05, num=100)
                
                f2, ax2 = plt.subplots()
                ax2.scatter(autocorr_max_lag, ccnorms_change)
                ax2.set_xlabel('Lag of max autocorrelation')
                ax2.set_ylabel('CCnorm difference')
                ax2.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID + '; p = ' + str(est2.pvalues[1]))
                ax2.plot(xseq, a+b*xseq, color='b')
                ax2.spines[['right', 'top']].set_visible(False)
            

#############################################################################################################################

def comparisons(datasets, models, results_dict, av_results_dict, nr_dict):
    '''
    Parameters
    ----------
    datasets : list of datasets
    models : list of models
    results_dict : dictionary of CCnormsof all neurons for each dataset and model (4 sets of CCnorms -> training, training 
    over all folds and lambdas, validation, validation over all folds and lambdas)
    av_results_dict : dictionary of average CCnorms (over all neurons) for each dataset and model (validation average, 
    validation SEM, training average, training SEM)
    nr_dict : dictionary of noise ratios of neurons for each dataset

    Returns
    -------
    m_i_n_dict : dictionary containing, for each dataset and model comparison, the most improved neurons 
    l_i_n_dict : dictionary containing, for each dataset and model comparison, the least improved (most unimproved) neurons 
    ttest_dict : dictionary containing, for each dataset and model comparison, the t-test results
    ranktest_dict : dictionary containing, for each dataset and model comparison, the rank-test results
    nr_corr_dict : dictionary containing, for each dataset and model comparison, the correlation coefficient of the CCnorm
    change and the noise ratios for all neurons
    prop_better_dict : dictionary containing, for each dataset and model comparison, the proportion of neurons better
    modelled by the first model over the second
    '''
    
    m_i_n_dict = {}
    l_i_n_dict = {}
    ttest_dict = {}
    ranktest_dict = {}
    nr_corr_dict = {}
    prop_better_dict = {}

    for dataset in datasets: # loop over datasets
        models_m_i_n_dict = {}
        models_l_i_n_dict = {}
        models_ttest_dict = {}
        models_ranktest_dict = {}
        models_nr_corr_dict = {}
        models_prop_better_dict = {}
        
        for i_model_ID in models: # loop over models
            if i_model_ID in ['pop_MGU', 'pop_GRU', 'pop_LSTM', 'pop_subLSTM', 'pop_f_LSTM', 'pop_rc_LSTM']:
                for j_model_ID in models: # loop over models
                    val_ccnorms_1 = np.array(results_dict[dataset][i_model_ID][2]) # i_model validation CCnorms
                    val_ccnorms_2 = np.array(results_dict[dataset][j_model_ID][2]) # j_model validation CCnorms
                    
                    color = [item/40 for item in nr_dict[dataset]] # define colours linked to noise ratio values
                    
                    # scatter plot of validation CCnorms for i_model against j_model, colour coded by noise ratio
                    fig, ax = plt.subplots(layout='constrained')
                    ax.scatter(val_ccnorms_2, val_ccnorms_1, c=color, cmap=mpl.colormaps['Blues'], edgecolors='black')
                    #fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 40), cmap='Blues'), ax=ax, 
                                 #orientation='vertical', label='noise ratio')
                    # equal performance red dashed line
                    ax.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), 'r', linestyle='--', linewidth=3)
                    ax.axvline(x=av_results_dict[dataset][j_model_ID][0], color='k', linewidth=1.5) # average j_model CCnorm
                    ax.axhline(y=av_results_dict[dataset][i_model_ID][0], color='k', linewidth=1.5) # average i_model CCnorm
                    ax.scatter(av_results_dict[dataset][j_model_ID][0], av_results_dict[dataset][i_model_ID][0], s=25, 
                               color='lime', zorder=3) # single scatter dot at average j_model and i_model validation CCnorm
                    # make axes the same length ("equal" aspect ratio)
                    ax = plt.gca()
                    ax.set_aspect('equal')
                    ax.set_xlim([0, 1])
                    ax.set_ylim([0, 1])
                    ax.xaxis.set_tick_params(labelcolor='none')
                    ax.yaxis.set_tick_params(labelcolor='none')
                    ax.spines[['right', 'top']].set_visible(False)
                    
                    #ax.set_xlabel(j_model_ID)
                    #ax.set_ylabel(i_model_ID)
                    #ax.set_title('CCnorm - ' + dataset)
                    
                    val_ccnorms_diff = val_ccnorms_1 - val_ccnorms_2 # validation CCnorm difference
                    val_ccnorms_diff_mean = np.nanmean(val_ccnorms_diff) # mean difference
                    val_ccnorms_diff_sd = np.nanstd(val_ccnorms_diff, ddof=1) # STD of difference
                    
                    # uncomment to generate scatter plots of CCnorm difference against noise ratio
                    '''plt.figure()
                    plt.scatter(nr_dict[dataset], val_ccnorms_diff)
                    plt.axhline(y=0, color='g') # horizontal green line at 0 difference
                    plt.axhline(y=val_ccnorms_diff_mean, color='r') # horizontal red line at mean difference
                    #plt.axhline(y=val_ccnorms_diff_mean+1.5*val_ccnorms_diff_sd, color='r', linestyle='--')
                    #plt.axhline(y=val_ccnorms_diff_mean-1.5*val_ccnorms_diff_sd, color='r', linestyle='--')
                    plt.xlim([-1, 41])
                    plt.ylim([-0.5, 0.55])
                    plt.xlabel('noise ratio')
                    plt.ylabel('CCnorm difference')
                    plt.title('Performance difference - ' + dataset + ' - ' + i_model_ID + ' vs ' + j_model_ID)'''
                    
                    # most improved neurons: neurons falling more than 1.5 STDs above the mean CCnorm difference
                    '''models_m_i_n_dict[i_model_ID + ' vs ' + j_model_ID] = \
                        [np.where(val_ccnorms_diff > (val_ccnorms_diff_mean + 1.5*val_ccnorms_diff_sd))[0], \
                         str(np.sum(val_ccnorms_diff > 0)) + '/' + str(len(val_ccnorms_diff))]'''
                            
                    models_m_i_n_dict[i_model_ID + ' vs ' + j_model_ID] = \
                        [np.where(val_ccnorms_diff > (val_ccnorms_diff_mean + 1*val_ccnorms_diff_sd))[0], \
                         str(np.sum(val_ccnorms_diff > 0)) + '/' + str(len(val_ccnorms_diff))]
                            
                    # least improved neurons: neurons falling more than 1.5 STDs below the mean CCnorm difference
                    models_l_i_n_dict[i_model_ID + ' vs ' + j_model_ID] = \
                        [np.where(val_ccnorms_diff < (val_ccnorms_diff_mean - 1.5*val_ccnorms_diff_sd))[0]]
                        
                    # two-tailed paired t-test on validation CCnorm comparison
                    models_ttest_dict[i_model_ID + ' vs ' + j_model_ID] = \
                        scipy.stats.ttest_rel(results_dict[dataset][i_model_ID][2], results_dict[dataset][j_model_ID][2])
                    
                    # Spearman rank-order test on validation CCnorm comparison
                    models_ranktest_dict[i_model_ID + ' vs ' + j_model_ID] = \
                        scipy.stats.spearmanr(results_dict[dataset][i_model_ID][2], results_dict[dataset][j_model_ID][2])
                    
                    # Pearson correlation between validation CCnorm difference and noise ratios
                    models_nr_corr_dict[i_model_ID + ' vs ' + j_model_ID] = \
                        scipy.stats.pearsonr(val_ccnorms_diff, nr_dict[dataset])
                    
                    # proportion of neurons better modelled by i_model
                    models_prop_better_dict[i_model_ID + ' vs ' + j_model_ID] = \
                        len(np.where(val_ccnorms_diff > 0)[0])/len(val_ccnorms_diff)
                    
        m_i_n_dict[dataset] = models_m_i_n_dict
        l_i_n_dict[dataset] = models_l_i_n_dict
        ttest_dict[dataset] = models_ttest_dict
        ranktest_dict[dataset] = models_ranktest_dict
        nr_corr_dict[dataset] = models_nr_corr_dict
        prop_better_dict[dataset] = models_prop_better_dict
    
    return m_i_n_dict, l_i_n_dict, ttest_dict, ranktest_dict, nr_corr_dict, prop_better_dict

#############################################################################################################################
                                    
def plot_neurons(datasets, models, models_neuron_list, fits_dict, results_dict, data_dict, best_lambda_dict, device):
    from util_funcs import calc_CC_norm, get_stim_response_grid_concat, calc_CC_norm_test
    
    reduced = False
    zero_padding = True
    
    for dataset in datasets:
        for model_i in models:
            meta_info_i = fits_dict[dataset][model_i][-1]
            
            model_type_i = meta_info_i['model_type']
            type_data_i = meta_info_i['type_data']
            t_lims_i = meta_info_i['time_axis']
            t_i = np.mean([t_lims_i[:-1], t_lims_i[1:]], axis=0) 
            n_F_i = meta_info_i['n_F']
            f_range_i = np.logspace(np.log10(500), np.log10(22000), n_F_i)
            clip_bins_i = meta_info_i['clip_bins']
            shift_bins_i = meta_info_i['shift_bins']
            n_h_i = meta_info_i['n_h']
            padding_bins_i = meta_info_i['padding_bins']
            norm_i = meta_info_i['norm']
            
            stim_dataset_i = data_dict[dataset][model_i][0]['test_stim']
            resp_dataset_i = data_dict[dataset][model_i][0]['test_resp']
            
            all_stim_dataset_i = data_dict[dataset][model_i][0]['multi_stim']
            all_resp_dataset_i = data_dict[dataset][model_i][0]['multi_resp']
            
            stimuli_dict_i = data_dict[dataset][model_i][1]
            stimuli_i = data_dict[dataset][model_i][1]['test_stimuli']
            folds_i = data_dict[dataset][model_i][2]
            
            model_params_i = fits_dict[dataset][model_i][1]['all_model_params']
            
            clip_size_i = int(stim_dataset_i.size(3)/len(stimuli_dict_i['test_stimuli'][0]))
            clip_size_i_conv = clip_size_i - meta_info_i['padding_bins']
            
            mean_cv_i = data_dict[dataset][model_i][3][0]
            std_cv_i = data_dict[dataset][model_i][3][0]
            
            for model_j in models:
                if model_i != model_j and (model_i in ['pop_GRU', 'pop_f_LSTM', 'pop_rc_LSTM']):
                    meta_info_j = fits_dict[dataset][model_j][-1]
                    
                    model_type_j = meta_info_j['model_type']
                    type_data_j = meta_info_j['type_data']
                    t_lims_j = meta_info_j['time_axis']
                    t_j = np.mean([t_lims_j[:-1], t_lims_j[1:]], axis=0) 
                    n_F_j = meta_info_j['n_F']
                    f_range_j = np.logspace(np.log10(500), np.log10(22000), n_F_j)
                    clip_bins_j = meta_info_j['clip_bins']
                    shift_bins_j = meta_info_j['shift_bins']
                    n_h_j = meta_info_j['n_h']
                    padding_bins_j = meta_info_j['padding_bins']
                    norm_j = meta_info_j['norm']
                    
                    stim_dataset_j = data_dict[dataset][model_j][0]['test_stim']
                    resp_dataset_j = data_dict[dataset][model_j][0]['test_resp']
                    
                    stimuli_dict_j = data_dict[dataset][model_j][1]
                    stimuli_j = data_dict[dataset][model_j][1]['test_stimuli']
                    folds_j = data_dict[dataset][model_j][2]
                    
                    model_params_j = fits_dict[dataset][model_j][1]['all_model_params']
                    
                    clip_size_j = int(stim_dataset_j.size(3)/len(stimuli_dict_j['test_stimuli'][0]))
                    clip_size_j_conv = clip_size_j - meta_info_j['padding_bins']
                    
                    mean_cv_j = data_dict[dataset][model_j][3][0]
                    std_cv_j = data_dict[dataset][model_j][3][0]
                    
                    m_i_n = models_neuron_list[dataset][model_i + ' vs ' + model_j][0]
                    
                    #########################################################################################################
                    bestlambda_i = best_lambda_dict[dataset][model_i]
                    m_i_n_bestlambda_i = bestlambda_i[m_i_n]

                    bestlambda_j = best_lambda_dict[dataset][model_j]
                    m_i_n_bestlambda_j = bestlambda_j[m_i_n]
                    
                    for nID in range(len(m_i_n)):
                        for fID in range(folds_i):
                            fit_params_model_i = fits_dict[dataset][model_i][0][fID][m_i_n_bestlambda_i[nID]][0]
                            
                            fitted_model_i = getattr(all_models, model_i)(model_params_i)
                            fitted_model_i.to(device)
                            fitted_model_i.reload(fit_params_model_i)
                            
                            input_data_i = stim_dataset_i[fID]
                            target_data_i = resp_dataset_i[:, fID]

                            outputs_i, targets_i = fitted_model_i.forward_loop(input_data_i, target_data_i, 'val')
                            
                            traces_i = fitted_model_i.np_forward_loop(input_data_i)
                            predicted_i = np.transpose(outputs_i.detach().cpu().numpy())
                            
                            #################
                            fit_params_model_j = fits_dict[dataset][model_j][0][fID][m_i_n_bestlambda_j[nID]][0]
                            
                            fitted_model_j = getattr(all_models, model_j)(model_params_j)
                            fitted_model_j.to(device)
                            fitted_model_j.reload(fit_params_model_j)
                            
                            input_data_j = stim_dataset_j[fID]
                            target_data_j = resp_dataset_j[:, fID]
                            outputs_j, targets_j = fitted_model_j.forward_loop(input_data_j, target_data_j, 'val')
                            
                            predicted_j = np.transpose(outputs_j.detach().cpu().numpy())
                            
                            spiketimes_nID = all_resp_dataset_i[m_i_n[nID]]
                            
                            data = [all_resp_dataset_i[m_i_n[nID]][stim] for stim in stimuli_i[fID]]
                            _, actual_reps = get_stim_response_grid_concat(data, all_stim_dataset_i, stimuli_i, 
                                                                           fID, n_h_i, meta_info_i['bin_size'], 'val', 
                                                                           norm_i, mean_cv_i, std_cv_i, clip_bins_i, 
                                                                           shift_bins_i, zero_padding, padding_bins_i, 
                                                                           t_lims_i)
                            
                            all_stim_ccnorm_i, _, _ = calc_CC_norm(actual_reps, predicted_i[m_i_n[nID]])
                            if model_type_j == 'not_conv':
                                all_stim_ccnorm_j, _, _ = calc_CC_norm(actual_reps, predicted_j)
                            else:
                                conv_reshape_resps = np.reshape(np.transpose(predicted_j[m_i_n[nID]]), \
                                                                (len(stimuli_i[fID])*clip_size_j_conv))
                                all_stim_ccnorm_j, _, _ = calc_CC_norm(actual_reps, conv_reshape_resps)
                            
                            fig = plt.figure(figsize=(15, 12))
                            fig.suptitle('Neuron ' + str(m_i_n[nID]) + ' - CCnorms ' + str(round(all_stim_ccnorm_i, 2)) + \
                                       ', ' + str(round(all_stim_ccnorm_j, 2)), fontsize=10)
                            
                            outer_grid = fig.add_gridspec(4, 3, wspace=0.2, hspace=0.4, left=0.05, right=0.95, top=0.9, \
                                                          bottom=0.05)
                            
                            for stim_ID in range(len(stimuli_i[fID])):
                                actual = resp_dataset_i[m_i_n[nID], fID, \
                                                        stim_ID*clip_size_i:(stim_ID+1)*clip_size_i].detach().cpu().numpy()
                                
                                spiketimes_stimID = spiketimes_nID[stimuli_i[fID][stim_ID]]
                                
                                # plot results of neurons
                                grid_row_idx = stim_ID // outer_grid._ncols
                                grid_col_idx = stim_ID % outer_grid._ncols
                                
                                inner_grid = outer_grid[int(grid_row_idx), int(grid_col_idx)].subgridspec(3, 1)
                                ax1, ax2, ax3 = inner_grid.subplots(sharex=True)
                                
                                for rID in range(len(spiketimes_stimID)):
                                    ax1.scatter(spiketimes_stimID[rID], rID*np.ones((len(spiketimes_stimID[rID]))), 
                                                color='k', s=5)
                                ax1.spines[['right', 'top']].set_visible(False)
                                
                                ax2.plot(t_i, actual, label='FR')
                                ax2.plot(t_i, predicted_i[m_i_n[nID], stim_ID*clip_size_i:(stim_ID+1)*clip_size_i], 
                                         label=model_i)
                                if model_type_j == 'not_conv':
                                    ax2.plot(t_j, predicted_j[m_i_n[nID], stim_ID*clip_size_j:(stim_ID+1)*clip_size_j], 
                                             label=model_j)
                                else:
                                    ax2.plot(t_j, predicted_j[m_i_n[nID], :, stim_ID], label=model_j)

                                ax2.spines[['right', 'top']].set_visible(False)
                                if stim_ID == len(stimuli_i[fID]) - 1:
                                    ax2.legend(loc=(1.5, 0))
                                
                                im = ax3.imshow(all_stim_dataset_i[:, n_h_i-1:, stimuli_i[fID][stim_ID]], cmap='jet', 
                                                aspect='auto', interpolation='none', extent=extents(t_i) + \
                                                extents(f_range_i), origin='lower')
                                if grid_row_idx != 3:
                                    ax3.set_xticklabels([])
                                #f.colorbar(im, ax=ax3)
                                
                                ccnorm_i, _, _ = calc_CC_norm(actual_reps[stim_ID*clip_size_i:(stim_ID+1)*clip_size_i, :], 
                                                              predicted_i[m_i_n[nID], 
                                                                          stim_ID*clip_size_i:(stim_ID+1)*clip_size_i])
                                    
#############################################################################################################################

def plot_loss_curves(datasets, models, fits_dict):
    for dataset_ID in datasets:
        for model_ID in models:
            model_fits_dict = fits_dict[dataset_ID][model_ID][0]
            lambda_sequence = fits_dict[dataset_ID][model_ID][1]['lambdas']
            
            for fID in range(len(model_fits_dict)):
                for lamb in range(len(model_fits_dict[fID])):
                    train_loss_curve = model_fits_dict[fID][lamb][1]
                    tune_loss_curve = model_fits_dict[fID][lamb][2]
                    
                    plt.figure()
                    plt.plot(train_loss_curve)
                    plt.plot(tune_loss_curve)
                    plt.xlabel('epoch')
                    plt.ylabel('loss')
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.legend(['training', 'tuning'])
                    plt.title(dataset_ID + ' - ' + model_ID + ' - ' + 'Fold ' + str(fID+1) + ', lambda ' + 
                              str(lambda_sequence[lamb]))
                    plt.show()
                    
#############################################################################################################################

def depth_sw_analysis(ccnorms_change_measure, models_results_dict, models_built_data, dataset_ID, models, plot_sw):
    plot_mode = 'av'
    exclude_few = 'exclude'
    
    colours = {'twoD_CNN': 'teal', 'oneD_CNN': 'teal', 'pop_f_LSTM': 'salmon', 'pop_rc_LSTM': 'salmon'}
    
    if dataset_ID == 'NS2_include_single':
        depth_sw_file = 'Data/NS2_raw/' + dataset_ID + '_depth_sw.pckl'
    else:
        depth_sw_file = 'Data/' + dataset_ID + '/' + dataset_ID + '_depth_sw.pckl'
    
    f = open(depth_sw_file, 'rb')
    depth_sw = pickle.load(f)
    f.close()
    
    neuron_list = models_built_data[models[0]][-1]['neurons_ID_sess']
    noise_ratios = models_built_data[models[0]][-1]['noise_ratios']
    
    neuron_depths = []
    neuron_layers = []
    neuron_sws = []
    
    neuron_idxs = [] # holds indexes of neurons for which we have depth, layer, and sw info
    
    for n_idx in range(len(neuron_list)):
        neuron_ID = neuron_list[n_idx][0]
        
        if neuron_ID in depth_sw[0]:
            neuron_depths.append(depth_sw[0][neuron_ID])
            neuron_layers.append(depth_sw[1][neuron_ID])
            neuron_sws.append(depth_sw[2][neuron_ID])
            
            neuron_idxs.append(n_idx)
    n_IDs = []       
    for n_idx in range(len(neuron_list)):
        neuron_ID = neuron_list[n_idx][0]
        
        if neuron_ID in depth_sw[0]:
            n_IDs.append(neuron_list[n_idx])
    
    win_overlap = 50 # depth averaging window overlap, in microns
    win_size = 200 # depth averaging window size, in microns
    
    # uncomment for plots with NS2 range
    '''if dataset_ID == 'NS3' or dataset_ID == 'NS3_PEG':
        small_depths = np.where(np.array(neuron_depths) < -700)[0]
        large_depths = np.where(np.array(neuron_depths) > 600)[0]
        extra_depths = np.concatenate((small_depths, large_depths))
        
        new_neuron_depths = np.delete(np.array(neuron_depths), extra_depths)
        new_neuron_layers = np.delete(np.array(neuron_layers), extra_depths)
        new_neuron_idxs = np.delete(np.array(neuron_idxs), extra_depths)

        neuron_depths = list(new_neuron_depths)
        neuron_layers = list(new_neuron_layers)
        neuron_idxs = list(new_neuron_idxs)'''

    if (np.max(neuron_depths) - np.min(neuron_depths)) % win_overlap == 0:
        left_min = np.min(neuron_depths)
        left_max = np.max(neuron_depths) - win_size
        
        depth_av_win_left = np.arange(left_min, left_max + win_overlap, win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        '''depth_av_win_left = np.arange(np.min(neuron_depths) - win_overlap/2, np.max(neuron_depths) + win_overlap/2, 
                                      win_overlap)
        depth_av_win_right = depth_av_win_left + win_size'''
        
        depth_av_win_centre = depth_av_win_left + win_size/2
    else:
        min_limit = roundup(np.min(neuron_depths))
        max_limit = roundup(np.max(neuron_depths))
        
        left_min = min_limit
        left_max = max_limit - win_size
        
        depth_av_win_left = np.arange(left_min, left_max + win_overlap, win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        #depth_av_win_left = np.arange(min_limit - win_overlap/2, max_limit + win_overlap/2, win_overlap)
        #depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
        
        depth_av_win_right[-1] = depth_av_win_right[-1] + 1
    
    sw_win_overlap = 0.05
    sw_win_size = 0.2
    if (np.max(neuron_sws) - np.min(neuron_sws)) % sw_win_overlap == 0:
        left_min = np.min(neuron_sws)
        left_max = np.max(neuron_sws) - sw_win_size
        
        sw_av_win_left = np.arange(left_min, left_max + sw_win_overlap, sw_win_overlap)
        sw_av_win_right = sw_av_win_left + sw_win_size
        
        sw_av_win_centre = sw_av_win_left + sw_win_size/2
    else:
        min_limit = round(np.min(neuron_sws), 0)
        max_limit = round(np.max(neuron_sws), 0)
        
        left_min = min_limit
        left_max = max_limit - sw_win_size

        sw_av_win_left = np.arange(left_min, left_max + sw_win_overlap, sw_win_overlap)
        sw_av_win_right = sw_av_win_left + sw_win_size
        
        sw_av_win_centre = sw_av_win_left + sw_win_size/2
        
        sw_av_win_right[-1] = sw_av_win_right[-1] + 1
        
    # define layer-based colourmap - black is 13, red is 4, blue is 56
    layers_cmap = mcolors.ListedColormap(['black', 'red', 'blue'])
    layer_colours = []
    for neuron in neuron_layers:
        if neuron == '13':
            layer_colours.append(0)
        elif neuron == '44':
            layer_colours.append(1)
        elif neuron == '56':
            layer_colours.append(2)
    
    CCnorms = {}
    
    depth_win_axis = {}
    depth_CCnorm_diff = {}
    depth_CCnorm_diff_se = {}
    
    sw_win_axis = {}
    sw_CCnorm_diff = {}
    sw_CCnorm_diff_se = {}
    
    f1, ax1 = plt.subplots()
    f2, ax2 = plt.subplots()
    for i_model_ID in models:
        model_ccnorms = models_results_dict[i_model_ID][2]
        
        used_ccnorms = model_ccnorms[neuron_idxs]
        used_nr = np.array(noise_ratios)[neuron_idxs]
        
        CCnorms[i_model_ID] = used_ccnorms

        ########## Plots for each model
        
        # Plot of depth, noise ratio, and spike width graphs
        '''f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        
        # CCnorm vs depth
        X2 = sm.add_constant(neuron_depths) # add bias to independent variable
        xseq = np.linspace(np.min(neuron_depths)-0.05, np.max(neuron_depths)+0.05, num=100)
        est = sm.OLS(used_ccnorms, X2) # Ordinary Least Squares fit
        est2 = est.fit()
        a = est2.params[0] # bias
        b = est2.params[1] # slope
        
        ax1.scatter(neuron_depths, used_ccnorms)
        ax1.plot(xseq, a+b*xseq, color='r')
        ax1.set_xlabel('depth (micron from layer 3/4)')
        ax1.set_ylabel('CCnorm')
        ax1.set_title('p = ' + str(est2.pvalues[1]))
        
        # CCnorm vs noise ratio, colour coded by layer
        scatter = ax2.scatter(used_nr, used_ccnorms, c=layer_colours, cmap=layers_cmap)
        ax2.set_xlabel('noise ratio')
        handles, labels = scatter.legend_elements()
        ax2.legend(handles = handles, labels = ['1/3', '4', '5/6'])
        
        # CCnorm vs spike width
        X2 = sm.add_constant(neuron_sws) # add bias to independent variable
        xseq = np.linspace(np.min(neuron_sws)-0.05, np.max(neuron_sws)+0.05, num=100)
        est = sm.OLS(used_ccnorms, X2) # Ordinary Least Squares fit
        est2 = est.fit()
        a = est2.params[0] # bias
        b = est2.params[1] # slope
        
        ax3.scatter(neuron_sws, used_ccnorms)
        ax3.plot(xseq, a+b*xseq, color='r')
        ax3.set_xlabel('spike width (ms)')
        ax3.set_title('p = ' + str(est2.pvalues[1]))
        
        f.suptitle(dataset_ID + ' - ' + i_model_ID)'''
        
        
        # Plot of depth graph only
        if plot_mode == 'full':
            # DEPTH
            X2 = sm.add_constant(neuron_depths) # add bias to independent variable
            xseq = np.linspace(np.min(neuron_depths)-0.05, np.max(neuron_depths)+0.05, num=100)
            est = sm.OLS(used_ccnorms, X2) # Ordinary Least Squares fit
            est2 = est.fit()
            a = est2.params[0] # bias
            b = est2.params[1] # slope
            
            f1, ax1 = plt.subplots()
            scatter = ax1.scatter(neuron_depths, used_ccnorms, c=layer_colours, cmap=layers_cmap)
            ax1.plot(xseq, a+b*xseq, color='r')
            #ax1.set_xlim([-700, 600]) # uncomment for plots with NS2 depth range
            ax1.set_xlabel('depth (micron from layer 3/4)')
            ax1.set_ylabel('CCnorm')
            handles, labels = scatter.legend_elements()
            ax1.legend(handles = handles, labels = ['1/3', '4', '5/6'])
            ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
            ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            
            #f1.savefig('/home/lorenzom/Downloads/Temp/Depth_sw/CCnorm/' + dataset_ID + '_' + i_model_ID + '_depth.png')
            
            # SW
            if plot_sw:
                X2 = sm.add_constant(neuron_sws) # add bias to independent variable
                xseq = np.linspace(np.min(neuron_sws)-0.05, np.max(neuron_sws)+0.05, num=100)
                est = sm.OLS(used_ccnorms, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                b = est2.params[1] # slope
                
                f2, ax2 = plt.subplots()
                scatter = ax2.scatter(neuron_sws, used_ccnorms, c=layer_colours, cmap=layers_cmap)
                ax2.plot(xseq, a+b*xseq, color='r')
                #ax2.set_xlim([-700, 600]) # uncomment for plots with NS2 depth range
                ax2.set_xlabel('spike width (ms)')
                ax2.set_ylabel('CCnorm')
                ax2.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
                ax2.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
                
                #f2.savefig('/home/lorenzom/Downloads/Temp/Depth_sw/CCnorm/' + dataset_ID + '_' + i_model_ID + '_depth.png')
        
        elif plot_mode == 'av':
            # DEPTH
            depth_av_ccnorm = np.zeros((len(depth_av_win_centre)))
            depth_se_ccnorm = np.zeros((len(depth_av_win_centre)))
            
            if exclude_few == 'exclude':
                for win in range(len(depth_av_ccnorm)):
                    '''idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]'''
                    
                    idxs_1 = np.where((np.array(neuron_depths) >= depth_av_win_left[win]))[0]
                    idxs_2 = np.where((np.array(neuron_depths) < depth_av_win_right[win]))[0]
                    idxs = np.intersect1d(idxs_1, idxs_2)
                    
                    if len(idxs) >= 3:
                        depth_av_ccnorm[win] = np.nanmean(used_ccnorms[idxs])
                        depth_se_ccnorm[win] = np.nanstd(used_ccnorms[idxs], ddof=1)/np.sqrt(len(idxs))
                    else:
                        depth_av_ccnorm[win] = np.nan
                        depth_se_ccnorm[win] = np.nan

            else:
                for win in range(len(depth_av_ccnorm)):
                    '''idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where ((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]'''
                    
                    idxs_1 = np.where((np.array(neuron_depths) >= depth_av_win_left[win]))[0]
                    idxs_2 = np.where((np.array(neuron_depths) < depth_av_win_right[win]))[0]
                    idxs = np.intersect1d(idxs_1, idxs_2)
                    
                    depth_av_ccnorm[win] = np.nanmean(used_ccnorms[idxs])
                    depth_se_ccnorm[win] = np.nanstd(used_ccnorms[idxs], ddof=1)/np.sqrt(len(idxs))
            
            #f1, ax1 = plt.subplots()
            ax1.plot(depth_av_win_centre, depth_av_ccnorm, linewidth=3, color=colours[i_model_ID])
            ax1.fill_between(depth_av_win_centre, depth_av_ccnorm-depth_se_ccnorm, depth_av_ccnorm+depth_se_ccnorm, 
                             color=colours[i_model_ID], alpha=0.2)
            #ax1.set_xlim([-700, 600]) # uncomment for plots with NS2 depth range
            #ax1.set_xlabel('depth (micron from layer 3/4)')
            #ax1.set_ylabel('CCnorm')
            #ax1.set_title(dataset_ID + ' - ' + i_model_ID)
            ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            
            ax1.set_xticks([-750, -500, -250, 0, 250, 500, 750]) # NS3-PEG, NS3
            #ax1.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8]) # NS2
            #ax1.set_yticks([0.5, 0.55, 0.6, 0.65, 0.7]) # NS3, NS2 range
            #ax1.set_yticks([0.45, 0.55, 0.65, 0.75]) # NS3
            #ax1.set_yticks([0.45, 0.5, 0.55, 0.6, 0.65]) # NS3-PEG, NS2 range
            ax1.set_yticks([0.35, 0.45, 0.55, 0.65, 0.75]) # NS3-PEG
            #ax1.set_xticks(np.arange(-750, 1000, 250))
            ax1.xaxis.set_tick_params(labelcolor='none')
            ax1.yaxis.set_tick_params(labelcolor='none')
            
            #f1.savefig('/home/lorenzom/Downloads/Temp/Depth_sw/CCnorm/Av/' + dataset_ID + '_' + i_model_ID + '_depth_win' + \
                       #str(win_size) + '_' + exclude_few + '.png')
            
            # SW
            if plot_sw:
                sw_av_ccnorm = np.zeros((len(sw_av_win_centre)))
                sw_se_ccnorm = np.zeros((len(sw_av_win_centre)))
                
                if exclude_few == 'exclude':
                    for win in range(len(sw_av_ccnorm)):
                        idxs_1 = np.where((np.array(neuron_sws) >= sw_av_win_left[win]))[0]
                        idxs_2 = np.where((np.array(neuron_sws) < sw_av_win_right[win]))[0]
                        idxs = np.intersect1d(idxs_1, idxs_2)
                        
                        if len(idxs) >= 3:
                            sw_av_ccnorm[win] = np.nanmean(used_ccnorms[idxs])
                            sw_se_ccnorm[win] = np.nanstd(used_ccnorms[idxs], ddof=1)/np.sqrt(len(idxs))
                        else:
                            sw_av_ccnorm[win] = np.nan
                            sw_se_ccnorm[win] = np.nan
    
                else:
                    for win in range(len(sw_av_ccnorm)):
                        idxs_1 = np.where((np.array(neuron_sws) >= sw_av_win_left[win]))[0]
                        idxs_2 = np.where((np.array(neuron_sws) < sw_av_win_right[win]))[0]
                        idxs = np.intersect1d(idxs_1, idxs_2)
                        
                        sw_av_ccnorm[win] = np.nanmean(used_ccnorms[idxs])
                        sw_se_ccnorm[win] = np.nanstd(used_ccnorms[idxs], ddof=1)/np.sqrt(len(idxs))
                
                
                ax2.plot(sw_av_win_centre, sw_av_ccnorm, linewidth=3, color=colours[i_model_ID])
                ax2.fill_between(sw_av_win_centre, sw_av_ccnorm-sw_se_ccnorm, sw_av_ccnorm+sw_se_ccnorm, 
                                 color=colours[i_model_ID], alpha=0.2)
                #ax2.set_xlim([-700, 600]) # uncomment for plots with NS2 depth range
                #ax2.set_xlabel('spike width (ms)')
                #ax2.set_ylabel('CCnorm')
                #ax2.set_title(dataset_ID + ' - ' + i_model_ID)
                ax2.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
                #ax2.set_yticks([0.55, 0.65, 0.75]) # NS3, NS2
                ax2.set_yticks([0.5, 0.6, 0.7]) # NS3-PEG
                ax2.set_xticks([0.2, 0.4, 0.6, 0.8])
                ax2.xaxis.set_tick_params(labelcolor='none')
                ax2.yaxis.set_tick_params(labelcolor='none')
        
        
        for j_model_ID in models: # loop over models
            if j_model_ID != i_model_ID:
                # difference in CCnorm between two models, for each neuron
                ccnorms_diff = models_results_dict[i_model_ID][2] - models_results_dict[j_model_ID][2]
                
                # percentage change in CCnorm between two models, for each neuron
                ccnorms_percent_change = (models_results_dict[i_model_ID][2] - \
                                          models_results_dict[j_model_ID][2])/models_results_dict[j_model_ID][2]*100
                
                # choose CCnorm change measure 
                if ccnorms_change_measure == 'CCnorm_diff':
                    ccnorms_change = ccnorms_diff
                elif ccnorms_change_measure == 'CCnorm_percent_change':
                    ccnorms_change = ccnorms_percent_change
                
                used_ccnorms_change = ccnorms_change[neuron_idxs]
                
                ########## Plots for each model comparison
                
                # Plot of depth, noise ratio, and spike width graphs
                '''
                f, (ax1, ax2, ax3) = plt.subplots(1, 3)
                
                # CCnorm difference vs depth
                X2 = sm.add_constant(neuron_depths) # add bias to independent variable
                xseq = np.linspace(np.min(neuron_depths)-0.05, np.max(neuron_depths)+0.05, num=100)
                est = sm.OLS(used_ccnorms_change, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                b = est2.params[1] # slope
                
                ax1.scatter(neuron_depths, used_ccnorms_change)
                ax1.plot(xseq, a+b*xseq, color='r')
                ax1.set_xlabel('depth (micron from layer 3/4)')
                ax1.set_ylabel('CCnorm diff')
                ax1.set_title('p = ' + str(est2.pvalues[1]))
                
                # CCnorm difference vs noise ratio, colour coded by layer
                scatter = ax2.scatter(used_nr, used_ccnorms_change, c=layer_colours, cmap=layers_cmap)
                ax2.set_xlabel('noise ratio')
                handles, labels = scatter.legend_elements()
                ax2.legend(handles = handles, labels = ['1/3', '4', '5/6'])
                
                # CCnorm difference vs spike width
                X2 = sm.add_constant(neuron_sws) # add bias to independent variable
                xseq = np.linspace(np.min(neuron_sws)-0.05, np.max(neuron_sws)+0.05, num=100)
                est = sm.OLS(used_ccnorms_change, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                b = est2.params[1] # slope
                
                ax3.scatter(neuron_sws, used_ccnorms_change)
                ax3.plot(xseq, a+b*xseq, color='r')
                ax3.set_xlabel('spike width (ms)')
                ax3.set_title('p = ' + str(est2.pvalues[1]))
                
                f.suptitle(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID)'''
                
                # Plot of depth only
                if plot_mode == 'full':
                    X2 = sm.add_constant(neuron_depths) # add bias to independent variable
                    xseq = np.linspace(np.min(neuron_depths)-0.05, np.max(neuron_depths)+0.05, num=100)
                    est = sm.OLS(used_ccnorms_change, X2) # Ordinary Least Squares fit
                    est2 = est.fit()
                    a = est2.params[0] # bias
                    b = est2.params[1] # slope
                    
                    f1, ax1 = plt.subplots()
                    scatter = ax1.scatter(neuron_depths, used_ccnorms_change, c=layer_colours, cmap=layers_cmap)
                    ax1.plot(xseq, a+b*xseq, color='r')
                    ax1.set_xlim([-700, 600]) # uncomment for plots with NS2 depth range
                    ax1.set_xlabel('depth (micron from layer 3/4)')
                    ax1.set_ylabel('CCnorm diff')
                    handles, labels = scatter.legend_elements()
                    ax1.legend(handles = handles, labels = ['1/3', '4', '5/6'])
                    ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID + ' - p = ' + str(est2.pvalues[1]))
                    ax1.spines[['right', 'top']].set_visible(False)
                    
                    #f1.savefig('/home/lorenzom/Downloads/Temp/Depth_sw/CCnorm/' + dataset_ID + '_' + i_model_ID + '_' + \
                               #j_model_ID + '_depth.png')
                
                elif plot_mode == 'av':
                    # DEPTH
                    depth_av_ccnorm_change = np.zeros((len(depth_av_win_centre)))
                    depth_se_ccnorm_change = np.zeros((len(depth_av_win_centre)))
                    
                    if exclude_few == 'exclude':
                        for win in range(len(depth_av_ccnorm)):
                            '''idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                                np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                            idxs = idxs[0]'''
                            
                            idxs_1 = np.where((np.array(neuron_depths) >= depth_av_win_left[win]))[0]
                            idxs_2 = np.where((np.array(neuron_depths) < depth_av_win_right[win]))[0]
                            idxs = np.intersect1d(idxs_1, idxs_2)
                            
                            if len(idxs) >= 3:
                                depth_av_ccnorm_change[win] = np.nanmean(used_ccnorms_change[idxs])
                                depth_se_ccnorm_change[win] = np.nanstd(used_ccnorms_change[idxs], ddof=1)/np.sqrt(len(idxs))
                            else:
                                depth_av_ccnorm_change[win] = np.nan
                                depth_se_ccnorm_change[win] = np.nan
                                
                    else:
                        for win in range(len(depth_av_ccnorm)):
                            '''idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                                np.where ((np.array(neuron_depths) < depth_av_win_right[win]))
                            idxs = idxs[0]'''
                            
                            idxs_1 = np.where((np.array(neuron_depths) >= depth_av_win_left[win]))[0]
                            idxs_2 = np.where((np.array(neuron_depths) < depth_av_win_right[win]))[0]
                            idxs = np.intersect1d(idxs_1, idxs_2)
                            
                            depth_av_ccnorm_change[win] = np.nanmean(used_ccnorms_change[idxs])
                            depth_se_ccnorm_change[win] = np.nanstd(used_ccnorms_change[idxs], ddof=1)/np.sqrt(len(idxs))
                    
                    f3, ax3 = plt.subplots()
                    ax3.plot(depth_av_win_centre, depth_av_ccnorm_change, linewidth=3)
                    ax3.fill_between(depth_av_win_centre, depth_av_ccnorm_change-depth_se_ccnorm_change, 
                                     depth_av_ccnorm_change+depth_se_ccnorm_change, alpha=0.2)
                    #ax3.set_xlim([-700, 600]) # uncomment for plots with NS2 depth range
                    ax3.set_xlabel('depth (micron from layer 3/4)')
                    ax3.set_ylabel('CCnorm')
                    #ax3.set_ylim([-0.105, 0.035])
                    ax3.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID)
                    #ax3.xaxis.set_tick_params(labelcolor='none')
                    #ax3.yaxis.set_tick_params(labelcolor='none')
                    ax3.spines[['right', 'top']].set_visible(False)
                    
                    #f3.savefig('/home/lorenzom/Downloads/Temp/Depth_sw/CCnorm/Av/' + dataset_ID + '_' + i_model_ID + '_' + \
                               #j_model_ID + '_depth_win' + str(win_size) + '_' + exclude_few + '.png')
                               
                    depth_win_axis[i_model_ID + ' vs ' + j_model_ID] = depth_av_win_centre
                    depth_CCnorm_diff[i_model_ID + ' vs ' + j_model_ID] = depth_av_ccnorm_change
                    depth_CCnorm_diff_se[i_model_ID + ' vs ' + j_model_ID] = depth_se_ccnorm_change
                    
                    # SW
                    if plot_sw:
                        sw_av_ccnorm_change = np.zeros((len(sw_av_win_centre)))
                        sw_se_ccnorm_change = np.zeros((len(sw_av_win_centre)))
                        
                        if exclude_few == 'exclude':
                            for win in range(len(sw_av_ccnorm)):
                                idxs_1 = np.where((np.array(neuron_sws) >= sw_av_win_left[win]))[0]
                                idxs_2 = np.where((np.array(neuron_sws) < sw_av_win_right[win]))[0]
                                idxs = np.intersect1d(idxs_1, idxs_2)
                                
                                if len(idxs) >= 3:
                                    sw_av_ccnorm_change[win] = np.nanmean(used_ccnorms_change[idxs])
                                    sw_se_ccnorm_change[win] = np.nanstd(used_ccnorms_change[idxs], ddof=1)/np.sqrt(len(idxs))
                                else:
                                    sw_av_ccnorm_change[win] = np.nan
                                    sw_se_ccnorm_change[win] = np.nan
                                    
                        else:
                            for win in range(len(sw_av_ccnorm)):
                                idxs_1 = np.where((np.array(neuron_sws) >= sw_av_win_left[win]))[0]
                                idxs_2 = np.where((np.array(neuron_sws) < sw_av_win_right[win]))[0]
                                idxs = np.intersect1d(idxs_1, idxs_2)
                                
                                sw_av_ccnorm_change[win] = np.nanmean(used_ccnorms_change[idxs])
                                sw_se_ccnorm_change[win] = np.nanstd(used_ccnorms_change[idxs], ddof=1)/np.sqrt(len(idxs))
                        
                        f4, ax4 = plt.subplots()
                        ax4.plot(sw_av_win_centre, sw_av_ccnorm_change, linewidth=3)
                        ax4.fill_between(sw_av_win_centre, sw_av_ccnorm_change-sw_se_ccnorm_change, 
                                         sw_av_ccnorm_change+sw_se_ccnorm_change, alpha=0.2)
                        #ax4.set_xlim([-700, 600]) # uncomment for plots with NS2 depth range
                        ax4.set_xlabel('sw (ms)')
                        ax4.set_ylabel('CCnorm')
                        #ax4.set_ylim([-0.105, 0.035])
                        ax4.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID)
                        #ax4.xaxis.set_tick_params(labelcolor='none')
                        #ax4.yaxis.set_tick_params(labelcolor='none')
                        ax4.spines[['right', 'top']].set_visible(False)
                                   
                        sw_win_axis[i_model_ID + ' vs ' + j_model_ID] = sw_av_win_centre
                        sw_CCnorm_diff[i_model_ID + ' vs ' + j_model_ID] = sw_av_ccnorm_change
                        sw_CCnorm_diff_se[i_model_ID + ' vs ' + j_model_ID] = sw_se_ccnorm_change
   
    return neuron_depths, depth_win_axis, depth_CCnorm_diff, depth_CCnorm_diff_se, neuron_sws, sw_win_axis, sw_CCnorm_diff, \
        sw_CCnorm_diff_se, CCnorms, neuron_layers
                        
#############################################################################################################################

def depth_sw_sess_analysis(ccnorms_change_measure, models_results_dict, models_built_data, dataset_ID, models):
    
    plot_mode = 'full'
    exclude_few = ''
    
    if dataset_ID == 'NS2_include_single':
        depth_sw_file = 'Data/NS2_raw/' + dataset_ID + '_depth_sw.pckl'
    else:
        depth_sw_file = 'Data/' + dataset_ID + '/' + dataset_ID + '_depth_sw.pckl'
    
    f = open(depth_sw_file, 'rb')
    depth_sw = pickle.load(f)
    f.close()
    
    neuron_list = models_built_data[models[0]][-1]['neurons_ID_sess']
    noise_ratios = models_built_data[models[0]][-1]['noise_ratios']

    neuron_depths = []
    neuron_layers = []
    neuron_sws = []
    
    neuron_idxs = [] # holds indexes of neurons for which we have depth and sw info
    neuron_sess = [] # holds session indexes of neurons for which we have depth and sw info
    
    for n_idx in range(len(neuron_list)):
        neuron_ID = neuron_list[n_idx][0]
        
        if neuron_ID in depth_sw[0]:
            neuron_depths.append(depth_sw[0][neuron_ID])
            neuron_layers.append(depth_sw[1][neuron_ID])
            neuron_sws.append(depth_sw[2][neuron_ID])
            
            neuron_idxs.append(n_idx)
            neuron_sess.append(neuron_list[n_idx][1])
            
    sessions = np.unique(neuron_sess)
    
    win_overlap = 50 # depth averaging window overlap, in microns
    win_size = 200 # depth averaging window size, in microns
    
    if (np.max(neuron_depths) - np.min(neuron_depths)) % win_overlap == 0:
        depth_av_win_left = np.arange(np.min(neuron_depths) - win_overlap/2, np.max(neuron_depths) + win_overlap/2, 
                                      win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
    else:
        min_limit = roundup(np.min(neuron_depths))
        max_limit = roundup(np.max(neuron_depths))
        
        depth_av_win_left = np.arange(min_limit - win_overlap/2, max_limit + win_overlap/2, win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
            
    # define layer-based colourmap - black is 13, red is 4, blue is 56
    layers_cmap = mcolors.ListedColormap(['black', 'red', 'blue'])
    layer_colours = []
    for neuron in neuron_layers:
        if neuron == '13':
            layer_colours.append(0)
        elif neuron == '44':
            layer_colours.append(1)
        elif neuron == '56':
            layer_colours.append(2)
    
    for i_model_ID in models:
        model_ccnorms = models_results_dict[i_model_ID][2]

        used_ccnorms = model_ccnorms[neuron_idxs]
        used_nr = np.array(noise_ratios)[neuron_idxs]
        
        for sess_ID in range(len(sessions)):
            sess = sessions[sess_ID]
            
            sess_depths = []
            sess_layer_colours = []
            sess_used_ccnorms = []
            sess_used_nr = []
            
            for nID in range(len(neuron_idxs)):
                if neuron_sess[nID] == sess:
                    sess_depths.append(neuron_depths[nID])
                    sess_layer_colours.append(layer_colours[nID])
                    sess_used_ccnorms.append(used_ccnorms[nID])
                    sess_used_nr.append(used_nr[nID])
            
            sess_used_ccnorms = np.array(sess_used_ccnorms)
            ########## Plots for each model
            
            # Plot of depth graph only
            if plot_mode == 'full':
                if len(sess_used_ccnorms) > 1:
                    X2 = sm.add_constant(sess_depths) # add bias to independent variable
                    xseq = np.linspace(np.min(sess_depths)-0.05, np.max(sess_depths)+0.05, num=100)
                    est = sm.OLS(sess_used_ccnorms, X2) # Ordinary Least Squares fit
                    est2 = est.fit()
                    a = est2.params[0] # bias
                    b = est2.params[1] # slope
                    
                    f1, ax1 = plt.subplots()
                    scatter = ax1.scatter(sess_depths, sess_used_ccnorms, c=sess_layer_colours, cmap=layers_cmap)
                    ax1.plot(xseq, a+b*xseq, color='r')
                    ax1.set_ylim([0, 1])
                    ax1.set_xlabel('depth (micron from layer 3/4)')
                    ax1.set_ylabel('CCnorm')
                    handles, labels = scatter.legend_elements()
                    ax1.legend(handles = handles, labels = ['1/3', '4', '5/6'])
                    ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]) + str(sess))
                    ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            
            elif plot_mode == 'av':
                depth_av_ccnorm = np.zeros((len(depth_av_win_centre)))
                depth_se_ccnorm = np.zeros((len(depth_av_win_centre)))
                
                if exclude_few == 'exclude':
                    for win in range(len(depth_av_ccnorm)):
                        idxs = np.where((np.array(sess_depths) >= depth_av_win_left[win])) and \
                            np.where((np.array(sess_depths) < depth_av_win_right[win]))
                        idxs = idxs[0]
                        if len(idxs) >= 3:
                            depth_av_ccnorm[win] = np.nanmean(sess_used_ccnorms[idxs])
                            depth_se_ccnorm[win] = np.nanstd(sess_used_ccnorms[idxs], ddof=1)/np.sqrt(len(idxs))
                        else:
                            depth_av_ccnorm[win] = np.nan
                            depth_se_ccnorm[win] = np.nan
    
                else:
                    for win in range(len(depth_av_ccnorm)):
                        idxs = np.where((np.array(sess_depths) >= depth_av_win_left[win])) and \
                            np.where((np.array(sess_depths) < depth_av_win_right[win]))
                        idxs = idxs[0]
                        
                        if len(idxs) > 0:
                            depth_av_ccnorm[win] = np.nanmean(sess_used_ccnorms[idxs])
                            depth_se_ccnorm[win] = np.nanstd(sess_used_ccnorms[idxs], ddof=1)/np.sqrt(len(idxs))
                        else:
                            depth_av_ccnorm[win] = np.nan
                            depth_se_ccnorm[win] = np.nan
                
                f1, ax1 = plt.subplots()
                ax1.plot(depth_av_win_centre, depth_av_ccnorm)
                ax1.fill_between(depth_av_win_centre, depth_av_ccnorm-depth_se_ccnorm, depth_av_ccnorm+depth_se_ccnorm, 
                                 alpha=0.2)
                ax1.set_xlabel('depth (micron from layer 3/4)')
                ax1.set_ylabel('CCnorm')
                ax1.set_title(dataset_ID + ' - ' + i_model_ID)
                ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            
        for j_model_ID in models: # loop over models
            if j_model_ID != i_model_ID:
                # difference in CCnorm between two models, for each neuron
                ccnorms_diff = models_results_dict[i_model_ID][2] - models_results_dict[j_model_ID][2]
                
                # percentage change in CCnorm between two models, for each neuron
                ccnorms_percent_change = (models_results_dict[i_model_ID][2] - \
                                          models_results_dict[j_model_ID][2])/models_results_dict[j_model_ID][2]*100
                
                # choose CCnorm change measure 
                if ccnorms_change_measure == 'CCnorm_diff':
                    ccnorms_change = ccnorms_diff
                elif ccnorms_change_measure == 'CCnorm_percent_change':
                    ccnorms_change = ccnorms_percent_change
                
                used_ccnorms_change = ccnorms_change[neuron_idxs]
                
                for sess_ID in range(len(sessions)):
                    sess = sessions[sess_ID]
                    
                    sess_depths = []
                    sess_layer_colours = []
                    sess_used_ccnorms_change = []
                    
                    for nID in range(len(neuron_idxs)):
                        if neuron_sess[nID] == sess:
                            sess_depths.append(neuron_depths[nID])
                            sess_layer_colours.append(layer_colours[nID])
                            sess_used_ccnorms_change.append(used_ccnorms_change[nID])
                        
                    sess_used_ccnorms_change = np.array(sess_used_ccnorms_change)
                    ########## Plots for each model comparison
                    
                    # Plot of depth only
                    if plot_mode == 'full':
                        if len(sess_used_ccnorms_change) > 1:
                            X2 = sm.add_constant(sess_depths) # add bias to independent variable
                            xseq = np.linspace(np.min(sess_depths)-0.05, np.max(sess_depths)+0.05, num=100)
                            est = sm.OLS(sess_used_ccnorms_change, X2) # Ordinary Least Squares fit
                            est2 = est.fit()
                            a = est2.params[0] # bias
                            b = est2.params[1] # slope
                            
                            f1, ax1 = plt.subplots()
                            scatter = ax1.scatter(sess_depths, sess_used_ccnorms_change, c=sess_layer_colours, cmap=layers_cmap)
                            ax1.plot(xseq, a+b*xseq, color='r')
                            ax1.set_xlabel('depth (micron from layer 3/4)')
                            ax1.set_ylabel('CCnorm diff')
                            handles, labels = scatter.legend_elements()
                            ax1.legend(handles = handles, labels = ['1/3', '4', '5/6'])
                            ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID + ' - p = ' + str(est2.pvalues[1]))
                            ax1.spines[['right', 'top']].set_visible(False)
                    
                    elif plot_mode == 'av':
                        depth_av_ccnorm_change = np.zeros((len(depth_av_win_centre)))
                        depth_se_ccnorm_change = np.zeros((len(depth_av_win_centre)))
                        
                        if exclude_few == 'exclude':
                            for win in range(len(depth_av_ccnorm)):
                                idxs = np.where((np.array(sess_depths) >= depth_av_win_left[win])) and \
                                    np.where((np.array(sess_depths) < depth_av_win_right[win]))
                                idxs = idxs[0]
                                if len(idxs) >= 3:
                                    depth_av_ccnorm_change[win] = np.nanmean(sess_used_ccnorms_change[idxs])
                                    depth_se_ccnorm_change[win] = np.nanstd(sess_used_ccnorms_change[idxs], ddof=1)/np.sqrt(len(idxs))
                                else:
                                    depth_av_ccnorm_change[win] = np.nan
                                    depth_se_ccnorm_change[win] = np.nan
                                    
                        else:
                            for win in range(len(depth_av_ccnorm)):
                                idxs = np.where((np.array(sess_depths) >= depth_av_win_left[win])) and \
                                    np.where ((np.array(sess_depths) < depth_av_win_right[win]))
                                idxs = idxs[0]
                                depth_av_ccnorm_change[win] = np.nanmean(sess_used_ccnorms_change[idxs])
                                depth_se_ccnorm_change[win] = np.nanstd(sess_used_ccnorms_change[idxs], ddof=1)/np.sqrt(len(idxs))
                        
                        f1, ax1 = plt.subplots()
                        ax1.plot(depth_av_win_centre, depth_av_ccnorm_change)
                        ax1.fill_between(depth_av_win_centre, depth_av_ccnorm_change-depth_se_ccnorm_change, 
                                         depth_av_ccnorm_change+depth_se_ccnorm_change, alpha=0.2)
                        ax1.set_xlabel('depth (micron from layer 3/4)')
                        ax1.set_ylabel('CCnorm')
                        ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' vs ' + j_model_ID)
                        ax1.spines[['right', 'top']].set_visible(False)
                
#############################################################################################################################

def depth_cell_analysis(ccnorms_change_measure, HU_mode, models_corr_HUs, models_eff_HUs, models_fits_dict, 
                        models_built_data, best_lambdas, dataset_ID, models):
    
    plot_mode = 'av'
    exclude_few = ''
    
    # define p-value-based colourmap - black indicates non-significance, red indicates significance
    pvalues_cmap = mcolors.ListedColormap(['black', 'red'])
    
    if dataset_ID == 'NS2_include_single':
        depth_sw_file = 'Data/NS2_raw/' + dataset_ID + '_depth_sw.pckl'
    else:
        depth_sw_file = 'Data/' + dataset_ID + '/' + dataset_ID + '_depth_sw.pckl'
    
    f = open(depth_sw_file, 'rb')
    depth_sw = pickle.load(f)
    f.close()
    
    neuron_list = models_built_data[models[0]][-1]['neurons_ID_sess']
    noise_ratios = models_built_data[models[0]][-1]['noise_ratios']
    
    neuron_depths = []
    neuron_layers = []
    neuron_sws = []
    
    neuron_idxs = [] # holds indexes of neurons for which we have depth, layer, and sw info
    
    for n_idx in range(len(neuron_list)):
        neuron_ID = neuron_list[n_idx][0]
        
        if neuron_ID in depth_sw[0]:
            neuron_depths.append(depth_sw[0][neuron_ID])
            neuron_layers.append(depth_sw[1][neuron_ID])
            neuron_sws.append(depth_sw[2][neuron_ID])
            
            neuron_idxs.append(n_idx)
          
    win_overlap = 50 # depth averaging window overlap, in microns
    win_size = 200 # depth averaging window size, in microns
    
    if (np.max(neuron_depths) - np.min(neuron_depths)) % win_overlap == 0:
        depth_av_win_left = np.arange(np.min(neuron_depths) - win_overlap/2, np.max(neuron_depths) + win_overlap/2, 
                                      win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
    else:
        min_limit = roundup(np.min(neuron_depths))
        max_limit = roundup(np.max(neuron_depths))
        
        depth_av_win_left = np.arange(min_limit - win_overlap/2, max_limit + win_overlap/2, win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
          
    # define layer-based colourmap - black is 13, red is 4, blue is 56
    layers_cmap = mcolors.ListedColormap(['black', 'red', 'blue'])
    layer_colours = []
    for neuron in neuron_layers:
        if neuron == '13':
            layer_colours.append(0)
        elif neuron == '44':
            layer_colours.append(1)
        elif neuron == '56':
            layer_colours.append(2)
    
    for i_model_ID in models:
        i_model_params = models_fits_dict[i_model_ID][1]['all_model_params'] # hyperparameters of i_model
        i_model_type = models_fits_dict[i_model_ID][1]['model_type']
        if i_model_type == 'not_conv':
            i_hidden_size = i_model_params['hidden_size'] # network hidden size
        elif i_model_type == 'conv':
            i_hidden_size = np.size(models_fits_dict[i_model_ID][0][0][0][0]['w_l1'], 0)
        i_model_best_lambdas = best_lambdas[dataset_ID][i_model_ID] # i_model best lambda for each neuron
        
        # NumPy array of correlations between each neuron's recorded PSTH and each HU in i_model, for all folds and lambdas 
        # (dimensions -> (neurons, folds, lambdas, HUs))
        i_model_corr_HUs = models_corr_HUs[i_model_ID]['actual']
        
        used_corr_HUs = i_model_corr_HUs[neuron_idxs]
        
        if HU_mode == 'Eff':
            i_model_eff_HUs = models_eff_HUs[i_model_ID][0]
            i_HU_corr = []
        else:
            # array to hold correlations for specific fold and best lambda
            i_HU_corr = np.zeros((len(neuron_idxs), i_hidden_size))
        
        for nID in range(len(neuron_idxs)): # loop over neurons
            # correlations for specific neuron, fold (0, since NS2_include_single and NS3 only have one fold), and the
            # neuron's best lambda
            if HU_mode == 'Eff':
                eff_HUs = i_model_eff_HUs[i_model_best_lambdas[neuron_idxs[nID]]][neuron_idxs[nID]]
                i_HU_corr.append(used_corr_HUs[nID, 0, i_model_best_lambdas[neuron_idxs[nID]]][eff_HUs])
            elif HU_mode == 'All':
                i_HU_corr[nID, :] = used_corr_HUs[nID, 0, i_model_best_lambdas[neuron_idxs[nID]]]

        if HU_mode == 'Eff':
            i_HU_corr_kurtosis = np.zeros((len(neuron_idxs)))
            i_HU_corr_abs_max = np.zeros((len(neuron_idxs)))
            i_HU_corr_av = np.zeros((len(neuron_idxs)))
            
            for nID in range(len(neuron_idxs)):
                i_HU_corr_kurtosis[nID] = scipy.stats.kurtosis(i_HU_corr[nID], axis=0)
                i_HU_corr_abs_max[nID] = np.max(np.abs(i_HU_corr[nID]))
                i_HU_corr_av[nID] = np.nanmean(np.abs(i_HU_corr[nID]))
        
        elif HU_mode == 'All':
            # kurtosis of correlations for each neuron - measure of sparsity of correlations
            i_HU_corr_kurtosis = scipy.stats.kurtosis(i_HU_corr, axis=1)
            # significance of kurtosis for each neuron - transformed into a Boolean significance array
            kurtosis_sig = scipy.stats.kurtosistest(i_HU_corr, axis=1, alternative='greater')[1] < 0.01
            
            # maximum absolute correlation, over all HUs, for each neuron
            i_HU_corr_abs_max = np.max(np.abs(i_HU_corr), axis=1)
            
            # average absolute correlation, over all HUs, for each neuron
            i_HU_corr_av = np.nanmean(np.abs(i_HU_corr), 1)
        
        X2 = sm.add_constant(neuron_depths) # add bias to independent variable
        xseq = np.linspace(np.min(neuron_depths)-0.05, np.max(neuron_depths)+0.05, num=100)
        
        ############################### Kurtosis plots
        
        est = sm.OLS(i_HU_corr_kurtosis, X2) # Ordinary Least Squares fit
        est2 = est.fit()
        a = est2.params[0] # bias
        b = est2.params[1] # slope
        
        if plot_mode == 'full':
            f1, ax1 = plt.subplots()
            scatter = ax1.scatter(neuron_depths, i_HU_corr_kurtosis, c=layer_colours, cmap=layers_cmap) # layers colourmap
            #ax1.scatter(neuron_depths, i_HU_corr_kurtosis, c=kurtosis_sig, cmap=pvalues_cmap) # p values colourmap
            ax1.plot(xseq, a+b*xseq, color='r')
            ax1.set_xlabel('depth (micron from layer 3/4)')
            ax1.set_ylabel('cell state correlation kurtosis')
            handles, labels = scatter.legend_elements()
            ax1.legend(handles = handles, labels = ['1/3', '4', '5/6'])
            ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
            ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            
        elif plot_mode == 'av':
            depth_av_k = np.zeros((len(depth_av_win_centre)))
            depth_se_k = np.zeros((len(depth_av_win_centre)))
            
            if exclude_few == 'exclude':
                for win in range(len(depth_av_k)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    
                    if len(idxs) >= 3:
                        depth_av_k[win] = np.nanmean(i_HU_corr_kurtosis[idxs])
                        depth_se_k[win] = np.nanstd(i_HU_corr_kurtosis[idxs], ddof=1)/np.sqrt(len(idxs))
                    else:
                        depth_av_k[win] = np.nan
                        depth_se_k[win] = np.nan
            else:
                for win in range(len(depth_av_k)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    depth_av_k[win] = np.nanmean(i_HU_corr_kurtosis[idxs])
                    depth_se_k[win] = np.nanstd(i_HU_corr_kurtosis[idxs], ddof=1)/np.sqrt(len(idxs))
            
            f1, ax1 = plt.subplots()
            ax1.plot(depth_av_win_centre, depth_av_k)
            ax1.fill_between(depth_av_win_centre, depth_av_k-depth_se_k, depth_av_k+depth_se_k, alpha=0.2)
            ax1.set_xlabel('depth (micron from layer 3/4)')
            ax1.set_ylabel('cell state correlation kurtosis')
            ax1.set_title(dataset_ID + ' - ' + i_model_ID)
            ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
        
        ############################### Maximum absolute correlation plots
        
        est = sm.OLS(i_HU_corr_abs_max, X2) # Ordinary Least Squares fit
        est2 = est.fit()
        a = est2.params[0] # bias
        b = est2.params[1] # slope
        
        if plot_mode == 'full':
            f2, ax2 = plt.subplots()
            scatter = ax2.scatter(neuron_depths, i_HU_corr_abs_max, c=layer_colours, cmap=layers_cmap) # layers colourmap
            #ax2.scatter(neuron_depths, i_HU_corr_abs_max, c=kurtosis_sig, cmap=pvalues_cmap) # p values colourmap
            ax2.plot(xseq, a+b*xseq, color='r')
            ax2.set_xlabel('depth (micron from layer 3/4)')
            ax2.set_ylabel('cell state max abs correlation')
            handles, labels = scatter.legend_elements()
            ax2.legend(handles = handles, labels = ['1/3', '4', '5/6'])
            ax2.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
            ax2.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
        
        elif plot_mode == 'av':
            depth_av_abs_max = np.zeros((len(depth_av_win_centre)))
            depth_se_abs_max = np.zeros((len(depth_av_win_centre)))
            
            if exclude_few == 'exclude':
                for win in range(len(depth_av_abs_max)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    
                    if len(idxs) >= 3:
                        depth_av_abs_max[win] = np.nanmean(i_HU_corr_abs_max[idxs])
                        depth_se_abs_max[win] = np.nanstd(i_HU_corr_abs_max[idxs], ddof=1)/np.sqrt(len(idxs))
                    else:
                        depth_av_abs_max[win] = np.nan
                        depth_se_abs_max[win] = np.nan
            else:
                for win in range(len(depth_av_abs_max)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    depth_av_abs_max[win] = np.nanmean(i_HU_corr_abs_max[idxs])
                    depth_se_abs_max[win] = np.nanstd(i_HU_corr_abs_max[idxs], ddof=1)/np.sqrt(len(idxs))
            
            f2, ax2 = plt.subplots()
            ax2.plot(depth_av_win_centre, depth_av_abs_max)
            ax2.fill_between(depth_av_win_centre, depth_av_abs_max-depth_se_abs_max, depth_av_abs_max+depth_se_abs_max, 
                             alpha=0.2)
            ax2.set_xlabel('depth (micron from layer 3/4)')
            ax2.set_ylabel('cell state max abs correlation')
            ax2.set_title(dataset_ID + ' - ' + i_model_ID)
            ax2.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
        
        ############################### Average absolute correlation plots
        
        est = sm.OLS(i_HU_corr_av, X2) # Ordinary Least Squares fit
        est2 = est.fit()
        a = est2.params[0] # bias
        b = est2.params[1] # slope
        
        if plot_mode == 'full':
            f3, ax3 = plt.subplots()
            scatter = ax3.scatter(neuron_depths, i_HU_corr_av, c=layer_colours, cmap=layers_cmap) # layers cmap
            #ax3.scatter(neuron_depths, i_HU_corr_av, c=kurtosis_sig, cmap=pvalues_cmap) # p values cmap
            ax3.plot(xseq, a+b*xseq, color='r')
            ax3.set_xlabel('depth (micron from layer 3/4)')
            ax3.set_ylabel('cell state av correlation')
            handles, labels = scatter.legend_elements()
            ax3.legend(handles = handles, labels = ['1/3', '4', '5/6'])
            ax3.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
            ax3.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
        
        elif plot_mode == 'av':
            depth_av_av = np.zeros((len(depth_av_win_centre)))
            depth_se_av = np.zeros((len(depth_av_win_centre)))
            
            if exclude_few == 'exclude':
                for win in range(len(depth_av_av)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    
                    if len(idxs) >= 3:
                        depth_av_av[win] = np.nanmean(i_HU_corr_av[idxs])
                        depth_se_av[win] = np.nanstd(i_HU_corr_av[idxs], ddof=1)/np.sqrt(len(idxs))
                    else:
                        depth_av_av[win] = np.nan
                        depth_se_av[win] = np.nan
            else:
                for win in range(len(depth_av_av)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    depth_av_av[win] = np.nanmean(i_HU_corr_av[idxs])
                    depth_se_av[win] = np.nanstd(i_HU_corr_av[idxs], ddof=1)/np.sqrt(len(idxs))
            
            f3, ax3 = plt.subplots()
            ax3.plot(depth_av_win_centre, depth_av_av)
            ax3.fill_between(depth_av_win_centre, depth_av_av-depth_se_av, depth_av_av+depth_se_av, alpha=0.2)
            ax3.set_xlabel('depth (micron from layer 3/4)')
            ax3.set_ylabel('cell state av correlation')
            ax3.set_title(dataset_ID + ' - ' + i_model_ID)
            ax3.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
        
#############################################################################################################################

def depth_t_gate_analysis(ccnorms_change_measure, models_corr_gates, models_fits_dict, models_built_data, best_lambdas, 
                          dataset_ID, models):
    
    plot_mode = 'av'
    exclude_few = ''
    
    # define p-value-based colourmap - black indicates non-significance, red indicates significance
    pvalues_cmap = mcolors.ListedColormap(['black', 'red'])
    
    if dataset_ID == 'NS2_include_single':
        depth_sw_file = 'Data/NS2_raw/' + dataset_ID + '_depth_sw.pckl'
    else:
        depth_sw_file = 'Data/' + dataset_ID + '/' + dataset_ID + '_depth_sw.pckl'
    
    f = open(depth_sw_file, 'rb')
    depth_sw = pickle.load(f)
    f.close()
    
    neuron_list = models_built_data[models[0]][-1]['neurons_ID_sess']
    noise_ratios = models_built_data[models[0]][-1]['noise_ratios']
    
    neuron_depths = []
    neuron_layers = []
    neuron_sws = []
    
    neuron_idxs = [] # holds indexes of neurons for which we have depth, layer, and sw info
    
    for n_idx in range(len(neuron_list)):
        neuron_ID = neuron_list[n_idx][0]
        
        if neuron_ID in depth_sw[0]:
            neuron_depths.append(depth_sw[0][neuron_ID])
            neuron_layers.append(depth_sw[1][neuron_ID])
            neuron_sws.append(depth_sw[2][neuron_ID])
            
            neuron_idxs.append(n_idx)
    
    win_overlap = 50 # depth averaging window overlap, in microns
    win_size = 200 # depth averaging window size, in microns
    
    if (np.max(neuron_depths) - np.min(neuron_depths)) % win_overlap == 0:
        depth_av_win_left = np.arange(np.min(neuron_depths) - win_overlap/2, np.max(neuron_depths) + win_overlap/2, 
                                      win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
    else:
        min_limit = roundup(np.min(neuron_depths))
        max_limit = roundup(np.max(neuron_depths))
        
        depth_av_win_left = np.arange(min_limit - win_overlap/2, max_limit + win_overlap/2, win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
    
    # define layer-based colourmap - black is 13, red is 4, blue is 56
    layers_cmap = mcolors.ListedColormap(['black', 'red', 'blue'])
    layer_colours = []
    for neuron in neuron_layers:
        if neuron == '13':
            layer_colours.append(0)
        elif neuron == '44':
            layer_colours.append(1)
        elif neuron == '56':
            layer_colours.append(2)
    
    for i_model_ID in models:
        i_model_params = models_fits_dict[i_model_ID][1]['all_model_params'] # hyperparameters of i_model
        i_input_size = i_model_params['input_size'] # network hidden size
        i_model_best_lambdas = best_lambdas[dataset_ID][i_model_ID] # i_model best lambda for each neuron
        
        # NumPy array of correlations between each neuron's recorded PSTH and each HU in i_model, for all folds and lambdas 
        # (dimensions -> (neurons, folds, lambdas, HUs))
        i_model_corr_gates = models_corr_gates[i_model_ID][0]
        
        i_t_corr = np.zeros((len(neuron_idxs), i_input_size)) # array to hold correlations for specific fold and best lambda
        
        for nID in range(len(neuron_idxs)): # loop over neurons
            # correlations for specific neuron, fold (0, since NS2_include_single and NS3 only have one fold), and the
            # neuron's best lambda
            i_t_corr[nID, :] = i_model_corr_gates[i_model_best_lambdas[neuron_idxs[nID]]]['t'][neuron_idxs[nID]]

        # kurtosis of correlations for each neuron - measure of sparsity of correlations
        i_t_corr_kurtosis = scipy.stats.kurtosis(i_t_corr, axis=1)
        # significance of kurtosis for each neuron - transformed into a Boolean significance array
        kurtosis_sig = scipy.stats.kurtosistest(i_t_corr, axis=1, alternative='greater')[1] < 0.01
        
        X2 = sm.add_constant(neuron_depths) # add bias to independent variable
        xseq = np.linspace(np.min(neuron_depths)-0.05, np.max(neuron_depths)+0.05, num=100)
        
        ############################### Kurtosis plots
        
        est = sm.OLS(i_t_corr_kurtosis, X2) # Ordinary Least Squares fit
        est2 = est.fit()
        a = est2.params[0] # bias
        b = est2.params[1] # slope
        
        if plot_mode == 'full':
            f1, ax1 = plt.subplots()
            #ax1.scatter(neuron_depths, i_t_corr_kurtosis, c=kurtosis_sig, cmap=pvalues_cmap) # p values colourmap
            scatter = ax1.scatter(neuron_depths, i_t_corr_kurtosis, c=layer_colours, cmap=layers_cmap) # layers colourmap
            ax1.plot(xseq, a+b*xseq, color='r')
            ax1.set_xlabel('depth (micron from layer 3/4)')
            ax1.set_ylabel('t gate correlation kurtosis')
            handles, labels = scatter.legend_elements()
            ax1.legend(handles = handles, labels = ['1/3', '4', '5/6'])
            ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
            ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            
        else:
            depth_av_k = np.zeros((len(depth_av_win_centre)))
            depth_se_k = np.zeros((len(depth_av_win_centre)))
            
            if exclude_few == 'exclude':
                for win in range(len(depth_av_k)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    
                    if len(idxs) >= 3:
                        depth_av_k[win] = np.nanmean(i_t_corr_kurtosis[idxs])
                        depth_se_k[win] = np.nanstd(i_t_corr_kurtosis[idxs], ddof=1)/np.sqrt(len(idxs))
                    else:
                        depth_av_k[win] = np.nan
                        depth_se_k[win] = np.nan
            else:
                for win in range(len(depth_av_k)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    depth_av_k[win] = np.nanmean(i_t_corr_kurtosis[idxs])
                    depth_se_k[win] = np.nanstd(i_t_corr_kurtosis[idxs], ddof=1)/np.sqrt(len(idxs))
            
            f1, ax1 = plt.subplots()
            ax1.plot(depth_av_win_centre, depth_av_k)
            ax1.fill_between(depth_av_win_centre, depth_av_k-depth_se_k, depth_av_k+depth_se_k, alpha=0.2)
            ax1.set_xlabel('depth (micron from layer 3/4)')
            ax1.set_ylabel('t gate correlation kurtosis')
            ax1.set_title(dataset_ID + ' - ' + i_model_ID)
            ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
        
        ############################### Maximum absolute correlation plots
        
        # maximum absolute correlation, over all HUs, for each neuron
        i_t_corr_abs_max = np.max(np.abs(i_t_corr), axis=1)
        
        est = sm.OLS(i_t_corr_abs_max, X2) # Ordinary Least Squares fit
        est2 = est.fit()
        a = est2.params[0] # bias
        b = est2.params[1] # slope
        
        if plot_mode == 'full':
            f2, ax2 = plt.subplots()
            #ax2.scatter(neuron_depths, i_t_corr_abs_max, c=kurtosis_sig, cmap=pvalues_cmap) # p values colourmap
            scatter = ax2.scatter(neuron_depths, i_t_corr_abs_max, c=layer_colours, cmap=layers_cmap) # layers colourmap
            ax2.plot(xseq, a+b*xseq, color='r')
            ax2.set_xlabel('depth (micron from layer 3/4)')
            ax2.set_ylabel('t gate max abs correlation')
            handles, labels = scatter.legend_elements()
            ax2.legend(handles = handles, labels = ['1/3', '4', '5/6'])
            ax2.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
            ax2.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
        
        else:
            depth_av_abs_max = np.zeros((len(depth_av_win_centre)))
            depth_se_abs_max = np.zeros((len(depth_av_win_centre)))
            
            if exclude_few == 'exclude':
                for win in range(len(depth_av_abs_max)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    
                    if len(idxs) >= 3:
                        depth_av_abs_max[win] = np.nanmean(i_t_corr_abs_max[idxs])
                        depth_se_abs_max[win] = np.nanstd(i_t_corr_abs_max[idxs], ddof=1)/np.sqrt(len(idxs))
                    else:
                        depth_av_abs_max[win] = np.nan
                        depth_se_abs_max[win] = np.nan
            else:
                for win in range(len(depth_av_abs_max)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    depth_av_abs_max[win] = np.nanmean(i_t_corr_abs_max[idxs])
                    depth_se_abs_max[win] = np.nanstd(i_t_corr_abs_max[idxs], ddof=1)/np.sqrt(len(idxs))
            
            f2, ax2 = plt.subplots()
            ax2.plot(depth_av_win_centre, depth_av_abs_max)
            ax2.fill_between(depth_av_win_centre, depth_av_abs_max-depth_se_abs_max, depth_av_abs_max+depth_se_abs_max, 
                             alpha=0.2)
            ax2.set_xlabel('depth (micron from layer 3/4)')
            ax2.set_ylabel('t gate max abs correlation')
            ax2.set_title(dataset_ID + ' - ' + i_model_ID)
            ax2.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
        
        ############################### Average absolute correlation plots
        
        # average absolute correlation, over all HUs, for each neuron
        i_t_corr_av = np.nanmean(np.abs(i_t_corr), 1)
        
        est = sm.OLS(i_t_corr_av, X2) # Ordinary Least Squares fit
        est2 = est.fit()
        a = est2.params[0] # bias
        b = est2.params[1] # slope
        
        if plot_mode == 'full':
            f3, ax3 = plt.subplots()
            #ax3.scatter(neuron_depths, i_t_corr_av, c=kurtosis_sig, cmap=pvalues_cmap) # p values colourmap
            scatter = ax3.scatter(neuron_depths, i_t_corr_av, c=layer_colours, cmap=layers_cmap) # layers colourmap
            ax3.plot(xseq, a+b*xseq, color='r')
            ax3.set_xlabel('depth (micron from layer 3/4)')
            ax3.set_ylabel('t gate av correlation')
            handles, labels = scatter.legend_elements()
            ax3.legend(handles = handles, labels = ['1/3', '4', '5/6'])
            ax3.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
            ax3.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            
        else:
            depth_av_av = np.zeros((len(depth_av_win_centre)))
            depth_se_av = np.zeros((len(depth_av_win_centre)))
            
            if exclude_few == 'exclude':
                for win in range(len(depth_av_av)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    
                    if len(idxs) >= 3:
                        depth_av_av[win] = np.nanmean(i_t_corr_av[idxs])
                        depth_se_av[win] = np.nanstd(i_t_corr_av[idxs], ddof=1)/np.sqrt(len(idxs))
                    else:
                        depth_av_av[win] = np.nan
                        depth_se_av[win] = np.nan
            else:
                for win in range(len(depth_av_av)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    depth_av_av[win] = np.nanmean(i_t_corr_av[idxs])
                    depth_se_av[win] = np.nanstd(i_t_corr_av[idxs], ddof=1)/np.sqrt(len(idxs))
            
            f3, ax3 = plt.subplots()
            ax3.plot(depth_av_win_centre, depth_av_av)
            ax3.fill_between(depth_av_win_centre, depth_av_av-depth_se_av, depth_av_av+depth_se_av, alpha=0.2)
            ax3.set_xlabel('depth (micron from layer 3/4)')
            ax3.set_ylabel('t gate av correlation')
            ax3.set_title(dataset_ID + ' - ' + i_model_ID)
            ax3.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
        
#############################################################################################################################

def depth_gates_analysis(ccnorms_change_measure, HU_mode, models_corr_gates, models_eff_HUs, models_fits_dict, 
                         models_built_data, best_lambdas, dataset_ID, models):
    
    plot_mode = 'av'
    exclude_few = ''
    
    # define p-value-based colourmap - black indicates non-significance, red indicates significance
    pvalues_cmap = mcolors.ListedColormap(['black', 'red'])
    
    if dataset_ID == 'NS2_include_single':
        depth_sw_file = 'Data/NS2_raw/' + dataset_ID + '_depth_sw.pckl'
    else:
        depth_sw_file = 'Data/' + dataset_ID + '/' + dataset_ID + '_depth_sw.pckl'
    
    f = open(depth_sw_file, 'rb')
    depth_sw = pickle.load(f)
    f.close()
    
    neuron_list = models_built_data[models[0]][-1]['neurons_ID_sess']
    noise_ratios = models_built_data[models[0]][-1]['noise_ratios']
    
    neuron_depths = []
    neuron_layers = []
    neuron_sws = []
    
    neuron_idxs = [] # holds indexes of neurons for which we have depth, layer, and sw info
    
    for n_idx in range(len(neuron_list)):
        neuron_ID = neuron_list[n_idx][0]
        
        if neuron_ID in depth_sw[0]:
            neuron_depths.append(depth_sw[0][neuron_ID])
            neuron_layers.append(depth_sw[1][neuron_ID])
            neuron_sws.append(depth_sw[2][neuron_ID])
            
            neuron_idxs.append(n_idx)
    
    win_overlap = 50 # depth averaging window overlap, in microns
    win_size = 200 # depth averaging window size, in microns
    
    if (np.max(neuron_depths) - np.min(neuron_depths)) % win_overlap == 0:
        depth_av_win_left = np.arange(np.min(neuron_depths) - win_overlap/2, np.max(neuron_depths) + win_overlap/2, 
                                      win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
    else:
        min_limit = roundup(np.min(neuron_depths))
        max_limit = roundup(np.max(neuron_depths))
        
        depth_av_win_left = np.arange(min_limit - win_overlap/2, max_limit + win_overlap/2, win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
    
    # define layer-based colourmap - black is 13, red is 4, blue is 56
    layers_cmap = mcolors.ListedColormap(['black', 'red', 'blue'])
    layer_colours = []
    for neuron in neuron_layers:
        if neuron == '13':
            layer_colours.append(0)
        elif neuron == '44':
            layer_colours.append(1)
        elif neuron == '56':
            layer_colours.append(2)
    
    for i_model_ID in models:
        i_model_params = models_fits_dict[i_model_ID][1]['all_model_params'] # hyperparameters of i_model
        i_model_type = models_fits_dict[i_model_ID][1]['model_type']
        if i_model_type == 'not_conv':
            i_hidden_size = i_model_params['hidden_size'] # network hidden size
        elif i_model_type == 'conv':
            i_hidden_size = np.size(models_fits_dict[i_model_ID][0][0][0][0]['w_l1'], 0)
        i_model_best_lambdas = best_lambdas[dataset_ID][i_model_ID] # i_model best lambda for each neuron
        
        # NumPy array of correlations between each neuron's recorded PSTH and each HU in i_model, for all folds and lambdas 
        # (dimensions -> (neurons, folds, lambdas, HUs))
        i_model_corr_gates = models_corr_gates[i_model_ID][0]
        
        if plot_mode == 'av':
            f_k_av, axs_k_av = plt.subplots()
            axs_k_av.set_xlabel('depth (micron from layer 3/4)')
            axs_k_av.set_ylabel('gate correlation kurtosis')
            axs_k_av.set_title(dataset_ID + ' - ' + i_model_ID)
            axs_k_av.spines[['right', 'top']].set_visible(False)
            
            f_max_abs_av, axs_max_abs_av = plt.subplots()
            axs_max_abs_av.set_xlabel('depth (micron from layer 3/4)')
            axs_max_abs_av.set_ylabel('gate max abs correlation')
            axs_max_abs_av.set_title(dataset_ID + ' - ' + i_model_ID)
            axs_max_abs_av.spines[['right', 'top']].set_visible(False)
            
            f_av_av, axs_av_av = plt.subplots()
            axs_av_av.set_xlabel('depth (micron from layer 3/4)')
            axs_av_av.set_ylabel('gate av correlation')
            axs_av_av.set_title(dataset_ID + ' - ' + i_model_ID)
            axs_av_av.spines[['right', 'top']].set_visible(False)
        
        for gate_key in i_model_corr_gates[0]:
            if HU_mode == 'Eff':
                i_model_eff_HUs = models_eff_HUs[i_model_ID][0]
                i_gate_corr = []
            else:
                # array to hold correlations for specific fold and best lambda
                i_gate_corr = np.zeros((len(neuron_idxs), i_hidden_size))
            
            for nID in range(len(neuron_idxs)): # loop over neurons
                # correlations for specific neuron, fold (0, since NS2_include_single and NS3 only have one fold), and the
                # neuron's best lambda
                if HU_mode == 'Eff':
                    eff_HUs = i_model_eff_HUs[i_model_best_lambdas[neuron_idxs[nID]]][neuron_idxs[nID]]
                    i_gate_corr.append(
                        i_model_corr_gates[i_model_best_lambdas[neuron_idxs[nID]]][gate_key][neuron_idxs[nID]][eff_HUs])
                elif HU_mode == 'All':
                    i_gate_corr[nID, :] = \
                        i_model_corr_gates[i_model_best_lambdas[neuron_idxs[nID]]][gate_key][neuron_idxs[nID]]
            
            if HU_mode == 'Eff':
                i_gate_corr_kurtosis = np.zeros((len(neuron_idxs)))
                i_gate_corr_abs_max = np.zeros((len(neuron_idxs)))
                i_gate_corr_av = np.zeros((len(neuron_idxs)))
                
                for nID in range(len(neuron_idxs)):
                    i_gate_corr_kurtosis[nID] = scipy.stats.kurtosis(i_gate_corr[nID], axis=0)
                    i_gate_corr_abs_max[nID] = np.max(np.abs(i_gate_corr[nID]))
                    i_gate_corr_av[nID] = np.nanmean(np.abs(i_gate_corr[nID]))
                
            elif HU_mode == 'All':
                # kurtosis of correlations for each neuron - measure of sparsity of correlations
                i_gate_corr_kurtosis = scipy.stats.kurtosis(i_gate_corr, axis=1)
                # significance of kurtosis for each neuron - transformed into a Boolean significance array
                kurtosis_sig = scipy.stats.kurtosistest(i_gate_corr, axis=1, alternative='greater')[1] < 0.01
                
                # maximum absolute correlation, over all HUs, for each neuron
                i_gate_corr_abs_max = np.max(np.abs(i_gate_corr), axis=1)
                
                # average absolute correlation, over all HUs, for each neuron
                i_gate_corr_av = np.nanmean(np.abs(i_gate_corr), 1)
            
            X2 = sm.add_constant(neuron_depths) # add bias to independent variable
            xseq = np.linspace(np.min(neuron_depths)-0.05, np.max(neuron_depths)+0.05, num=100)
            
            ############################### Kurtosis plots
            
            est = sm.OLS(i_gate_corr_kurtosis, X2) # Ordinary Least Squares fit
            est2 = est.fit()
            a = est2.params[0] # bias
            b = est2.params[1] # slope
            
            if plot_mode == 'full':
                f1, ax1 = plt.subplots()
                scatter = ax1.scatter(neuron_depths, i_gate_corr_kurtosis, c=layer_colours, cmap=layers_cmap)
                #ax1.scatter(neuron_depths, i_gate_corr_kurtosis, c=kurtosis_sig, cmap=pvalues_cmap)
                ax1.plot(xseq, a+b*xseq, color='r')
                ax1.set_xlabel('depth (micron from layer 3/4)')
                ax1.set_ylabel(gate_key + ' gate correlation kurtosis')
                handles, labels = scatter.legend_elements()
                ax1.legend(handles = handles, labels = ['1/3', '4', '5/6'])
                ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
                ax1.spines[['right', 'top']].set_visible(False)
            
            elif plot_mode == 'av':
                depth_av_k = np.zeros((len(depth_av_win_centre)))
                depth_se_k = np.zeros((len(depth_av_win_centre)))
                
                if exclude_few == 'exclude':
                    for win in range(len(depth_av_k)):
                        idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                            np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                        idxs = idxs[0]
                        
                        if len(idxs) >= 3:
                            depth_av_k[win] = np.nanmean(i_gate_corr_kurtosis[idxs])
                            depth_se_k[win] = np.nanstd(i_gate_corr_kurtosis[idxs], ddof=1)/np.sqrt(len(idxs))
                        else:
                            depth_av_k[win] = np.nan
                            depth_se_k[win] = np.nan
                else:
                    for win in range(len(depth_av_k)):
                        idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                            np.where ((np.array(neuron_depths) < depth_av_win_right[win]))
                        idxs = idxs[0]
                        depth_av_k[win] = np.nanmean(i_gate_corr_kurtosis[idxs])
                        depth_se_k[win] = np.nanstd(i_gate_corr_kurtosis[idxs], ddof=1)/np.sqrt(len(idxs))
                
                axs_k_av.plot(depth_av_win_centre, depth_av_k, label=gate_key)
                axs_k_av.fill_between(depth_av_win_centre, depth_av_k-depth_se_k, depth_av_k+depth_se_k, alpha=0.2)
            
            ############################### Maximum absolute correlation plots
            
            est = sm.OLS(i_gate_corr_abs_max, X2) # Ordinary Least Squares fit
            est2 = est.fit()
            a = est2.params[0] # bias
            b = est2.params[1] # slope
            
            if plot_mode == 'full':
                f2, ax2 = plt.subplots()
                scatter = ax2.scatter(neuron_depths, i_gate_corr_abs_max, c=layer_colours, cmap=layers_cmap)
                #ax2.scatter(neuron_depths, i_gate_corr_abs_max, c=kurtosis_sig, cmap=pvalues_cmap)
                ax2.plot(xseq, a+b*xseq, color='r')
                ax2.set_xlabel('depth (micron from layer 3/4)')
                ax2.set_ylabel(gate_key + ' gate max abs correlation')
                handles, labels = scatter.legend_elements()
                ax2.legend(handles = handles, labels = ['1/3', '4', '5/6'])
                ax2.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
                ax2.spines[['right', 'top']].set_visible(False)
            
            elif plot_mode == 'av':
                depth_av_abs_max = np.zeros((len(depth_av_win_centre)))
                depth_se_abs_max = np.zeros((len(depth_av_win_centre)))
                
                if exclude_few == 'exclude':
                    for win in range(len(depth_av_k)):
                        idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                            np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                        idxs = idxs[0]
                        
                        if len(idxs) >= 3:
                            depth_av_abs_max[win] = np.nanmean(i_gate_corr_abs_max[idxs])
                            depth_se_abs_max[win] = np.nanstd(i_gate_corr_abs_max[idxs], ddof=1)/np.sqrt(len(idxs))
                        else:
                            depth_av_abs_max[win] = np.nan
                            depth_se_abs_max[win] = np.nan
                else:
                    for win in range(len(depth_av_k)):
                        idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                            np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                        idxs = idxs[0]
                        depth_av_abs_max[win] = np.nanmean(i_gate_corr_abs_max[idxs])
                        depth_se_abs_max[win] = np.nanstd(i_gate_corr_abs_max[idxs], ddof=1)/np.sqrt(len(idxs))
                
                axs_max_abs_av.plot(depth_av_win_centre, depth_av_abs_max, label=gate_key)
                axs_max_abs_av.fill_between(depth_av_win_centre, depth_av_abs_max-depth_se_abs_max, 
                                            depth_av_abs_max+depth_se_abs_max, alpha=0.2)
            
            ############################### Average absolute correlation plots
            
            est = sm.OLS(i_gate_corr_av, X2) # Ordinary Least Squares fit
            est2 = est.fit()
            a = est2.params[0] # bias
            b = est2.params[1] # slope
            
            if plot_mode == 'full':
                f3, ax3 = plt.subplots()
                scatter = ax3.scatter(neuron_depths, i_gate_corr_av, c=layer_colours, cmap=layers_cmap)
                #ax3.scatter(neuron_depths, i_gate_corr_av, c=kurtosis_sig, cmap=pvalues_cmap)
                ax3.plot(xseq, a+b*xseq, color='r')
                ax3.set_xlabel('depth (micron from layer 3/4)')
                ax3.set_ylabel(gate_key + ' gate av correlation')
                handles, labels = scatter.legend_elements()
                ax3.legend(handles = handles, labels = ['1/3', '4', '5/6'])
                ax3.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
                ax3.spines[['right', 'top']].set_visible(False)
                
            elif plot_mode == 'av':
                depth_av_av = np.zeros((len(depth_av_win_centre)))
                depth_se_av = np.zeros((len(depth_av_win_centre)))
                
                if exclude_few == 'exclude':
                    for win in range(len(depth_av_k)):
                        idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                            np.where ((np.array(neuron_depths) < depth_av_win_right[win]))
                        idxs = idxs[0]
                        
                        if len(idxs) >= 3:
                            depth_av_av[win] = np.nanmean(i_gate_corr_av[idxs])
                            depth_se_av[win] = np.nanstd(i_gate_corr_av[idxs], ddof=1)/np.sqrt(len(idxs))
                        else:
                            depth_av_av[win] = np.nan
                            depth_se_av[win] = np.nan
                else:
                    for win in range(len(depth_av_k)):
                        idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                            np.where ((np.array(neuron_depths) < depth_av_win_right[win]))
                        idxs = idxs[0]
                        depth_av_av[win] = np.nanmean(i_gate_corr_av[idxs])
                        depth_se_av[win] = np.nanstd(i_gate_corr_av[idxs], ddof=1)/np.sqrt(len(idxs))
                
                axs_av_av.plot(depth_av_win_centre, depth_av_av, label=gate_key)
                axs_av_av.fill_between(depth_av_win_centre, depth_av_av-depth_se_av, depth_av_av+depth_se_av, alpha=0.2)
        
        if plot_mode == 'av':
            axs_k_av.legend()
            axs_max_abs_av.legend()
            axs_av_av.legend()
                
#############################################################################################################################

def depth_gatesmodels_analysis(ccnorms_change_measure, HU_mode, models_gates_weights, models_eff_HUs, models_fits_dict, 
                               models_built_data, best_lambdas, dataset_ID, models):
    
    plot_mode = 'full'
    exclude_few = ''
    
    # define p-value-based colourmap - black indicates non-significance, red indicates significance
    pvalues_cmap = mcolors.ListedColormap(['black', 'red'])
    
    if dataset_ID == 'NS2_include_single':
        depth_sw_file = 'Data/NS2_raw/' + dataset_ID + '_depth_sw.pckl'
    else:
        depth_sw_file = 'Data/' + dataset_ID + '/' + dataset_ID + '_depth_sw.pckl'
    
    f = open(depth_sw_file, 'rb')
    depth_sw = pickle.load(f)
    f.close()
    
    neuron_list = models_built_data[models[0]][-1]['neurons_ID_sess']
    noise_ratios = models_built_data[models[0]][-1]['noise_ratios']
    
    neuron_depths = []
    neuron_layers = []
    neuron_sws = []
    
    neuron_idxs = [] # holds indexes of neurons for which we have depth, layer, and sw info
    
    for n_idx in range(len(neuron_list)):
        neuron_ID = neuron_list[n_idx][0]
        
        if neuron_ID in depth_sw[0]:
            neuron_depths.append(depth_sw[0][neuron_ID])
            neuron_layers.append(depth_sw[1][neuron_ID])
            neuron_sws.append(depth_sw[2][neuron_ID])
            
            neuron_idxs.append(n_idx)
    
    win_overlap = 50 # depth averaging window overlap, in microns
    win_size = 200 # depth averaging window size, in microns
    
    if (np.max(neuron_depths) - np.min(neuron_depths)) % win_overlap == 0:
        depth_av_win_left = np.arange(np.min(neuron_depths) - win_overlap/2, np.max(neuron_depths) + win_overlap/2, 
                                      win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
    else:
        min_limit = roundup(np.min(neuron_depths))
        max_limit = roundup(np.max(neuron_depths))
        
        depth_av_win_left = np.arange(min_limit - win_overlap/2, max_limit + win_overlap/2, win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
    
    # define layer-based colourmap - black is 13, red is 4, blue is 56
    layers_cmap = mcolors.ListedColormap(['black', 'red', 'blue'])
    layer_colours = []
    for neuron in neuron_layers:
        if neuron == '13':
            layer_colours.append(0)
        elif neuron == '44':
            layer_colours.append(1)
        elif neuron == '56':
            layer_colours.append(2)
    
    for i_model_ID in models:
        if i_model_ID == 'pop_gatesLSTM' or i_model_ID == 'pop_gatessubLSTM':
            gate_keys = ['i', 'f', 'o']
        elif i_model_ID == 'pop_gatesGRU':
            gate_keys = ['r', 'z']
        elif i_model_ID == 'pop_gatesfLSTM':
            gate_keys = ['i', 'f', 'o', 'c']
        
        i_model_params = models_fits_dict[i_model_ID][1]['all_model_params'] # hyperparameters of i_model
        i_model_type = models_fits_dict[i_model_ID][1]['model_type']
        if i_model_type == 'not_conv':
            i_hidden_size = i_model_params['hidden_size'] # network hidden size
        elif i_model_type == 'conv':
            i_hidden_size = np.size(models_fits_dict[i_model_ID][0][0][0][0]['w_l1'], 0)
        i_model_best_lambdas = best_lambdas[dataset_ID][i_model_ID] # i_model best lambda for each neuron

        # Gates plotted separately
        i_model_gates_weights = models_gates_weights[i_model_ID][0]
        
        if plot_mode == 'av':
            f_av, axs_av = plt.subplots()
            axs_av.set_xlabel('depth (micron from layer 3/4)')
            axs_av.set_ylabel('gate weight percentage')
            #axs_av.set_ylim([0.08, 0.18])
            axs_av.set_title(dataset_ID + ' - ' + i_model_ID)
            axs_av.spines[['right', 'top']].set_visible(False)
        
        for gate_ID in range(np.size(i_model_gates_weights, 1)):
            if HU_mode == 'Eff':
                i_model_eff_HUs = models_eff_HUs[i_model_ID][0]
                i_gate_corr = []
            else:
                # array to hold correlations for specific fold and best lambda
                i_gate_weights = np.zeros((len(neuron_idxs)))
            
            for nID in range(len(neuron_idxs)): # loop over neurons
                # correlations for specific neuron, fold and the neuron's best lambda
                if HU_mode == 'Eff':
                    continue
                elif HU_mode == 'All':
                    i_gate_weights[nID] = i_model_gates_weights[neuron_idxs[nID], gate_ID]
            
            X2 = sm.add_constant(neuron_depths) # add bias to independent variable
            xseq = np.linspace(np.min(neuron_depths)-0.05, np.max(neuron_depths)+0.05, num=100)
            
            est = sm.OLS(i_gate_weights, X2) # Ordinary Least Squares fit
            est2 = est.fit()
            a = est2.params[0] # bias
            b = est2.params[1] # slope
            
            if plot_mode == 'full':
                f1, ax1 = plt.subplots()
                scatter = ax1.scatter(neuron_depths, i_gate_weights, c=layer_colours, cmap=layers_cmap)
                #ax1.scatter(neuron_depths, i_gate_weights, c=kurtosis_sig, cmap=pvalues_cmap)
                ax1.plot(xseq, a+b*xseq, color='r')
                ax1.set_xlabel('depth (micron from layer 3/4)')
                ax1.set_ylabel(gate_keys[gate_ID] + ' gate weight percentage')
                handles, labels = scatter.legend_elements()
                ax1.legend(handles = handles, labels = ['1/3', '4', '5/6'])
                ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
                ax1.spines[['right', 'top']].set_visible(False)
            
            elif plot_mode == 'av':
                depth_av_gate_weights = np.zeros((len(depth_av_win_centre)))
                depth_se_gate_weights = np.zeros((len(depth_av_win_centre)))
                
                if exclude_few == 'exclude':
                    for win in range(len(depth_av_gate_weights)):
                        idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                            np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                        idxs = idxs[0]
                        
                        if len(idxs) >= 3:
                            depth_av_gate_weights[win] = np.nanmean(i_gate_weights[idxs])
                            depth_se_gate_weights[win] = np.nanstd(i_gate_weights[idxs], ddof=1)/np.sqrt(len(idxs))
                        else:
                            depth_av_gate_weights[win] = np.nan
                            depth_se_gate_weights[win] = np.nan
                else:
                    for win in range(len(depth_av_gate_weights)):
                        idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                            np.where ((np.array(neuron_depths) < depth_av_win_right[win]))
                        idxs = idxs[0]
                        depth_av_gate_weights[win] = np.nanmean(i_gate_weights[idxs])
                        depth_se_gate_weights[win] = np.nanstd(i_gate_weights[idxs], ddof=1)/np.sqrt(len(idxs))
                
                axs_av.plot(depth_av_win_centre, depth_av_gate_weights, label=gate_keys[gate_ID])
                axs_av.fill_between(depth_av_win_centre, depth_av_gate_weights-depth_se_gate_weights, 
                                      depth_av_gate_weights+depth_se_gate_weights, alpha=0.2)
        
        if plot_mode == 'av':
            axs_av.legend()
                    
         # All gates together                                                                                           
        i_model_all_gates_weights = models_gates_weights[i_model_ID][1]
        
        if HU_mode == 'Eff':
            i_model_eff_HUs = models_eff_HUs[i_model_ID][0]
            i_gate_corr = []
        else:
            # array to hold correlations for specific fold and best lambda
            i_all_gate_weights = np.zeros((len(neuron_idxs)))
        
        for nID in range(len(neuron_idxs)): # loop over neurons
            # correlations for specific neuron, fold, and the neuron's best lambda
            if HU_mode == 'Eff':
                continue
            elif HU_mode == 'All':
                i_all_gate_weights[nID] = i_model_all_gates_weights[neuron_idxs[nID]]

        est = sm.OLS(i_all_gate_weights, X2) # Ordinary Least Squares fit
        est2 = est.fit()
        a = est2.params[0] # bias
        b = est2.params[1] # slope
        
        if plot_mode == 'full':
            f1, ax1 = plt.subplots()
            scatter = ax1.scatter(neuron_depths, i_all_gate_weights, c=layer_colours, cmap=layers_cmap)
            #ax1.scatter(neuron_depths, i_gate_weights, c=kurtosis_sig, cmap=pvalues_cmap)
            ax1.plot(xseq, a+b*xseq, color='r')
            ax1.set_xlabel('depth (micron from layer 3/4)')
            ax1.set_ylabel('all gates weight percentage')
            handles, labels = scatter.legend_elements()
            ax1.legend(handles = handles, labels = ['1/3', '4', '5/6'])
            ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
            ax1.spines[['right', 'top']].set_visible(False)
        
        elif plot_mode == 'av':
            depth_av_all_gate_weights = np.zeros((len(depth_av_win_centre)))
            depth_se_all_gate_weights = np.zeros((len(depth_av_win_centre)))
            
            if exclude_few == 'exclude':
                for win in range(len(depth_av_all_gate_weights)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    
                    if len(idxs) >= 3:
                        depth_av_all_gate_weights[win] = np.nanmean(i_all_gate_weights[idxs])
                        depth_se_all_gate_weights[win] = np.nanstd(i_all_gate_weights[idxs], ddof=1)/np.sqrt(len(idxs))
                    else:
                        depth_av_all_gate_weights[win] = np.nan
                        depth_se_all_gate_weights[win] = np.nan
            else:
                for win in range(len(depth_av_all_gate_weights)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where ((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    depth_av_all_gate_weights[win] = np.nanmean(i_all_gate_weights[idxs])
                    depth_se_all_gate_weights[win] = np.nanstd(i_all_gate_weights[idxs], ddof=1)/np.sqrt(len(idxs))
            
            f_av, axs_av = plt.subplots()
            axs_av.set_xlabel('depth (micron from layer 3/4)')
            axs_av.set_ylabel('all gates weight percentage')
            axs_av.set_title(dataset_ID + ' - ' + i_model_ID)
            axs_av.spines[['right', 'top']].set_visible(False)
            axs_av.plot(depth_av_win_centre, depth_av_all_gate_weights)
            axs_av.fill_between(depth_av_win_centre, depth_av_all_gate_weights-depth_se_all_gate_weights, 
                                depth_av_all_gate_weights+depth_se_all_gate_weights, alpha=0.2)
            
#############################################################################################################################          

def depth_silence_analysis(models_corr_stims_silence, models_built_data, dataset_ID, models):
    
    plot_mode = 'av'
    exclude_few = 'exclude'
    
    # define p-value-based colourmap - black indicates non-significance, red indicates significance
    pvalues_cmap = mcolors.ListedColormap(['black', 'red'])
    
    if dataset_ID == 'NS2_include_single':
        depth_sw_file = 'Data/NS2_raw/' + dataset_ID + '_depth_sw.pckl'
    else:
        depth_sw_file = 'Data/' + dataset_ID + '/' + dataset_ID + '_depth_sw.pckl'
    
    f = open(depth_sw_file, 'rb')
    depth_sw = pickle.load(f)
    f.close()
    
    neuron_list = models_built_data[models[0]][-1]['neurons_ID_sess']
    
    neuron_depths = []
    neuron_layers = []
    neuron_sws = []
    
    neuron_idxs = [] # holds indexes of neurons for which we have depth, layer, and sw info
    
    for n_idx in range(len(neuron_list)):
        neuron_ID = neuron_list[n_idx][0]
        
        if neuron_ID in depth_sw[0]:
            neuron_depths.append(depth_sw[0][neuron_ID])
            neuron_layers.append(depth_sw[1][neuron_ID])
            neuron_sws.append(depth_sw[2][neuron_ID])
            
            neuron_idxs.append(n_idx)
          
    win_overlap = 50 # depth averaging window overlap, in microns
    win_size = 200 # depth averaging window size, in microns
    
    if (np.max(neuron_depths) - np.min(neuron_depths)) % win_overlap == 0:
        depth_av_win_left = np.arange(np.min(neuron_depths) - win_overlap/2, np.max(neuron_depths) + win_overlap/2, 
                                      win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
    else:
        min_limit = roundup(np.min(neuron_depths))
        max_limit = roundup(np.max(neuron_depths))
        
        depth_av_win_left = np.arange(min_limit - win_overlap/2, max_limit + win_overlap/2, win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
          
    # define layer-based colourmap - black is 13, red is 4, blue is 56
    layers_cmap = mcolors.ListedColormap(['black', 'red', 'blue'])
    layer_colours = []
    for neuron in neuron_layers:
        if neuron == '13':
            layer_colours.append(0)
        elif neuron == '44':
            layer_colours.append(1)
        elif neuron == '56':
            layer_colours.append(2)
    
    for i_model_ID in models:
        # NumPy array of correlations between each neuron's recorded PSTH and each HU in i_model, for all folds and lambdas 
        # (dimensions -> (neurons, folds, lambdas, HUs))
        i_model_corr_silence = models_corr_stims_silence[i_model_ID][0]
        
        used_corr_silence = i_model_corr_silence[neuron_idxs]
        
        X2 = sm.add_constant(neuron_depths) # add bias to independent variable
        xseq = np.linspace(np.min(neuron_depths)-0.05, np.max(neuron_depths)+0.05, num=100)
        
        est = sm.OLS(used_corr_silence, X2) # Ordinary Least Squares fit
        est2 = est.fit()
        a = est2.params[0] # bias
        b = est2.params[1] # slope
        
        if plot_mode == 'full':
            f1, ax1 = plt.subplots()
            scatter = ax1.scatter(neuron_depths, used_corr_silence, c=layer_colours, cmap=layers_cmap) # layers colourmap
            #ax1.scatter(neuron_depths, i_HU_corr_kurtosis, c=kurtosis_sig, cmap=pvalues_cmap) # p values colourmap
            ax1.plot(xseq, a+b*xseq, color='r')
            ax1.set_xlabel('depth (micron from layer 3/4)')
            ax1.set_ylabel('SDRM')
            handles, labels = scatter.legend_elements()
            ax1.legend(handles = handles, labels = ['1/3', '4', '5/6'])
            ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
            ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            
        elif plot_mode == 'av':
            depth_av_corr_silence = np.zeros((len(depth_av_win_centre)))
            depth_se_corr_silence = np.zeros((len(depth_av_win_centre)))
            
            if exclude_few == 'exclude':
                for win in range(len(depth_av_corr_silence)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    
                    if len(idxs) >= 3:
                        depth_av_corr_silence[win] = np.nanmean(used_corr_silence[idxs])
                        depth_se_corr_silence[win] = np.nanstd(used_corr_silence[idxs], ddof=1)/np.sqrt(len(idxs))
                    else:
                        depth_av_corr_silence[win] = np.nan
                        depth_se_corr_silence[win] = np.nan
            else:
                for win in range(len(depth_av_corr_silence)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    depth_av_corr_silence[win] = np.nanmean(used_corr_silence[idxs])
                    depth_se_corr_silence[win] = np.nanstd(used_corr_silence[idxs], ddof=1)/np.sqrt(len(idxs))
            
            f1, ax1 = plt.subplots()
            ax1.plot(depth_av_win_centre, depth_av_corr_silence)
            ax1.fill_between(depth_av_win_centre, depth_av_corr_silence-depth_se_corr_silence, 
                             depth_av_corr_silence+depth_se_corr_silence, alpha=0.2)
            ax1.set_xlabel('depth (micron from layer 3/4)')
            ax1.set_ylabel('SDRM')
            ax1.set_title(dataset_ID + ' - ' + i_model_ID)
            ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
                       
#############################################################################################################################          

def depth_ON_OFF_analysis(models_corr_ON_OFF_stims, models_built_data, dataset_ID, models):
    
    plot_mode = 'av'
    exclude_few = 'exclude'
    
    # define p-value-based colourmap - black indicates non-significance, red indicates significance
    pvalues_cmap = mcolors.ListedColormap(['black', 'red'])
    
    if dataset_ID == 'NS2_include_single':
        depth_sw_file = 'Data/NS2_raw/' + dataset_ID + '_depth_sw.pckl'
    else:
        depth_sw_file = 'Data/' + dataset_ID + '/' + dataset_ID + '_depth_sw.pckl'
    
    f = open(depth_sw_file, 'rb')
    depth_sw = pickle.load(f)
    f.close()
    
    neuron_list = models_built_data[models[0]][-1]['neurons_ID_sess']
    
    neuron_depths = []
    neuron_layers = []
    neuron_sws = []
    
    neuron_idxs = [] # holds indexes of neurons for which we have depth, layer, and sw info
    
    for n_idx in range(len(neuron_list)):
        neuron_ID = neuron_list[n_idx][0]
        
        if neuron_ID in depth_sw[0]:
            neuron_depths.append(depth_sw[0][neuron_ID])
            neuron_layers.append(depth_sw[1][neuron_ID])
            neuron_sws.append(depth_sw[2][neuron_ID])
            
            neuron_idxs.append(n_idx)
          
    win_overlap = 50 # depth averaging window overlap, in microns
    win_size = 200 # depth averaging window size, in microns
    
    if (np.max(neuron_depths) - np.min(neuron_depths)) % win_overlap == 0:
        depth_av_win_left = np.arange(np.min(neuron_depths) - win_overlap/2, np.max(neuron_depths) + win_overlap/2, 
                                      win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
    else:
        min_limit = roundup(np.min(neuron_depths))
        max_limit = roundup(np.max(neuron_depths))
        
        depth_av_win_left = np.arange(min_limit - win_overlap/2, max_limit + win_overlap/2, win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
          
    # define layer-based colourmap - black is 13, red is 4, blue is 56
    layers_cmap = mcolors.ListedColormap(['black', 'red', 'blue'])
    layer_colours = []
    for neuron in neuron_layers:
        if neuron == '13':
            layer_colours.append(0)
        elif neuron == '44':
            layer_colours.append(1)
        elif neuron == '56':
            layer_colours.append(2)
    
    for i_model_ID in models:
        # NumPy array of correlations between each neuron's recorded PSTH and each HU in i_model, for all folds and lambdas 
        # (dimensions -> (neurons, folds, lambdas, HUs))
        i_model_corr_ON_OFF = models_corr_ON_OFF_stims[i_model_ID][0]
        
        used_corr_ON_OFF = i_model_corr_ON_OFF[neuron_idxs]
        
        X2 = sm.add_constant(neuron_depths) # add bias to independent variable
        xseq = np.linspace(np.min(neuron_depths)-0.05, np.max(neuron_depths)+0.05, num=100)
        
        est = sm.OLS(used_corr_ON_OFF, X2) # Ordinary Least Squares fit
        est2 = est.fit()
        a = est2.params[0] # bias
        b = est2.params[1] # slope
        
        if plot_mode == 'full':
            f1, ax1 = plt.subplots()
            scatter = ax1.scatter(neuron_depths, used_corr_ON_OFF, c=layer_colours, cmap=layers_cmap) # layers colourmap
            #ax1.scatter(neuron_depths, i_HU_corr_kurtosis, c=kurtosis_sig, cmap=pvalues_cmap) # p values colourmap
            ax1.plot(xseq, a+b*xseq, color='r')
            ax1.set_xlabel('depth (micron from layer 3/4)')
            ax1.set_ylabel('sudden changes corr')
            handles, labels = scatter.legend_elements()
            ax1.legend(handles = handles, labels = ['1/3', '4', '5/6'])
            ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
            ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            
        elif plot_mode == 'av':
            depth_av_corr_ON_OFF = np.zeros((len(depth_av_win_centre)))
            depth_se_corr_ON_OFF = np.zeros((len(depth_av_win_centre)))
            
            if exclude_few == 'exclude':
                for win in range(len(depth_av_corr_ON_OFF)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    
                    if len(idxs) >= 3:
                        depth_av_corr_ON_OFF[win] = np.nanmean(used_corr_ON_OFF[idxs])
                        depth_se_corr_ON_OFF[win] = np.nanstd(used_corr_ON_OFF[idxs], ddof=1)/np.sqrt(len(idxs))
                    else:
                        depth_av_corr_ON_OFF[win] = np.nan
                        depth_se_corr_ON_OFF[win] = np.nan
            else:
                for win in range(len(depth_av_corr_ON_OFF)):
                    idxs = np.where((np.array(neuron_depths) >= depth_av_win_left[win])) and \
                        np.where((np.array(neuron_depths) < depth_av_win_right[win]))
                    idxs = idxs[0]
                    depth_av_corr_ON_OFF[win] = np.nanmean(used_corr_ON_OFF[idxs])
                    depth_se_corr_ON_OFF[win] = np.nanstd(used_corr_ON_OFF[idxs], ddof=1)/np.sqrt(len(idxs))
            
            f1, ax1 = plt.subplots()
            ax1.plot(depth_av_win_centre, depth_av_corr_ON_OFF)
            ax1.fill_between(depth_av_win_centre, depth_av_corr_ON_OFF-depth_se_corr_ON_OFF, 
                             depth_av_corr_ON_OFF+depth_se_corr_ON_OFF, alpha=0.2)
            ax1.set_xlabel('depth (micron from layer 3/4)')
            ax1.set_ylabel('sudden changes corr')
            ax1.set_title(dataset_ID + ' - ' + i_model_ID)
            ax1.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
                
#############################################################################################################################

def depth_rcLSTM_analysis(rcLSTM_weights, models_fits_dict, models_built_data, best_lambdas, dataset_ID, models):
    
    plot_mode = 'av'
    exclude_few = 'exclude'
    plots_together = True
    
    # define p-value-based colourmap - black indicates non-significance, red indicates significance
    pvalues_cmap = mcolors.ListedColormap(['black', 'red'])
    
    if dataset_ID == 'NS2_include_single':
        depth_sw_file = 'Data/NS2_raw/' + dataset_ID + '_depth_sw.pckl'
    else:
        depth_sw_file = 'Data/' + dataset_ID + '/' + dataset_ID + '_depth_sw.pckl'
    
    f = open(depth_sw_file, 'rb')
    depth_sw = pickle.load(f)
    f.close()
    
    neuron_list = models_built_data[models[0]][-1]['neurons_ID_sess']
    
    neuron_depths = []
    neuron_layers = []
    neuron_sws = []
    
    neuron_idxs = [] # holds indexes of neurons for which we have depth, layer, and sw info
    
    for n_idx in range(len(neuron_list)):
        neuron_ID = neuron_list[n_idx][0]
        
        if neuron_ID in depth_sw[0]:
            neuron_depths.append(depth_sw[0][neuron_ID])
            neuron_layers.append(depth_sw[1][neuron_ID])
            neuron_sws.append(depth_sw[2][neuron_ID])
            
            neuron_idxs.append(n_idx)
    
    # uncomment for plots with NS2 range
    if dataset_ID == 'NS3' or dataset_ID == 'NS3_PEG':
        small_depths = np.where(np.array(neuron_depths) < -700)[0]
        large_depths = np.where(np.array(neuron_depths) > 600)[0]
        extra_depths = np.concatenate((small_depths, large_depths))
        
        new_neuron_depths = np.delete(np.array(neuron_depths), extra_depths)
        new_neuron_layers = np.delete(np.array(neuron_layers), extra_depths)
        new_neuron_idxs = np.delete(np.array(neuron_idxs), extra_depths)

        neuron_depths = list(new_neuron_depths)
        neuron_layers = list(new_neuron_layers)
        neuron_idxs = list(new_neuron_idxs)
    
    win_overlap = 50 # depth averaging window overlap, in microns
    win_size = 200 # depth averaging window size, in microns
    
    if (np.max(neuron_depths) - np.min(neuron_depths)) % win_overlap == 0:
        left_min = np.min(neuron_depths)
        left_max = np.max(neuron_depths)- win_size
        
        depth_av_win_left = np.arange(left_min, left_max + win_overlap, win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        '''depth_av_win_left = np.arange(np.min(neuron_depths) - win_overlap/2, np.max(neuron_depths) + win_overlap/2, 
                                      win_overlap)
        depth_av_win_right = depth_av_win_left + win_size'''
        
        depth_av_win_centre = depth_av_win_left + win_size/2
    else:
        min_limit = roundup(np.min(neuron_depths))
        max_limit = roundup(np.max(neuron_depths))
        
        left_min = min_limit
        left_max = max_limit - win_size
        
        depth_av_win_left = np.arange(left_min, left_max + win_overlap, win_overlap)
        depth_av_win_right = depth_av_win_left + win_size
        
        #depth_av_win_left = np.arange(min_limit - win_overlap/2, max_limit + win_overlap/2, win_overlap)
        #depth_av_win_right = depth_av_win_left + win_size
        
        depth_av_win_centre = depth_av_win_left + win_size/2
        
        depth_av_win_right[-1] = depth_av_win_right[-1] + 1
        
    # define layer-based colourmap - black is 13, red is 4, blue is 56
    layers_cmap = mcolors.ListedColormap(['black', 'red', 'blue'])
    layer_colours = []
    for neuron in neuron_layers:
        if neuron == '13':
            layer_colours.append(0)
        elif neuron == '44':
            layer_colours.append(1)
        elif neuron == '56':
            layer_colours.append(2)
    
    labels = ['fc', 'subLSTM']
    
    for i_model_ID in models:
        if not plots_together:
            for label in labels:
                w = rcLSTM_weights[i_model_ID][label]
                used_w = w[neuron_idxs]
                
                X2 = sm.add_constant(neuron_depths) # add bias to independent variable
                xseq = np.linspace(np.min(neuron_depths)-0.05, np.max(neuron_depths)+0.05, num=100)
                
                est = sm.OLS(used_w, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                b = est2.params[1] # slope
                
                if plot_mode == 'full':
                    f1, ax1 = plt.subplots()
                    scatter = ax1.scatter(neuron_depths, used_w, c=layer_colours, cmap=layers_cmap)
                    #ax1.scatter(neuron_depths, used_w, c=kurtosis_sig, cmap=pvalues_cmap)
                    ax1.plot(xseq, a+b*xseq, color='r')
                    ax1.set_xlabel('depth (micron from layer 3/4)')
                    ax1.set_ylabel(label)
                    handles, labels = scatter.legend_elements()
                    ax1.legend(handles = handles, labels = ['1/3', '4', '5/6'])
                    ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
                    ax1.spines[['right', 'top']].set_visible(False)
                
                elif plot_mode == 'av':
                    depth_av_w = np.zeros((len(depth_av_win_centre)))
                    depth_se_w = np.zeros((len(depth_av_win_centre)))
                    
                    if exclude_few == 'exclude':
                        for win in range(len(depth_av_w)):
                            idxs_1 = np.where((np.array(neuron_depths) >= depth_av_win_left[win]))[0]
                            idxs_2 = np.where((np.array(neuron_depths) < depth_av_win_right[win]))[0]
                            idxs = np.intersect1d(idxs_1, idxs_2)
                            
                            if len(idxs) >= 3:
                                depth_av_w[win] = np.nanmean(used_w[idxs])
                                depth_se_w[win] = np.nanstd(used_w[idxs], ddof=1)/np.sqrt(len(idxs))
                            else:
                                depth_av_w[win] = np.nan
                                depth_se_w[win] = np.nan
                    else:
                        for win in range(len(depth_av_w)):
                            idxs_1 = np.where((np.array(neuron_depths) >= depth_av_win_left[win]))[0]
                            idxs_2 = np.where((np.array(neuron_depths) < depth_av_win_right[win]))[0]
                            idxs = np.intersect1d(idxs_1, idxs_2)
                            
                            depth_av_w[win] = np.nanmean(used_w[idxs])
                            depth_se_w[win] = np.nanstd(used_w[idxs], ddof=1)/np.sqrt(len(idxs))
                    
                    f_av, axs_av = plt.subplots()
                    axs_av.set_xlabel('depth (micron from layer 3/4)')
                    axs_av.set_ylabel(label)
                    axs_av.set_title(dataset_ID + ' - ' + i_model_ID)
                    axs_av.spines[['right', 'top']].set_visible(False)
                    axs_av.plot(depth_av_win_centre, depth_av_w)
                    axs_av.fill_between(depth_av_win_centre, depth_av_w-depth_se_w, depth_av_w+depth_se_w, alpha=0.2)
       
        elif plots_together:
            if plot_mode == 'full':
                used_w_labels = {}
                
                for label in labels:
                    w = rcLSTM_weights[i_model_ID][label]
                    used_w_labels[label] = w[neuron_idxs]
                
                used_w_diff = used_w_labels[labels[1]] - used_w_labels[labels[0]]
                
                X2 = sm.add_constant(neuron_depths) # add bias to independent variable
                xseq = np.linspace(np.min(neuron_depths)-0.05, np.max(neuron_depths)+0.05, num=100)
                
                est = sm.OLS(used_w_diff, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                b = est2.params[1] # slope
                
                f1, ax1 = plt.subplots()
                scatter = ax1.scatter(neuron_depths, used_w_diff, c=layer_colours, cmap=layers_cmap)
                #ax1.scatter(neuron_depths, used_w_diff, c=kurtosis_sig, cmap=pvalues_cmap)
                ax1.plot(xseq, a+b*xseq, color='r')
                ax1.set_xlabel('depth (micron from layer 3/4)')
                ax1.set_ylabel('var_diff')
                handles, labels = scatter.legend_elements()
                ax1.legend(handles = handles, labels = ['1/3', '4', '5/6'])
                ax1.set_title(dataset_ID + ' - ' + i_model_ID + ' - p = ' + str(est2.pvalues[1]))
                ax1.spines[['right', 'top']].set_visible(False)
                
                # Depth-averaged variance difference
                depth_av_w_diff = np.zeros((len(depth_av_win_centre)))
                depth_se_w_diff = np.zeros((len(depth_av_win_centre)))
                
                if exclude_few == 'exclude':
                    for win in range(len(depth_av_w_diff)):
                        idxs_1 = np.where((np.array(neuron_depths) >= depth_av_win_left[win]))[0]
                        idxs_2 = np.where((np.array(neuron_depths) < depth_av_win_right[win]))[0]
                        idxs = np.intersect1d(idxs_1, idxs_2)
                        
                        if len(idxs) >= 3:
                            depth_av_w_diff[win] = np.nanmean(used_w_diff[idxs])
                            depth_se_w_diff[win] = np.nanstd(used_w_diff[idxs], ddof=1)/np.sqrt(len(idxs))
                        else:
                            depth_av_w_diff[win] = np.nan
                            depth_se_w_diff[win] = np.nan
                else:
                    for win in range(len(depth_av_w_diff)):
                        idxs_1 = np.where((np.array(neuron_depths) >= depth_av_win_left[win]))[0]
                        idxs_2 = np.where((np.array(neuron_depths) < depth_av_win_right[win]))[0]
                        idxs = np.intersect1d(idxs_1, idxs_2)
                        
                        depth_av_w_diff[win] = np.nanmean(used_w_diff[idxs])
                        depth_se_w_diff[win] = np.nanstd(used_w_diff[idxs], ddof=1)/np.sqrt(len(idxs))
                
                f_av, axs_av = plt.subplots()
                axs_av.set_xlabel('depth (micron from layer 3/4)')
                axs_av.set_ylabel('var_diff')
                axs_av.set_title(dataset_ID + ' - ' + i_model_ID)
                axs_av.spines[['right', 'top']].set_visible(False)
                axs_av.plot(depth_av_win_centre, depth_av_w_diff, label=label)
                axs_av.fill_between(depth_av_win_centre, depth_av_w_diff-depth_se_w_diff, depth_av_w_diff+depth_se_w_diff, 
                                    alpha=0.2)
                #axs_av.legend()
                
                split_neurons = int(0.1*len(neuron_depths))
                blocks = int(len(neuron_depths)/split_neurons)
                
                depth_block_idxs = []
                
                for block in range(blocks):
                    if block == 0:
                        depth_block_idxs.append(np.argsort(neuron_depths)[-split_neurons:])   
                    elif block > 0 and block < blocks-1:
                        depth_block_idxs.append(np.argsort(neuron_depths)[-(block+1)*split_neurons:-block*split_neurons])
                    elif block == blocks-1:
                        depth_block_idxs.append(np.argsort(neuron_depths)[:-block*split_neurons])
                
                blocks_depths = []
                blocks_w_diff = []
                
                for block in range(blocks):
                    blocks_depths.append(np.array(neuron_depths)[depth_block_idxs[block]])
                    blocks_w_diff.append(used_w_diff[depth_block_idxs[block]])
                
                blocks_depths = list(reversed(blocks_depths))
                block_av_depth = np.array([np.nanmean(block) for block in blocks_depths])
                
                blocks_w_diff = list(reversed(blocks_w_diff))
                block_av_w_diff = np.array([np.nanmean(block) for block in blocks_w_diff])
                block_se_w_diff = np.array([np.nanstd(block, ddof=1)/np.sqrt(len(block)) for block in blocks_w_diff])
                
                f_block, axs_block = plt.subplots()
                axs_block.set_xlabel('depth (micron from layer 3/4)')
                axs_block.set_ylabel('var_diff')
                axs_block.set_title(dataset_ID + ' - ' + i_model_ID)
                axs_block.spines[['right', 'top']].set_visible(False)
                axs_block.plot(block_av_depth, block_av_w_diff)
                axs_block.fill_between(block_av_depth, block_av_w_diff-block_se_w_diff, 
                                       block_av_w_diff+block_se_w_diff, alpha=0.2)
                
            elif plot_mode == 'av':
                f_av, axs_av = plt.subplots()
                label_colours = {labels[0]: '#069AF3', labels[1]: '#808080'}
                for label in labels:
                    w = rcLSTM_weights[i_model_ID][label]
                    used_w = w[neuron_idxs]
                    #import pdb
                    #pdb.set_trace()
                    depth_av_w = np.zeros((len(depth_av_win_centre)))
                    depth_se_w = np.zeros((len(depth_av_win_centre)))
                    
                    if exclude_few == 'exclude':
                        for win in range(len(depth_av_w)):
                            #import pdb
                            #pdb.set_trace()
                            idxs_1 = np.where((np.array(neuron_depths) >= depth_av_win_left[win]))[0]
                            idxs_2 = np.where((np.array(neuron_depths) < depth_av_win_right[win]))[0]
                            idxs = np.intersect1d(idxs_1, idxs_2)
                            
                            if len(idxs) >= 3:
                                depth_av_w[win] = np.nanmean(used_w[idxs])
                                depth_se_w[win] = np.nanstd(used_w[idxs], ddof=1)/np.sqrt(len(idxs))
                            else:
                                depth_av_w[win] = np.nan
                                depth_se_w[win] = np.nan
                    else:
                        for win in range(len(depth_av_w)):
                            idxs_1 = np.where((np.array(neuron_depths) >= depth_av_win_left[win]))[0]
                            idxs_2 = np.where((np.array(neuron_depths) < depth_av_win_right[win]))[0]
                            idxs = np.intersect1d(idxs_1, idxs_2)
                            
                            depth_av_w[win] = np.nanmean(used_w[idxs])
                            depth_se_w[win] = np.nanstd(used_w[idxs], ddof=1)/np.sqrt(len(idxs))
                    
                    axs_av.set_xlabel('depth (micron from layer 3/4)')
                    axs_av.set_ylabel('w_sum')
                    axs_av.set_title(dataset_ID + ' - ' + i_model_ID)
                    #axs_av.set_xlim([-700, 650])
                    
                    #axs_av.set_yticks([0.35, 0.45, 0.55, 0.65])
                    #axs_av.set_yticks([0.3, 0.45, 0.6, 0.75])
                    #axs_av.set_xticks([-600, -300, 0, 300, 600])
                    #axs_av.xaxis.set_tick_params(labelcolor='none')
                    #axs_av.yaxis.set_tick_params(labelcolor='none')
                    
                    axs_av.spines[['right', 'top']].set_visible(False)
                    axs_av.plot(depth_av_win_centre, depth_av_w, label=label, linewidth=3, color=label_colours[label])
                    axs_av.fill_between(depth_av_win_centre, depth_av_w-depth_se_w, depth_av_w+depth_se_w, alpha=0.2, 
                                        color=label_colours[label])
                    axs_av.legend()
        
#############################################################################################################################

def roundup(x):
    if x % 100 == 0:
        a = x
    else:
        if x > 0:
            a = x + 100 - x % 100
        else:
            flip_x = -x
            a = -(flip_x + 100 - flip_x % 100)
            
    return a

#############################################################################################################################

def fake_parula():
    from matplotlib.colors import LinearSegmentedColormap
    
    cm_data = [[ 0.26710521,  0.03311059,  0.6188155 ],
       [ 0.26493929,  0.04780926,  0.62261795],
       [ 0.26260545,  0.06084214,  0.62619176],
       [ 0.26009691,  0.07264411,  0.62951561],
       [ 0.25740785,  0.08360391,  0.63256745],
       [ 0.25453369,  0.09395358,  0.63532497],
       [ 0.25147146,  0.10384228,  0.6377661 ],
       [ 0.24822014,  0.11337029,  0.6398697 ],
       [ 0.24478105,  0.12260661,  0.64161629],
       [ 0.24115816,  0.131599  ,  0.6429888 ],
       [ 0.23735836,  0.14038009,  0.64397346],
       [ 0.23339166,  0.14897137,  0.64456048],
       [ 0.22927127,  0.15738602,  0.64474476],
       [ 0.22501278,  0.16563165,  0.64452595],
       [ 0.22063349,  0.17371215,  0.64390834],
       [ 0.21616055,  0.18162302,  0.64290515],
       [ 0.21161851,  0.18936156,  0.64153295],
       [ 0.20703353,  0.19692415,  0.63981287],
       [ 0.20243273,  0.20430706,  0.63776986],
       [ 0.19784363,  0.211507  ,  0.63543183],
       [ 0.19329361,  0.21852157,  0.63282872],
       [ 0.18880937,  0.2253495 ,  0.62999156],
       [ 0.18442119,  0.23198815,  0.62695569],
       [ 0.18014936,  0.23844124,  0.62374886],
       [ 0.17601569,  0.24471172,  0.62040016],
       [ 0.17204028,  0.25080356,  0.61693715],
       [ 0.16824123,  0.25672163,  0.6133854 ],
       [ 0.16463462,  0.26247158,  0.60976836],
       [ 0.16123449,  0.26805963,  0.60610723],
       [ 0.15805279,  0.27349243,  0.60242099],
       [ 0.15509948,  0.27877688,  0.59872645],
       [ 0.15238249,  0.28392004,  0.59503836],
       [ 0.14990781,  0.28892902,  0.59136956],
       [ 0.14767951,  0.29381086,  0.58773113],
       [ 0.14569979,  0.29857245,  0.58413255],
       [ 0.1439691 ,  0.30322055,  0.58058191],
       [ 0.14248613,  0.30776167,  0.57708599],
       [ 0.14124797,  0.31220208,  0.57365049],
       [ 0.14025018,  0.31654779,  0.57028011],
       [ 0.13948691,  0.32080454,  0.5669787 ],
       [ 0.13895174,  0.32497744,  0.56375063],
       [ 0.13863958,  0.32907012,  0.56060453],
       [ 0.138537  ,  0.3330895 ,  0.55753513],
       [ 0.13863384,  0.33704026,  0.55454374],
       [ 0.13891931,  0.34092684,  0.55163126],
       [ 0.13938212,  0.34475344,  0.54879827],
       [ 0.14001061,  0.34852402,  0.54604503],
       [ 0.14079292,  0.35224233,  0.54337156],
       [ 0.14172091,  0.35590982,  0.54078769],
       [ 0.14277848,  0.35953205,  0.53828312],
       [ 0.14395358,  0.36311234,  0.53585661],
       [ 0.1452346 ,  0.36665374,  0.5335074 ],
       [ 0.14661019,  0.3701591 ,  0.5312346 ],
       [ 0.14807104,  0.37363011,  0.52904278],
       [ 0.1496059 ,  0.3770697 ,  0.52692951],
       [ 0.15120289,  0.3804813 ,  0.52488853],
       [ 0.15285214,  0.38386729,  0.52291854],
       [ 0.15454421,  0.38722991,  0.52101815],
       [ 0.15627225,  0.39056998,  0.5191937 ],
       [ 0.15802555,  0.39389087,  0.5174364 ],
       [ 0.15979549,  0.39719482,  0.51574311],
       [ 0.16157425,  0.40048375,  0.51411214],
       [ 0.16335571,  0.40375871,  0.51254622],
       [ 0.16513234,  0.40702178,  0.51104174],
       [ 0.1668964 ,  0.41027528,  0.50959299],
       [ 0.16864151,  0.41352084,  0.50819797],
       [ 0.17036277,  0.41675941,  0.50685814],
       [ 0.1720542 ,  0.41999269,  0.50557008],
       [ 0.17370932,  0.42322271,  0.50432818],
       [ 0.17532301,  0.42645082,  0.50313007],
       [ 0.17689176,  0.42967776,  0.50197686],
       [ 0.17841013,  0.43290523,  0.5008633 ],
       [ 0.17987314,  0.43613477,  0.49978492],
       [ 0.18127676,  0.43936752,  0.49873901],
       [ 0.18261885,  0.44260392,  0.49772638],
       [ 0.18389409,  0.44584578,  0.49673978],
       [ 0.18509911,  0.44909409,  0.49577605],
       [ 0.18623135,  0.4523496 ,  0.494833  ],
       [ 0.18728844,  0.45561305,  0.49390803],
       [ 0.18826671,  0.45888565,  0.49299567],
       [ 0.18916393,  0.46216809,  0.49209268],
       [ 0.18997879,  0.46546084,  0.49119678],
       [ 0.19070881,  0.46876472,  0.49030328],
       [ 0.19135221,  0.47208035,  0.48940827],
       [ 0.19190791,  0.47540815,  0.48850845],
       [ 0.19237491,  0.47874852,  0.4876002 ],
       [ 0.19275204,  0.48210192,  0.48667935],
       [ 0.19303899,  0.48546858,  0.48574251],
       [ 0.19323526,  0.48884877,  0.48478573],
       [ 0.19334062,  0.49224271,  0.48380506],
       [ 0.19335574,  0.49565037,  0.4827974 ],
       [ 0.19328143,  0.49907173,  0.48175948],
       [ 0.19311664,  0.50250719,  0.48068559],
       [ 0.192864  ,  0.50595628,  0.47957408],
       [ 0.19252521,  0.50941877,  0.47842186],
       [ 0.19210087,  0.51289469,  0.47722441],
       [ 0.19159194,  0.516384  ,  0.47597744],
       [ 0.19100267,  0.51988593,  0.47467988],
       [ 0.19033595,  0.52340005,  0.47332894],
       [ 0.18959113,  0.5269267 ,  0.47191795],
       [ 0.18877336,  0.530465  ,  0.47044603],
       [ 0.18788765,  0.53401416,  0.46891178],
       [ 0.18693822,  0.53757359,  0.46731272],
       [ 0.18592276,  0.54114404,  0.46563962],
       [ 0.18485204,  0.54472367,  0.46389595],
       [ 0.18373148,  0.5483118 ,  0.46207951],
       [ 0.18256585,  0.55190791,  0.4601871 ],
       [ 0.18135481,  0.55551253,  0.45821002],
       [ 0.18011172,  0.55912361,  0.45615277],
       [ 0.17884392,  0.56274038,  0.45401341],
       [ 0.17755858,  0.56636217,  0.45178933],
       [ 0.17625543,  0.56998972,  0.44946971],
       [ 0.174952  ,  0.57362064,  0.44706119],
       [ 0.17365805,  0.57725408,  0.44456198],
       [ 0.17238403,  0.58088916,  0.4419703 ],
       [ 0.17113321,  0.58452637,  0.43927576],
       [ 0.1699221 ,  0.58816399,  0.43648119],
       [ 0.1687662 ,  0.5918006 ,  0.43358772],
       [ 0.16767908,  0.59543526,  0.43059358],
       [ 0.16667511,  0.59906699,  0.42749697],
       [ 0.16575939,  0.60269653,  0.42428344],
       [ 0.16495764,  0.6063212 ,  0.42096245],
       [ 0.16428695,  0.60993988,  0.41753246],
       [ 0.16376481,  0.61355147,  0.41399151],
       [ 0.16340924,  0.61715487,  0.41033757],
       [ 0.16323549,  0.62074951,  0.40656329],
       [ 0.16326148,  0.62433443,  0.40266378],
       [ 0.16351136,  0.62790748,  0.39864431],
       [ 0.16400433,  0.63146734,  0.39450263],
       [ 0.16475937,  0.63501264,  0.39023638],
       [ 0.16579502,  0.63854196,  0.38584309],
       [ 0.16712921,  0.64205381,  0.38132023],
       [ 0.168779  ,  0.64554661,  0.37666513],
       [ 0.17075915,  0.64901912,  0.37186962],
       [ 0.17308572,  0.65246934,  0.36693299],
       [ 0.1757732 ,  0.65589512,  0.36185643],
       [ 0.17883344,  0.65929449,  0.3566372 ],
       [ 0.18227669,  0.66266536,  0.35127251],
       [ 0.18611159,  0.66600553,  0.34575959],
       [ 0.19034516,  0.66931265,  0.34009571],
       [ 0.19498285,  0.67258423,  0.3342782 ],
       [ 0.20002863,  0.67581761,  0.32830456],
       [ 0.20548509,  0.67900997,  0.3221725 ],
       [ 0.21135348,  0.68215834,  0.31587999],
       [ 0.2176339 ,  0.68525954,  0.30942543],
       [ 0.22432532,  0.68831023,  0.30280771],
       [ 0.23142568,  0.69130688,  0.29602636],
       [ 0.23893914,  0.69424565,  0.28906643],
       [ 0.2468574 ,  0.69712255,  0.28194103],
       [ 0.25517514,  0.69993351,  0.27465372],
       [ 0.26388625,  0.70267437,  0.26720869],
       [ 0.27298333,  0.70534087,  0.25961196],
       [ 0.28246016,  0.70792854,  0.25186761],
       [ 0.29232159,  0.71043184,  0.2439642 ],
       [ 0.30253943,  0.71284765,  0.23594089],
       [ 0.31309875,  0.71517209,  0.22781515],
       [ 0.32399522,  0.71740028,  0.21959115],
       [ 0.33520729,  0.71952906,  0.21129816],
       [ 0.3467003 ,  0.72155723,  0.20298257],
       [ 0.35846225,  0.72348143,  0.19466318],
       [ 0.3704552 ,  0.72530195,  0.18639333],
       [ 0.38264126,  0.72702007,  0.17822762],
       [ 0.39499483,  0.72863609,  0.17020921],
       [ 0.40746591,  0.73015499,  0.1624122 ],
       [ 0.42001969,  0.73158058,  0.15489659],
       [ 0.43261504,  0.73291878,  0.14773267],
       [ 0.44521378,  0.73417623,  0.14099043],
       [ 0.45777768,  0.73536072,  0.13474173],
       [ 0.47028295,  0.73647823,  0.1290455 ],
       [ 0.48268544,  0.73753985,  0.12397794],
       [ 0.49497773,  0.73854983,  0.11957878],
       [ 0.5071369 ,  0.73951621,  0.11589589],
       [ 0.51913764,  0.74044827,  0.11296861],
       [ 0.53098624,  0.74134823,  0.11080237],
       [ 0.5426701 ,  0.74222288,  0.10940411],
       [ 0.55417235,  0.74308049,  0.10876749],
       [ 0.56550904,  0.74392086,  0.10885609],
       [ 0.57667994,  0.74474781,  0.10963233],
       [ 0.58767906,  0.74556676,  0.11105089],
       [ 0.59850723,  0.74638125,  0.1130567 ],
       [ 0.609179  ,  0.74719067,  0.11558918],
       [ 0.61969877,  0.74799703,  0.11859042],
       [ 0.63007148,  0.74880206,  0.12200388],
       [ 0.64030249,  0.74960714,  0.12577596],
       [ 0.65038997,  0.75041586,  0.12985641],
       [ 0.66034774,  0.75122659,  0.1342004 ],
       [ 0.67018264,  0.75203968,  0.13876817],
       [ 0.67990043,  0.75285567,  0.14352456],
       [ 0.68950682,  0.75367492,  0.14843886],
       [ 0.69900745,  0.75449768,  0.15348445],
       [ 0.70840781,  0.75532408,  0.15863839],
       [ 0.71771325,  0.75615416,  0.16388098],
       [ 0.72692898,  0.75698787,  0.1691954 ],
       [ 0.73606001,  0.75782508,  0.17456729],
       [ 0.74511119,  0.75866562,  0.17998443],
       [ 0.75408719,  0.75950924,  0.18543644],
       [ 0.76299247,  0.76035568,  0.19091446],
       [ 0.77183123,  0.76120466,  0.19641095],
       [ 0.78060815,  0.76205561,  0.20191973],
       [ 0.78932717,  0.76290815,  0.20743538],
       [ 0.79799213,  0.76376186,  0.21295324],
       [ 0.8066067 ,  0.76461631,  0.21846931],
       [ 0.81517444,  0.76547101,  0.22398014],
       [ 0.82369877,  0.76632547,  0.2294827 ],
       [ 0.832183  ,  0.7671792 ,  0.2349743 ],
       [ 0.8406303 ,  0.76803167,  0.24045248],
       [ 0.84904371,  0.76888236,  0.24591492],
       [ 0.85742615,  0.76973076,  0.25135935],
       [ 0.86578037,  0.77057636,  0.25678342],
       [ 0.87410891,  0.77141875,  0.2621846 ],
       [ 0.88241406,  0.77225757,  0.26755999],
       [ 0.89070781,  0.77308772,  0.27291122],
       [ 0.89898836,  0.77391069,  0.27823228],
       [ 0.90725475,  0.77472764,  0.28351668],
       [ 0.91550775,  0.77553893,  0.28875751],
       [ 0.92375722,  0.7763404 ,  0.29395046],
       [ 0.9320227 ,  0.77712286,  0.29909267],
       [ 0.94027715,  0.7779011 ,  0.30415428],
       [ 0.94856742,  0.77865213,  0.3091325 ],
       [ 0.95686038,  0.7793949 ,  0.31397459],
       [ 0.965222  ,  0.7800975 ,  0.31864342],
       [ 0.97365189,  0.78076521,  0.32301107],
       [ 0.98227405,  0.78134549,  0.32678728],
       [ 0.99136564,  0.78176999,  0.3281624 ],
       [ 0.99505988,  0.78542889,  0.32106514],
       [ 0.99594185,  0.79046888,  0.31648808],
       [ 0.99646635,  0.79566972,  0.31244662],
       [ 0.99681528,  0.80094905,  0.30858532],
       [ 0.9970578 ,  0.80627441,  0.30479247],
       [ 0.99724883,  0.81161757,  0.30105328],
       [ 0.99736711,  0.81699344,  0.29725528],
       [ 0.99742254,  0.82239736,  0.29337235],
       [ 0.99744736,  0.82781159,  0.28943391],
       [ 0.99744951,  0.83323244,  0.28543062],
       [ 0.9973953 ,  0.83867931,  0.2812767 ],
       [ 0.99727248,  0.84415897,  0.27692897],
       [ 0.99713953,  0.84963903,  0.27248698],
       [ 0.99698641,  0.85512544,  0.26791703],
       [ 0.99673736,  0.86065927,  0.26304767],
       [ 0.99652358,  0.86616957,  0.25813608],
       [ 0.99622774,  0.87171946,  0.25292044],
       [ 0.99590494,  0.87727931,  0.24750009],
       [ 0.99555225,  0.88285068,  0.2418514 ],
       [ 0.99513763,  0.8884501 ,  0.23588062],
       [ 0.99471252,  0.89405076,  0.2296837 ],
       [ 0.99421873,  0.89968246,  0.2230963 ],
       [ 0.99370185,  0.90532165,  0.21619768],
       [ 0.99313786,  0.91098038,  0.2088926 ],
       [ 0.99250707,  0.91666811,  0.20108214],
       [ 0.99187888,  0.92235023,  0.19290417],
       [ 0.99110991,  0.92809686,  0.18387963],
       [ 0.99042108,  0.93379995,  0.17458127],
       [ 0.98958484,  0.93956962,  0.16420166],
       [ 0.98873988,  0.94533859,  0.15303117],
       [ 0.98784836,  0.95112482,  0.14074826],
       [ 0.98680727,  0.95697596,  0.12661626]]

    test_cm = LinearSegmentedColormap.from_list(__file__, cm_data)

    return test_cm

#############################################################################################################################

def spont_FR_analysis(models_results_dict, dataset_ID, dataset_params):
    if dataset_ID == 'NS3' or dataset_ID == 'NS3_PEG':
        spont_file = 'Data/' + dataset_ID + '/' + dataset_ID + '_prebuild_spontaneous.pckl'
    elif dataset_ID == 'NS2_include_single':
        spont_file = 'Data/NS2_raw/NS2_prebuild_spontaneous.pckl'

    f = open(spont_file, 'rb')
    spont_spikes = pickle.load(f)
    f.close()
    
    nIDs = dataset_params['neurons_ID_sess']
    spont_FRs = []
    
    for nID, sess in nIDs:
        sess -= 1
        spont_times = spont_spikes['silence_times'][sess]
        sess_nIDs = spont_spikes['neuron_IDs'][sess]
        sess_spikes = spont_spikes['all_sess_resps'][sess]
        
        total_spont_time = sum([time[1] - time[0] for time in spont_times])
        
        nID_sess_idx = sess_nIDs.index(nID)
        nID_spikes = sess_spikes[nID_sess_idx]
        num_spikes = sum([len(silence_spikes) for silence_spikes in nID_spikes])
        
        spont_FRs.append(num_spikes/total_spont_time)
    
    for model in models_results_dict:
        CCnorms = models_results_dict[model][2]
        
        f, ax = plt.subplots()
        ax.scatter(spont_FRs, CCnorms)
        ax.set_xlabel('spontaneous FR')
        ax.set_ylabel('CCnorm')
        ax.set_title(dataset_ID + ' - ' + model)
        
        X2 = sm.add_constant(spont_FRs) # add bias to independent variable
        est = sm.OLS(CCnorms, X2) # Ordinary Least Squares fit
        est2 = est.fit()
        a = est2.params[0] # bias
        b = est2.params[1] # slope
        # x-sequence for plotting line of best fit
        xseq = np.linspace(np.min(spont_FRs) - 0.05, np.max(spont_FRs) + 0.05, num=100)
        
        ax.plot(xseq, a+b*xseq, color='b')
        ax.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
        
        for other_model in models_results_dict:
            if other_model != model:
                CCnorm_diff = models_results_dict[other_model][2] - CCnorms
                
                f, ax = plt.subplots()
                ax.scatter(spont_FRs, CCnorm_diff)
                #ax.set_xlabel('spontaneous FR')
                #ax.set_ylabel('CCnorm diff')
                
                X2 = sm.add_constant(spont_FRs) # add bias to independent variable
                est = sm.OLS(CCnorm_diff, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                b = est2.params[1] # slope
                # x-sequence for plotting line of best fit
                xseq = np.linspace(np.min(spont_FRs) - 0.05, np.max(spont_FRs) + 0.05, num=100)

                ax.plot(xseq, a+b*xseq, color='k')
                ax.spines[['right', 'top']].set_visible(False)
                #ax.set_xticks([0, 20, 40, 60, 80, 100]) # NS3
                #ax.set_xticks([0, 10, 20, 30, 40, 50]) # NS3-PEG
                ax.set_xticks([0, 10, 20, 30, 40, 50, 60]) # NS2
                #ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6]) # NS3
                #ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5]) # NS3-PEG
                ax.set_yticks([-0.2, 0, 0.2, 0.4]) # NS2
                ax.xaxis.set_tick_params(labelcolor='none')
                ax.yaxis.set_tick_params(labelcolor='none')
                #ax.set_title(dataset_ID + ' - ' + other_model + ' vs ' + model + ' - p = ' + str(est2.pvalues[1]))

#############################################################################################################################

def plot_param_count(models_param_count, rc_LSTM_param_count, models_results_dict, rc_LSTM_performance, dataset_ID):
    models_med_results = {model_ID: np.nanmedian(models_results_dict[model_ID][2]) for model_ID in models_results_dict}
    
    if dataset_ID == 'NS3' or dataset_ID == 'NS2_include_single':
        models_param_count.update(rc_LSTM_param_count)
        models_med_results.update(rc_LSTM_performance)
    
    model_colours = {'pop_LN': 'tab:blue', 'pop_NRF': 'tab:blue', 'pop_RNN': 'tab:green', 'pop_MGU': 'tab:red', 
                     'pop_GRU': 'tab:red', 'pop_LSTM': 'tab:red', 'pop_subLSTM': 'tab:red', 'pop_f_LSTM': 'tab:red', 
                     'pop_rc_LSTM': 'tab:red', 'rc_LSTM 32': 'tab:red', 'rc_LSTM 64': 'tab:red', 'rc_LSTM 128': 'tab:red', 
                     'rc_LSTM 512': 'tab:red', 'oneD_CNN': 'tab:blue', 'oneD_x2_CNN': 'tab:blue', 'twoD_CNN': 'tab:blue', 
                     'PM twoD_CNN': 'tab:blue', 'PM oneD_CNN': 'tab:blue'}
    
    f, ax = plt.subplots()
    f.set_size_inches(12, 4.8)
    for model_ID in models_param_count:
        ax.scatter(models_param_count[model_ID], models_med_results[model_ID], color=model_colours[model_ID], s=50)
    #ax.set_xlabel('Parameter count')
    #ax.set_ylabel('Median CCnorm')
    #ax.set_title(dataset_ID)
    if dataset_ID == 'NS3' or dataset_ID == 'NS2_include_single':
        ax.set_xticks([0, 100000, 200000, 300000, 400000, 500000, 600000])
        ax.set_yticks([0.5, 0.6, 0.7])
    else:
        ax.set_xlim([0, 350000])
        ax.set_xticks([0, 100000, 200000, 300000])
        ax.set_yticks([0.45, 0.5, 0.55, 0.6])
    ax.spines[['right', 'top']].set_visible(False)
    ax.xaxis.set_tick_params(labelcolor='none')
    ax.yaxis.set_tick_params(labelcolor='none')

#############################################################################################################################

def plot_t_corr(t_corr_resps, t_corr_stims, n_neurons, best_lambdas, dataset_ID, models, pre_act, all_f):
    if pre_act:
        gate_key = 't_pre_act'
    else:
        gate_key = 't'
    
    if dataset_ID == 'NS3':
        mode_lamb = 15
    else:
        mode_lamb = 14
        
    redblue_cmap = redblue(256)
        
    for model_ID in models:
        model_corr_resps = t_corr_resps[model_ID][0]
        model_corr_stims = t_corr_stims[model_ID][0]
        
        if all_f:
            redblue_cmap.set_bad('black')
            redblue_cmap.set_over('green')
            
            f, ax = plt.subplots(figsize=(12, 5))
            '''im = ax.imshow(model_corr_stims[mode_lamb][gate_key], cmap='Greens', aspect='auto', interpolation='none', 
                      origin='lower', vmin=-1, vmax=1)'''
            im = ax.imshow(model_corr_stims[mode_lamb][gate_key], cmap=redblue_cmap, aspect='auto', interpolation='none', 
                      origin='lower', vmin=-1, vmax=1)
            #cbar = f.colorbar(im, ax=ax)
            #ax.set_ylabel('h step')
            #ax.set_xlabel('f chan')
            #ax.set_title(dataset_ID + ', ' + model_ID + ', input gate & stimuli correlation')
            ax.xaxis.set_tick_params(labelcolor='none')
            ax.yaxis.set_tick_params(labelcolor='none')
            
            corr_stims = model_corr_stims[mode_lamb][gate_key]
            
            open_gate_idxs = np.argwhere(corr_stims == 1.5)
            
            for idx in range(np.size(open_gate_idxs, 0)):
                corr_stims[open_gate_idxs[idx][0], open_gate_idxs[idx][1]] = np.nan
            
            f_av_corr = np.nanmean(corr_stims, 1)
            h_av_corr = np.nanmean(corr_stims, 0)
            
            f, ax = plt.subplots(figsize=(6, 2.5))
            ax.scatter(np.arange(1, len(f_av_corr)+1 , 1), f_av_corr)
            ax.set_yticks([0, 0.2, 0.4]) # NS3, NS3 - 0.1
            #ax.set_yticks([-0.3, 0, 0.3, 0.6]) # NS3_PEG
            #ax.set_yticks([-0.2, 0, 0.2]) # NS2
            #ax.set_yticks([0, 0.2]) # NS2 - 0.1
            #ax.set_ylabel('CC')
            #ax.set_title(dataset_ID + ', ' + model_ID + ', input gate & stimuli correlation - f averaged')
            
            X2 = sm.add_constant(np.arange(1, len(f_av_corr)+1 , 1)) # add bias to independent variable
            est = sm.OLS(f_av_corr, X2) # Ordinary Least Squares fit
            est2 = est.fit()
            a = est2.params[0] # bias
            b = est2.params[1] # slope
            # x-sequence for plotting line of best fit
            xseq = np.linspace(np.min(np.arange(1, len(f_av_corr)+1 , 1))-0.05, 
                               np.max(np.arange(1, len(f_av_corr)+1 , 1))+0.05, num=100)
            ax.plot(xseq, a+b*xseq, color='b')
            ax.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            ax.xaxis.set_tick_params(labelcolor='none')
            ax.yaxis.set_tick_params(labelcolor='none')
            
            f, ax = plt.subplots(figsize=(6, 2.5))
            ax.scatter(np.arange(1, len(h_av_corr)+1 , 1), h_av_corr)
            ax.set_xticks([1, 4, 7, 10, 13, 16]) # NS3
            ax.set_yticks([0, 0.2, 0.4]) # NS3, NS3 - 0.1
            #ax.set_yticks([-0.2, 0, 0.2, 0.4]) # NS3_PEG
            #ax.set_yticks([-0.4, -0.2, 0, 0.2]) # NS2
            #ax.set_yticks([-0.2, 0, 0.2]) # NS2 - 0.1
            #ax.set_ylabel('CC')
            #ax.set_title(dataset_ID + ', ' + model_ID + ', input gate & stimuli correlation - h averaged')
            
            X2 = sm.add_constant(np.arange(1, len(h_av_corr)+1 , 1)) # add bias to independent variable
            est = sm.OLS(h_av_corr, X2) # Ordinary Least Squares fit
            est2 = est.fit()
            a = est2.params[0] # bias
            b = est2.params[1] # slope
            # x-sequence for plotting line of best fit
            xseq = np.linspace(np.min(np.arange(1, len(h_av_corr)+1 , 1))-0.05, 
                               np.max(np.arange(1, len(h_av_corr)+1 , 1))+0.05, num=100)
            ax.plot(xseq, a+b*xseq, color='b')
            ax.spines[['right', 'top']].set_visible(False) # set top and right plot axes to invisible (prevents box layout)
            ax.xaxis.set_tick_params(labelcolor='none')
            ax.yaxis.set_tick_params(labelcolor='none')

        else:
            f, ax = plt.subplots()
            ax.scatter(np.arange(len(model_corr_stims[mode_lamb][gate_key])), model_corr_stims[mode_lamb][gate_key])
            ax.set_ylabel('CC')
            ax.set_title(dataset_ID + ', ' + model_ID + ', input gate & stimuli correlation')
            #ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75]) # NS3
            #ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8]) # NS3_PEG
            #ax.set_yticks([-0.8, -0.4, 0, 0.4]) # NS2
            #ax.xaxis.set_tick_params(labelcolor='none')
            #ax.yaxis.set_tick_params(labelcolor='none')
            ax.spines[['right', 'top']].set_visible(False)
