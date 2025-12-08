import numpy as np
import random
import torch
import torch.nn as nn
import all_models
from util_funcs import get_stim_response_grid_concat, calc_CC_norm

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pop_cross_val_loop(dataset, lambda_sequence, model_ID, params, loss_fcn, optim, num_epochs, stim_batch_size, 
                       resp_batch_size, learning_rate, early_stop, grad_set_to_none, grad_clipping, adaptive_lr, val_stimuli, 
                       train_stimuli, neuron_data, stimuli_cochleagrams, n_h, bin_size, norm, mean_cv, std_cv, clip_bins, 
                       shift_bins, zero_padding, padding_bins, model_type, t_axis, split):
    '''
    Parameters
    ----------
    dataset : whole dataset
    lambda_sequence : array of L1 regularisation strength values
    model_ID : model name
    params : model parameters
    loss_fcn : loss function
    optim : optimiser name
    num_epochs : number of training epochs
    stim_batch_size : size of stimulus minibatch (in this case, in time bins)
    resp_batch_size : size of response minibatch (in this case, in time bins)
    learning_rate : optimisation learning rate
    early_stop : boolean variable, controls use of early stopping
    grad_set_to_none : boolean variable, controls whether gradients are set to 0 or None when resetting in optimisation
    grad_clipping : boolean variable, controls use of gradient clipping
    adaptive_lr : boolean variable, controls whether learning rate is adaptive (dependent on tuning loss) during training
    val_stimuli : validation stimuli
    train_stimuli : training stimuli
    neuron_data : spike time responses of all neurons to each repeat of each stimulus (dimensions -> (neurons, stimuli, 
    repeats))
    stimuli_cochleagrams : array of cochleagrams (dimensions -> (f, t, stimuli))
    n_h : number of history steps used in stimulus tensorisation
    bin_size : time bin size used to compute cochleagrams and PSTHs
    norm : type of stimulus normalisation to zero-mean and unit variance, either 'all' or 'each'
    mean_cv : mean over all stimuli in the cross-validation set
    std_cv : standard deviation over all stimuli in the cross-validation set
    clip_bins : number of bins to be clipped from the start of the stimuli and responses
    shift_bins : number of bins that the response is to be shifted by with respect to the stimuli
    zero_padding : boolean variable, indicates whether stimuli were zero-padded
    padding_bins : number of bins added to zero-padded stimuli
    model_type : model type - convolutional or not convolutional
    t_axis : time axis used to compute cochleagrams
    
    Returns
    -------
    ccnorms : cross-validation CCnorms for each model run (each fold and each lambda value)
    model_fits_info : fitted model parameters for each model run
    loss_lr_curves : training loss, tuning loss, and learning rate over training epochs for each model run
    '''
    
    train_stim_dataset = dataset['train_stim'] # dimensions -> (folds, f, history, concatenated t)
    train_resp_dataset = dataset['train_resp'] # dimensions -> (neurons, folds, concatenated t)
    
    tune_stim_dataset = dataset['tune_stim'] # dimensions -> (folds, f, history, concatenated t)
    tune_resp_dataset = dataset['tune_resp'] # dimensions -> (neurons, folds, concatenated t)
    
    val_stim_dataset = dataset['val_stim'] # dimensions -> (folds, f, history, concatenated t)
    val_resp_dataset = dataset['val_resp'] # dimensions -> (neurons, folds, concatenated t)
    
    train_val_stim_dataset = dataset['train_val_stim'] # dimensions -> (folds, f, history, concatenated t)
    train_val_resp_dataset = dataset['train_val_resp'] # dimensions -> (neurons, folds, concatenated t)
    
    n_neurons = train_resp_dataset.size(0)
    folds = train_stim_dataset.size(0)
    
    # tensors to hold CCnorms
    neuron_val_all_lamb_performance = np.zeros((n_neurons, folds, len(lambda_sequence)))
    neuron_train_all_lamb_performance = np.zeros((n_neurons, folds, len(lambda_sequence)))
    
    # tensors to hold training and tuning loss, and learning rate values
    training_loss = np.zeros((folds, len(lambda_sequence), num_epochs))
    tuning_loss = np.zeros((folds, len(lambda_sequence), num_epochs))
    learning_rate_seq = np.zeros((folds, len(lambda_sequence), num_epochs))

    model_fits_info = []
    
    for fID in range(folds): # loop over folds
        print("fID = ")
        print(fID)
        
        f_train_stim_data = train_stim_dataset[fID] # training fold stimulus data
        f_train_resp_data = train_resp_dataset[:, fID] # training fold response data, for all neurons
        
        f_tune_stim_data = tune_stim_dataset[fID] # tuning fold stimulus data
        f_tune_resp_data = tune_resp_dataset[:, fID] # tuning fold response data, for all neurons
        
        f_val_stim_data = val_stim_dataset[fID] # validation fold stimulus data
        f_val_resp_data = val_resp_dataset[:, fID] # validation fold response data, for all neurons
        
        f_train_val_stim_data = train_val_stim_dataset[fID] # training/validation fold stimulus data
        f_train_val_resp_data = train_val_resp_dataset[:, fID] # training/validation fold response data, for all neurons

        fold_model_fit_info = []
        
        for lamb in range(len(lambda_sequence)): # loop over regularisation parameters
            print("lambda = ")
            print(lamb)
            
            model = getattr(all_models, model_ID)(params) # define and initialise model from model_ID
            model.to(device)
            loss = loss_fcn
            optimiser = getattr(torch.optim, optim)(model.parameters(), lr=learning_rate) # define optimiser from optim

            # train model and get fitted parameters, as well as training loss, tuning loss, and learning rate over epochs
            fitted_params, train_loss, tune_loss, lr_sequence = train_loop(model, loss, optimiser, num_epochs, 
                                                                           stim_batch_size, resp_batch_size, 
                                                                           f_train_stim_data, f_train_resp_data, 
                                                                           f_tune_stim_data, f_tune_resp_data, 
                                                                           lambda_sequence[lamb], grad_set_to_none, 
                                                                           grad_clipping, adaptive_lr, early_stop)
            
            training_loss[fID, lamb, :] = train_loss
            tuning_loss[fID, lamb, :] = tune_loss
            learning_rate_seq[fID, lamb, :] = lr_sequence

            # if early stopping is used, reload fitted parameters from best epoch; no need for this if early stopping is not
            # in use, as the latest fitted parameters will be the ones to use
            if early_stop:
                model.reload(fitted_params)
            
            # evaluate model on validation data, unused in training - get CCnorms for each neuron on validation data
            val_ccnorms = val_loop(model, loss, f_val_stim_data, f_val_resp_data, n_neurons, folds, fID, val_stimuli, 
                                   neuron_data, stimuli_cochleagrams, params['his_steps'], bin_size, norm, mean_cv, std_cv, 
                                   clip_bins, shift_bins, zero_padding, padding_bins, model_type, t_axis, split)

            # evaluate model on training data - get CCnorms for each neuron on training data
            train_ccnorms = val_loop(model, loss, f_train_val_stim_data, f_train_val_resp_data, n_neurons, folds, fID, 
                                     train_stimuli, neuron_data, stimuli_cochleagrams, params['his_steps'], bin_size, norm, 
                                     mean_cv, std_cv, clip_bins, shift_bins, zero_padding, padding_bins, model_type, t_axis,
                                     split)

            neuron_val_all_lamb_performance[:, fID, lamb] = val_ccnorms
            neuron_train_all_lamb_performance[:, fID, lamb] = train_ccnorms
            
            model_info = model.get_params()
            model_info_np = {item: value.detach().cpu().numpy() for (item, value) in model_info.items()}
            
            fold_model_fit_info.append([model_info_np, training_loss[fID, lamb], tuning_loss[fID, lamb], 
                                        learning_rate_seq[fID, lamb]])
             
        model_fits_info.append(fold_model_fit_info)   
    
    ccnorms = {'val': neuron_val_all_lamb_performance, 'train': neuron_train_all_lamb_performance}
    loss_lr_curves = {'train' : training_loss, 'tune' : tuning_loss, 'lr' : learning_rate_seq}
    
    return ccnorms, model_fits_info, loss_lr_curves

#############################################################################################################################

def train_loop(model, criterion, optimiser, num_epochs, stim_batch_size, resp_batch_size, train_stim_data, train_resp_data, 
               tune_stim_data, tune_resp_data, lamb, grad_set_to_none, grad_clipping, adaptive_lr, early_stop):
    '''
    Parameters
    ----------
    model : defined and initialised model
    criterion : loss function
    optimiser : defined optimiser
    num_epochs : number of training epochs
    stim_batch_size : size of stimulus minibatch (in this case, in time bins)
    resp_batch_size : size of response minibatch (in this case, in time bins)
    train_stim_data : training stimuli (dimensions -> (f, history, concatenated t))
    train_resp_data : training responses (dimensions -> (neurons, concatenated t))
    tune_stim_data : tuning stimuli (dimensions -> (f, history, concatenated t))
    tune_resp_data : tuning responses (dimensions -> (neurons, concatenated t))
    lamb : regularisation strength
    grad_set_to_none : boolean variable, controls whether gradients are set to 0 or None when resetting in optimisation
    grad_clipping : boolean variable, controls use of gradient clipping
    adaptive_lr : boolean variable, controls whether learning rate is adaptive (dependent on tuning loss) during training
    early_stop : boolean variable, controls use of early stopping
    
    Returns
    -------
    fitted_params : fitted model parameters (weights and biases)
    train_loss : training loss over training epochs
    tune_loss : tuning loss over training epochs
    lr_seq : learning rate over training epochs
    '''
    
    # tuning and early stopping variables
    best_tuning_loss = np.Inf # best (smallest) tuning loss - initialised to infinity
    best_epoch = -1 # training epoch showing smallest loss - initialised to -1
    best_network = None # fitted parameters of model at epoch producing best tuning loss - initialised to None
    epochs_since_best_tuning_loss = 0 # number of training epochs since best_epoch - initialised to 0
    
    train_loss = np.zeros((num_epochs))
    train_loss_no_reg = np.zeros((num_epochs))
    tune_loss = np.zeros((num_epochs))
    tune_loss_no_reg = np.zeros((num_epochs))
    lr_seq = np.zeros((num_epochs))
    
    early_stop_flag = False # boolean variable, indicates whether model has been early-stopped
    
    for epoch in range(num_epochs): # loop over training epochs
        '''if epoch % 20 == 0:
            print(epoch)'''

        minibatch_num = int(np.floor(train_stim_data.size(2)/stim_batch_size)) # number of minibatches in training data
        minibatch_list = list(range(minibatch_num)) # list of minibatches
        random.shuffle(minibatch_list) # order in which minibatches are input for model training is randomised at each epoch

        loss_temp = [] # running loss for epoch
        loss_temp_no_reg = [] # running loss for epoch, ignoring regularisation (useful to explore model differences)

        for mb_ID in minibatch_list: # loop over minibatches
            # input stimuli, dimensions -> (f, history, t)
            mb_inputs = train_stim_data[:, :, stim_batch_size*(mb_ID):stim_batch_size*(mb_ID+1)]
            # target responses, dimensions -> (neurons, t)
            target_data = train_resp_data[:, resp_batch_size*(mb_ID):resp_batch_size*(mb_ID+1)]

            # get model output (response prediction) and target responses reshaped for loss calculation; model stage is set
            # to "train"
            # outputs: model response predictions, dimensions -> (t, neurons)
            # targets: target responses, dimensions -> (t, neurons)
            outputs, targets = model.forward_loop(mb_inputs, target_data, 'train')

            reg_loss = model.regul(lamb) # regularisation contribution to loss
            loss = criterion(outputs, targets) + reg_loss # full loss including criterion and regularisation
            
            # append minibatch losses (with and without regularisation) to epoch running loss lists
            loss_temp.append(loss.item())
            loss_temp_no_reg.append(criterion(outputs, targets).item())
            
            optimiser.zero_grad(set_to_none=grad_set_to_none) # reset gradients of all optimised parameters
            
            loss.backward() # loss gradient calculation via backpropagation
            if grad_clipping is True: # if gradient clipping is used, clip gradient values at 1
                nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0) # gradient clipping
            optimiser.step() # optimisation step, defined by parameter gradients and learning rate
            
        train_loss[epoch] = np.mean(loss_temp) # epoch loss is the average of all minibatch losses
        train_loss_no_reg[epoch] = np.mean(loss_temp_no_reg)
        lr_seq[epoch] = optimiser.param_groups[-1]['lr'] # learning rate value at epoch
        
        with torch.no_grad():
            # input tuning stimuli, dimensions -> (f, history, concatenated t)
            # target responses, dimensions -> (neurons, concatenated t)
            # get model output (response prediction) and target responses reshaped for loss calculation; model stage is set
            # to "val"
            # outputs: model response predictions, dimensions -> (concatenated t, neurons)
            # targets: target responses, dimensions -> (concatenated t, neurons)
            outputs, targets = model.forward_loop(tune_stim_data, tune_resp_data, 'val')

            reg_loss = model.regul(lamb) # regularisation contribution to loss
            loss = criterion(outputs, targets) + reg_loss # full loss including criterion and regularisation
            
            tune_loss[epoch] = loss.item() # epoch tuning loss
            tune_loss_no_reg[epoch] = criterion(outputs, targets).item() # epoch tuning loss without regularisation
            
            # loss gradient: difference between current tuning loss and tuning loss at previous training epoch
            if epoch > 0:
                loss_gradient = loss.item() - tune_loss[epoch-1]
            else:
                loss_gradient = 0

            # if an adaptive learning rate schedule is used, change learning rate depending on loss gradient
            if adaptive_lr is True:
                optimiser.vary_LR(loss_gradient) # to call this, adapted (from torch.optim) optimisers must be used
            
            # if tuning loss is smaller than the current best tuning loss, update tuning loss-related variables
            if loss.item() < best_tuning_loss:
                best_epoch = epoch # current epoch becomes best epoch
                best_tuning_loss = loss.item() # current loss becomes best tuning loss
                epochs_since_best_tuning_loss = 0 # no epochs have passed since best epoch
                
                best_network = model.get_params() # model parameters at this epoch become best model parameters
            else: # if tuning loss is worse than the best one, simply update the number of epochs since the best one
                epochs_since_best_tuning_loss = epochs_since_best_tuning_loss + 1

            # if early stopping is used, and more than 1/10 of epochs have passed and more than 1/10 of epochs have passed 
            # since the best epoch, the model is early-stopped
            if early_stop and epoch > (num_epochs/10) and epochs_since_best_tuning_loss > (num_epochs/10):
                fitted_params = best_network # fitted parameters are those extracted from best network
                early_stop_flag = True # model has been early-stopped
                break # break epoch loop

    if early_stop_flag is False: # if model has not been early-stopped, extract fitted parameters from latest version
        fitted_params = model.get_params()
         
    return fitted_params, train_loss, tune_loss, lr_seq

#############################################################################################################################

def val_loop(model, criterion, stim_data, resp_data, n_neurons, folds, fID, stimuli, neuron_data_list, stimuli_cochleagrams, 
             n_h, bin_size, norm, mean_cv, std_cv, clip_bins, shift_bins, zero_padding, padding_bins, model_type, t_axis,
             split):
    '''
    Parameters
    ----------
    model : defined and initialised model
    criterion : defined loss function
    stim_data : stimulus data to validate model performance (dimensions -> (f, history, concatenated t))
    resp_data : response data to validate model performance (dimensions -> (neurons, concatenated t))
    n_neurons : number of neurons
    folds : number of folds
    fID : fold
    stimuli : list of stimuli to use in validation
    neuron_data_list : spike times of each neuron in response to each repeat of each stimulus
    stimuli_cochleagrams : all stimulus cochleagrams (dimensions -> (f, t, stimuli))
    n_h : number of history steps used in tensorisation
    bin_size : time bin size used to compute cochleagrams and PSTHs
    norm : type of stimulus normalisation to zero-mean and unit variance, either 'all' or 'each'
    mean_cv : mean over all stimuli in the cross-validation set
    std_cv : standard deviation over all stimuli in the cross-validation set
    clip_bins : number of bins to be clipped from the start of the stimuli and responses
    shift_bins : number of bins that the response is to be shifted by with respect to the stimuli
    zero_padding : boolean variable, indicates whether stimuli were zero-padded
    padding_bins : number of bins added to zero-padded stimuli
    model_type : model type - convolutional or not convolutional
    t_axis : time axis used to compute cochleagrams

    Returns
    -------
    all_neuron_ccnorms : CCnorms for each neuron calculated on stim_data and resp_data
    '''
    
    all_neuron_ccnorms = np.zeros((n_neurons))
    
    if model_type == 'not_conv':
        with torch.no_grad():
            # input stimuli, dimensions -> (f, history, concatenated t)
            # target responses, dimensions -> (neurons, concatenated t)
            # get model output (response prediction) and target responses reshaped for loss calculation; model stage is set
            # to "val"
            # outputs: model response predictions, dimensions -> (concatenated t, neurons)
            # targets: target responses, dimensions -> (concatenated t, neurons)
            outputs, targets = model.forward_loop(stim_data, resp_data, 'val')
            
            for nID in range(n_neurons):
                # convert model response prediction for one neuron to NumPy array
                np_outputs = np.squeeze(outputs[:, nID].numpy(force=True))

                # list of spike time responses of neuron to stimuli in fold, for each repeat
                if fID is None:
                    data = [neuron_data_list[nID][stim] for stim in stimuli]
                else:
                    data = [neuron_data_list[nID][stim] for stim in stimuli[fID]]

                # get NumPy array of target responses of one neuron to each repeat of stimuli, dimensions -> (concatenated t, 
                # repeats)
                _, np_targets = get_stim_response_grid_concat(data, stimuli_cochleagrams, stimuli, fID, n_h, bin_size, 'val', 
                                                              norm, mean_cv, std_cv, clip_bins, shift_bins, zero_padding, 
                                                              padding_bins, t_axis)

                # get CCnorm, CCabs, and CCmax for neuron
                ccnorm, ccabs, ccmax = calc_CC_norm(np_targets, np_outputs)
                all_neuron_ccnorms[nID] = ccnorm
                    
    if model_type == 'conv': # if model is convolutional, model output has different dimensions     
        with torch.no_grad():
            # input stimuli, dimensions -> (f, history, concatenated t)
            # target responses, dimensions -> (neurons, concatenated t)
            # get model output (response prediction) and target responses reshaped for loss calculation; model stage is set
            # to "val"
            # outputs: model response predictions, dimensions -> (n, t, neurons)
            # targets: target responses, dimensions -> (n, t, neurons) where n is stimuli

            outputs, targets = model.forward_loop(stim_data, resp_data, 'val')

            for nID in range(n_neurons):
                # list of spike time responses of neuron to stimuli in fold, for each repeat
                if fID is None:
                    data = [neuron_data_list[nID][stim] for stim in stimuli]
                else:
                    data = [neuron_data_list[nID][stim] for stim in stimuli[fID]]
                
                # get NumPy array of target responses of one neuron to each repeat of stimuli, dimensions -> (concatenated t, 
                # repeats)
                _, np_targets = get_stim_response_grid_concat(data, stimuli_cochleagrams, stimuli, fID, n_h, bin_size, 'val', 
                                                              norm, mean_cv, std_cv, clip_bins, shift_bins, zero_padding, 
                                                              padding_bins, t_axis)
                
                resp_size = int(np.size(np_targets, 0)/outputs.size(0)) # response length (in time bins)
                
                # difference in length between actual and predicted response
                if split != 1:
                    resp_output_diff = int((resp_size - outputs.size(1))/(1/split-1))
                else:
                    resp_output_diff = resp_size - outputs.size(1)
                
                # NumPy arrays to hold concatenated predicted and actual responses; concatenation step is necessary because
                # convolutional network output has stimuli separated
                predicted = np.zeros((outputs.size(1)*outputs.size(0))) # dimensions -> (concatenated t)
                # dimensions -> (concatenated t, repeats)
                actual = np.zeros((outputs.size(1)*outputs.size(0), len(data[0])))
                
                for sID in range(outputs.size(0)):
                    # firing rate prediction of neuron in response to stimulus
                    predicted[sID*outputs.size(1):(sID+1)*outputs.size(1)] = outputs[sID, :, nID].detach().cpu().numpy()
                    
                    resp_sID = np_targets[sID*resp_size:(sID+1)*resp_size, :] # actual response of neuron to stimulus

                    # length difference taken into account by clipping bins at the start of the actual response
                    actual[sID*outputs.size(1):(sID+1)*outputs.size(1), :] = resp_sID[resp_output_diff:, :]

                # get CCnorm, CCabs, and CCmax for neuron
                ccnorm, ccabs, ccmax = calc_CC_norm(actual, predicted)
                all_neuron_ccnorms[nID] = ccnorm
    
    return all_neuron_ccnorms