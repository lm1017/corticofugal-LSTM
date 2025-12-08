def tensorize_mod(X_ft, n_h):
    '''
    Add a history dimension to a 2D stimulus grid - this is a Python adaptation of the tensorize_mod.m MATLAB function 
    written by Monzilur Rahman and available at 
    https://github.com/monzilur/cochlear_models/blob/master/LN_model/tensorize_mod.m
    
    Parameters
    ----------
    X_ft : stimulus cochleagram, freq x time
    n_h : number of history steps
     
    Returns
    -------
    X_fht : tensorised stimulus, freq x history x time
    '''
    
    import numpy as np
    
    n_f = np.size(X_ft, 0)
    n_t = np.size(X_ft, 1)
    
    n_start = 0
    n_end = n_h
    
    X_fht = np.zeros((n_f, n_h, n_t-n_h+1))
    
    for i in range(n_t-n_h+1):
        X_fht[:, 0:n_h, i] = X_ft[:, n_start:n_end]
        n_start = n_start + 1
        n_end = n_end + 1
        
    return X_fht