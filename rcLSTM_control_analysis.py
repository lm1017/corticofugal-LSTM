import pickle
import scipy
import numpy as np

n_HUs = [32, 64, 128, 256, 512]

model = 'pop_rc_LSTM'
dataset = 'NS2_include_single'

ttest_dict = {}
ranktest_dict = {}

for HU_num_i in n_HUs:
    filename_i = 'Results/rc_LSTM_controls/Test/final_fit_' + model + '_' + \
        dataset + '_' + str(HU_num_i) + '.pckl'
    f = open(filename_i, 'rb')
    obj = pickle.load(f)
    f.close()
    
    results_i = obj[1]
    print(np.nanmean(results_i))
    print(np.nanmedian(results_i))
    
    for HU_num_j in n_HUs:
        filename_j = 'Results/rc_LSTM_controls/Test/final_fit_' + model + '_' + dataset + '_' + str(HU_num_j) + '.pckl'
        f = open(filename_j, 'rb')
        obj = pickle.load(f)
        f.close()
        
        results_j = obj[1]
        
        ttest_dict[str(HU_num_i) + ' vs ' + str(HU_num_j)] = [scipy.stats.ttest_rel(results_i, results_j)[0], \
                                                              scipy.stats.ttest_rel(results_i, results_j)[1]]
        if HU_num_i != HU_num_j:
            ranktest_dict[str(HU_num_i) + ' vs ' + str(HU_num_j)] = [scipy.stats.wilcoxon(results_i, results_j)[0], 
                                                                     scipy.stats.wilcoxon(results_i, results_j)[1]]