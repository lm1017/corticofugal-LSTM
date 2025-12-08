import numpy as np
import scipy.stats as stats
import pickle

model_list = ['pop_LSTM', 'pop_subLSTM', 'pop_f_LSTM', 'pop_rc_LSTM']
n_HUs = 150

loss_array = []
avg_loss = []
se_loss = []
for model_ID in model_list:
    results_file = 'Fit_files/Test/8_150_test_no_loop_results_' + model_ID + '_8_5_250.pckl'
    f = open(results_file, 'rb')
    obj = pickle.load(f)
    f.close()

    test_loss = obj[2]
    loss_array.append(test_loss)
    avg_loss.append(np.nanmean(test_loss))
    se_loss.append(np.nanstd(test_loss, ddof=1)/np.sqrt(len(test_loss)))

ttest_dict = {}
ranktest_dict = {}

for i in range(len(loss_array)):
    i_model = model_list[i]
    i_model_loss = loss_array[i]
    
    for j in range(len(loss_array)):
        j_model = model_list[j]
        j_model_loss = loss_array[j]
        
        ttest_dict[i_model + ' vs ' + j_model] = [stats.ttest_rel(i_model_loss, j_model_loss)[0],
                                                  stats.ttest_rel(i_model_loss, j_model_loss)[1]]
        
        print('T test, ' + i_model + ' vs ' + j_model + ': ' + str(stats.ttest_rel(i_model_loss, j_model_loss)[0]) + ', ' + \
              str(stats.ttest_rel(i_model_loss, j_model_loss)[1]))
            
        if i_model != j_model:
            ranktest_dict[i_model + ' vs ' + j_model] = [stats.wilcoxon(i_model_loss, j_model_loss)[0], \
                                                         stats.wilcoxon(i_model_loss, j_model_loss)[1]]
                
            print('Rank test, ' + i_model + ' vs ' + j_model + ': ' + str(stats.wilcoxon(i_model_loss, j_model_loss)[0]) + \
                  ', ' + str(stats.wilcoxon(i_model_loss, j_model_loss)[1]))
