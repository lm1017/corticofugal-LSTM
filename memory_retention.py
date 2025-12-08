import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from denseweight import DenseWeight

NS3_whole_av = {'pop_f_LSTM': {'1': 0.657, '0.5': 0.647, '0.25': 0.644, '0.125': 0.628, '0.0625': 0.591, 
                                  '0.03125': 0.521, '0.333': 0.652, '0.111': 0.621, '0.037': 0.559}}

NS3_whole_med = {'pop_f_LSTM': {'1': 0.679, '0.5': 0.658, '0.25': 0.661, '0.125': 0.655, '0.0625': 0.613,
                                   '0.03125': 0.534, '0.333': 0.674, '0.111': 0.637, '0.037': 0.571}}

#############################################################################################################################

NS3_PEG_whole_av = {'pop_f_LSTM': {'1': 0.569, '0.5': 0.553, '0.25': 0.540, '0.125': 0.518, '0.0625': 0.476,
                                      '0.03125': 0.338, '0.333': 0.531, '0.111': 0.508, '0.037': 0.391}}

NS3_PEG_whole_med = {'pop_f_LSTM': {'1': 0.591, '0.5': 0.554, '0.25': 0.563, '0.125': 0.527, '0.0625': 0.497,
                                       '0.03125': 0.321, '0.333': 0.549, '0.111': 0.504, '0.037': 0.405}}

#############################################################################################################################

NS2_whole_av = {'pop_f_LSTM': {'1': 0.664, '0.5': 0.663, '0.25': 0.658, '0.125': 0.650, '0.0625': 0.643, 
                                  '0.03125': 0.623, '0.015625': 0.598, '0.0078125': 0.461,
                                  '0.333': 0.654, '0.111': 0.653, '0.037': 0.630}}

NS2_whole_med = {'pop_f_LSTM': {'1': 0.665, '0.5': 0.668, '0.25': 0.659, '0.125': 0.643, '0.0625': 0.640, 
                                   '0.03125': 0.624, '0.015625': 0.596, '0.0078125': 0.450,
                                   '0.333': 0.655, '0.111': 0.646, '0.037': 0.636}}

#############################################################################################################################

datasets = ['NS3', 'NS3_PEG', 'NS2']
all_results = dict((d_ID, 0) for d_ID in datasets)

for d_ID in datasets:
    if d_ID == 'NS3':
        all_results[d_ID] = {'av': NS3_whole_av, 
                             'med': NS3_whole_med}
        
    if d_ID == 'NS3_PEG':
        all_results[d_ID] = {'av': NS3_PEG_whole_av, 
                             'med': NS3_PEG_whole_med}
        
    if d_ID == 'NS2':
        all_results[d_ID] = {'av': NS2_whole_av, 
                             'med': NS2_whole_med}
            
#############################################################################################################################    

dataset_colours = dict.fromkeys(datasets)
dataset_colours['NS2'] = 'k'
dataset_colours['NS3'] = '#C20078'
dataset_colours['NS3_PEG'] = '#15B01A'

model_markers = {'pop_f_LSTM': 'o'}

bounded = True

def exp_func(x, a, b, c):
    if bounded:
        return a*(1 - b*np.exp(x*(-1/c)))
    else:
        return a - b*np.exp(x*(-1/c))

def power_func(x, a, b, c):
    if bounded:
        return a*(1 - b*np.power((x + 1), c))
    else:
        return a - b*np.power((x + 1), c)

fig, axs = plt.subplots()

#datasets = ['NS3', 'NS3_PEG']
datasets = ['NS3', 'NS3_PEG', 'NS2']

for d_ID in datasets:
    d_ID_results = all_results[d_ID]
    
    for key in d_ID_results:
        if key == 'av':
            for key_1 in d_ID_results[key]:
                if key_1 == 'pop_f_LSTM':
                    results = d_ID_results[key][key_1]
                    keys_nums = []
                    for key_2 in results.keys():
                        if d_ID == 'NS3' or d_ID == 'NS3_PEG':
                            keys_nums.append(1/(1/float(key_2))*1000)
                        else:
                            keys_nums.append(4/(1/float(key_2))*1000)
                    
                    axs.scatter(keys_nums, results.values(), marker=model_markers[key_1], color=dataset_colours[d_ID])
                    
                    y_data = np.array(list(results.values()))
                    
                    n_fits = 1000
                    
                    p_opts_exp = []
                    sum_res_exp = []
                    
                    p_opts_pow = []
                    sum_res_pow = []
                    
                    for fit in range(n_fits):
                        if bounded:
                            a0 = np.random.uniform(0, 1)
                            b0 = np.random.uniform(0, 1)
                            c0_pow = np.random.uniform(-1, 0)
                        else:
                            a0 = np.random.uniform(0.5*np.max(y_data), 1.5*np.max(y_data))
                            b0 = np.random.uniform(0.5*(np.max(y_data) - np.min(y_data)), 
                                                   1.5*(np.max(y_data) - np.min(y_data)))
                            c0_pow = np.random.uniform(-3, 0)
                        
                        c0_exp = np.random.uniform(500, 2500)
                        
                        dw = DenseWeight(alpha=1)
                        weights = dw.fit(keys_nums)
                        
                        
                        
                        if bounded:
                            popt_pow, pcov, info_pow, mesg, ier = curve_fit(power_func, keys_nums, y_data,
                                                                            np.array([a0, b0, c0_pow]), sigma=1/weights, 
                                                                            bounds=([0, 0, -1], [1, 1, 0]), full_output=True)
                            
                            popt_exp, pcov, info_exp, mesg, ier = curve_fit(exp_func, keys_nums, y_data,
                                                                            np.array([a0, b0, c0_exp]), sigma=1/weights, 
                                                                            bounds=([0, 0, 0], [1, 1, np.inf]),
                                                                            full_output=True)
                            
                        else:
                            popt_pow, pcov, info_pow, mesg, ier = curve_fit(power_func, keys_nums, y_data,
                                                                            np.array([a0, b0, c0_pow]), sigma=1/weights, 
                                                                            full_output=True)
                            
                            popt_exp, pcov, info_exp, mesg, ier = curve_fit(exp_func, keys_nums, y_data,
                                                                            np.array([a0, b0, c0_exp]), sigma=1/weights, 
                                                                            full_output=True)
                        
                        p_opts_exp.append(popt_exp)
                        sum_res_exp.append(np.sum(np.square(info_exp['fvec'])))
                        
                        p_opts_pow.append(popt_pow)
                        sum_res_pow.append(np.sum(np.square(info_pow['fvec'])))
                    
                    popt_exp = p_opts_exp[np.argmin(sum_res_exp)]
                    popt_pow = p_opts_pow[np.argmin(sum_res_pow)]

                    res_exp = np.min(sum_res_exp)
                    res_pow = np.min(sum_res_pow)
                    
                    if 'NS2' in datasets:
                        chunk_axis = np.linspace(0, 4100, num=4100)
                    else:
                        chunk_axis = np.linspace(0, 1500, num=1500)
                        
                    axs.plot(chunk_axis, power_func(chunk_axis, *popt_pow), color=dataset_colours[d_ID], linewidth=3,
                             label=key_1 + ' - ' + d_ID + ' - ' + str(round(popt_pow[2], 2)))
                    
                    if bounded:
                        x_val = np.exp(np.log(0.2/popt_pow[1])/popt_pow[2]) - 1
                    else:
                        x_val = np.exp(np.log(0.1*popt_pow[0]/popt_pow[1])/popt_pow[2]) - 1
                    
                    axs.vlines(x=x_val, ymin=0.2, ymax=power_func(x_val, *popt_pow), linestyle='--', linewidth=3,
                               color=dataset_colours[d_ID])
                    print(d_ID + ' ' + key + ' - ' + str(x_val))
                    
                    axs.spines[['right', 'top']].set_visible(False)
            
            #axs.set_xlabel('chunk length (ms)')
            #axs.set_ylabel('CCnorm')
            axs.set_ylim([0.2, 0.7])
            #axs.set_xscale('log')
            #axs.set_title(key)
            axs.xaxis.set_tick_params(labelcolor='none')
            axs.yaxis.set_tick_params(labelcolor='none')
            #axs.legend(loc=(0.7, 1.02))