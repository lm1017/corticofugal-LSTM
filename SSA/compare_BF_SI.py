import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from SSA_utils import pitman_morgan_test
from scipy import stats

dataset_ID = 'NS3_PEG'
models = ['pop_f_LSTM', 'oneD_CNN']

start = 'silence'
norm = 0

plot_by_sess = False
plot_violin = True

models_SI = {}

for model_ID in models:
    SI_file = 'BF_SI/' + start + '_start/Norm_' + str(norm) + '/' + dataset_ID + '_cond_2_SI_' + model_ID + '.pckl'
    f = open(SI_file, 'rb')
    all_SI = pickle.load(f)
    f.close()
    
    models_SI[model_ID] = all_SI
   
f, ax = plt.subplots()
ax.scatter(models_SI[models[1]], models_SI[models[0]])
if (ax.get_ylim()[1] - ax.get_ylim()[0]) > (ax.get_xlim()[1] - ax.get_xlim()[0]):
    ax.set_xlim(ax.get_ylim())
else:
    ax.set_ylim(ax.get_xlim())
ax = plt.gca()
ax.set_aspect('equal')
'''ax.set_yticks([0, 0.1, 0.2])
ax.set_xticks([0, 0.1, 0.2])'''
'''ax.set_yticks([-0.1, 0, 0.1])
ax.set_xticks([-0.1, 0, 0.1])'''
ax.set_yticks([-0.1, 0])
ax.set_xticks([-0.1, 0])
#ax.set_xlabel(models[1])
#ax.set_ylabel(models[0])
#ax.set_title('BF SI - norm ' + str(norm))
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_tick_params(labelcolor='none')
ax.yaxis.set_tick_params(labelcolor='none')

f, ax = plt.subplots()
ax.boxplot([models_SI[models[0]], models_SI[models[1]]])
ax.set_ylabel('SI')
ax.set_xticklabels(models)
ax.spines[['right', 'top']].set_visible(False)

t_stat, p = pitman_morgan_test(models_SI[models[0]], models_SI[models[1]])
ax.set_title('BF SI - norm ' + str(norm) + ', p = ' + str(p))

if plot_violin:
    f, ax = plt.subplots()
    ax.scatter(np.random.normal(loc=1, scale=0.1, size=len(models_SI[models[0]])), models_SI[models[0]], color='k', 
               alpha=0.1, s=20)
    ax.scatter(np.random.normal(loc=2, scale=0.1, size=len(models_SI[models[1]])), models_SI[models[1]], color='k', 
               alpha=0.1, s=20)
    ax.violinplot([models_SI[models[0]], models_SI[models[1]]], showmeans=True)
    #ax.set_ylabel('SI')
    ax.set_yticks([0, 0.1, 0.2]) # NS3
    #ax.set_yticks([-0.1, 0, 0.1]) # NS3-PEG
    #ax.set_yticks([-0.1, 0]) # NS2
    ax.set_xticks([1, 2])
    ax.set_xticklabels(models)
    ax.spines[['right', 'top']].set_visible(False)
    #ax.set_title('BF SI - norm ' + str(norm) + ', p = ' + str(p))
    ax.xaxis.set_tick_params(labelcolor='none')
    ax.yaxis.set_tick_params(labelcolor='none')

ttest_1_0 = stats.ttest_1samp(models_SI[models[0]], popmean=0)
ttest_1_1 = stats.ttest_1samp(models_SI[models[1]], popmean=0)

cond_2 = [boxplot_stats(models_SI[models[0]])[0], boxplot_stats(models_SI[models[1]])[0]]    