#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:31:28 2026

@author: lorenzom
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from SSA_utils import pitman_morgan_test, ula_stats

dataset_ID = 'NS2_include_single'
models = ['pop_t_subLSTM', 'twoD_CNN']

protocol = 'oddball'
start = 'sound'

plot_all_conds = False
plot_by_sess = True
plot_violin = True

best_lambs = {'NS3': {'pop_t_subLSTM': 15, 'twoD_CNN': 15}, 'NS3_PEG': {'pop_t_subLSTM': 14, 'oneD_CNN': 16}, 
              'NS2_include_single': {'pop_t_subLSTM': 14, 'twoD_CNN': 15}}

models_SI = {}

for model_ID in models:
    SI_file = 'SI/' + start + '_start/' + dataset_ID + '_' + protocol + '_' + model_ID + '.pckl'
    f = open(SI_file, 'rb')
    all_SI = pickle.load(f)
    f.close()
    
    best_lamb_SI = all_SI[str(best_lambs[dataset_ID][model_ID])]
    
    models_SI[model_ID] = best_lamb_SI
    
if protocol == 'oddball':
    for cond in models_SI['pop_t_subLSTM']:
        if plot_all_conds:
            f, ax = plt.subplots()
            ax.scatter(models_SI[models[1]][cond], models_SI[models[0]][cond])
            if (ax.get_ylim()[1] - ax.get_ylim()[0]) > (ax.get_xlim()[1] - ax.get_xlim()[0]):
                ax.set_xlim(ax.get_ylim())
            else:
                ax.set_ylim(ax.get_xlim())
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.set_xlabel(models[1])
            ax.set_ylabel(models[0])
            ax.set_title('SI - ' + protocol + ', ' + cond)
        
        if cond == 'cond 2':
            cond_2 = [boxplot_stats(models_SI[models[0]][cond])[0], boxplot_stats(models_SI[models[1]][cond])[0]]
        
            f, ax = plt.subplots()
            ax.boxplot([models_SI[models[0]][cond], models_SI[models[1]][cond]])
            ax.set_ylabel('SI')
            ax.set_xticklabels(models)
            ax.spines[['right', 'top']].set_visible(False)
            
            t_stat, p = pitman_morgan_test(models_SI[models[0]][cond], models_SI[models[1]][cond])
            ax.set_title(protocol + ', ' + cond + ', p = ' + str(p))
            
            if plot_violin:
                f, ax = plt.subplots()
                ax.violinplot([models_SI[models[0]][cond], models_SI[models[1]][cond]], showmeans=True)
                ax.set_ylabel('SI')
                ax.set_xticks([1, 2])
                ax.set_xticklabels(models)
                ax.spines[['right', 'top']].set_visible(False)
                ax.set_title(protocol + ', ' + cond + ', p = ' + str(p))
        
elif protocol == 'switch_oddball':
    if plot_all_conds:
        f, ax = plt.subplots()
        ax.scatter(models_SI[models[1]], models_SI[models[0]])
        if (ax.get_ylim()[1] - ax.get_ylim()[0]) > (ax.get_xlim()[1] - ax.get_xlim()[0]):
            ax.set_xlim(ax.get_ylim())
        else:
            ax.set_ylim(ax.get_xlim())
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlabel(models[1])
        ax.set_ylabel(models[0])
        ax.set_title('SI - ' + protocol)
    
    f, ax = plt.subplots()
    ax.boxplot([models_SI[models[0]], models_SI[models[1]]])
    ax.set_ylabel('SI')
    ax.set_xticklabels(models)
    ax.spines[['right', 'top']].set_visible(False)
    
    t_stat, p = pitman_morgan_test(models_SI[models[0]], models_SI[models[1]])
    ax.set_title(protocol + ', p = ' + str(p))
    
    if plot_violin:
        f, ax = plt.subplots()
        ax.violinplot([models_SI[models[0]], models_SI[models[1]]], showmeans=True)
        ax.set_ylabel('SI')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(models)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_title(protocol + ', p = ' + str(p))
    
ula_cond_2 = ula_stats()

fig, ax = plt.subplots()
ax.bxp(ula_cond_2 + cond_2, showfliers=True)
ax.set_ylim([-0.85, 0.85])
ax.set_ylabel('SI')
ax.spines[['right', 'top']].set_visible(False)
ax.set_title('Ulanovsky and Cond 2, ' + models[0] + ' and ' + models[1])
plt.show()


# Plot by session
sess_file = 'Neuron_sessions/' + dataset_ID + '_neuron_sessions.pckl'
f = open(sess_file, 'rb')
neuron_sess = pickle.load(f)
f.close()

neuron_sess = [neuron[1] for neuron in neuron_sess]
unique_sess = np.unique(neuron_sess)

cond_2_sess = [{sess: [] for sess in unique_sess}, {sess: [] for sess in unique_sess}]
for nID in range(len(neuron_sess)):
    sess = neuron_sess[nID]
    
    cond_2_sess[0][sess].append(models_SI[models[0]]['cond 2'][nID])
    cond_2_sess[1][sess].append(models_SI[models[1]]['cond 2'][nID])
    
bp_stats_cond_2_sess = [{sess: None for sess in unique_sess}, {sess: None for sess in unique_sess}]
for sess in unique_sess:
    bp_stats_cond_2_sess[0][sess] = boxplot_stats(cond_2_sess[0][sess])[0]
    bp_stats_cond_2_sess[1][sess] = boxplot_stats(cond_2_sess[1][sess])[0]
    
bp_stats_cond_2_sess[0] = [bp_stats_cond_2_sess[0][sess] for sess in bp_stats_cond_2_sess[0]]
bp_stats_cond_2_sess[1] = [bp_stats_cond_2_sess[1][sess] for sess in bp_stats_cond_2_sess[1]]

if plot_by_sess:
    fig, ax = plt.subplots()
    ax.bxp(bp_stats_cond_2_sess[0] + bp_stats_cond_2_sess[1], showfliers=True)
    ax.set_ylim([-0.85, 0.85])
    ax.set_ylabel('SI')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title('Cond 2 by session, ' + models[0] + ' and ' + models[1])
    plt.show()

if plot_violin:
    ula_medians = [sess['med'] for sess in ula_cond_2]
    
    fig, ax = plt.subplots()
    ax.violinplot([ula_medians, models_SI[models[0]]['cond 2'], models_SI[models[1]]['cond 2']], showmeans=True)
    ax.set_ylim([-0.85, 0.85])
    ax.set_ylabel('SI')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Ulanovsky'] + models)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title('Ulanovsky and Cond 2, ' + models[0] + ' and ' + models[1])
    plt.show()