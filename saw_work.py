# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:45:20 2020

@author: dakar
"""

import pandas as pd
import os
import numpy as np
from plotnine import *
import plotnine as p9
import matplotlib.pyplot as plt
import copy

import sys
sys.path.append(r'C:\Users\dakar\Desktop\Extra work\KState\MADM (IMSE)\Spring 2020\mcda')
# from promethee import promethee
from saw import SAW
import utils


in_file = 'Alternative_Data.ods'
os.chdir(r'C:\Users\dakar\Desktop\Extra work\KState\MADM (IMSE)\Spring 2020\Assignements\Project\Pt 1')
df = pd.read_excel(in_file,'Alt Values with Qualitative Scales',engine='odf') # actual altenative data
vf_df = pd.read_excel(in_file,'Value Functions',engine='odf') # value functions
# print(df.head())
alternatives = list(df['Short Name']) # list of alternatives
criteria = list(df.columns[3:]) # list of criteria

weight_df = pd.read_excel(in_file,'Weights',engine='odf',index_col=0)
crit_groups = weight_df.loc['group',:]
weights = pd.Series(weight_df.loc['weight',:].transpose())

vf_points = {crit:{'x':[x for x in list(vf_df[crit+'_x']) if str(x) != 'nan'],
                  'y':[y for y in list(vf_df[crit+'_y']) if str(y) != 'nan']} for crit in criteria}


max_sens = 5
sensitivity_weights = {crit:utils.criterion_sensitivity(crit,weights,criteria,
                                                        step=weights[crit]/10,
                                                        max_each_dir = max_sens) for crit in criteria}
sensitivity_scores = {crit:{val: SAW(alternatives = alternatives,
                                    criteria = criteria,alt_raw_df = df,
                                    criteria_values = {crit:{'x':sorted(vf_points[crit]['x']),
                                                             'y':[y for _,y in sorted(zip(vf_points[crit]['x'],
                                                                                          vf_points[crit]['y']))]} for crit in criteria},
                                    weights = sensitivity_weights[crit][val]) for val in list(range(-max_sens,max_sens+1))} for crit in criteria}

plot_df = {val:utils.make_plot_df(alt_value_df=sensitivity_scores[criteria[0]][val]['weighted_crit_vals'],
                            criteria=criteria,weights=weights,sensitivity=False,
                            alt_order=list(sensitivity_scores[criteria[0]][0]['alt_vals'].sort_values(ascending=False).index)) for val in sensitivity_scores[criteria[0]].keys()}

plots = {val:utils.plot_alt_benefit(plot_df[val],
                                    title='Benefit by Alternative\n'+criteria[0]+'={}'.format(np.round(sensitivity_weights[criteria[0]].loc[criteria[0],val],2)),
                                    which='both') for val in plot_df.keys()}
plots[-2]
plots[0]
# res = SAW(alternatives = alternatives,
#                                     criteria = criteria,alt_raw_df = df,
#                                     criteria_values = {crit:{'x':sorted(vf_points[crit]['x']),
#                                                              'y':[y for _,y in sorted(zip(vf_points[crit]['x'],
#                                                                                           vf_points[crit]['y']))]} for crit in criteria},
#                                     weights = weights)
#
vf_sens_points = copy.deepcopy(vf_points)
vf_sens_points['List Price']['x'] = [x if x != 175000.0 else 120000.0 for x in vf_sens_points['List Price']['x']]

vf_sensitivity_scores = {crit:{val: SAW(alternatives = alternatives,
                                    criteria = criteria,alt_raw_df = df,
                                    criteria_values = {crit:{'x':sorted(vf_sens_points[crit]['x']),
                                                             'y':[y for _,y in sorted(zip(vf_sens_points[crit]['x'],
                                                                                          vf_sens_points[crit]['y']))]} for crit in criteria},
                                    weights = sensitivity_weights[crit][val]) for val in list(range(-max_sens,max_sens+1))} for crit in criteria}

vf_sens_plot_df = {val:utils.make_plot_df(alt_value_df=vf_sensitivity_scores[criteria[0]][val]['weighted_crit_vals'],
                            criteria=criteria,weights=weights,sensitivity=False,
                            alt_order=list(sensitivity_scores[criteria[0]][0]['alt_vals'].sort_values(ascending=False).index)) for val in vf_sensitivity_scores[criteria[0]].keys()}


vf_sens_plots = {val:utils.plot_alt_benefit(vf_sens_plot_df[val],
                                    title='Benefit by Alternative\n'
                                    +criteria[0]+' Value Function Sensitivity\n'+
                                    '$w_i$={}'.format(np.round(sensitivity_weights[criteria[0]].loc[criteria[0],val],2)),
                                    which='total',legend=False) for val in vf_sens_plot_df.keys()}

##### Weight Sensitivity #####
crit = 'Floorplan Openness'
floorplan_plot_dfs = {val:utils.make_plot_df(alt_value_df=sensitivity_scores[crit][val]['weighted_crit_vals'],
                            criteria=criteria,weights=weights,sensitivity=False,
                            alt_order=list(sensitivity_scores[crit][0]['alt_vals'].sort_values(ascending=False).index)) for val in sensitivity_scores[crit].keys()}
for val,dataframe in floorplan_plot_dfs.items():
    dataframe['Criterion Weight'] = np.round(sensitivity_weights[crit].loc[crit,val],3)
floorplan_sens_df = pd.concat([floorplan_plot_dfs[0],floorplan_plot_dfs[4]],
                              axis='rows')
g = utils.plot_alt_benefit(floorplan_sens_df,title='{} Sensitivity'.format(crit),
                           which='total',sensitivity=True,legend=False)
g
