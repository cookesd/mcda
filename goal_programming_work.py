# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:23:55 2020

@author: dakar
"""

import pandas as pd
import os
import numpy as np
from plotnine import *
import plotnine as p9
import matplotlib.pyplot as plt

import sys
sys.path.append(r'C:\Users\dakar\Desktop\Extra work\KState\MADM (IMSE)\Spring 2020\mcda')
from goal_programming import goal_programming
import utils


pd.set_option("display.max_colwidth", 10000) # so the words for value scales all show up

in_file = 'Alternative_Data.ods'
os.chdir(r'C:\Users\dakar\Desktop\Extra work\KState\MADM (IMSE)\Spring 2020\Assignements\Project\Pt 1')
df = pd.read_excel(in_file,'Alt Values with Qualitative Scales',engine='odf') # actual altenative data
# vf_df = pd.read_excel(in_file,'Value Functions',engine='odf') # value functions
# print(df.head())
alternatives = list(df['Short Name']) # list of alternatives
criteria = list(df.columns[3:]) # list of criteria

weight_df = pd.read_excel(in_file,'Weights',engine='odf',index_col=0)
crit_groups = weight_df.loc['group',:]
weights = pd.Series(weight_df.loc['weight',:].transpose())

goal_df = weight_df.loc[['goal','monotonic'],:].transpose()

res = goal_programming(orig_val_df,orig_goals,monotonicity,method='a',classes = None,weights=None,single=True,num_sols = 1,verbose=False,normalize = False)





res = {normalized:{method:None for method in ['v','a']} for normalized in [True,False]}
dev_plot_df = pd.DataFrame(columns = ['Criteria', 'Type', 'Alternative', 'Deviation', 'Norm', 'Method'])

for normalized in res.keys():
    for method in res[normalized].keys():
        the_val_df = df.copy()
        the_val_df.index = the_val_df['Short Name']
        the_val_df = the_val_df[criteria] # gets just the criteria columns

        the_goals = goal_df['goal'].copy() # need this because I'm rewriting the goals in the function and d
#         print(the_val_df)
        res[normalized][method] = goal_programming(the_val_df,the_goals,
                                                   monotonicity = goal_df['monotonic'],
                                                   num_sols = len(alternatives),weights=weights,single=True,
                                                   method = method, normalize=normalized)
        
        # for dev_plot_df
        temp = pd.melt(res[normalized][method]['deltas'],
                       id_vars=['Criteria','Type'],var_name='Alternative',value_name='Deviation')
        temp['Norm'] = {True:'Normalized',False:'Not Normalized'}[normalized]
        temp['Method'] = {'a':'Archimedean','v':'Chebyshev'}[method]
        temp['Deviation'] = np.abs(temp['Deviation'])

        temp['print_val'] = np.round(temp['Deviation'],2)
        
        dev_plot_df = pd.concat([dev_plot_df,temp],axis='index',ignore_index=True,sort=False)
dev_plot_df['Alternative'] = pd.Categorical(dev_plot_df['Alternative'],
                                             categories = reversed(list(dev_plot_df.loc[(dev_plot_df['Criteria']=='total') & (dev_plot_df['Norm'] == 'Normalized') & (dev_plot_df['Method'] == 'Archimedean'),:].sort_values(by='Deviation',ascending=False)['Alternative'])), #[c for c in res[normalized][method]['deltas'].columns if c not in ['Criteria','Type']],
                                             ordered = True)
dev_plot_df['Criteria'] = pd.Categorical(dev_plot_df['Criteria'],
                                          categories = reversed(['max','total'] + list(weights.sort_values(ascending=False).index)),
                                          ordered=True)

obj_plot_df = pd.concat([pd.concat([pd.concat([pd.Series(res[normalized][method]['objective values'],name='Objective Values'),
                                               pd.Series({k:k+1 for k in res[normalized][method]['objective values'].keys()},name='Ranks'),
                                               pd.Series({i:j[0] for i,j in enumerate(res[normalized][method]['ranked alternatives'])},name='Alts'),
                                               pd.Series({i:max(res[normalized][method]['objective values'].values()) for i in range(len(res[normalized][method]['objective values']))},name='Text_y'),
                                               pd.Series({i:{'a':'Archimedean','v':'Chebyshev'}[method] for i in range(len(res[normalized][method]['ranked alternatives']))},name='Method'),
                                               pd.Series({i:{True:'Normalized',False:'Not Normalized'}[normalized] for i in range(len(res[normalized][method]['ranked alternatives']))},name='Normalized')],axis='columns')
                                    for method in res[normalized].keys()],axis='index',ignore_index=True) for normalized in res.keys()],axis='index',ignore_index=True)

chevy_plot_df = dev_plot_df.loc[(dev_plot_df['Norm']=='Normalized') & (dev_plot_df['Method']=='Chebyshev') & (dev_plot_df['Type']=='max'),:].sort_values(by='print_val')
chevy_plot_df['Criteria'] = pd.Series({46: 'Storage Space', 82: 'Storage Space',
                                       58: 'List Price', 10:'List Price', 94: 'List Price', 34: 'List Price',
                                       70: 'List Price', 22: 'List Price'})
chevy_plot_df['Criteria'] = pd.Categorical(chevy_plot_df['Criteria'],
                                           categories = reversed(list(weights.sort_values(ascending=False).index)),
                                           ordered=True)
# chevy_plot_df['Criteria'] = pd.Categorical(chevy_plot_df['Criteria'],
#                                            categories=['max'])
normalize_chebyshev_plot = (ggplot(chevy_plot_df.sort_values(by='Alternative',ascending=True),
                                   aes(x='Alternative',y='print_val',fill='Criteria')) 
                            + geom_bar(stat='identity')
                            + ggtitle('Max Deviation by Alternative')
                            + scale_fill_brewer(type='qual',palette='Paired',name='Deviation',drop=False,guide=False)
                            + theme(axis_text_x=element_text(rotation=45,hjust=1),
                                    axis_text_y=element_text(rotation=45,vjust=1))
                            + ylab('Deviation')
                            + guides(color=None))
normalize_chebyshev_plot


arc_plot_df = dev_plot_df.loc[(dev_plot_df['Norm']=='Normalized') & (dev_plot_df['Method']=='Archimedean') & (dev_plot_df['Type']=='Criteria'),:]
# arc_plot_df['Alternative'] = pd.Categorical(arc_plot_df['Alternative'],
#                                               categories=arc_plot_df['Alternative'].sort_values(ascending=True),ordered=True)
arc_plot_df['Criteria'] = pd.Categorical(arc_plot_df['Criteria'],
                                         categories = reversed(list(weights.sort_values(ascending=False).index)),
                                         ordered=True)
normalize_arc_plot = (ggplot(arc_plot_df,
                                   aes(x='Alternative',y='Deviation',fill='Criteria')) 
                            + geom_col(stat='identity')
                            + ggtitle('Weighted Deviation by Alternative')
                            + scale_fill_brewer(type='qual',palette='Paired',name='Deviation')
                            + theme(axis_text_x=element_text(rotation=45,hjust=1),
                                    axis_text_y=element_text(rotation=45,vjust=1))
                            + ylab('Deviation')
                            + guides(color=None))
normalize_arc_plot
