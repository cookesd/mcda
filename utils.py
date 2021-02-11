# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:03:07 2020

@author: dakar
"""

import pandas as pd
import numpy as np
import plotnine as p9

__all__ = ['criterion_sensitivity']

def criterion_sensitivity(criterion,weights,all_crits,step=.05,max_each_dir = 3):
    '''Changes the weight for the specified criterion (updating others proportionally to keep sum = 1)
    @ param data: The alternative raw values on each criterion
    @ param weights: The series of original criterion weights
    @ param all_crits: The list of criterion (must be same as data.columns and weights.index)
    @ param criterion: The desired criterion we will investigate the sensitivity
    @ param step: how large of increments to increase / decrease the criterion's weight'''
    
    other_crits = list(set(all_crits) - {criterion}) # removes the desired criterion from the list of other criteria
    beg_weights = dict(weights)
        # gets the weight proportion of other crits when criterion removed
    other_crit_proportions = {crit:beg_weights[crit] / (1-beg_weights[criterion]) for crit in other_crits}
    
    res_weights = pd.DataFrame(index=all_crits)
    res_weights[0] = pd.Series(weights)
    # print(res_weights)
    for i in range(1,max_each_dir+1):
        if beg_weights[criterion] + step*(i+1) <= 1:
            res_weights[i] = _get_new_weights(criterion,beg_weights,all_crits,step*(i+1),
                                             other_crits,other_crit_proportions)
        if beg_weights[criterion] - step*(i-1) >= 0:
            res_weights[-i] = _get_new_weights(criterion,beg_weights,all_crits,-step*(i+1),
                                              other_crits,other_crit_proportions)
            
    return(res_weights)

def _get_new_weights(criterion,beg_weights,all_crits,step,other_crits,props):
    crit_weight = beg_weights[criterion] + step
    new_weights = pd.Series({crit:beg_weights[crit] - step*props[crit] for crit in other_crits})
    new_weights[criterion]=crit_weight
    return(new_weights)


def make_plot_df(alt_value_df,criteria,weights,sensitivity=False,alt_order=None):
    '''Melts the alternative value df into long format
    So I can make stacked bar charts of weights on each criteria
    and the overall weights
    
    @ param alt_value_df: the dataframe holding weighted values
    for each alternative on each criteria and overall benefit of alternative.
    The alternative names must be a column in the df
    @ returns: melted df columns <Alternative (Alt name),
    Criterion (Criterion Name or Alternative Benefit),
    value (The weighted value for the crit or total alt value),
    type (Criterion Value or Total Value)
    '''
    
    if sensitivity:
        ids = ['Alternative','Criterion Weight']
    else:
        ids = 'Alternative'
    if alt_order==None:
        categories = list(alt_value_df.sort_values(by='Alternative Benefit',ascending=False).index)
    else:
        categories = alt_order
    plot_crits = ['Alternative Benefit']+criteria
    plot_df = pd.melt(alt_value_df,id_vars = ids,var_name = 'Criterion',value_name='Benefit')

        # Puts the criterion in order by the weight given to the criterion
    plot_df['Criterion'] = pd.Categorical(plot_df['Criterion'],
                                          categories = reversed(['Alternative Benefit']+list(weights.sort_values(ascending=False).index)),
                                          ordered=True)#reversed(plot_crits)) # makes criterion an ordered categorical var so can order stacked bars
        
        # Distinguishes the Criterion Values from the Total Value to facet the plot
    plot_df['type'] = pd.Series({i:'Weighted Criterion Value' if plot_df.loc[i,'Criterion'] != 'Alternative Benefit' else 'Total Value' for i in plot_df.index })
    plot_df['type'] = pd.Categorical(plot_df['type'],categories = ['Weighted Criterion Value','Total Value']) # orders so crit value plotted on left
        # make the Alternative column a ordered categorical to sort the bars (by best to worst alternative)
    plot_df['Alternative'] = pd.Categorical(plot_df['Alternative'],
                                            categories=categories,
                                            ordered=True) # puts alternatives in order of which has largest value
    plot_df['print_value'] = round(plot_df['Benefit'],2)
        # Sorts the dataframe by the criterion so the stacked bars display in order by criterion weight (the categorical column)
    plot_df.sort_values(by='Criterion',inplace=True) # puts stacked bars in order of largest weight (using the Criterion categorical column)
    
    return(plot_df)

def plot_alt_benefit(plot_df,title='Benefit by Alternative',which='both',sensitivity = False,legend=True):
    '''Builds a stacked bar chart of the alternative benefits
    @ param plot_df: The df containing benefits for each alt by the criteria and total benefit
    @ param title: The title for the graph
    @ param which: which parts to plot. Acceptable values are
    'total' for just total value.
    'criteria' for just criteria level stacked bars'
    'both' for total and criteria. The graphs will be faceted in this case
    
    Returns the ggplot graph to be displayed elsewhere'''
    
    _facet = which == 'both'
    if which == 'both':
        plot_df = plot_df
    elif which == 'total':
        plot_df = plot_df.loc[plot_df['type']=='Total Value']
    elif which == 'criteria':
        plot_df = plot_df.loc[plot_df['type']=='Weighted Criterion Value']
    else:
        print(which, 'is not an approved value for which.\n Enter "total", "criteria", or "both"')
        return(None)
    

    
    if legend:
        g = (p9.ggplot(plot_df,p9.aes(x='Alternative',y='Benefit',fill='Criterion'))
             + p9.geom_col(stat='identity',position=p9.position_stack(vjust=.5)) # makes stacked bar plot
             + p9.scale_fill_brewer(type='qual',palette='Paired')) # changes the color palette to one for qualitative scales)
    else:
        g = (p9.ggplot(plot_df,p9.aes(x='Alternative',y='Benefit',fill='Criterion'))
             + p9.geom_col(p9.aes(show_legend=False),stat='identity',position=p9.position_stack(vjust=.5)) # makes stacked bar plot
             + p9.scale_fill_brewer(type='qual',palette='Paired',guide=False) # changes the color palette to one for qualitative scales
             + p9.theme(legend_position=None)
         )
        
        # Builds the base plot
    g = (g
         # + p9.geom_col(stat='identity',position=p9.position_stack(vjust=.5)) # makes stacked bar plot
         # + p9.scale_fill_brewer(type='qual',palette='Paired') # changes the color palette to one for qualitative scales
         + p9.geom_text(p9.aes(label='print_value'),position=p9.position_stack(vjust=.5),size=6,hjust='center') # adds weighted value to bars
         + p9.ggtitle(title) # makes the title
         + p9.theme(axis_text_x=p9.element_text(rotation=45,hjust=1)) # rotates x axis labels
    )
    # Adds the facet if required
    if sensitivity:
        if _facet:
            return((g + p9.facet_grid('type~Criterion Weight')))
        else:
            return((g + p9.facet_grid('Criterion Weight~')))
    elif _facet:
        return((g + p9.facet_grid('~type')))
    else:
        return(g)


if __name__ == '__main__':
    b = [np.random.randint(0,100) for i in range(5)]
    c = [i/sum(b) for i in b]
    crits = ['test'+str(i) for i in range(len(b))]
    crit_df = criterion_sensitivity(criterion=crits[0],weights=dict(zip(crits,c)),all_crits=crits)
    crit_df.transpose().sort_index().plot(kind='bar')

    
