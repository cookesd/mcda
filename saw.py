# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:11:33 2020

@author: dakar
"""

import pandas as pd
import numpy as np



def SAW(alternatives,criteria,weights,alt_raw_df,criteria_values=None,criteria_value_functions=None):
    '''
    Conduct the Simple Additive Weighting MCDA method
    
    Determines a benefit score for an alternative by computing the sum product
    of alternative scores on the criteria and the criteria weights.

    Parameters
    ----------
    alternatives : iterable
        The alternatives to evaluate.
    criteria : iterable
        The criteria to evaluate the alternatives upon.
    alt_raw_df : DataFrame
        Alternative (indices) values on the criteria (columns)
    criteria_values : dict of dicts
        Key, value pairs are the criteria and a second dict keyed by {'x','y'}
        whose values are iterables listing the x (raw scores)
        and y (criteria benefit) values to interpolate from to determine an
        alternative's benefit.
    criteria_value_functions : dict
        Key value pairs are the criteria and a function that accepts an alternatives
        raw score on the criteria and returns the benefit value on a scale of 0 to 100.
    weights : dict-like
        Key, value pairs are the criteria and their weights

    Returns
    -------
    res : dict 
        contains the folowing values keyed by {'alt_vals','weighted_crit_vals','crit_vals'}
        respectively
    alt_values : Series
        The alternative benefit scores.
    alt_weighted_crit_val : DataFrame
        The weighted benefit score for each alternative on each criteria and the sum.
    alt_crit_val : DataFrame
        The unweighted benefit score for each alternative on each criteria.
    '''
    alt_values = pd.Series({alt:None for alt in alternatives})
    
    # get actual values for each alternative on each criteria
    # by interpolating from the value functions
    # translate raw scores into value scores for each alternative on each criterion
    if criteria_values:
        alt_crit_val = pd.DataFrame({crit:{alt:val for alt,val in zip(alternatives,
                                                                      np.interp(list(alt_raw_df[crit]),
                                                                             criteria_values[crit]['x'],
                                                                             criteria_values[crit]['y']))} for crit in criteria})
    else: # if don't supply criteria_values, must supply criter_value_functions
        try:
            alt_crit_val = pd.DataFrame({crit:{alt:criteria_value_functions[crit](alt_raw_df.loc[alt,crit]) for alt in alternatives} for crit in criteria})
        except NameError:
            print('You must supply either criteria_values or criteria_value_functions.')
        
        
    # Get the weighted value for each alternative on each criteria
    alt_weighted_crit_val = pd.DataFrame({col:alt_crit_val[col]*weights[col] for col in alt_crit_val.columns})
    
    # Get the actual summed values for each alternative
    alt_values = alt_weighted_crit_val.sum(axis=1)
    # Add this as a column to the alt_weighted_crit_val df
    alt_weighted_crit_val['Alternative Benefit'] = alt_values#alt_weighted_crit_val.sum(axis=1)
    alt_weighted_crit_val['Alternative'] = alt_weighted_crit_val.index
    res = {'alt_vals':alt_values,
           'weighted_crit_vals': alt_weighted_crit_val,
           'crit_vals':alt_crit_val}
    return(res)


