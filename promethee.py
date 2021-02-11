# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:50:29 2020

@author: dakar

PROMETHEE Outranking Multiple Attribute Decision Making method
"""

import pandas as pd
import numpy as np
import itertools


def promethee(crits,alts,crit_params,alt_vals):
    '''
    Creates an ordering of alternatives using the PROMETHEE
    method of MCDA.

    Parameters
    ----------
    crits : iterable
        The criteria to evaluate the alternatives
    alts : iterable
        The alternatives for evaluation
    crit_params : pd.DataFrame
        DataFrame holding the parameters for each criteria.
        Columns: Criteria (same as crits) along columns
        Index:
            'w_i': the criteria weights
            'method': the method to preference function ('linear' or 'gaussian')
            'direction': either 'positive' or 'negative' for larger is better or smaller is better
                         or a float signifying the desired value on a nonmonotonic scale
            'p_i': (for 'linear' method) the outranking threshold for the criterion 
            'q_i': (for 'linear' method) the indifference threshold for the criterion
            'p_i': (for 'gaussian' method) the inflection point of diminishing returns for the criterion
    alt_vals : pd.DataFrame
        Dataframe containing the alternative values on each criterion scale (z_i(a))
        Columns: Criteria (same as crits)
        Index: Alternatives (same as alts)

    Returns
    -------
    Dict with keys:
        'pref_functs': dict keyed by criteria of preference functions for each
        'alt_prefs': MultiIndex (alternatives a,b) pd.DataFrame with P_i(a,b)
        'pref_inds': MultiIndex (alternaties a,b) pd.DataFrame with P(a,b)
        'outflows': pd.DataFrame with Q^+(a) and Q^-(a) (alternatives as index)
        'pairwise_comps': MultiIndex (alternatives a,b) pd.Series with pairwise comparison between alternatives
        'order': List of sets holding the ordering of alternatives

    '''
    

    
    
    # Get preference functions and comparisons
    pref_functs = {crit:get_pref_funct(**crit_params[crit].to_dict()) for crit in crits}
    p_iab = pd.DataFrame({crit:{(a,b):pref_functs[crit](alt_vals.loc[a,crit],
                                                        alt_vals.loc[b,crit]) for a,b in itertools.permutations(alts,2)} for crit in crits})
    
    # Get Pairwise Prefence Indices
    p_ab = pd.Series({(a,b):get_pairwise_pref(weights = crit_params.loc['w_i',:],p_iab=p_iab.loc[(a,b),:]) for a,b in itertools.permutations(alts,2)})
    
    # Calculate outranking flows
    q_pos = pd.Series({alt:get_pos_outflows(p_ab.loc[[alt]]) for alt in alts},name='Positive Outranking Flows')
    q_neg = pd.Series({alt:get_neg_outflows(p_ab.loc[(alts,[alt])]) for alt in alts},name='Negative Outranking Flows')
    outflows = pd.concat([q_pos,q_neg],axis='columns')
    
    # Get overall pairwise comparison and partial order
    pairwise_comps = pd.Series({(a,b):get_pairwise_comparison(q_pos_a=q_pos[a],q_pos_b=q_pos[b],q_neg_a=q_neg[a],q_neg_b=q_neg[b]) for a,b in itertools.permutations(alts,2)},
                          name='Pairwise Comparisons')
    order = determine_order(pairwise_comps)
    
    res = {'pref_functs': pref_functs,
           'alt_prefs': p_iab,
           'pref_inds': p_ab,
           'outflows': outflows,
           'pairwise_comps': pairwise_comps,
           'order': order
           }
    return(res)

#### Develop Preference Functions and Pairwise Preference Relationship for each crit ####
def get_pref_funct(method,direction = 'increasing',**kwargs):
    '''Develops the preference function for a given criterion determined by the parameters
    @ param method: str 'linear' or 'gaussian' for piecewise linear or gaussian respectively
    @ param direction: str 'increasing' or 'decreasing'  describing which values are prefered
    or float value determinining the desired value for this scale
    @ param **kwargs: dict / series with keys 'p_i' and 'q_i' for 'linear' method and 's_i' for gaussian
    'p_i' is the threshold for P_i = 1, 'q_i' is the threshold for P_i = 0
    's_i' is the parameter for the function $1 - e^{\frac{-\left(z_i(a) - z_i(b) \right)^2}{2s^2}}$
    it influences the inflection point in the distribution
    
    returns a function that accepts z_i(a) and z_i(b) to determine P_i(a,b)
    The function will accept the parameters but it stores the ones supplied
    '''

    if method == 'linear':
        for kw in ['p_i','q_i']:
            # makes sure the parameters are passed (could instead fill them with None if not supplied and require later)
            assert kw in kwargs, 'Method {} requires parameter {}'.format(method,kw)
        
        if direction == 'increasing':
#             return(lambda z_ia,z_ib,p=kwargs['p'],q=kwargs['q']: 0 if (z_ia - z_ib <= q) else (1 if (z_ia - z_ib >= p or p==q) else  (z_ia - z_ib - q)/(p-q)))
            def increasing_function(z_ia,z_ib,p=kwargs['p_i'],q=kwargs['q_i']):
                '''Preference function for a criterion where larger values are better
                @ param z_ia: numeric. The value for the first alternative on the criterion scale
                @ param z_ib: numeric. The value for the second alternative on the criterion scale
                @ param p: numeric. The threshold for P_i = 1. Defaults to value supplied when created the function
                @ param q: numeric. The threshold for P_i = 0. Defaults to value supplied when created the function
                
                Returns the value (between 0 and 1) for the preference of alternative a over alternative b on the criterion
                0 means a not preferred over b; 1 means a completely prefered over b; other value specifies the intensity
                '''
                
                if z_ia - z_ib <= q:
                    return(0)
                elif (z_ia-z_ib >= p) or p==q:
                    return(1)
                else:
                    return((z_ia - z_ib - q)/(p-q))
            
            return(increasing_function) # actually returns the function
            
        elif direction == 'decreasing':
#             return(lambda z_ia,z_ib,p=kwargs['p'],q=kwargs['q']: 0 if (-(z_ia - z_ib) <= q) else (1 if (-(z_ia - z_ib) >= p or p==q) else  -(z_ia - z_ib - q)/(p-q)))
             
            def decreasing_function(z_ia,z_ib,p=kwargs['p_i'],q=kwargs['q_i']):
                '''Preference function for a criterion where smaller values are better
                @ param z_ia: numeric. The value for the first alternative on the criterion scale
                @ param z_ib: numeric. The value for the second alternative on the criterion scale
                @ param p: numeric. The threshold for P_i = 1. Defaults to value supplied when created the function
                @ param q: numeric. The threshold for P_i = 0. Defaults to value supplied when created the function
                
                Returns the value (between 0 and 1) for the preference of alternative a over alternative b on the criterion
                0 means a not preferred over b; 1 means a completely prefered over b; other value specifies the intensity
                '''
                
                if -(z_ia - z_ib) <= q:
                    return(0)
                elif (-(z_ia-z_ib) >= p) or p==q:
                    return(1)
                else:
                    return((-(z_ia - z_ib) - q)/(p-q))
            return(decreasing_function)
#             return(decreasing_function)
                
            
            
        else: # direction == 'some number'
            assert float(direction), 'direction must be either increasing, decreasing, or a numeric value specifying the desired value'
            # Function 
#             return(lambda z_ia,z_ib,des = float(direction),p=kwargs['p'],q=kwargs['q']: 0 if (-(abs(z_ia-des) - abs(z_ib-des)) <= q) else (1 if (-(abs(z_ia-des) - abs(z_ib-des)) >= p or p==q) else  -(abs(z_ia-des) - abs(z_ib-des) - q)/(p-q)))
            def non_monotonic_function(z_ia,z_ib,des=float(direction),p=kwargs['p_i'],q=kwargs['q_i']):
                '''Preference function for a criterion with non-monotonic value
                @ param z_ia: numeric. The value for the first alternative on the criterion scale
                @ param z_ib: numeric. The value for the second alternative on the criterion scale
                @ param des: numeric. The desired value for the criterion. Defaults to value supplied when created the function
                @ param p: numeric. The threshold for P_i = 1. Defaults to value supplied when created the function
                @ param q: numeric. The threshold for P_i = 0. Defaults to value supplied when created the function
                
                Returns the value (between 0 and 1) for the preference of alternative a over alternative b on the criterion
                0 means a not preferred over b; 1 means a completely prefered over b; other value specifies the intensity
                '''
                
                if -(abs(z_ia-des) - abs(z_ib-des)) <= q:
                    return(0)
                elif -(abs(z_ia-des) - abs(z_ib-des)) >= p or p==q:
                    return(1)
                else:
                    return((-(abs(z_ia-des) - abs(z_ib-des)) - q)/(p-q))
                
            return(non_monotonic_function)
            
            
    if method == 'gaussian':
        for kw in ['s_i']:
            # makes sure the parameters are passed
            assert kw in kwargs, 'Method {} requires parameter {}'.format(method,kw)
        
        return(lambda z_ia,z_ib,s=kwargs['s']: 0 if z_ib > z_ia else 1 - np.exp(-(z_ia-z_ib)**2 / (2*(s**2))))


def get_pref(funct,z_ia,z_ib):
    '''Not neccesary, but returns the preference value P_i(a,b)
    between alternatives a and b
    
    @ param funct: function accepting the values z_ia and z_ib and returns the preference value
    @ param z_ia: float. The value of alternative a on the criterion scale
    @ param z_ia: float. The value of alternative b on the criterion scale
    
    Returns the preference value
    
    '''
    return(funct(z_ia,z_ib))

##### Calculate Pairwise Preference Index for alternatives #####
def get_pairwise_pref(weights,p_iab):
    '''Calculates the preference between alternatives a and b given the criteria weights and
    the preference values on the criteria
    
    @ param weights: pd.Series indexed by the criteria containing the criteria weights
    @ param p_iab: pd.Series indexed by the criteria holding the value P_i(a,b)
    
    returns the pairwise preference index P(a,b)
    '''
    
    crits = list(weights.index)
    p_ab = sum([weights[crit]*p_iab[crit] for crit in crits]) / weights.sum()
    return(p_ab)

##### Calculate positive $\left(Q^+(a)\right)$ and negative $\left(Q^-(a)\right)$ outranking flows #####
def get_pos_outflows(other_prefs):
    '''Gets the Positive outranking flow for alternative a
    (sum of P(a,b) for b != a)
    
    @ param other_prefs: iterable of values P(a,b) for b != a'''
    return(sum(other_prefs))

def get_neg_outflows(other_prefs):
    '''Gets the Positive outranking flow for alternative a
    (sum of P(b,a) for b == a)
    
    @ param other_prefs: iterable of values P(b,a) for b != a'''
    return(sum(other_prefs))

##### Determine *partial* order (allows incomparability) #####
def get_pairwise_comparison(q_pos_a,q_pos_b,q_neg_a,q_neg_b):
    '''
    Determines the outranking relationship of alternatives
    a on b using the positive and negative outranking flows
    for the two alternatives.

    Parameters
    ----------
    q_pos_a : float
        Positive outranking flow for alternative a
    q_pos_b : float
        Positive outranking flow for alternative b
    q_neg_a : float
        Negative outranking flow for alternative a
    q_neg_b : float
        Negative outranking flow for alternative b

    Returns
    -------
    String in ['outranks','indifferent','incomparable'] based on the outranking
    relationship of alternative a to b.

    '''
    
    a_better_on_pos = q_pos_a > q_pos_b
    a_better_on_neg = q_neg_a < q_neg_b
    equal_on_pos = q_pos_a == q_pos_b
    equal_on_neg = q_neg_a == q_neg_b
    
    res = ""
    if (a_better_on_pos and a_better_on_neg) or (a_better_on_pos and equal_on_neg) or (equal_on_pos and a_better_on_neg):
        res='outranks'
    elif equal_on_pos and equal_on_neg:
        res='indifferent'
    else: #(a_better_on_pos and not a_better_on_neg) or (not a_better_on_pos and not a_better_on_neg)
        res='incomparable'
    return(res)

def determine_order(pairwise_s):
    '''
    Creates an ordering of the alternatives based on the given
    pairwise comparisons

    Parameters
    ----------
    pairwise_s : pd.Series
        A multi-index series  with indices (a,b) containing the 
        pairwise comparison for alternative a to b

    Returns
    -------
    List of sets containing the ordered alternatives where
    mutually incomparable alternatives are in the same set.
    May encounter issues for groups of 3 or more alternatives where
    some pairs are mutually incomparable, but one or more in the set
    outranks another in the set.
    
    Should compare with the pairwise comparison info for certainty.

    '''
    
    alts = set(pairwise_s.index.get_level_values(0)) # gets the alternatives from the first level of the list
    order = []
    
    while alts:
        times_outranked = pd.Series({alt:pairwise_s.loc[(list(set(alts) - {alt}),[alt])].value_counts()['outranks'] if 'outranks' in pairwise_s.loc[(list(set(alts) - {alt}),[alt])].value_counts() else 0 for alt in alts})
        best = set(times_outranked[times_outranked==0].index)
        order.append(best)
        alts -= best 
    return(order)



if __name__ == '__main__':
    df = pd.read_excel('PROMETHEE Example.xlsx','Data',index_col=0)
    df2 = df.drop(columns='Comfort')
    crits = list(df2.columns)
    crit_params = df2.loc[['w_i','q_i','p_i'],:]
    crit_params = pd.concat([crit_params,pd.DataFrame({crit:{'direction':direct,
                                                             'method':'linear'} for crit,direct in zip(crits,['decreasing','decreasing','increasing','increasing'])})],axis='index')
    alt_vals = df2.drop(index=['w_i','q_i','p_i'],inplace=False)
    alts = list(alt_vals.index)
    
    res = promethee(crits=crits,alts=alts,
                    crit_params=crit_params,
                    alt_vals=alt_vals)
    print('***** P_i(a,b) *****:')
    print(res['alt_prefs'].unstack())
    
    print('***** P(a,b) *****')
    print(res['pref_inds'].unstack())
    
    print('***** Outflows *****')
    print(res['outflows'])
    
    print('***** Pairwise Comps *****')
    print(res['pairwise_comps'].unstack())
    
    print('***** Order *****')
    print(res['order'])
    
    
    