# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 12:26:28 2020

@author: dakar

ELECTRE Outranking Multiple Attribute Decision Making Method
"""

import pandas as pd
# import numpy as np
import itertools
# import numbers

def electre(weights,alt_vals,c_threshold=0.7,d_threshold=0.1,weak_c_threshold=0.6,weak_d_threshold=0.3,cardinal_scale=True,ordinal_crit_thresh=None,version=1):
    '''
    Conduct the ELECTRE MCDA method.
    
    Determines the kernel of mutually incomparable / non-outranked alternatives
    and the complete ordering of alternatives.

    Parameters
    ----------
    weights : Series
        The relative importance weight keyed by the criteria.
        They should be non-negative and sum to 1.
    alt_vals : DataFrame
        The value for each alternative on each criterion.
        Indices are the alternatives and columns are the criteria.
    c_threshold : float, optional
        The concordance threshold. If not supplied, set to 0.7.
    d_threshold : float, optional
        The discordance threshold. If not supplied, set to 0.1.
    weak_c_threshold : float, optional (default=None)
        The weak concordance threshold used for ELECTRE II.
        Omitted if version==1. If not supplied, set to 0.6.
    weak_d_threshold : Series, optional (default=None)
        The weak discordance thresholds used for ELECTRE II.
        Same indices as d_thresholds. Omitted if output='kernel'. If not supplied, set to 0.3.
    cardinal_scale : bool, optional (defalut=True)
        Are criteria measured on a cardinal scale
    ordinal_crit_thresh : Series, optional
        The discordance thresholds for the criteria (indices) with an ordinal scale.
        Required if cardinal_scale==False. The default is None.
    version : int {1,2}, optional (default=1)
        Which version of ELECTRE to conduct. The default is 1.

    Returns
    -------
    res : dict
        Result of the ELECTRE problem. Keys include:
        indices : The concordance and discordance indices
        relations : the pairwise ranking relationship of alternatives
        kernels : the kernels if version==1. None otherwise
        orders : dict containing the ascending, descending, and complete orders if version != 2. None otherwise
    '''
    res_keys = ['indices','relations','kernels','orders']
    res = dict(zip(res_keys,[None]*len(res_keys)))
    
    crits = list(weights.index)
    # ordinal_crits = list(ordinal_crit_thresh.keys())
    # non_ordinal_crits = list(set(crits) - set(ordinal_crits))
    alts = list(alt_vals.index)
    alt_pairs = list(itertools.permutations(alts,2))
    
    # if not c_thresholds:
    #     c_thresholds = pd.Series({crit:0.7 for crit in crits})
    # if not d_thresholds:
    #     d_thresholds = pd.Series({crit:0.1 for crit in crits})
    # if not weak_c_thresholds:
    #     weak_c_thresholds = pd.Series({crit:0.6 for crit in crits})
    # if not weak_d_thresholds:
    #     weak_d_thresholds = pd.Series({crit:0.3 for crit in crits})
    
    # Get concordance indices
    c_ab = pd.Series({(a,b):get_concordance(weights,
                                            alt_vals.loc[a,:],
                                            alt_vals.loc[b,:]) for (a,b) in alt_pairs},
                     name='Concordance_Index')
    # Get discordance indices
    if cardinal_scale:
        d_ab = get_discordances(crits,alts,
                                weights,
                                alt_vals)
    else:
        assert isinstance(ordinal_crit_thresh,pd.Series), 'cardinal_scale==False requires dict of criteria thresholds in ordinal_crit_thresh'
        d_ab = pd.Series({(a,b):get_discordance_ordinal(alt_vals.loc[a,:],
                                                         alt_vals.loc[b,:],
                                                         ordinal_crit_thresh)\
                          for (a,b) in alt_pairs},
                         name='Discordance_Index')
    # puts the concordance/discordance indices in a single DataFrame
    res['indices'] = pd.concat([c_ab,d_ab],axis='columns')
    
    # Get outranking relationships and kernel or complete ordering
    if version==1:
        strong_relation = pd.Series({(a,b):get_relation(c_ab[(a,b)],d_ab[(a,b)],
                                                        c_threshold,d_threshold,
                                                        version=version)\
                                     for (a,b) in alt_pairs},name='relation')
        kernels = get_kernel(strong_relation)
        res['relations'] = strong_relation
        res['kernels'] = kernels
    else:
        strong_relation = pd.Series({(a,b):get_relation(c_ab[(a,b)],d_ab[(a,b)],
                                                        c_threshold,d_threshold,
                                                        version=version,c_ba=c_ab[(b,a)])\
                                     for (a,b) in alt_pairs},name='Strong_Relation')
        weak_relation = pd.Series({(a,b):get_relation(c_ab[(a,b)],d_ab[(a,b)],
                                                      weak_c_threshold,weak_d_threshold,
                                                      version=version,c_ba=c_ab[(b,a)])\
                                   for (a,b) in alt_pairs},name='Weak_Relation')
        res['relations'] = pd.concat([strong_relation,weak_relation],axis='columns')
    
        ascending_order = get_ascending_order(alts,strong_relation,weak_relation)
        descending_order = get_descending_order(alts,strong_relation,weak_relation)
        complete_order = get_complete_order(ascending_order,descending_order)
        res['orders'] = dict(zip(['ascending','descending','complete'],[ascending_order,descending_order,complete_order]))
        
    
        
    return(res)
    
def get_concordance(weights,a_vals,b_vals):
    '''
    Calculate the concordance index.
    
    Calculates the Concordance Index (value between 0 and 1) comparing if a
    outranks b defined as the sum of weights for criteria that a does at least
    as well as b divided by the sum of all the weights.

    Parameters
    ----------
    weights : Series
        The criteria weights. Indexed by the criteria.
    a_vals : Series
        The values for alternative a on the criteria. Indexed by the criteria.
    b_vals : Series
        The values for alternative b on the criteria. Indexed by the criteria.

    Returns
    -------
    c_ab : float
        The concordance index comparing if alternative a outranks b.
    '''   
    weights = pd.Series(weights)
    q_ab = [crit for crit in weights.index if a_vals[crit] >= b_vals[crit]]
    num = weights[q_ab].sum()
    denom = weights.sum()
    c_ab = num / denom
    return(c_ab)

def get_discordances(crits,alts,weights,alt_vals):
    '''
    Calculate the concordance index.
    
    Calculates the Concordance Index (value between 0 and 1) comparing if a
    outranks b defined as the sum of weights for criteria that a does at least
    as well as b divided by the sum of all the weights.

    Parameters
    ----------
    crits : iterable
        The criteria.
    alts : iterable
        The alternatives
    weights : Series
        The criteria weights.
    alt_vals : DataFrame
        The values for alternative b on the criteria.
        Indexed by the alternatives with criteria as the columns.

    Returns
    -------
    discordance_indices : Series
        Multi-indexed series of discordance indices (indices are tuple of alternative
        permutations of size 2).
    '''
    alt_pairs = list(itertools.permutations(alts,2))
    # r_abs = pd.Series({(a,b):[crit for crit in crits if alt_vals.loc[b,crit] > alt_vals.loc[a,crit]]\
    #                    for (a,b) in alt_pairs},name='q_ab')
    weighted_discordances = pd.DataFrame({crit:{(a,b):weights[crit]*(max(0,alt_vals.loc[b,crit]-alt_vals.loc[a,crit])) \
                                                for (a,b) in alt_pairs} for crit in crits})
    pair_max = weighted_discordances.max(axis='columns')
    grand_max = pair_max.max() # the max across all the pairs and alternatives
    try:
        discordance_indices = pd.Series(pair_max / grand_max,name='Discordance_Index')
    except ZeroDivisionError:
        print('All alternatives perform exactly the same on all criteria')
    
    return(discordance_indices)


def get_discordance_ordinal(a_vals,b_vals,threshold):
    '''
    Calculate the concordance index.
    
    Calculates the Concordance Index (value between 0 and 1) comparing if a
    outranks b defined as the sum of weights for criteria that a does at least
    as well as b divided by the sum of all the weights.

    Parameters
    ----------
    a_vals : Series
        The values for alternative a on the criteria. Indexed by the criteria.
    b_vals : Series
        The values for alternative b on the criteria. Indexed by the criteria.
    threshold : Series
        The threshold values for discordance on each criteria

    Returns
    -------
    c_ab : float
        The concordance index comparing if alternative a outranks b.
    '''
    exceed_threshold = (b_vals - a_vals) >= threshold
    d_ab = {True:1,False:0}[exceed_threshold.sum() > 0]
    return(d_ab)



    
def get_relation(c_ab,d_ab,c_thresh = 0.7,d_thresh = 0.1,version = 1,c_ba=None):
    '''
    Determine if a outranks b.
    
    Determines if the concordance and discordance relationships for alternative a
    w.r.t. alternative b are better (higher and lower respectively) than the threshold.
    If the version is 

    Parameters
    ----------
    c_ab : float
        The concordance index of alternative a w.r.t. alternative b.
    d_ab : float
        The discordance index of alternative a w.r.t. alternative b.
    c_thresh : float, optional
        The concordance threshold. The default is 0.7.
    d_thresh : float, optional
        The discordance threshold. The default is 0.1.
    version : int {1,2}, optional
        The version of ELECTRE to use. The default is 1.
    c_ba : float, optional
        The concordance index of alternative b w.r.t. alternative a. The default is None

    Returns
    -------
    outranks : bool
        Whether or not alternative a outranks alternative b

    '''
    outranks = (c_ab > c_thresh) & (d_ab < d_thresh)
    if version==2:
        if c_ba:
            outranks = outranks & (c_ab > c_ba)
        else:
            print('You want the ELECTRE II outranking relation but did not\
                  specify a value for c_ba. Returning ELECTRE I.')
    return(outranks)

def get_kernel(relations):
    '''
    Determine the kernel based on the outranking relationship.
    
    Finds the set of mutually incomparable alternatives where every alternative
    outside of the set is outranked by some alternative in the set. The kernel
    is unique if and only if no directed cycles exist in the directed graph made
    from the outranking relationships. This method is inefficient in that it
    explicitely checks if all combinations of alternatives is a kernel.

    Parameters
    ----------
    relations : Series
        Multi-index Series keyed by the alternative pairs (a,b).
        True if a outranks b. False otherwise.

    Returns
    -------
    kernels : list of lists
        The sets of mutually incomparable alternatives where every alternative
         outside of the set is outranked by some alternative in the set.

    '''
    alts = list(relations.index.get_level_values(0).unique())
    # outrank = {alt:[b for b in alts if relations[(alt,b)] and alt != b] for alt in alts} # gets alternatives where the outranking relationship is true and it's not the same alternative
    # outranked_by = {alt:[b for b in alts if relations[(b,alt)] and alt != b] for alt in alts}
    
    # not_outranked = [key for key in outranked_by.keys() if len(outranked_by[key])==0]
    
    all_combinations = [list(itertools.combinations(alts,i)) for i in range(1,len(alts)+1)]
    all_combinations = list(itertools.chain.from_iterable(all_combinations))
    kernels = [combination for combination in all_combinations if is_kernel(combination,set(alts) - set(combination),relations)]
    return(kernels)

def is_kernel(poss_kernel,other_alts,outranks):
    '''
    Check if a set of alternatives is a kernel.

    Parameters
    ----------
    poss_kernel : list
        The set of alternatives to be evaluated as a kernel.
    other_alts : list
        The remaining alternatives.
    outranks : Multi-index Series
        Multi-index Series keyed by the alternative pairs (a,b).
        True if a outranks b. False otherwise.

    Returns
    -------
    bool True if the alternatives are a kernel.

    '''
        # True if every alt in other_alts is outranked by some alt in comb_alts
    crit_1 = all([any([outranks[(alt,o_alt)] for alt in poss_kernel]) for o_alt in other_alts])
        # True if every pair of alts in poss_kernel is incomparable (neither outranks each other)
    if len(poss_kernel) > 1:
        crit_2 = all([(not outranks[(a,b)]) and (not outranks[(b,a)]) for a,b in itertools.combinations(poss_kernel,2)])
    else:
        crit_2 = True
    return(crit_1 and crit_2)


def get_descending_order(alts,strong_relation,weak_relation):
    '''
    Determine the descending order of alternatives

    Parameters
    ----------
    alts : iterable
        The alternatives to order.
    strong_relation : Multi-index Series
        Multi-index Series keyed by the alternative pairs (a,b).
        True if a outranks b according to strong thresholds. False otherwise.
    weak_relation : Multi-index Series
        Multi-index Series keyed by the alternative pairs (a,b).
        True if a outranks b according to weak thresholds. False otherwise.

    Returns
    -------
    order : list of sets
        The descending ordering of alternatives. Alternatives in the same list are incomparable.
    '''
    alts = set(alts)
    order = []
    while alts:
            # set of remaining alternatives that are NOT strongly outranked
        f = set([alt for alt in alts if (1-strong_relation.loc[(list(set(alts)-set([alt])),[alt])]).astype(bool).all()])
            # set of alternatives in f that are NOT weakly outranked by alternatives in f
        f_prime = set([alt for alt in f if (1-weak_relation.loc[(list(set(f)-set([alt])),[alt])]).astype(bool).all()])
        order.append(f_prime) # adds this/these alternatives to the end of the ordering
        alts = alts - f_prime
    return(order)

def get_ascending_order(alts,strong_relation,weak_relation):
    '''
    Determine the ascending order of alternatives

    Parameters
    ----------
    alts : iterable
        The alternatives to order.
    strong_relation : Multi-index Series
        Multi-index Series keyed by the alternative pairs (a,b).
        True if a outranks b according to strong thresholds. False otherwise.
    weak_relation : Multi-index Series
        Multi-index Series keyed by the alternative pairs (a,b).
        True if a outranks b according to weak thresholds. False otherwise.

    Returns
    -------
    order : list of sets
        The ascending ordering of alternatives. Alternatives in the same list are incomparable.
    '''
    alts = set(alts)
    order = []
    while alts:
            # set of remaining alternatives that do NOT strongly outrank any remaining alternatives
        g = set([alt for alt in alts if (1-strong_relation.loc[[alt],(list(set(alts)-set([alt])))]).astype(bool).all()])
            # set of alternatives in g that do NOT weakly outrank any alternatives in g
        g_prime = set([alt for alt in g if (1-weak_relation.loc[[alt],(list(set(g)-set([alt])))]).astype(bool).all()])
        order.insert(0,g_prime)
        alts = alts - g_prime
    return(order)

def get_complete_order(ascending_order,descending_order):
    '''
    Determine the complete ordering of alternatives.
    
    Creates a complete ordering from the ascending ordering and descending ordering
    Uses the intersection of the orderings to reconcile differences
    An alternative a outranks b iff the it is in the same or a better class
    in both orderings.

    Parameters
    ----------
    ascending_order : iterable of iterables
        The ascending ordering of alternatives. Alternatives in the same list are incomparable.
    descending_order : iterable of iterables
        The descending ordering of alternatives. Alternatives in the same list are incomparable.

    Returns
    -------
    intersection : list of lists
        The complete ordering of alternatives.

    '''
        # makes a dict keyed by alternatives saying which order it is in the order
    ascend_alt_dict = {alt:[i for i,j in enumerate(ascending_order) if alt in j] for alt in itertools.chain.from_iterable(ascending_order)}
    descend_alt_dict = {alt:[i for i,j in enumerate(descending_order) if alt in j] for alt in itertools.chain.from_iterable(descending_order)}

        
    # get outranking relationship for each pair of alternatives (True if a outranks b in both)
    alts = set(itertools.chain.from_iterable(ascending_order))
    inds = list(itertools.permutations(alts,2)) # gets all permutaions of length 2 of alts (gets both directions)
    u_outranks_v = pd.Series({(u,v):(ascend_alt_dict[u] <= ascend_alt_dict[v]) & (descend_alt_dict[u] <= descend_alt_dict[v]) for u,v in inds},
                         name='comparison')
    # makes the intersection ordering by iteratively removing alternatives that
    # are not outranked by remaining alternatives
    intersection = []
    while alts:
            # gets all the alts that are not outranked by anything that's remaining
        next_alts = [alt for alt in alts if (1-u_outranks_v.loc[(list(alts),[alt])]).astype(bool).all()]
        intersection.append(next_alts)
        alts -= set(next_alts)
    return(intersection)


if __name__ == '__main__':
    df = pd.read_excel('24 ELECTRE Example.xlsx','Data',header=1,index_col=1,nrows=8).iloc[:,1:]
    # Extract the weights into a series and remove the row from
    weights = df.loc['Weights',:].transpose()
    df.drop(index='Weights',inplace=True)
    # Read the rating scale
    scale_df = pd.read_excel('24 ELECTRE Example.xlsx','Data',
                                         skip_rows=11,header=11,index_col=0).iloc[:,0:1]
    # Add a column ordering the scale
    scale_df = pd.concat([scale_df,pd.Series(list(range(1,len(scale_df.index)+1)),index=scale_df.index)],axis='columns')
    scale_df.columns = ['Name','Value']
    # Replace the original values with numbers for elementwise comparisons
    num_df = pd.DataFrame({col:df[col].map(lambda x: scale_df.loc[x,'Value']) for col in df.columns})
    
    res = electre(weights=weights,
                  alt_vals=num_df,
                  c_threshold=0.8,d_threshold=0.1,
                  weak_c_threshold=0.7,weak_d_threshold=0.5,
                  cardinal_scale=False,ordinal_crit_thresh=pd.Series({crit:3 for crit in weights.index}),
                  version=2)
    print('***** INDICES *****:')
    print(res['indices'].unstack())
    
    print('***** RELATIONS *****')
    print(res['relations'].unstack())
    
    print('***** KERNELS *****')
    print(res['kernels'])
    
    print('***** ORDERS *****')
    for key,value in res['orders'].items():
        print(key,': ',value,'\n')
    # print(res['orders'])
