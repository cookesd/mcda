# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:18:03 2020

@author: dakar
"""

import pandas as pd
import numpy as np
import pyomo.environ as pyo


def goal_programming(orig_val_df,orig_goals,monotonicity,method='a',classes = None,weights=None,single=True,num_sols = 1,verbose=False,normalize = False):
    '''Implements goal programming methods to determine the alternative that is closest to goals
    
    @param val_df: DF holding values for each alternative (rows) on each criteria (columns)
    @param goals: a series of goal values for each criteria
    @param monotonicity: a series keyed by the criteria stating if it's monotonically positive, negative, or no
    @param method: either 'a' for Archimedean (minimizes sum of weighted deviations), 'v' for Chevyshev (minimizes max deviation),
                    'p' for Preemptive (minimizes weighted deviations in preemptive order; requires classes)
    
    returns value for each alternative in the optimal solution'''
    
    if normalize == True:
        bound = 10
    else:
        bound = orig_val_df['List Price'].max() - orig_goals['List Price']#.apply(max,axis='columns')
    epsilon = 0.00001
    calc_type = {'a':'Archimedian','v':'Chebyshev','p':'Preemptive'}
    
    ### Make the model ###
    model = pyo.ConcreteModel(name='Goal Programming Model')
    alts = list(orig_val_df.index)
    crits = list(orig_val_df.columns)

    
    model.x = pyo.Var(alts,within=pyo.Binary)
    model.delta_pos = pyo.Var(crits,within = pyo.NonNegativeReals) # the positive delta variables (for values less than greater is better crits)
    model.delta_neg = pyo.Var(crits,within = pyo.NonNegativeReals) # the negative delta variables (for values greater than less is better crits)
    model.Delta = pyo.Var(within=pyo.NonNegativeReals) # for chebyshev goal programming
    
    model.pos_del_bin = pyo.Var(crits,within=pyo.Binary)
    model.neg_del_bin = pyo.Var(crits,within=pyo.Binary)
    
    ######## Normalize based on monotonicity ##########
    val_df = orig_val_df.copy()
    goals = orig_goals.copy()
    if normalize:
        for crit in crits:
#             print('before ({})'.format(crit))
#             print(val_df[crit])
            if monotonicity[crit] == 'positive':
                val_df[crit] = bound*(orig_val_df[crit] / orig_goals[crit]) # the value for an alternative is 1 if = to goal, >1 if better and \in [0,1) if less than goal
                goals[crit] = bound # normalize the goal value to 1 and want val + delta_pos >= goal
#                 print(val_df[crit])

            elif monotonicity[crit] == 'negative':
                val_df[crit] = bound - bound*(orig_goals[crit] / orig_val_df[crit].where(orig_val_df[crit] != 0, epsilon)) # replaces 0's with very small number (new val is 0 if = goal, 1 if = \infty, < 0 if less than goal)
                goals[crit] = 0 # normalize the goal value to 0 and want val - delta_neg <= goal
#                 print(val_df[crit])
            else: # the criteria is non-monotonic so we need to get a scale that accounts for values above and below the goal
            # use normal distribution and distribute about that
            # Pg. 16 (Hwang, C.L and Yoon, K.P., Multiple Attribute Decision Making: An Introduction)
            # Sage University Paper Series. Quantitative Applications in the Social Sciences
                if val_df[crit].std() == 0:
                    print('All values for criteria {} are equal, so not using in analysis'.format(crit))
                    crits.remove(crit)
                else:
                    z = (orig_val_df[crit] - orig_goals[crit]) / orig_val_df[crit].std()
                    val_df[crit] = bound*(np.exp(-(z**2)/2))
                    goals[crit] = bound # normalize the goal value to 1 and want val + delta_pos - delta_neg == goal
#                     print(val_df[crit])
    else:
        val_df = orig_val_df.copy()
        goals = orig_goals.copy()
        
#     if verbose:
#         print(val_df)
#         print(pd.concat([goals,monotonicity],axis='columns'))
                
    if verbose:
        print(val_df)
        print(goals)
    
    model.cuts = pyo.ConstraintList()
#     model.pprint()
    def obj_rule(model,c=None):
        if method == 'a':
            return(sum([weights[crit]*(model.delta_pos[crit] + model.delta_neg[crit]) for crit in crits]))
        elif method == 'v':
            # returns the largest weighted deviation
            return(model.Delta)
        elif method == 'p':
            # returns the obj for the current class
            return(sum([weights[crit]*(model.delta_pos[crit] + model.delta_neg[crit]) for crit in c])) 
    model.obj = pyo.Objective(rule=obj_rule,sense=pyo.minimize) # need to fix to pass the class for preemptive
    
    
    def delta_rule(model,crit):
        if monotonicity[crit] == 'positive':
            return(sum([model.x[alt]*val_df.loc[alt,crit] for alt in alts]) + model.delta_pos[crit] - model.delta_neg[crit] >= goals[crit] + epsilon)
        elif monotonicity[crit] == 'negative':
            return(sum([model.x[alt]*val_df.loc[alt,crit] for alt in alts]) + model.delta_pos[crit] - model.delta_neg[crit] <= goals[crit] + epsilon)
        else:
            return(sum([model.x[alt]*val_df.loc[alt,crit] for alt in alts]) + model.delta_pos[crit] - model.delta_neg[crit] == goals[crit] + epsilon)
    model.delta_cons = pyo.Constraint(crits,rule=delta_rule)
    
    def Delta_rule(model,crit):
        return(weights[crit]*(model.delta_pos[crit] + model.delta_pos[crit]) <= model.Delta)
    model.Delta_cons = pyo.Constraint(crits,rule=Delta_rule)
    
    def pos_del_bin_rule(model,crit):
        return(model.delta_pos[crit] <= 2*bound*model.pos_del_bin[crit])
    model.pos_del_bin_cons = pyo.Constraint(crits,rule=pos_del_bin_rule)
    
    def neg_del_bin_rule(model,crit):
        return(model.delta_neg[crit] <= 2*bound*model.neg_del_bin[crit])
    model.neg_del_bin_cons = pyo.Constraint(crits,rule=neg_del_bin_rule)
    
    def del_bin_rule(model,crit):
        return(model.pos_del_bin[crit] + model.neg_del_bin[crit] == 1)
    model.del_bin_cons = pyo.Constraint(crits,rule=del_bin_rule)
    
#     def Delta_rule2(model,crit):
#         return(weights[crit]*(model.delta_neg[crit]) <= model.Delta)
#     model.Delta_cons2 = pyo.Constraint(crits,rule=Delta_rule2)
    
    def single_sol_rule(model):
        '''Says you can only select one solution'''
        return(sum([model.x[alt] for alt in alts]) ==1)
    if single:
        model.single_sol = pyo.Constraint(rule=single_sol_rule)
    
    ##### Solve the model #####
    sols = [] # empty list to hold the solutions
    ranked_alts = [None for i in range(num_sols)]
    obj_vals = {i:None for i in range(num_sols)}
#     deltas = pd.DataFrame(columns = [i for i in range(num_sols)])
#     deltas = pd.DataFrame(index=crits+['max','total'])
    deltas = pd.DataFrame({'Criteria':{i:i for i in crits+['max','total']},
                          'Type':{i:'Criteria' if i in crits else i for i in crits+['max','total']}})
    max_crits = {alt:None for alt in alts}
    for i in range(num_sols):
#         print(i,method,normalized)
        solver = pyo.SolverFactory('glpk')
        sols.append(solver.solve(model))
        # get a dict of alternatives with non-zero values
        
        ranked_alts[i] = [alt for alt in alts if pyo.value(model.x[alt]) ==1]

        
        # adds constraint that at least one decision variable must be different
        model.cuts.add(len(ranked_alts[i]) - sum([model.x[alt] for alt in ranked_alts[i]]) + sum([model.x[alt] for alt in alts if alt not in ranked_alts[i]]) >=1)
        obj_vals[i] = pyo.value(model.obj)
        
        temp = pd.Series({crit:-weights[crit]*pyo.value(model.delta_neg[crit]) if pyo.value(model.delta_neg[crit]) > epsilon
                                                     else weights[crit]*pyo.value(model.delta_pos[crit]) for crit in crits})
        if verbose:
            if (ranked_alts[i][0] in ['Meadow Ln 2']) & (normalize == True):
                print('\n*******', ranked_alts[i][0],'******')
                print(temp)
                print(pd.Series({crit:(-weights[crit]*pyo.value(model.delta_neg[crit]),
                                       weights[crit]*pyo.value(model.delta_pos[crit])) for crit in crits}))
        deltas[ranked_alts[i][0]] = pd.concat([temp,pd.Series({'max':np.abs(temp).max(),'total':np.abs(temp).sum()})],axis='index')
        max_crits[ranked_alts[i][0]] = [crit for crit in crits if temp[crit]==np.abs(temp).max()]
    output = {'model':model,'solutions':sols,'ranked alternatives':ranked_alts,'objective values':obj_vals,'deltas':deltas,'max_crits':max_crits}
    return(output)