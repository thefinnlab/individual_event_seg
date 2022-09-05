''' Note - the outputs of this .py script is available in data/figure_2_4.../linear_models'''
''' We also use relative paths for people to rerun this code and provide the matrices. If you want to run this, just make sure save-out directory is changed unless you want to overwrite the existent outputs that we provide.'''

import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.utils import check_random_state
import statsmodels.api as sm
import statsmodels.formula.api as smf


#### Define functions
def permute_data(matrix, perm):
    '''perm = random state'''

    data1 = pd.DataFrame(matrix)
    random_state = check_random_state(perm)
    data_row_id = range(data1.shape[0])
    permuted_ix = random_state.choice(data_row_id, size=len(data_row_id), replace=False)
    new_fmri_dist = data1.iloc[permuted_ix, permuted_ix].values
    
    return new_fmri_dist 


def bootstrap_data(data,boot):
    '''boot = random_stat'''
    """Bootstrap helper function. Sample with replacement and apply function"""

    
    random_state = check_random_state(boot)
    data_row_id = range(data.shape[0])
    boot_mat = data[random_state.choice(data_row_id, size=len(data_row_id), replace=True)]
    
    return boot_mat


'''Step 4: Doing a linear regression to get the beta values and residual values'''

#doing ordinary least squares model

''' Set up a function to put together a linear model '''


def ols_linear_reg(to_predict, predictors, df, print_model):
    
    ''' to_predict should be the Y value that you want to predict'''
    ''' fixed should be a list of values that you want to include in your model in the order that you want them in.
        Example: ['Motion','USE'] to get Motion + USE'''
    ''' df is the pandas dataframe with the columns that correspond to what is in to_predict and fixed_'''
    ''' fixed of interest should be a string that has the variable that you care about (if there is one)'''
    ''' returns the beta value and the model outputs'''
    #inside function
    predictors_joined = " ".join(sum([[i, '+'] for i in predictors], [])[:-1])
    if print_model==True:
        print('The model you are running is:', f'{to_predict}~{predictors_joined}+const')

    #df = sm.add_constant(df)
        
    reg1 = smf.ols(f'{to_predict}~{predictors_joined}',data=df).fit()


    intercept = reg1.params['Intercept']
    beta_dict = {}
    for val in predictors:
        beta_dict[val] = reg1.params[val]
    if print_model==True:
        print(beta_dict)
    return reg1, beta_dict, reg1.resid, intercept


''' Step 5: Get a p-value!'''
# build up a distribution of the permutated betas and see if the "real" beta is comparable
def perm_p_val(null_dist,obs_diff):
    n_perms = len(null_dist)
    
    sign = np.sign(obs_diff-np.mean(null_dist))
    if sign==1:
        count = (null_dist >= obs_diff).sum()
    elif sign==-1:
        count = (null_dist <= obs_diff).sum()
    else:
        print('CHECK ERROR')
    p = (count + 1)/(n_perms + 1) 

    return p, sign

def boot_p_val(boot_dist,obs_diff):
    null_dist = boot_dist-obs_diff
    
    nboot = len(null_dist)
    
    sign = np.sign(obs_diff-np.mean(null_dist))
    if sign == 1:
        count = (null_dist >= obs_diff).sum()
    elif sign==-1:
        count = (null_dist <= obs_diff).sum()
    else:
        print('CHECK ERROR')
    
    pval = (count+1)/(nboot +1)
    return pval, sign

def get_matchz_matrix(mov,node):
    matchz_dict = np.load('../../data/event_seg/matchz_matrices_concat.npy',allow_pickle=True).item()
    matchz_node = matchz_dict[mov][f'node_{node}']
    
    return matchz_node


def get_motion(mov,method):
    ''' Method can be ISC or AnnaK'''
    motionISC_matrix = np.load(f'../../data/fig_2_4_neural_event_boundaries/matrices/motion/motion{method}_matrix_{mov}.npy') 
    
    return motionISC_matrix


def get_memory_AnnaK(mov):
    memoryAnnaK_matrix = np.load(f'../../data/fig_2_4_neural_event_boundaries/matrices/memory/memoryAnnaK_matrix_{mov}.npy') 
    
    return memoryAnnaK_matrix

def mean_center(lower_tri):
    ''' Make sure you are feeding in the lower triangle of the matrix'''
    return [val-np.nanmean(lower_tri) for val in lower_tri]

def drop_nans_for_mantel(x):
    where_nan = np.argwhere((np.isnan(x)))

    if len(where_nan) > 0:
        print('nan',node)
        for c in range(len(where_nan)):
            x[where_nan[c]] = np.nanmin(x)
    return x
    
    
''' Defining Variables'''
predictors = ['motionISC','motionAnnaK','memoryAnnaK']

mov_val = int(sys.argv[1])
movies = ['growth','defeat','iteration','lemonade']
mov = movies[mov_val]

motionISC_matrix = get_motion(mov,'ISC')
motionAnnaK_matrix = get_motion(mov,'AnnaK')
memoryAnnaK_matrix = get_memory_AnnaK(mov)

low_t_inds = np.tril_indices(motionISC_matrix.shape[0], k=-1)
dict_mats = {}
dict_mats['motionISC'] = motionISC_matrix
dict_mats['motionAnnaK'] = motionAnnaK_matrix
dict_mats['memoryAnnaK'] = memoryAnnaK_matrix


dict_of_all_values = {}


for node in range(1,101):
    dict_of_all_values[node] = {}

for node in range(1,101):
    matchz = drop_nans_for_mantel(get_matchz_matrix(mov,node))
    
    ''' Getting the true residuals'''
    df_OG = pd.DataFrame()
    #df_OG['OG_val'] = mean_center(matchz[low_t_inds])
    df_OG['OG_val'] = matchz[low_t_inds]

    for vals in predictors:
        df_OG[vals] = mean_center(dict_mats[vals][low_t_inds])
    reg_mod, beta,res, intercept_true = ols_linear_reg('OG_val',predictors,df_OG, print_model=True)


    df_boot = pd.DataFrame()
    nboots = 10000
    boot_intercept_null = []
    for nboot in range(nboots):
        matchz_boot_x = bootstrap_data(matchz,boot=nboot)

        #df_boot['matchz_boot'] = mean_center(matchz_boot_x[low_t_inds])
        df_boot['matchz_boot'] = matchz_boot_x[low_t_inds]


        ''' Bootstrap the other effects in the model'''

        boot_dict = {}
        ''' Note dict_mat was made at the top and has MotionISC in there. Did it this way in case we want to reference multiple variables'''
        for vals in predictors:
            boot_dict[vals] = bootstrap_data(dict_mats[vals],boot=nboot)
            df_boot[vals] = mean_center(boot_dict[vals][low_t_inds])

        reg_boot, beta_boot, res_boot, intercept = ols_linear_reg('matchz_boot',predictors,df_boot, print_model=False)

        boot_intercept_null.append(np.median(intercept))



    p,sign = boot_p_val(np.array(boot_intercept_null),intercept_true) #CHECK THIS
    print(f'p_value is {round(p,4)}')

    dict_of_all_values[node]['beta'] = beta
    dict_of_all_values[node]['res'] = res
    dict_of_all_values[node]['p'] = p
    dict_of_all_values[node]['sign'] = sign
    dict_of_all_values[node]['intercept'] = intercept_true


    #Save:

    to_predict = 'matchz'

    #change the directories here !! 
    save_dir = '../../data/fig_2_4_neural_event_boundaries/_linear_models/'
    label = ''.join([to_predict]+['+']+predictors)
    print(label)
    np.save(save_dir + label + f'_{mov}_output_vals.npy',dict_of_all_values)
        
