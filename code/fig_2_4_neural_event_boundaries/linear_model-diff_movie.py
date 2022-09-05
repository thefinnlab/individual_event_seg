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
        
    #reg1 = smf.ols(f'{to_predict}~{predictors_joined}+const',data=df).fit()
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
    
    
''' Defining Variables'''

movie_combos = [('growth','defeat'),('growth','iteration'),('growth','lemonade'), 
                ('defeat','iteration'),('iteration','lemonade'),('defeat','lemonade')]

mov_idx = int(sys.argv[1])

mov1 = movie_combos[mov_idx][0]
mov2 = movie_combos[mov_idx][1]
print(mov1)
print(mov2)
predictors = ['diff_motionISC','diff_motionAnnaK','diff_memoryAnnaK']


motionISC_matrix_x = get_motion(mov1,'ISC') 
motionISC_matrix_y = get_motion(mov2,'ISC') 
motionAnnaK_matrix_x = get_motion(mov1,'AnnaK') 
motionAnnaK_matrix_y = get_motion(mov2,'AnnaK')  
memoryAnnaK_matrix_x = get_memory_AnnaK(mov1)
memoryAnnaK_matrix_y = get_memory_AnnaK(mov2)

diff_motionISC = np.subtract(motionISC_matrix_x,motionISC_matrix_y)
diff_motionAnnaK = np.subtract(motionAnnaK_matrix_x,motionAnnaK_matrix_y)
diff_memoryAnnaK = np.subtract(memoryAnnaK_matrix_x,memoryAnnaK_matrix_y)


dict_mats = {}
dict_mats['diff_motionISC'] = diff_motionISC #note it is the difference!
dict_mats['diff_motionAnnaK'] = diff_motionAnnaK #note it is the difference!
dict_mats['diff_memoryAnnaK'] = diff_memoryAnnaK #note it is the difference!


dict_of_all_values = {}

for node in range(1,101):
    dict_of_all_values[node] = {}
    
for node in range(1,101):
    ''' Load in the data that you need'''
    matchz_x = get_matchz_matrix(mov1,node)
    low_t_inds = np.tril_indices(matchz_x.shape[0], k=-1)
    matchz_y = get_matchz_matrix(mov2,node)

    df_boot = pd.DataFrame()

    nboots = 10000
    boot_intercept_null = []
    for nboot in range(nboots):
        ''' Get the difference between match_z matrices'''
        matchz_boot_x = bootstrap_data(matchz_x,boot=nboot)
        matchz_boot_y = bootstrap_data(matchz_y,boot=nboot)
        diff_matchz_boot = np.subtract(matchz_boot_x,matchz_boot_y)

        df_boot['diff_matchz_boot'] = diff_matchz_boot[low_t_inds] #do not want to mean center the target variable

        ''' Bootstrap the other effects in the model'''
        for vals in predictors:
            #want to mean center the predictors
            df_boot[vals] = mean_center(bootstrap_data(dict_mats[vals],boot=nboot)[low_t_inds])

        
        ''' Add in all of the bootstrapped values to the model (which is all in a bootstrapped df)'''
        reg_boot, beta_boot, res_boot, intercept = ols_linear_reg('diff_matchz_boot',predictors,df_boot, print_model=False)
        boot_intercept_null.append(np.median(intercept)) 

    ''' Getting the true residuals'''
    OG_diff = np.subtract(matchz_x,matchz_y)
    df_OG = pd.DataFrame()
    df_OG['OG_diff'] = OG_diff[low_t_inds]
    for vals in predictors:
        df_OG[vals] = mean_center(dict_mats[vals][low_t_inds]) #grabbing the value from above
    reg_mod, beta, res, intercept_true = ols_linear_reg('OG_diff',predictors,df_OG, print_model=True)

    plt.hist(boot_intercept_null)
    plt.vlines(x=intercept_true,ymin=0,ymax=300,color='red')

    p,sign = boot_p_val(np.array(boot_intercept_null),intercept_true)
    print(f'p_value is {round(p,4)}')

    dict_of_all_values[node]['beta'] = beta
    dict_of_all_values[node]['res'] = res
    dict_of_all_values[node]['p'] = p
    dict_of_all_values[node]['intercept'] = intercept_true
    dict_of_all_values[node]['sign'] = sign
    
    
    to_predict = 'matchz_diff'

    save_dir = '../../data/fig_2_4_neural_event_boundaries/_linear_models/'
    label = ''.join([to_predict]+['+']+predictors)
    print(label)
    np.save(save_dir + label + f'_{mov1}_{mov2}_output_vals.npy',dict_of_all_values)
        
