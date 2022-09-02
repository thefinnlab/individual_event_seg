'''This script takes the boundaries for each individual that were saved out and generates a z-score subject-by-subject "alignment" matrix'''


import warnings
import sys 
import os    
import glob
import numpy as np
from scipy.stats import zscore, norm



data_dir = '/dartfs/FinnLab/clara/K99_EventSeg/data/'

K99_dir = '/dartfs/rc/lab/F/FinnLab/K99_NSD/data_for_Josie/'

schaeffer_dir = '/dartfs/rc/lab/F/FinnLab/clara/K99_EventSeg/data/_schaeffer_rois/'
schaeffer_dir_local = '/dartfs/rc/lab/F/FinnLab/clara/K99_EventSeg/data/_schaeffer_rois/_comparisons_no_range_no_train/'
schaeffer_save_HMM = '/dartfs/rc/lab/F/FinnLab/clara/K99_EventSeg/data/_schaeffer_rois/_comparisons_no_range_no_train/_HMM_plots/'


def match_z(proposed_bounds, gt_bounds, num_TRs):
    nPerm = 1000
    threshold = 3
    np.random.seed(0)

    gt_lengths = np.diff(np.concatenate(([0],gt_bounds,[num_TRs])))
    match = np.zeros(nPerm + 1)
    for p in range(nPerm + 1):
        gt_bounds = np.cumsum(gt_lengths)[:-1]
        for b in gt_bounds:
            if np.any(np.abs(proposed_bounds - b) <= threshold):
                match[p] += 1
        match[p] /= len(gt_bounds)
        gt_lengths = np.random.permutation(gt_lengths)
    
    return (match[0]-np.mean(match[1:]))/np.std(match[1:])

mov = sys.argv[1] #options - growth, iteration, defeat, lemonade
node_list = range(int(sys.argv[2]),int(sys.argv[2])+1) #options - 100 nodes

subs = list(range(0,43)) 
A = np.load(os.path.join(schaeffer_dir + "%s_node1_movie_cutTRs_7N.npy") %(mov),allow_pickle=True)
nTRs = A.shape[1]

def match_loop(nodes):
    nodes = [nodes]
    print(nodes)
    bound_types = []
    for sub in subs:
        bounds = np.load(schaeffer_dir_local + '_individual_subjs_HMM/'+'%s/sub%s_event_boundaries_sm_%s_HMM_zscore.npy' %(mov,sub,mov),allow_pickle=True)

        for node in nodes:        
            bd = bounds.item(0)[node]
            bound_types.append(bd)
    
    matchz_mat = np.zeros((len(bound_types), len(bound_types)))
    for i, b1 in enumerate(bound_types):
        for j, b2 in enumerate(bound_types):
            if i != j:
                matchz_mat[i,j] = match_z(b1, b2, nTRs)
    print(matchz_mat.shape)
    np.save(schaeffer_dir_local + "_matchz_individuals/"+ "%s/matchz_ind_43by43_node%s_%s_no_range_no_train.npy" %(mov,nodes[0],mov),matchz_mat)
        
    
from joblib import Parallel, delayed

Parallel(n_jobs=30, require='sharedmem')(delayed(match_loop)(nodes) for nodes in node_list)



