''' Note: to run this function you need subject x TR x voxel arrays for each node. You can also use the number of events per ROI that are available in data.'''
''' The boundaries per individual per ROI that were generated from this function is what was used in the ss_matchz.py script to generate subject-by-subject alignment matrices. We provide these matrices in the data. '''
''' Even though we are using absolute paths here, we also provide a dictionary that combined the outputs from this script - found here: event_seg/boundary_locations_per_individual.npy'''

import sys
import os
from functools import reduce
import numpy as np
from brainiak.eventseg.event import EventSegment
from scipy.stats import zscore, norm
import sys

schaefer_dir = '/dartfs/rc/lab/F/FinnLab/clara/K99_EventSeg/data/_schaefer_rois/'
schaefer_dir_local = '/dartfs/rc/lab/F/FinnLab/clara/K99_EventSeg/data/_schaefer_rois/_comparisons_no_range_no_train'
schaefer_save_HMM = '/dartfs/rc/lab/F/FinnLab/clara/K99_EventSeg/data/_schaefer_rois/_comparisons_no_range_no_train/_HMM_plots/'


mov = sys.argv[1] #can be iteration, growth, defeat, lemonade
movs = [mov]

min_here = 6
if mov == 'growth':
    max_here = 100
elif mov == 'lemonade':
    max_here = 90
elif mov == 'defeat':
    max_here = 95
elif mov == 'iteration':
    max_here = 150



def fun_avg(nodes,sub,results,bounds_sm):


    for node in nodes:
        
        
        
        
        # This loads in the number of events for each ROI - as determined in the HMM_avg_node_range.py script. We saved out the median number of events in every ROI for others to use (found in data)
        a = np.load(schaefer_dir + '_event_avg/%s/%s_node%s_event_distribution.npy' %(mov,mov,node))
        if len(a) < 100:
            print('ERROR: This is not long enough please test this out: %s %s' %(mov,node))
    


        peak = int(np.round(np.median(a))) 

        k_array=np.arange(peak,peak+1,1)
        k_array

        test_ll = np.zeros(len(k_array))
        node_loaded = np.load(os.path.join(schaefer_dir,"%s_node%s_movie_cutTRs_7N.npy" %(mov,str(node))))
        
        ### Z-SCORING HERE TEMPORARILY IN ORDER TO REMOVE VOXELS THAT ARE BAD
        n_subs, n_ts, n_vox = node_loaded.shape
        bad_vox = []
        node_z = zscore(node_loaded,axis=1)
        for sub_h in range(n_subs):
            for vox in range(n_vox):
                if (np.unique(np.isnan(node_z[sub_h, :, vox]))) == [True]: #if True i.e. that they are all nans
                    bad_vox.append(vox)
        bad_vox = np.unique(bad_vox) #get only the unique values
        if len(bad_vox) > 0:
            print('bad_vox', bad_vox)
            node_loaded = np.delete(node_loaded[:, :, :],bad_vox,axis=2) #need to remove from node_loaded because do not want the z-scored
            print(node_loaded.shape) #correctly removed the length of bad_vox! 
                

        
        for i, k in enumerate(k_array):
            movie_HMM = EventSegment(k)
            movie_test = np.mean(node_loaded[sub:sub+1], axis = 0)
            movie_HMM.fit(movie_test)
            _, test_ll[i] = movie_HMM.find_events(movie_test)
        max_ind_fin = np.argmax(test_ll)
        print('max_ind_fin', max_ind_fin) #will always be 0 now! 
        print('k_array[max_ind_fin]', k_array[max_ind_fin])
        print('ll',test_ll)
    
        
        ### saving out the event boundaries with the SM
        #reasoning to use the split merge option: http://www.chrisbaldassano.com/blog/2020/05/19/splitmerge/
        HMMsm = EventSegment(n_events = k_array[max_ind_fin], split_merge = True)
        HMMsm.fit(movie_test)
        bounds_s = np.where(np.diff(np.argmax(HMMsm.segments_[0], axis = 1)))[0]
        bounds_sm[node] = bounds_s
        print('bounds_s',bounds_s)
        #This output is what is used in ss_matchz.py
        np.save(schaefer_dir_local + '/_individual_subjs_HMM/'+'%s/sub%s_event_boundaries_sm_%s_HMM_zscore.npy' %(mov,sub,mov),bounds_sm)
        
        ### events per ROI
        results[node] = k_array[max_ind_fin]
        print(results[node])
        np.save(schaefer_dir_local + '/_individual_subjs_HMM/'+ '%s/sub%s_events_per_roi_%s_HMM_zscore.npy'%(mov,sub,mov),results)
        

        ll[node] = test_ll
        print(ll)
        ## ll 
        np.save(schaefer_dir_local + '/_individual_subjs_HMM/'+'%s/sub%s_event_boundaries_ll_%s_HMM_zscore.npy' %(mov,sub,mov),ll)



for n in (movs):
    mov = n
    print(mov)
    nodes = list(range(1,1001))
    results = {} #keys are nodes
    bounds_sm = {}
    ll = {}

    subs = range(int(sys.argv[2]),int(sys.argv[2])+1) #added this in so that can use the array option
    print(subs)
    for sub in subs:
        for node in nodes:
            results[node] = []
            #bounds[node] = []
            bounds_sm[node] = []
            ll[node] = []
        for node in nodes:
            result = fun_avg([node],sub,results,bounds_sm)
