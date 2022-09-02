''' Note the median of the outputs from this function are saved out in data. Just including it to see how we fit the HMM. Code inspired by: https://naturalistic-data.org/content/Event_Segmentation.html'''


import sys 
import os    
import glob
import numpy as np
from brainiak.eventseg.event import EventSegment
from scipy.stats import zscore, norm
from matplotlib import pyplot as plt



schaefer_dir = '/dartfs/rc/lab/F/FinnLab/clara/K99_EventSeg/data/_schaefer_rois/'


mov = sys.argv[1] #this grabs either iteration, growth, defeat or lemonade as a string

min_here = 6
if mov == 'growth':
    max_here = 100
elif mov == 'lemonade':
    max_here = 90
elif mov == 'defeat':
    max_here = 95
elif mov == 'iteration':
    max_here = 150
k_array = np.arange(min_here, max_here, 5)
test_ll = np.zeros(len(k_array))
print(mov)


node = int(sys.argv[2]) #get the node number

node_loaded = np.load(os.path.join(schaefer_dir,"%s_node%s_movie_cutTRs_7N.npy" %(mov,str(node))))



### Z-SCORING HERE TEMPORARILY IN ORDER TO REMOVE VOXELS THAT ARE BAD (SEE ABOVE)
n_subs, n_ts, n_vox = node_loaded.shape
bad_vox = []
node_z = zscore(node_loaded,axis=1)
for sub_h in range(n_subs):
    for vox in range(n_vox):
        if (np.unique(np.isnan(node_z[sub_h, :, vox]))) == [True]: #means that they are True i.e. that they are all nans
            bad_vox.append(vox)
bad_vox = np.unique(bad_vox) #get only the unique values
if len(bad_vox) > 0:
    print('bad_vox', bad_vox)
    node_loaded = np.delete(node_loaded[:, :, :],bad_vox,axis=2) #need to remove from node_loaded because do not want the z-scored
    print(node_loaded.shape) #correctly removed the length of bad_vox! 


test_ll = np.zeros(len(k_array))
node_loaded = np.load(os.path.join(schaefer_dir, "%s_node%s_movie_cutTRs_7N.npy" % (mov, str(node))))
print(node_loaded.shape)


### training on half the data
### want to randomly pick subjects that fit in each split 

choices = np.array([22,21])
np.random.shuffle(choices)
choice = choices[0]
choice

k_val_array = []
choices = np.array([22,21])
comp = 100
for rand in range(comp): 
    print('rand',rand)
    np.random.shuffle(choices)
    choice = choices[0]
    np.random.shuffle(node_loaded)
    for i, k in enumerate(k_array):
        movie_train = np.mean(node_loaded[:choice], axis=0)
        movie_HMM = EventSegment(k)
        movie_HMM.fit(movie_train)
        movie_test = np.mean(node_loaded[choice:], axis=0)
        _, test_ll[i] = movie_HMM.find_events(movie_test)
    max_ind = np.argmax(test_ll)
    
    print('Max is %d events' % k_array[max_ind])
    print('finished with part 1 for node')
    
    if max_ind < 6: #so that it doesn't go below 6!
        k_small = np.arange((k_array[max_ind]), (k_array[max_ind]) + 5, 1)
    else:
        k_small = np.arange((k_array[max_ind]) - 5, (k_array[max_ind]) + 5, 1)
    test_ll_small = np.zeros(len(k_small))
    
    for i, k in enumerate(k_small):
        movie_train = np.mean(node_loaded[:choice], axis=0)
        movie_HMM = EventSegment(k)
        movie_HMM.fit(movie_train)
        movie_test = np.mean(node_loaded[choice:], axis=0)
        _, test_ll_small[i] = movie_HMM.find_events(movie_test)
    
    max_ind_fin = np.argmax(test_ll_small)
    print('Max is %d events' % k_small[max_ind_fin])
    test_ll_small[max_ind_fin]
    event_bounds = np.where(np.diff(np.argmax(movie_HMM.segments_[0], axis = 1)))[0]
    nTRs = movie_test.shape[0]

    k_val_array.append(k_small[max_ind_fin])
    np.save(schaefer_dir + '_event_avg/' + '/%s/%s_node%s_event_distribution.npy' %(mov,mov,node), k_val_array)


#plotted the distribution
#plt.hist(k_val_array)
#plt.title('node %s, mov = %s, comparisons = %s' %(node,mov,comp))
#plt.savefig(schaefer_dir + '_event_avg/' + '_avg_distributions/%s/%s_node%s_event_distribution.png' %(mov,mov,node))
np.save(schaefer_dir + '_event_avg/' + '/%s/%s_node%s_event_distribution.npy' %(mov,mov,node), k_val_array)

