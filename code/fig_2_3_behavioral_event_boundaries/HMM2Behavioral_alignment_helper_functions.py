import numpy as np
import warnings
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import math
from scipy.stats import norm
import nibabel as nib


def get_peak_locations(nTRs,results_plot, height=51, behavioral=False):

    ''' Account for HRF'''
    if behavioral==True:
        print('BEHAVIORAL! Adding 3 TRs to account for HRF')
        for sub in range(len(results_plot)): 
            temp = []
            for i in results_plot[sub]:
                if i < nTRs+3:
                    temp.append(i+ 3)
                else:
                    warnings.warn('WARNING: ISSUE WITH TRS- CHECK')
                    
            results_plot[sub] = temp
            '''ACCOUNTING FOR HRF'''

    
    counts = {}
    for TR in range(nTRs+1):
        counts[TR] = 0


    counts = {}

    for TR in range(nTRs+1):
        counts[TR] = 0

    '''Going through each subject and getting if there is an event at that TR or +- 3TR'''
    not_included = []
    for tot in range(len(results_plot)): 

        temp = results_plot[tot]
        temp = [round(num) for num in temp]
        for b in temp:
            for range_b in range(b-3,b+3):
                if (range_b >= 0) and (range_b<nTRs+1): 
                    ''' Accounting for within TR range re: +3-3'''
                    counts[range_b] += 1
                else:
                    not_included.append(range_b)
    print('FYI: You are removing below 0s or above the number of TRs, but still accounting for the within TR +-3 Range')
    print(f'The TRs removed are {not_included} \n. Confirm they are over the number of TRs - {nTRs}')
    
    values = [(j/len(results_plot))*100 for j in counts.values()]

    peaks, _ = find_peaks(list(values), height=height) #more than half of the pairs
    return (peaks,counts)

    
def plot_locations(mov, nTRs,results_plot,counts, peaks_new,height=51):
    
    plt.figure(figsize=(20, 5))
    values = [(j/40)*100 for j in counts.values()]
    plt.bar(counts.keys(), values)
    plt.xlabel('Time (s)')
    plt.ylabel('% Subs')
    
    plt.title('Number of times this TR was in sub segmentation [+- 3]')
    plt.suptitle(mov)

    height = height 

    #peaks, _ = find_peaks(list(values), height=height) #more than half of the pairs
    plt.plot(peaks_new, np.array(list(values))[peaks_new], "x", color='red')
    plt.show()
    
def clean_peak_loc(peaks, counts, range_to_remove,get_middle):
    to_remove = []
    to_add = []
    peaks = list(peaks)
    for ii,c in enumerate(peaks):
        if ii == len(peaks)-1:
            None

        elif  peaks[ii+1] - peaks[ii] <= range_to_remove:
            a = peaks[ii]
            b = peaks[ii+1]
            if list(counts.values())[a] < list(counts.values())[b]:
                to_remove.append(a)
            elif list(counts.values())[b] < list(counts.values())[a]:
                to_remove.append(b)
            else:
                if get_middle==True:
                    print(f'TRs {a} and {b} are the same height of {list(counts.values())[a]}. Choosing the TR in the middle for this - {math.ceil(np.median([a,b]))}')
                    to_remove.append(a)
                    to_remove.append(b)
                    '''Rounding .5s up not down'''
                    to_add.append(math.ceil(np.median([a,b])))
                #None
                
                
    print(f'Removing {len(to_remove)} timepoints')
    
    peaks_new = []
    for val in peaks:
        if val not in to_remove:
            peaks_new.append(val)
        else:
            None
        
    ''' Add in the TRs that are in the middle between two heights'''
    for val in to_add:
        peaks_new.append(val)
    
    peaks_new = np.sort(np.array(peaks_new))
    
    return peaks_new
def match_z(proposed_bounds, gt_bounds, num_TRs,nPerm):
    #print(nPerm)
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


def match_loop(mov,brain_list,beh_list): #This is for the quick plots! 
    
    if mov == 'growth':
        nTRs = 503
    elif mov == 'defeat':
        nTRs = 476
    elif mov == 'iteration':
        nTRs = 746
    elif mov == 'lemonade':
        nTRs = 449
    

    bound_types = []
    bound_types.append(brain_list)
    bound_types.append(beh_list)
    
    matchz_mat = np.zeros((len(bound_types), len(bound_types)))
    for i, b1 in enumerate(bound_types):
        for j, b2 in enumerate(bound_types):
            if i != j:
                matchz_mat[i,j] = match_z(b1, b2, nTRs,nPerm=1000)
    p = tuple(norm.sf(matchz_mat).tolist())
    return p, matchz_mat


def plot_corr_r_filter_p(OG_dict,pval_list, node_range):
    '''Filtering by p <. 05'''
    plotting_a = []
    list_of_reg = []
    for i in node_range:
        if pval_list[i-1] < .05:
            plotting_a.append(OG_dict[i])
            list_of_reg.append(i-1)
        else:
            plotting_a.append(np.nan)
    return color_rois(plotting_a), list_of_reg

def plot_corr_r_filter_FDR(OG_dict,pval_list, node_range=range(1,101)):
    from statsmodels.stats.multitest import multipletests
    q = multipletests(pval_list,method='fdr_bh')[1]
    list_of_reg = []
    isrsa_plot = []
    node_val = []
    for node in node_range:
        if q[node-1] > .05:
            isrsa_plot.append(np.nan)
        else:
            isrsa_plot.append(OG_dict[node])
            list_of_reg.append(node-1)
            node_val.append(OG_dict[node])

    return color_rois(isrsa_plot), list_of_reg,node_val

def color_rois(values):
    """
    This function assumes you are passing a vector "values" with the same length as the number of nodes in the atlas.
    """
    data_dir = '/dartfs/rc/lab/F/FinnLab/clara/K99_EventSeg/data/'
    atlas_fname = (data_dir + '_masks/'+ 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz')
    schaeffer = nib.load(atlas_fname)
    schaeffer100_data = schaeffer.get_fdata()
    img = np.zeros(schaeffer100_data.shape)
    for roi in range(len(values)):
        itemindex = np.where(schaeffer100_data==roi+1) # find voxels in this node (subtract 1 to account for zero-indexing)
        img[itemindex] = values[roi] # color them by the desired value 
    affine = schaeffer.affine
    img_nii = nib.Nifti1Image(img, affine)
    
    return img_nii





def pvalue_from_null_dist(obs_diff, null_dist):
    n_perms = len(null_dist) 
    count = (null_dist >= obs_diff).sum() #only want positive
    p = (count + 1)/(n_perms + 1) 
    return p


def get_binary_Density(nTRs,results_plot,plot=False,behavioral=False):
    
    '''Generating the Binary to Create a Density Curve Across Subjects'''

    if behavioral==True:
        val_temp = np.zeros(nTRs+4) 
        length_here = nTRs+4
    else:
        val_temp = np.zeros(nTRs)
        length_here = nTRs

    for tot in range(len(results_plot)): #doing minus 1 since the last one is compared to the rest earlier
        temp = results_plot[tot]
        temp = [round(num) for num in temp] 
        '''Rounding here'''

        for b in temp:
            for range_b in range(b-3,b+3):
                if (range_b >= 0) and (range_b<nTRs+1): 
                    ''' Accounting for within TR range re: +3-3'''
                    val_temp[range_b] += 1
                else:
                    print(range_b)

    
    val_temp = [((i/len(results_plot))*100) for i in val_temp]
    if plot==True:
        plt.figure(figsize=(20, 4), dpi=80)

        print(length_here)
        print(len(val_temp))
        plt.scatter(range(length_here), val_temp)
        x = plt.plot(range(length_here), val_temp)


        plt.xlabel('Time (s)')
        plt.ylabel('Percent (out of %s)' %(len(results_plot)))
        plt.title('Peaks based on Distribution of Values')

        plt.show()

    return val_temp