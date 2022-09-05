import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import nibabel as nib
import mantel
from statsmodels.stats.multitest import multipletests
from nilearn.plotting import plot_glass_brain, plot_stat_map,  view_img, view_img_on_surf, plot_surf, plot_surf_stat_map, plot_surf_roi, plot_surf_contours, view_surf
from nilearn.surface import vol_to_surf
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()


def linear_model_USE(mov):
    r_p_df = pd.read_csv('../../data/fig_6_USE_event_boundaries/_partial_mantel_outputs/' + f'{mov}_mantel_r_p_R_generated.csv')
    
    node_OG = r_p_df['rval']
    pval_list = r_p_df['pval']


    plotting_a = []
    listOf_reg = []
    val = []
    ''' Account for p-values calcuated with different order'''
    fdr_corrected_p = multipletests(pval_list,method='fdr_bh')[1]
    
    
    for i in range(100):
        if fdr_corrected_p[i] < .05:
            plotting_a.append(node_OG[i])
            val.append(node_OG[i])
            listOf_reg.append(i)
        else:
            plotting_a.append(np.nan)

    return plotting_a

def get_node_r_val(rois,mov):
    ''' This function is intended to get the value for each node on its own '''
    r_p_df = pd.read_csv('../../data/fig_6_USE_event_boundaries/_partial_mantel_outputs/' + f'{mov}_mantel_r_p_R_generated.csv')
    node_OG = r_p_df['rval']
    
    vals = []
    for val in rois:
        vals.append(node_OG[val])
    return vals


def color_rois(values):
    """
    This function assumes you are passing a vector "values" with the same length as the number of nodes in the atlas.
    """
    data_dir = '../../data/'

    atlas_fname = (data_dir + '_masks/'+ 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz')
    schaeffer = nib.load(atlas_fname)
    schaeffer100_data = schaeffer.get_fdata()
    img = np.zeros(schaeffer100_data.shape)
    for roi in range(len(values)):
        itemindex = np.where(schaeffer100_data==roi+1) 
        img[itemindex] = values[roi] # color them by the desired value 
    affine = schaeffer.affine
    img_nii = nib.Nifti1Image(img, affine)
    
    return img_nii


''' Note all of the contour functions are identical - there are differences per movie based on which ROIs needed to be contoured'''


def plot_isrsa_contours(isrsa_mantel_r2_brain,roi_list,mov, colorbar,vmax,vmin):
    '''roi_list should be list of ROIs that you want colored (leave the other ROIs as 0)'''
    '''Black - 1'''
    '''n_color - 2'''
    
    n_color = 'blue'
    
    cmap='RdBu_r'
    fig,ax = plt.subplots(nrows=1, ncols= 1, figsize=(6,6),subplot_kw={'projection': '3d'})
    fig.tight_layout(h_pad=None,w_pad=None)
    texture = vol_to_surf(isrsa_mantel_r2_brain, fsaverage.pial_left,interpolation='nearest',radius =1, n_samples=1)
    surf_plot1=plot_surf_roi(fsaverage.infl_left, texture, hemi='left',cmap = cmap, axes=ax, colorbar=False, vmin=vmin,  vmax = vmax,
                                 bg_map=fsaverage.sulc_left,darkness=.4)
    texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_left,interpolation='nearest',radius =1, n_samples=1)
    try: 
        plot_surf_contours(fsaverage.infl_left, texture, figure=surf_plot1,legend=False, axes=ax, levels=[1,2],colors=['black',n_color],linewidth=10)#,levels = [texture],  labels=['Fig 3 nodes'])
    except:
        plot_surf_contours(fsaverage.infl_left, texture, figure=surf_plot1,legend=False, axes=ax, levels=[1],colors=['black'],linewidth=10)#,levels = [texture],  labels=['Fig 3 nodes'])
         

    
    
    fig,ax = plt.subplots(nrows=1, ncols= 1, figsize=(6,6),subplot_kw={'projection': '3d'})
    fig.tight_layout(h_pad=None,w_pad=None)
    texture = vol_to_surf(isrsa_mantel_r2_brain, fsaverage.pial_right,interpolation='nearest',radius =1, n_samples=1)
    surf_plot1=plot_surf_roi(fsaverage.infl_right, texture, hemi='right',cmap = cmap, axes=ax, colorbar=False, vmin=vmin,  vmax = vmax,
                                 bg_map=fsaverage.sulc_right,darkness=.4)
    try: 
        texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_right,interpolation='nearest',radius =1, n_samples=1)
        plot_surf_contours(fsaverage.infl_right, texture, figure=surf_plot1,axes=ax, legend=False, levels=[1,2],colors=['black',n_color])#,levels = [texture],  labels=['Fig 3 nodes'])
    except:
        texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_right,interpolation='nearest',radius =1, n_samples=1)
        plot_surf_contours(fsaverage.infl_right, texture, figure=surf_plot1,legend=False,axes=ax,  levels=[1],colors=['black'])#,levels = [texture],  labels=['Fig 3 nodes'])


    fig,ax = plt.subplots(nrows=1, ncols= 1, figsize=(6,6),subplot_kw={'projection': '3d'})
    fig.tight_layout(h_pad=None,w_pad=None)
    texture = vol_to_surf(isrsa_mantel_r2_brain, fsaverage.pial_left,interpolation='nearest',radius =1, n_samples=1)
    surf_plot2=plot_surf_roi(fsaverage.infl_left, texture, hemi='left',cmap = cmap, axes=ax, colorbar=False,vmin=vmin,  vmax = vmax,
                                bg_map=fsaverage.sulc_left, view = 'medial',darkness=.4)
    try:
        texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_left,interpolation='nearest',radius =1, n_samples=1)
        plot_surf_contours(fsaverage.infl_left, texture, view = 'medial', figure=surf_plot2,legend=False, axes=ax, levels=[2],colors=[n_color])#,levels = [texture],  labels=['Fig 3 nodes'])
    except:
        None
 
    
    fig,ax = plt.subplots(nrows=1, ncols= 1, figsize=(6,6),subplot_kw={'projection': '3d'})
    fig.tight_layout(h_pad=None,w_pad=None)
    texture = vol_to_surf(isrsa_mantel_r2_brain, fsaverage.pial_right,interpolation='nearest',radius =1, n_samples=1)
    surf_plot4=plot_surf_roi(fsaverage.infl_right, texture, hemi='right',cmap = cmap, colorbar=False,vmin=vmin,axes=ax,   vmax = vmax,
                                bg_map=fsaverage.sulc_right, view ='medial',darkness=.4)
    try: 
        texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_right,interpolation='nearest',radius =1, n_samples=1)
        plot_surf_contours(fsaverage.infl_right, texture,view ='medial', figure=surf_plot4,legend=False, axes=ax, levels=[1],colors=['black'])#,levels = [texture],  labels=['Fig 3 nodes'])
    except:
        texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_right,interpolation='nearest',radius =1, n_samples=1)
        plot_surf_contours(fsaverage.infl_right, texture,view ='medial', figure=surf_plot4,legend=False, axes=ax, levels=[2],colors=[n_color])#,levels = [texture],  labels=['Fig 3 nodes'])


def plot_isrsa_contours_ite(isrsa_mantel_r2_brain,roi_list,mov, colorbar,vmax,vmin):
    '''roi_list should be list of ROIs that you want colored (leave the other ROIs as 0)'''
    '''Black - 1'''
    '''n_color - 2'''
    
    n_color = 'blue'
    
    cmap='RdBu_r'
    fig,ax = plt.subplots(nrows=1, ncols= 1, figsize=(6,6),subplot_kw={'projection': '3d'})
    fig.tight_layout(h_pad=None,w_pad=None)
    texture = vol_to_surf(isrsa_mantel_r2_brain, fsaverage.pial_left,interpolation='nearest',radius =1, n_samples=1)
    surf_plot1=plot_surf_roi(fsaverage.infl_left, texture, hemi='left',cmap = cmap, axes=ax, colorbar=False, vmin=vmin,  vmax = vmax,
                                 bg_map=fsaverage.sulc_left,darkness=.4)
    texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_left,interpolation='nearest',radius =1, n_samples=1)
    try: 
        plot_surf_contours(fsaverage.infl_left, texture, figure=surf_plot1,legend=False, axes=ax, levels=[1,2],colors=['black',n_color],linewidth=10)#,levels = [texture],  labels=['Fig 3 nodes'])
    except:
        plot_surf_contours(fsaverage.infl_left, texture, figure=surf_plot1,legend=False, axes=ax, levels=[1],colors=['black'],linewidth=10)#,levels = [texture],  labels=['Fig 3 nodes'])
         

    
    fig,ax = plt.subplots(nrows=1, ncols= 1, figsize=(6,6),subplot_kw={'projection': '3d'})
    fig.tight_layout(h_pad=None,w_pad=None)
    texture = vol_to_surf(isrsa_mantel_r2_brain, fsaverage.pial_right,interpolation='nearest',radius =1, n_samples=1)
    surf_plot1=plot_surf_roi(fsaverage.infl_right, texture, hemi='right',cmap = cmap, axes=ax, colorbar=False, vmin=vmin,  vmax = vmax,
                                 bg_map=fsaverage.sulc_right,darkness=.4)
    try: 
        texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_right,interpolation='nearest',radius =1, n_samples=1)
        plot_surf_contours(fsaverage.infl_right, texture, figure=surf_plot1,axes=ax, legend=False, levels=[1,2],colors=['black',n_color])#,levels = [texture],  labels=['Fig 3 nodes'])
    except:
        texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_right,interpolation='nearest',radius =1, n_samples=1)
        plot_surf_contours(fsaverage.infl_right, texture, figure=surf_plot1,legend=False,axes=ax,  levels=[1],colors=['black'])#,levels = [texture],  labels=['Fig 3 nodes'])


    fig,ax = plt.subplots(nrows=1, ncols= 1, figsize=(6,6),subplot_kw={'projection': '3d'})
    fig.tight_layout(h_pad=None,w_pad=None)
    texture = vol_to_surf(isrsa_mantel_r2_brain, fsaverage.pial_left,interpolation='nearest',radius =1, n_samples=1)
    surf_plot2=plot_surf_roi(fsaverage.infl_left, texture, hemi='left',cmap = cmap, axes=ax, colorbar=False,vmin=vmin,  vmax = vmax,
                                bg_map=fsaverage.sulc_left, view = 'medial',darkness=.4)
    try:
        texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_left,interpolation='nearest',radius =1, n_samples=1)
        plot_surf_contours(fsaverage.infl_left, texture, view = 'medial', figure=surf_plot2,legend=False, axes=ax, levels=[2],colors=[n_color])#,levels = [texture],  labels=['Fig 3 nodes'])
    except:
        None

    
    fig,ax = plt.subplots(nrows=1, ncols= 1, figsize=(6,6),subplot_kw={'projection': '3d'})
    fig.tight_layout(h_pad=None,w_pad=None)
    texture = vol_to_surf(isrsa_mantel_r2_brain, fsaverage.pial_right,interpolation='nearest',radius =1, n_samples=1)
    surf_plot4=plot_surf_roi(fsaverage.infl_right, texture, hemi='right',cmap = cmap, colorbar=False,vmin=vmin,axes=ax,   vmax = vmax,
                                bg_map=fsaverage.sulc_right, view ='medial',darkness=.4)
    try:
        texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_right,interpolation='nearest',radius =1, n_samples=1)
        plot_surf_contours(fsaverage.infl_right, texture,view ='medial', figure=surf_plot4,legend=False, axes=ax, levels=[2],colors=[n_color])#,levels = [texture],  labels=['Fig 3 nodes'])
    except:
        print('ERROR!')

def plot_isrsa_contours_lem(isrsa_mantel_r2_brain,roi_list,mov, colorbar,vmax,vmin):
    '''roi_list should be list of ROIs that you want colored (leave the other ROIs as 0)'''
    '''Black - 1'''
    '''n_color - 2'''
    
    n_color = 'blue'
    cmap='RdBu_r'

    fig,ax = plt.subplots(nrows=1, ncols= 1, figsize=(6,6),subplot_kw={'projection': '3d'})
    fig.tight_layout(h_pad=None,w_pad=None)
    texture = vol_to_surf(isrsa_mantel_r2_brain, fsaverage.pial_left,interpolation='nearest',radius =1, n_samples=1)
    surf_plot1=plot_surf_roi(fsaverage.infl_left, texture, hemi='left',cmap = cmap, colorbar=False, vmin=vmin,  vmax = vmax,
                                 bg_map=fsaverage.sulc_left,axes=ax,darkness=.4)
    try:
        texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_left,interpolation='nearest',radius =1, n_samples=1)
        plot_surf_contours(fsaverage.infl_left, texture, figure=surf_plot1,legend=False, axes=ax,levels=[1],colors=['k'])#,levels = [texture],  labels=['Fig 3 nodes'])
    except:
        texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_left,interpolation='nearest',radius =1, n_samples=1)
        plot_surf_contours(fsaverage.infl_left, texture, figure=surf_plot1,legend=False, axes=ax,levels=[2],colors=[n_color])#,levels = [texture],  labels=['Fig 3 nodes'])
        try:
            texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_left,interpolation='nearest',radius =1, n_samples=1)
            plot_surf_contours(fsaverage.infl_left, texture, figure=surf_plot1,legend=False, axes=ax,levels=[1,2],colors=['k',n_color])#,levels = [texture],  labels=['Fig 3 nodes'])
        except:
            None


    fig,ax = plt.subplots(nrows=1, ncols= 1, figsize=(6,6),subplot_kw={'projection': '3d'})
    fig.tight_layout(h_pad=None,w_pad=None)
    texture = vol_to_surf(isrsa_mantel_r2_brain, fsaverage.pial_right,interpolation='nearest',radius =1, n_samples=1)
    surf_plot1=plot_surf_roi(fsaverage.infl_right, texture, hemi='right',cmap = cmap, axes=ax,colorbar=False, vmin=vmin,  vmax = vmax, bg_map=fsaverage.sulc_right,darkness=.4)
    try: 
        texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_right,interpolation='nearest',radius =1, n_samples=1)
        plot_surf_contours(fsaverage.infl_right, texture, figure=surf_plot1,legend=False, axes=ax,levels=[2],colors=[n_color])#,levels = [texture],  labels=['Fig 3 nodes'])
    except:
        None


    
    fig,ax = plt.subplots(nrows=1, ncols= 1, figsize=(6,6),subplot_kw={'projection': '3d'})
    fig.tight_layout(h_pad=None,w_pad=None)
    texture = vol_to_surf(isrsa_mantel_r2_brain, fsaverage.pial_left,interpolation='nearest',radius =1, n_samples=1)
    surf_plot2=plot_surf_roi(fsaverage.infl_left, texture, hemi='left',cmap = cmap, colorbar=False,vmin=vmin,  vmax = vmax,
                                bg_map=fsaverage.sulc_left, view = 'medial',axes=ax,darkness=.4)
    try: 
        texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_left,interpolation='nearest',radius =1, n_samples=1)
        plot_surf_contours(fsaverage.infl_left, texture, figure=surf_plot2,legend=False, axes=ax,levels=[2],colors=[n_color])#,levels = [texture],  labels=['Fig 3 nodes'])
    except:
        None
        

    
    fig,ax = plt.subplots(nrows=1, ncols= 1, figsize=(6,6),subplot_kw={'projection': '3d'})
    fig.tight_layout(h_pad=None,w_pad=None)
    texture = vol_to_surf(isrsa_mantel_r2_brain, fsaverage.pial_right,interpolation='nearest',radius =1, n_samples=1)
    surf_plot4=plot_surf_roi(fsaverage.infl_right, texture, hemi='right',cmap = cmap, colorbar=False,vmin=vmin,  vmax = vmax,
                                bg_map=fsaverage.sulc_right, view ='medial',axes=ax,darkness=.4)
    try: 
        texture = vol_to_surf(color_rois(roi_list), fsaverage.pial_right,interpolation='nearest',radius =1, n_samples=1)
        plot_surf_contours(fsaverage.infl_right, texture, figure=surf_plot4,legend=False, axes=ax,view ='medial',levels=[2],colors=[n_color])#,levels = [texture],  labels=['Fig 3 nodes'])
    except:
        None



    


    