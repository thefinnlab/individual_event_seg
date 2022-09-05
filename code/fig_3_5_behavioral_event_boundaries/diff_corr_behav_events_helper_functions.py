import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

''' Defining the custom colormap'''
Rd_Bu_mod = LinearSegmentedColormap.from_list('Rd_Bu_mod', (
    # Edit this gradient at https://eltos.github.io/gradient/#Rd_Bu_mod=0:192F5A-23.9:7F9DD1-49.5:FEFDFF-50:767272-50.5:FEFDFF-67.4:E68367-82.8:BA2832-100:660A0D
    (0.000, (0.098, 0.184, 0.353)),
    (0.239, (0.498, 0.616, 0.820)),
    (0.495, (0.996, 0.992, 1.000)),
    (0.500, (0.463, 0.447, 0.447)),
    (0.505, (0.996, 0.992, 1.000)),
    (0.674, (0.902, 0.514, 0.404)),
    (0.828, (0.729, 0.157, 0.196)),
    (1.000, (0.400, 0.039, 0.051))))

def get_color(x):
    '''Get color for plotting for each movie'''
    if (x=='growth'):
        col_x='#00493E'
    elif x=='defeat':
        col_x = '#96223D'
    elif x=='iteration':
        col_x = '#0B0075'
    elif x=='lemonade':
        col_x = '#CE7700'
    return col_x

def corr_scatter_label(x,y, pval_dict_corr):
    try:
        r = pval_dict_corr[(x,y)]['r']
        p = pval_dict_corr[(x,y)]['p']
    except:
        r = pval_dict_corr[(y,x)]['r']
        p = pval_dict_corr[(y,x)]['p']       
    
    
    return p,r


def pval_star(p):
    if p > .05:
        star = 'n.s.'
    if p < .05:
        star = '*'
    if p < .01:
        star = '**'
    if p < .001:
        star = '***'
    if p < .0001 or p==.0001:
        star = '****'
    return star

def dist_plots(x, y, ax, num, c, pval_dict_diff, label=None, color=None, cmap=None, **kwargs):
    ''' Used to plot the density plots to show the difference between alignment values between the movies'''
    
    if (x.name=='growth'):
        col_x=get_color(x.name)
    elif x.name=='defeat':
        col_x=get_color(x.name)
    elif x.name=='iteration':
        col_x=get_color(x.name)
    elif x.name=='lemonade':
        col_x=get_color(x.name)
    if (y.name=='growth'):
        col_y=get_color(y.name)
    elif y.name=='defeat':
        col_y=get_color(y.name)
    elif y.name=='iteration':
        col_y=get_color(y.name)
    elif y.name=='lemonade':
        col_y=get_color(y.name)   

    try:
        r = pval_dict_diff[(x.name,y.name)]['beta']
        p = pval_dict_diff[(x.name,y.name)]['p']
    except:
        r = pval_dict_diff[(y.name,x.name)]['beta']
        p = pval_dict_diff[(y.name,x.name)]['p']   
    
    
    g2 = sns.distplot(x, hist = False, kde = True,color=col_x,
                 kde_kws = {'linewidth': 5},ax=ax)
    g2 = sns.distplot(y, hist = False, kde = True,color=col_y,
                 kde_kws = {'linewidth': 5},ax=ax)
    
    #FILL IN HERE
    # Get the two lines from the axes to generate shading
    l1 = g2.lines[0]
    l2 = g2.lines[1]
    g2.axvline(x=0,color='grey',linewidth=2)

    # Get the xy data from the lines so that we can shade
    x1 = l1.get_xydata()[:,0]
    y1 = l1.get_xydata()[:,1]
    x2 = l2.get_xydata()[:,0]
    y2 = l2.get_xydata()[:,1]
    g2.fill_between(x1,y1, color=col_x, alpha=0.05)
    g2.fill_between(x2,y2, color=col_y, alpha=0.05)

    g2.set_xlim(-5,10.5)
    g2.set_ylim(0,.35)
    
    
    g2.set_xticks(np.arange(-5, 11, 5))
    g2.set_yticks(np.arange(0, .35, .2))
    g2.set_xticklabels(np.arange(-5, 11, 5),fontsize=32)
    g2.set_yticklabels(np.arange(0, .35, .2),fontsize=32)
    
    if (c==2 and num==0):
        g2.set(xlabel=None)
        g2.set(ylabel=None)
    elif c==2:
        g2.set(xlabel=None)
        g2.set(ylabel=None)
        g2.set(yticklabels=[])
        g2.set(xticklabels=[])
    elif num==0:

        g2.set(yticklabels=[])
        g2.set(xticklabels=[])
        g2.set(xlabel=None)
        g2.set(ylabel=None)
    else:
        g2.set(yticklabels=[])
        g2.set(xticklabels=[])  
        g2.set(xlabel=None)
        g2.set(ylabel=None)
        g2.tick_params(bottom=False)
        

def mat_plots(x, ax, num, label=None, color=None, cmap=None, **kwargs):
    '''Used to just plot the alignment matrix on the diagonal'''
    
    '''get the name'''
    mov=x.name
    if num==3:
        val = True
    else:
        val = False
    
    matchz_node =  np.load('../../data/fig_3_5_behavioral_event_boundaries/behavioral_match_z/match_z_%s_all4movie.npy' %mov)
    g2 = sns.heatmap(matchz_node, ax=ax,yticklabels=False,xticklabels=False,vmax=10, vmin=-10,cmap=Rd_Bu_mod,cbar=val)

    g2.set(xticklabels=[])  
    g2.set(xlabel=None)
    g2.set(ylabel=None)
    g2.tick_params(bottom=False)
    
    