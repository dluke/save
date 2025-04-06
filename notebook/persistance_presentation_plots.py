## ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# apply "superstatistical" analysis ( https://www.nature.com/articles/ncomms8516#Sec19 ) to walking/crawling modes
#
# %%
# cloned from persistance_pt1, 
# clean out everything and make the plots presentation ready

# %%

sys.path.insert(0, os.path.abspath('tools/'))
import bayesloop
import sys
import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotutils
import matdef
import _fj

# %%
# paths
import pili
plot_target = os.path.normpath( os.path.join(pili.root, '../impress/paper_review/ims') )
print('plotting target', plot_target)

# %%
# load fanjin data
debug = False
N = 100 if debug else None
crawling_idx, crawling_trs = _fj.slicehelper.load_linearized_trs(
    'default_crawling_list', N)
walking_idx, walking_trs = _fj.slicehelper.load_linearized_trs(
    'default_walking_list', N)
all_idx = np.concatenate([crawling_idx, walking_idx])
all_trs = crawling_trs + walking_trs
print()
print("loaded {} crawling tracks".format(crawling_idx.size))
print("loaded {} walking tracks".format(walking_idx.size))
print("total {} tracks".format(len(all_trs)))

# %%
# search all the tracks for some which show clear steps in aspect ratio
whaspect = [tr['length']/tr['width'] for tr in all_trs]
whaspect_std = [np.std(whaspect_i) for whaspect_i in whaspect]
a_sorted = sorted(enumerate(whaspect_std), key=lambda t: t[1], reverse=True)
print(a_sorted[:10])

# %%
# and now we need to familiarise ourselves with the code given by
# https://www.nature.com/articles/ncomms8516#Sec19
sys.path.insert(0, os.path.abspath('tools/'))
analyser = bayesloop.BayesLoop()
print('The default limits on q and a are respectively, ',
      analyser.qBound, analyser.aBound)
print(analyser.aBound)
print(analyser.qBound)
# where the limits on q are expected to be [-1, 1] so its not entirely clear why [-1.5,1.5] is used
print('Default control parameters.')
print('pmin = {}'.format(analyser.pMin))
print('Box kernel halfwidths (Ra, Rq) = ({}, {})'.format(analyser.Ra, analyser.Rq))
print('default gridsize =', analyser.gridSize)
print('kernal size is in the context of gridsize and the limits so in fact kernel dimensions are ({},{})'.format(
    2 * analyser.Ra * (analyser.aBound[1] -
                       analyser.aBound[0])/analyser.gridSize,
    2 * analyser.Rq * (analyser.qBound[1]-analyser.qBound[0])/analyser.gridSize
))

# %%
# lets pick track #2 to work with because it appears to switch to walking and back
sorted_pick_id = 2
eye_track_id, eye_track_std = a_sorted[sorted_pick_id]
eye_data_id = all_idx[eye_track_id]
print('track data id =', eye_data_id)
eye_track = all_trs[eye_track_id]

# %%
# font setup
import matplotlib.font_manager as fm
# Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]
for f in fm.fontManager.ttflist:
    print(f.name)

mpl.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

# %%
# And again for reference, plot apsect ratio
ax = plt.gca()
eye_track_aspect = whaspect[eye_track_id]
ax.plot(0.1 * np.arange(eye_track_aspect.size),  eye_track_aspect)
ax.set_xlabel('time (s)')
ax.set_ylabel('length/width (microns)')
plt.show()
# %%
# and velocity so that we know if the bacterium stops moving

# %%
# xy data
trackxy = np.column_stack([eye_track['x'], eye_track['y']])
print('check xy data shape ', trackxy.shape)
print('write out this data so we can check it against the GUI tool')
target = 'tools/trackxy_{:04d}.dat'.format(eye_data_id)
print('writing track {:04d} to {}'.format(eye_data_id, target))
np.savetxt(target, trackxy)

# %%
# setup and run the analysis
# analysis takes velocity data
fjtimestep = 0.1
track_u = (trackxy[1:] - trackxy[:-1])/fjtimestep
speed = np.linalg.norm(track_u, axis=1)
print('velocity lims ({},{})'.format(speed.min(), speed.max()))
analyser.data = track_u
analyser.pMin = 1e-18  # see persistance.py
analyser.startAnalysis()

# %%
# %matplotlib inline
#
# mean parameters postMean
print(analyser.postMean.shape)
print(analyser.avgPost.shape)
# rescale eye_track_aspect to [0,1]
rescale = 1/np.quantile(eye_track_aspect, 0.95)
aspect_ghost = rescale * eye_track_aspect
kwghost = {'linewidth': 4, 'color': '0.2', 'alpha': 0.4}
linekw = {'linewidth': 4, 'alpha':0.8}


def plot_qa(axes, analyser, aspect_ghost=aspect_ghost):
    basis = np.arange(analyser.postMean[0].size)
    _plot_qa(axes, basis, analyser.postMean, analyser.pMin, aspect_ghost=aspect_ghost)

def _plot_qa(axes, basis, postMean, pMin, aspect_ghost=aspect_ghost, xlabel='time (s)', title=None):
    ax1 = axes
    q_mean, a_mean = postMean
    # ax1.axhline(rescale, linewidth=1, c='k', alpha=0.6, linestyle='--')
    ax1.set_xlabel(xlabel)
    if not (aspect_ghost is None):
        aspect_ghost = aspect_ghost[1:-1] # same shape as q,a
        ax1.plot(basis, aspect_ghost, label='rescaled length/width', **kwghost)
    if title is None: 
        title = 'pmin = {:.2E}'.format(pMin)
    ax1.set_title(title)
    ax1.set_ylim((0, 1.5))
    ax1.plot(basis, q_mean, label='q parameter', **linekw)
    ax1.plot(basis, a_mean, label='a parameter', **linekw)
    ax1.legend()

fig = plt.figure(figsize=(10,5))
ax = fig.add_axes([0,0,1,1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
title = r'p_min = {:.2E}, R = {:5.2f}'.format(analyser.pMin, analyser.get_R()[0])
timebasis = matdef.TIMESTEP * np.arange(analyser.postMean[0].size)
_plot_qa(ax, timebasis, analyser.postMean, analyser.pMin,
    title=title )
target = os.path.join(plot_target, 'qa_normal.png')
print('saving to ', target)
fig.savefig(target, bbox_inches='tight')

# %%
# check pmin
analyser.pMin = 1e-7
analyser.startAnalysis()
del analyser.postSequ

# %% plot^^
ax.clear()
title = r'p_min = {:.2E}, R = {:5.2f}'.format(analyser.pMin, analyser.get_R()[0])
_plot_qa(ax, timebasis, analyser.postMean, analyser.pMin,
    title=title )
target =  os.path.join(plot_target, 'qa_low_pmin.png')
fig.savefig(target, bbox_inches='tight')


# %%
# check box kernel
side_analyser = bayesloop.BayesLoop()
side_analyser.pMin = 1e-18
side_analyser.Ra = 0
side_analyser.Rq = 0
side_analyser.data = track_u
side_analyser.kernel_on = False
side_analyser.startAnalysis()
del side_analyser.postSequ

# %% plot^^
print(len(eye_track.step_idx))
ax.clear()
title = r'p_min = {:.2E}, R = {:5.2f}'.format(side_analyser.pMin, side_analyser.get_R()[0])
_plot_qa(ax, timebasis, side_analyser.postMean, side_analyser.pMin,
    title=title )
target =  os.path.join(plot_target, 'qa_no_kernel.png')
fig.savefig(target, bbox_inches='tight')



