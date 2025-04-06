# ---
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
# load fanjin matlab data

# %% 
import os
join = os.path.join 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import _fj
import matdef
import readmat

import shapeplot

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)


# %% 
# config
load_matlab_data = False

# %% 
_, trs = _fj.slicehelper.load_trs('all')
# %% 
original_data_dir = "originalnpy/"
if load_matlab_data:
    ftracking = _fj.init(first=False, alldirs=True, tracktype=matdef.ORIGINAL)
    # convert to tracks 
    original_trs = ftracking.get_tracklike()
    # pickle 
    target_dir = join("/home/dan/usb_twitching/fanjin/working_copy/", original_data_dir)
    npyform = 'track_{:05d}.pkl'
    for i, tr in enumerate(original_trs):
        path = join(target_dir, npyform.format(i))
        print("write to", path)
        with open(path, 'wb') as f:
            pickle.dump(tr, f)
else:
    # loading pickled data
    _, original_trs = _fj.slicehelper.load_original_trs('all')

# %% 
# plot denoised and original
candidate_idx = 2924
original_tr = original_trs[candidate_idx]
denoised_tr = trs[candidate_idx]

top = _fj.load_subset_idx()["top"]

# %% 
norm = np.linalg.norm
def plot_speed(ax, tr):
    speed = norm(tr.get_head_v(),axis=1)
    time = tr["time"][1:]
    ax.plot(time, speed)

fig, axes = plt.subplots(1, 2, figsize=(2*5, 5))
ax1, ax2 = axes
plot_speed(ax1, original_tr)
plot_speed(ax2, denoised_tr)

# %% 
ax = plt.gca()
shapeplot.longtracks(ax, [original_tr])
shapeplot.longtracks(ax, [denoised_tr])
plt.savefig('tmp_track.svg')


# %% 
N = 100
def local_plot(ax, trs, lkw={}):
    for tr in trs:
        x = tr['x']
        y = tr['y']
        l, = ax.plot(x,y, **lkw)
    ax.set_aspect("equal")
    return l

def plot_several_local(original_tr, denoised_tr):
    fig, axes= plt.subplots(3,1,figsize=(10,10))
    for i in range(3):
        ax = axes[i]
        start = i*N
        local_plot(ax, [original_tr.cut(start,start+N)])
        local_plot(ax, [denoised_tr.cut(start,start+N)])
plot_several_local(original_tr, denoised_tr)

# %% 
_N = 200
fig, ax = plt.subplots(figsize=(20,5))
lkw = {"linewidth":3}
l1 = local_plot(ax, [original_tr.cut(0,_N)], lkw)
l2 = local_plot(ax, [denoised_tr.cut(0,_N)], lkw)
ax.tick_params(axis='both', labelsize=24)
ax.legend([l1,l2], ["original", "smoothed"], fontsize=26)
notename = "matdata"
pub.save_figure("example_track_data_20s", notename)


# %% 
# and for "top" dataset
_idx = 1
_orig, _denoised = original_trs[top[_idx]], trs[top[_idx]]
plot_several_local(_orig, _denoised)

# %% 
# local linear smoothing

# %% [markdown]
# the suggestion exists in this trajectory that the denoising is smoothing away
# sharp linear segements.
# we want to extract those sharp linear segments, i.e. by local linearisation 

# %% 
disp = norm(original_tr.get_head_v(), axis=1)
xlim = (0,1.5)
sns.histplot(disp, binrange=xlim)
ax = plt.gca()
ax.set_xlabel("displacement / dt")
