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
# we are going to coarse grain the trajectory
# using some sort of kernel in "trajectory space" (1d embedding) or in the original 2d space
#  unfortunately there are various coarsening procedurues that we can come up with

# %% 
import os
join = os.path.join 
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import itertools
from copy import deepcopy

import _fj
import matdef
import readmat

import shapeplot
import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

from pili import support

import llist
norm = np.linalg.norm

from mdl import *


# %% 
trs_idx, original_trs = _fj.slicehelper.load_original_trs('all')
_, denoised_trs = _fj.slicehelper.load_trs('all')

# %% 
# plot denoised and original
candidate_idx = 2924
original_tr = original_trs[candidate_idx]
denoised_tr = denoised_trs[candidate_idx]

top = _fj.load_subset_idx()["top"]

# %% 
# %% 
wavelet_style = {"alpha":0.6, "linestyle":'--'}
N = 100
default_color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
def reset_color_cycle(): return itertools.cycle(default_color_cycle)
global_color = reset_color_cycle()

def plot_piecewise(ax, x, y, lkw={}):
    base_color = next(global_color)
    light = support.lighten_color(base_color, 0.5)
    _color = itertools.cycle([base_color, light])
    for i in range(len(x)-1):
        ax.plot([x[i],x[i+1]],[y[i],y[i+1]], c=next(_color), **lkw)
    ax.set_aspect("equal")

def new_axis():
    size = (10,10)
    fig, ax = plt.subplots(figsize=size)
    return ax

N = 1000
trdata = original_tr.cut(0,N)
x, y = trdata.get_head2d().T
ax = new_axis()
plot_piecewise(ax, x, y)

_x, _y = denoised_tr.cut(0,N).get_head2d().T
ax.plot(_x, _y, **wavelet_style)


# %% 

lptr = get_lptrack(trdata)
lplist = lptr.get_llist()


# %% 

# r = 0.24
r = 0.006129942280882657
coarse = recursive_coarsen(lptr, r)
ax = new_axis()
lkw = {'alpha':0.6}

global_color = reset_color_cycle()
plot_piecewise(ax, trdata['x'], trdata['y'], lkw)
plot_piecewise(ax, coarse.x, coarse.y, lkw)

# %% 

# def compute_separation(data_lptr, coarse_lptr):
#     N = len(data_lptr)
#     M = len(coarse_lptr)
#     # print("track sizes (N, M)", N, M)
#     coarse_pt = coarse_lptr.get_n2()
#     def _to_index(dt):
#          return int(round(10*dt))
#     def _adjacent(n): 
#         ad = [] if n-1 < 0 else [n-1]
#         if n < M-1:
#             ad.append(n)
#         if n+1 < M-1:
#             ad.append(n+1)
#         return ad
        
#     # for all data points in original trajectory compute the distance to the coarse trajectory
#     # we will speed this up by comparing only against trajectory-adjacent segments
#     # step along the original trajectory, while updating a mapping onto the coarse trajectory
#     segment_map = []
#     for i, _dt in enumerate(coarse_lptr.dt):
#         segment_map.extend([i]*_to_index(_dt)) 
#     # ^ this maps the last point in the original track to the last point in the 
#     # coarse track, but these points have no associated next tsegment
#     _i = 0
#     distance = []
#     for _dt, _x, _y in data_lptr:
#         segment = segment_map[_i]
#         # get adjacent segments
#         adjacent = _adjacent(segment)
#         a = coarse_pt[adjacent[0]:adjacent[-1]+1] 
#         b = coarse_pt[adjacent[0]+1:adjacent[-1]+2] 
#         p = np.array([_x, _y])
#         dist = min(support.lineseg_dists(p, a, b))
#         _i += 1
#         distance.append(dist)
#     return np.array(distance)
    


# %% 
# simplest possible method. vary the coarsening parameter and compute the description_length
trdata = original_tr.cut(0,1000)
lptr = get_lptrack(trdata)

delta = 0.01
coarsening = np.geomspace(0.005,0.24,20)

desc_length = []
for coarse_r in coarsening:
    print("attempt to coarsen track with {} nodes at {}".format(len(lptr), coarse_r))
    _coarse = recursive_coarsen(lptr, coarse_r)
    dl = description_length(lptr, _coarse, delta)
    desc_length.append(dl)

    
ax = new_axis()
ax.plot(coarsening, desc_length, marker='o')
ax.set_ylim((0,None))

# %% 
# vary both l and r

delta_space = np.linspace(0.005, 0.02, 4, True)
models = []

desc_length = []
for coarse_r in coarsening:
    _coarse = recursive_coarsen(lptr, coarse_r)
    models.append(_coarse)
    _dl = []
    for delta in delta_space:
        dl = description_length(lptr, _coarse, delta)
        _dl.append(dl)
    desc_length.append(_dl)
    # lines.append((coarsening, desc_length))

dldata = list(map(list, zip(*desc_length)))
lines = [(coarsening, dl) for dl in dldata]

# %% 
ax = new_axis()
_i = 0
for basis, data in lines:
    label = r"r = {:.3f}".format(delta_space[_i])
    ax.plot(basis, data, marker='o', label=label)
    _i += 1
ax.set_ylim((0,None))
labelsize=24
ax.set_ylabel("description length", fontsize=labelsize)
ax.set_xlabel(r"coarsening length", fontsize=labelsize)
ax.tick_params(axis='both', labelsize=24)
ax.legend(fontsize=24)
notename = "coarsening"
pub.save_figure("description_length", notename)

# %% 
# obtain the minimum

def min_model(line):
    base, dl = line
    i = np.argmin(dl)
    return i, base[i], dl[i]
i, l, dl = min_model(lines[1])
model = models[i]
print("min model length {} for a total description length {}".format(len(model), dl))

def local_plot(ax, x, y, lkw={}):
    l, = ax.plot(x,y, **lkw)
    ax.set_aspect("equal")
    return l

def local_plot_tr(ax, tr, lkw={}):
    return local_plot(ax, tr['x'], tr['y'], lkw)

# count points in models
M = [len(model) for model in models]
print('M', M)


# %% 
on = [True,True,True]
def plot_model(model, m_label="model", _N=200, switch=on):
    max_time = _N/10
    short = model.cut(0,max_time)
    lkw = {"alpha":0.4, "linewidth":8}
    fig, ax = plt.subplots(figsize=(20,5))
    labels = []
    handles = []
    if switch[0]:
        l1 = local_plot_tr(ax, original_tr.cut(0,_N), lkw)
        handles.append(l1)
        labels.append("original")
    if switch[1]:
        l2 = local_plot_tr(ax, denoised_tr.cut(0,_N), lkw)
        handles.append(l2)
        labels.append("smoothed")
    if switch[2]:
        l3 = local_plot(ax, short.x, short.y, lkw)
        handles.append(l3)
        labels.append(m_label)
    ax.tick_params(axis='both', labelsize=24)
    # labels =  ["original", "smoothed", r"$l = {:.3f}$".format(l)]
    ax.legend(handles, labels, fontsize=26)
    return ax
plot_model(models[-1], switch=[True,True,False])

pub.save_figure("example_track_data_20s", notename)

# %% [markdown]
# optimisation
# (i) with fixed M but moving the nodes
#   *  optimise MDL criteria directly
#   *  Use least squares principle

# (ii) reducing M by joining segements if it is free to do so

# %% 

_model = models[-2]
print("_model as M = ",len(_model))
_short = _model.cut(0,20)
print("shortened to M = ",len(_short))
short_lptr = lptr.cut(0,20)
_res = model_lsq(short_lptr, _short, intermediate={})
print("finished")
    

# %% 

def project(u, v):
    # print(np.sum(u*v)/(norm(u)*norm(v)))
    nv = norm(v)
    return np.sum(u*v)/(nv*nv) * v

# constraint on the end segments
def clip_model(model, data_lptr):
    model = deepcopy(model)
    model_data = model.get_n2()
    data = data_lptr.get_n2()
    def _adjust(s,a,b):
        u = s - b # data vector
        v = a - b # model vector
        new_v = project(u, v)
        return b + new_v
    a, b = model_data[0], model_data[1]
    _x, _y = _adjust(data[0],a,b)
    model.x[0] = _x
    model.y[0] = _y
    a, b = model_data[-1], model_data[-2]
    _x, _y = _adjust(data[-1],a,b)
    model.x[-1] = _x
    model.y[-1] = _y
    return model

# %% 

result_lptr = get_result_lptr(_res, _model)

#
_result = clip_model(result_lptr, short_lptr)

plot_model(_result)

# %% [markdown]
# we want to combine optimisation with coarsening together
# and then look for the MDL model

# %% 
l_space = np.geomspace(0.06, 0.24, 8,True)
delta = 0.03
_data = lptr.cut(0,20)
# %% 

models2 = []
coarse2 = []
res2 = []
desc_length = []
intermediates = []
for l in l_space:
    print("attempt to coarsen track with {} nodes at {:.3f}".format(len(_data), l))
    _coarse = recursive_coarsen(_data, l)
    coarse2.append(_coarse)
    inter = new_intermediate()
    res = model_lsq(_data, _coarse, intermediate=inter)
    res2.append(res)
    conf = {"tol": 1e-3}
    _model = get_result_lptr(res, _coarse)
    # _model = clip_model(_model, _data)
    models2.append(_model)
    dl = description_length(_data, _model, delta, intermediate=inter)
    intermediates.append(inter)
    desc_length.append(dl)
print("finished")

# %% 
M = [len(_m)  for _m in coarse2]
print(M)

fig, axes = plt.subplots(1,2, figsize=(2*5,5))
ax = axes[0]
ax.plot(l_space, desc_length, marker='o')
ax.set_xlabel("coarsening length")
ax.set_ylabel("description length")
ax = axes[1]
ax.plot(M, desc_length, marker='o')
ax.set_xlabel("M")
ax.set_ylabel("description length")

# %% 
# plot models

# for _m in coarse2:
#     plot_model(_m)

original = original_tr.cut(0,_N)
for i, _m in enumerate(models2):
    # ax = plot_model(_m, switch=[True,False,True])
    inter = intermediates[i]
    ax = plot_model_on_data(_m, original, intermediate=inter)
    l = l_space[i]
    title = "l, M = ({:.3f},{})".format(l, len(_m))
    ax.set_title(title, fontsize=24)

# %% 

_N = 200
i = -2
inter = intermediates[i]
plot_model_on_data(models2[i], original_tr.cut(0,_N), intermediate=inter)

# %%
# switch to focus on l = 0.197, M = 12
conf = {"tol": 1e-3}
l = l_space[-2]

_coarse = recursive_coarsen(_data, l)
inter = new_intermediate()
res = model_lsq(_data, _coarse, intermediate=inter)
_model = get_result_lptr(res, _coarse)
# _model = clip_model(_model, _data)
dl = description_length(_data, _model, delta, intermediate=inter)
dl
print("fin")


# %%
# and plot it again
ax = plot_model_on_data(_model, original, intermediate=inter)

# %% 
_a = np.array([])
null = LPtrack(_a, _a, _a)
ax = plot_model_on_data(null, original)

# %%
# we decided to sort by the largest distance from the model to last associated data point
# then do a single clipping pass

# if segment has no adjacent points, delete it

# def prune_segments(data, model, intermediate):
#     point_tracker = intermediate["point_tracker"]
#     [len(ptlist) for ptlist in point_tracker]
#     linkmodel = model.get_llist()
    # link_to_lptr(linkmodel)

# def clip_segments(data, model, intermediate):
#     point_tracker = inter['point_tracker']
#     model = deepcopy(model)
#     model_data = model.get_n2()
#     point_data = data.get_n2()
#     a = model_data[:-1]
#     b = model_data[1:]
#     M = len(model)

#     for i in range(M-1):
#         a[i]

# %%
