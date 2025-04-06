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
# MDL principle giving good results for piecewise linear modelling of trajectory data
# but out implmentation is O(N^2) and may have some edge case issues remaining
#
# we most likely want some kind of trajectory aware neighbour list
# although an 'off-the-shelf' neighbor list implementation might work for now

# %% 
import os
join = os.path.join 
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from copy import deepcopy

import _fj

import shapeplot
import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

from pili import support

import llist
norm = np.linalg.norm

import mdl

# %% 
# config
def new_axis():
    size = (10,10)
    fig, ax = plt.subplots(figsize=size)
    return ax


# %% 
# load data
trs_idx, original_trs = _fj.slicehelper.load_original_trs('all')

# %% 
candidate_idx = 2924
original_tr = original_trs[candidate_idx]
top = _fj.load_subset_idx()["top"]

# %% 
lptr = mdl.get_lptrack(original_tr)
_T = 200
original = original_tr.cut(0,10*_T)
_data = lptr.cut(0,_T)

l_space = np.geomspace(0.06, 0.24, 8,True)

l = l_space[-2]
r = 0.03

_conf = {"scipy": {"method":"L-BFGS-B"}, "coarsen_parameter": 'l'}

with support.Timer("local optimal lsqmdl") as t:
    dl, model, inter = mdl.local_optimal_lsqmdl(_data, l, r, conf=_conf)

# %%
# compute statistics
import twanalyse
def pwlstats(model):
    xy = model.get_n2()
    disp = (xy[1:] - xy[:-1])
    distance = norm(disp, axis=1)
    angle = twanalyse.disp_angle(disp)
    # TODO time/velocity statistics
    ldata = {}
    ldata["distance"] = distance
    ldata["angle"] = angle
    return ldata

def pwlplotstats(ldata):
    distance = ldata["distance"] 
    angle = ldata["angle"] 
    fig, axes = plt.subplots(1,2, figsize = (10,5))
    ax = axes[0]
    ax.set_xlabel("displacement")
    xlim = (0, np.max(distance))
    sns.histplot(distance, binrange=xlim, ax=ax)
    ax = axes[1]
    ax.set_xlabel("angles")
    xlim = [-np.pi, np.pi]
    ax.set_xlim(xlim)
    ax.set_xticks([-np.pi/2,0,np.pi/2])
    ax.set_xticklabels([r"-\pi/2", 0, r"\pi/2"])
    sns.histplot(angle,ax=ax)

X = pwlstats(model)
pwlplotstats(X)

# %%
ax = mdl.plot_model_on_data(model, original, intermediate=inter, config={"figsize":(20,5)})
# pub.save_figure("candidate_mdl_model", "mdlaccelerate", config={"svg":True})

# %%
# next optimise for l

l_initial_guess = 0.2
r = 0.03
_T = 40
original = original_tr.cut(0,10*_T) # plotting
_data = lptr.cut(0,_T)
_coarse = mdl.recursive_coarsen(_data, l_initial_guess)
M_initial_guess = len(_coarse) 
M_initial_guess 

# %%
# super_models = []
# scipy_conf = {"tol":1e-3, "method":"BFGS", "callback":observe}
_conf = {"scipy": {"method":"L-BFGS-B"}, "coarsen_parameter": 'M'}
def loss(_M, _data, r):
    M = _M
    dl, _model, inter = mdl.local_optimal_lsqmdl(_data, M, r, conf=_conf)
    return dl, _model, inter

# %%

args = (_data, r)
N = len(_data)
Intf = mdl.IntegerFunction(N, loss,  args=args)

M0 = M_initial_guess
Mmax = len(mdl.recursive_coarsen(_data, 0.3*l0))
Mmin = len(mdl.recursive_coarsen(_data, 2*l0))
bracket = [Mmin, Mmax]
print("bracket", bracket)

mdl.convex_minimise(Intf, bracket)

# %%
# Intf.cache_model
cv = Intf.cache_value
X, dl = list(cv.keys()), list(cv.values())
plt.scatter(X, dl)

# %%
def describe_convex_search(Intf):
    
    exect = sorted(Intf.cache_exect.items())
    print(exect)
    total_exect = sum([ex[1] for ex in exect])
    print(f"total exec time {total_exect:.2f}s")
    

describe_convex_search(Intf)

# %%
_M = 23
_model = Intf.cache_model[_M]
_inter = Intf.cache_inter[_M]
n = np.sum(_inter["is_outlier"])
print("M, n(outlier) = {}, {}   total = {}".format(_M, n, _M+n))
ax = mdl.plot_model_on_data(_model, original, intermediate=_inter, config={"figsize":(20,5)})

# %%
Seglist = mdl.SegmentList(_data, _model)
Seglist.update_distances(_model.get_n2().flatten())
# d = Seglist.get_closest_segment()
d = Seglist.get_closest_data()
# print(Seglist.distance_matrix)
