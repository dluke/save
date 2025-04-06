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
# prepare to implement a continuity term/constraint into our solver
# continuity and rempping seem intimiately connected so we should explore both ideas together

# %% 
import os
from copy import deepcopy
import random
import numpy as np
import pickle
join = os.path.join 
norm = np.linalg.norm
pi = np.pi

import scipy
import matplotlib.pyplot as plt
import seaborn as sns

import sctml
import sctml.publication as pub
import pili

import _fj
print("writing figures to", pub.writedir)

from pili import support
import llist
import mdl
import annealing 


# %%
def local_plot(anneal, data, _config={"h_outlier":True}, fsize=20):
    # note specific plotting function
    fig, ax = plt.subplots(figsize=(fsize,fsize))
    is_outlier = anneal.get_outliers()
    mdl.plot_model_on_data(ax, anneal.get_current_model(), data, 
        intermediate={'is_outlier':is_outlier}, config=_config)


# %% 
# load candidate track
r = 0.03
candidate_idx = 2924
original_tr = _fj.trackload_original([candidate_idx])[0]
lptr = mdl.get_lptrack(original_tr)

# %% 
#
def find_chunk_boundary_at(anneal, target_sidx, T=20):
    leftnode = anneal.sidxmap[target_sidx]
    anneal.seglist.update_node_time_at(anneal.llmodel.first)
    model = anneal.get_current_model()
    time = model.get_time()
    right_i = np.searchsorted(time, leftnode.value.get_time()+T, side='right')
    rightnode = anneal.sidxmap[model.sidx[right_i]]
    return leftnode, rightnode

def copy_chunk(leftnode, rightnode):
    # copy the linkded list model from left node to right node
    nodes = []
    _node = leftnode
    while _node != rightnode:
        nodes.append(_node.value)
        _node = _node.next
    nodes.append(rightnode.value)
    new = llist.dllist([deepcopy(_n) for _n in nodes])
    return new

def chunk_at(anneal, target_sidx, T=20):
    #
    leftnode, rightnode = find_chunk_boundary_at(anneal, target_sidx, T=20)
    _Ti = leftnode.value.get_time()
    _Tf = rightnode.value.get_time()
    data = lptr.cut(_Ti, _Tf)
    chunk_linked = copy_chunk(leftnode, rightnode)
    chunk_linked.last.value.sidx = -1
    chunk = annealing.array_model(chunk_linked)
    return data, chunk

# anneal = solver.anneal
# _data, chunk = chunk_at(anneal, 31, T=20)

# %% 

# %% 
annealing._debug = True
r = 0.03

save_chunk = []
save_data = []
for i in range(0, 10):
    print("----------------------------------------------------------------")
    print(f"chunk {i}")
    T = 20
    Ti = i * T
    Tf = (i+1) * T
    _data = lptr.cut(Ti, Tf)
    M = int(len(_data)/10) + 1
    _guess = mdl.recursive_coarsen(_data, M, parameter='M')

    chunk_anneal = annealing.Anneal(r)
    chunk_anneal.initialise(_guess, _data)
    rng = np.random.RandomState(0)
    chunk_solver = annealing.Solver(chunk_anneal, rng=rng)

    # loss function configuration
    loss_conf = {}
    loss_conf["contour_term"] = 0.01
    loss_conf["debug"] = False
    chunk_solver.anneal.default_loss_conf = loss_conf


    chunk_solver.multiple_linear_solve()
    chunk_solver.priority_solve()
    chunk_solver.remapping()
    chunk_solver.cleanup()

    save_data.append(_data)
    save_chunk.append(chunk_solver.anneal)
    # local_plot(chunk_solver.anneal, _data)

# %% 
# plot chunks
for i in range(len(save_chunk)):
    local_plot(save_chunk[i], save_data[i])
    plt.gca().set_title(f'chunk {i}')

# %% 
annealing.f = sys.stdout
annealing._debug = True

i = 8
T = 20
Ti = i * T
Tf = (i+1) * T
_data = lptr.cut(Ti, Tf)
M = int(len(_data)/10) + 1
_guess = mdl.recursive_coarsen(_data, M, parameter='M')


# setup solver
chunk_anneal = annealing.Anneal(r)
chunk_anneal.initialise(_guess, _data)
rng = np.random.RandomState(0)
chunk_solver = annealing.Solver(chunk_anneal, rng=rng)

# loss function configuration
loss_conf = {}
loss_conf["contour_term"] = 0.01
loss_conf["debug"] = True
chunk_solver.anneal.default_loss_conf = loss_conf

#
annealing._debug = False
chunk_solver.multiple_linear_solve()
annealing._debug = True
chunk_solver.priority_solve()
chunk_solver.dump_state(f"chunk_{i:02d}")

local_plot(chunk_solver.anneal, _data)

# %% 
chunk_solver.anneal.llmodel

# %% 
chunk_solver = annealing.Solver.load_state("chunk_01")

chunk_solver.priority_solve()

# %% 
local_plot(chunk_solver.anneal, _data)

# %% 
# # local_plot(solver.anneal, _data)
# points = [t[1] for t in chunk_solver._record]
# segments = [t[2] for t in chunk_solver._record]
# fig, ax = plt.subplots(figsize=(20,20))
# # _config = { "h_outlier": False, "h_points": [82], "h_nodes": [36]}
# _config = { "h_outlier": False, "h_points": points, "h_nodes": segments}
# mdl.plot_model_on_data(ax, chunk_solver.anneal.get_current_model(), _data,  config=_config)



