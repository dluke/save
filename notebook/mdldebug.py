
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
# test mdlideal, mostlikely a temporary debugging notebook, use this type of note sparingly

# %% 
import os
import random
import numpy as np
import pickle
join = os.path.join 
norm = np.linalg.norm
pi = np.pi

import scipy
import matplotlib.pyplot as plt

import mdl
import annealing 

# %% 
def load_data(path="annealing/current"):
    datapath = '_'.join([path, "data.pkl"])
    with open(datapath, 'rb') as f:
        data = pickle.load(f)
    return data

solver = annealing.Solver.load_state()
_data = load_data()

# %% 
sidx = 12
seglist = solver.anneal.seglist

_model = solver.anneal.get_current_model()
is_outlier = solver.anneal.get_outliers()
node = solver.anneal.get_segment_node(sidx)
print("model sidx", _model.sidx)

highlight_nodes = solver.anneal.get_nodes_at(sidx, n_local=3)
highlight_sidx = [_n.value.sidx for _n in highlight_nodes][:-1]
print([h.value.dt for h in highlight_nodes][:-1])
print("hightlight", highlight_sidx)

# %% 
fig, ax = plt.subplots(figsize=(20,20))
_config = {"h_seg": False, "h_outlier": False, "h_points":[61, 112], "h_nodes": highlight_sidx}

mdl.plot_model_on_data(ax, _model, _data, intermediate={'is_outlier':is_outlier}, config=_config)
# plt.scatter(node.value.x, node.value.y, marker="D", s=200)

# %% 
# fails?
solver.create_and_destroy(sidx)

# %% 
def local_plot(anneal, _config={}):
    # note specific plotting function
    fig, ax = plt.subplots(figsize=(20,20))
    is_outlier = anneal.get_outliers()
    mdl.plot_model_on_data(ax, anneal.get_current_model(), _data, 
        intermediate={'is_outlier':is_outlier}, config=_config)
local_plot(solver.anneal)

# %% 
def test():
    # simulate creation at sidx = 12
    _anneal = solver.anneal.clone()
    node = _anneal.get_segment_node(sidx)
    affected = _anneal.create(node)
    h_nodes = _anneal.expand_nodes(affected, n_local=3)
    dist = annealing.partial_array_model(h_nodes).get_distance()
    print(dist)
    print([h.value.dt for h in h_nodes])
    highlight_sidx = [_n.value.sidx for _n in h_nodes][:-1]

    _config = {"h_points":[61, 112], "h_nodes": highlight_sidx}
    local_plot(_anneal, _config)



# %% 
# solver.random_queue_solve()


# _anneal = annealing.Anneal()
# _anneal.initialise(c_model, _data)
# solver = annealing.Solver(_anneal, rng=rngstate)
