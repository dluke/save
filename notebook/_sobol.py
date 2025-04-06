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
# analyse first sobol sensitivity analysis

# %% 
import os, sys
import os.path
join = os.path.join
import numpy as np
import stats
import json

# %% 
# we pulled analysis data (local.json) back from the cluster using compute-sync.sh script
simdir = "/home/dan/usb_twitching/run/b2392cf/cluster/sobol"
with open(join(simdir, "lookup.json"), 'r') as f:
    lookup = json.load(f)

# %% 
# first job is to check that the simulation and analysis actually ran
print(len(lookup[0]))
stats_mask = np.array([os.path.exists(join(simdir, udir, "local.json")) for udir in lookup[0]])
print(np.count_nonzero(stats_mask))
