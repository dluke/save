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
# some miscellaneous prep for running mdl

# %%
import os
join = os.path.join
import json
import numpy as np
norm = np.linalg.norm

import pili
import _fj
import twanalyse
import stats

import matplotlib.pyplot as plt
import seaborn as sns

# %%
# sort all trajectories by velocity 
idx, tracks = _fj.slicehelper.load_trs('default_crawling_list')
# idx, tracks = _fj.slicehelper.load_trs('candidates_whitelist')

# %%
vel = [np.mean(norm(tr.get_head_v(),axis=1)) for tr in tracks]
vel_sorted = sorted(list(zip(idx, vel)), key=lambda t:t[1], reverse=True)

# %%
top_200_idx = np.array([t[0] for t in vel_sorted[:200]], dtype=int)
_fj.slicehelper.save('top_200_crawling', top_200_idx)
# %%

idx, toptrs  =  _fj.slicehelper.load_original_trs('top_200_crawling')

# %%
duration = [tr.get_duration() for tr in  toptrs]
ax = sns.histplot(duration)

# %%

tr =  _fj.trackload_original([2986])[0]
tr.get_duration()
# 10 hours to solve  500 second trajectory
34966/60/60  

# %%
