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
# a quick canvas of scipy.optimize methods to see what is efficient for our problem

# %% 
import os
join = os.path.join 
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd

from tabulate import tabulate

import _fj

# import shapeplot
# import sctml
# import sctml.publication as pub
# print("writing figures to", pub.writedir)

from pili import support
import mdl


# %% 
trs_idx, original_trs = _fj.slicehelper.load_original_trs('all')

# %% 
candidate_idx = 2924
original_tr = original_trs[candidate_idx]
top = _fj.load_subset_idx()["top"]

# %% 
lptr = mdl.get_lptrack(original_tr)
_T = 20
_data = lptr.cut(0,_T)

l = 0.2
r = 0.03

conf = {}

with support.Timer("local optimal lsqmdl") as t:
    dl, model, inter = mdl.local_optimal_lsqmdl(_data, l, r)


# %%

mdl.describe_optimize_result(inter)


# %%
def try_methods(args):
    methods = ["BFGS", "Nelder-Mead", "Powell", "L-BFGS-B"]
    _opt_conf = {"method": methods[0]}
    _conf = {"scipy" : _opt_conf}
    m_exect = []
    m_nit = []
    for method in methods:
        _conf["scipy"]["method"] = method
        dl, model, inter = mdl.local_optimal_lsqmdl(*args, conf=_conf)
        print(method, mdl.describe_optimize_result(inter))
        m_exect.append(inter["exect"])
        m_nit.append(inter["res"].nit)

    df = pd.DataFrame({"method":methods, "exec_t":m_exect, "n_iter":m_nit})
    return df
# %% 
args = (_data, l, r)
df =  try_methods(args)
df

# %% 
# and try for larger N
_T = 200
_data = lptr.cut(0,_T)
args = (_data, l, r)
df =  try_methods(args)
df



# %% 
# original = original_tr.cut(0,10*_T)

# %% 
