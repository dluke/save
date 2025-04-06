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
# draw the chi2(k) distributions so that we can think about how to modify it to be more forgiving for small k

# %%

import os
import sys
import numpy as np
import scipy.stats

join = lambda *x: os.path.abspath(os.path.join(*x))

from pili import support
import pili
import sctml.publication as pub

print("writing figures to", pub.writedir)

# %%

for k in range(10): 
    scipy.stats.chi2(k)

# %%
