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
# notebook for testing the speed of sample entropy computation

# %% 
import time
import information
import numpy as np
from information import Timer

# %% 
m = 2
r = 0.1
clock = []
lbasis = np.asarray(np.geomspace(100, int(1e4), 10), dtype=int)
for N in lbasis:
    L = np.random.normal(size=N)
    with Timer() as t:
        information.sampen(L, m, r)
    clock.append(t.time)
# %% 
ax = plt.gca()
ax.plot(lbasis, clock, marker='o')
ax.set_ylabel('time (s)')
ax.set_xlabel('sequence length')

