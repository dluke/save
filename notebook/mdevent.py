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
# This notebook is just designed for prototyping analysis code for mdevent output
# The first use for this output is computing the total "action", i.e. displacement/rotation associated with a single pilus cycle
# 
# %% 
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import command
import readtrack
import eventanalyse
import plotutils
import scipy.stats
from tqdm import tqdm

# %% 
testdata = "/home/dan/usb_twitching/run/new/mdevent"
with command.chdir(testdata):
    mddata = readtrack.mdeventset()
md = mddata[0]
print('data size', md.size)
# %% 
# convert to pili dictionary
# (?) should we consider that the same pilus can bind multiple times

def _pilusmd(md):
    retidx = md['process'] == 'retraction'
    retdata = md[retidx]
    # pidx = md['pidx']
    pilusdata = {}
    for row in retdata:
        if row['pidx'] not in pilusdata:
            pilusdata[row['pidx']] = [row]
        else:
            pilusdata[row['pidx']].append(row)
    for pidx in pilusdata.keys():
        pilusdata[pidx] = np.stack(pilusdata[pidx])
    return pilusdata
 
pilusdata = _pilusmd(md)

# %%
# sum the displacements for each pilus
def pildisplace(pilusdata):
    displace = {}
    for pidx in pilusdata.keys():
        pdata = pilusdata[pidx]
        dr = np.stack([pdata['d_x'], pdata['d_y'], pdata['d_z']],axis=1)
        d = np.linalg.norm(np.sum(dr,axis=0))
        displace[pidx] = d
    return displace

displace = pildisplace(pilusdata)


# %%
# plot the distribution of total displacement associaited with each pilus

ax = plt.gca()
disparr = np.array(list(displace.values()))
plotutils.ax_kdeplot(ax, disparr, hist=True, res=50)
ax.set_ylabel('P')
ax.set_xlabel(r'displacement (microns)')
print(scipy.stats.describe(disparr))
plt.show()

# %%
# investigate pili bound retracting fraction
testdata = "/home/dan/usb_twitching/run/new/angle_smoothed/range_02_pbrf/anchor_angle_smoothing_fraction_00.133"
with command.chdir(testdata):
    mddata = readtrack.mdeventset()
md = mddata[0]

# %%
print(md.get_dtype())
plt.hist(md['nbound'])
plt.show()
# %%
dt = 0.1

print('calculating states')
result = eventanalyse.movestate(md, dt)
print('finished.')
# %%

eventanalyse.plot_disp_nret(result)
plt.show()
# what fraction of intervals have 'small' displacement
disp, nret = result['disp'], result['nret']
dxthreshold = 0.01
smallidx = disp < dxthreshold
print('fraction of intervals with d < {} is {:.3f}'.format(
    dxthreshold, np.count_nonzero(smallidx)/disp.size))

# %%

eventanalyse._plot_nbound_pbrf(result, dxthreshold)
plt.show()
