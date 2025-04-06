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
# a notebook for trying out reconstruction of the body frame and pili

# %% 
import numpy as np
import matplotlib.pyplot as plt
import command
import readtrack
import twanalyse
import eventanalyse
import pilush

# %% 

# simdir = "/home/dan/usb_twitching/debug_record/two_parameter_model/two_parameters_talaI/single_3433301/withvtk/long"
simdir = "/home/dan/usb_twitching/debug_record/two_parameter_model/two_parameters_talaI/single_3433301/withvtk"
print('using data at ', simdir)

# %%
with command.chdir(simdir):
    print('reading dataset')
    evdataset = readtrack.eventset()
    trs = readtrack.trackset()
    ptrs  = readtrack.piliset()

evdata = evdataset[0]
tr = trs[0]
ptr = ptrs[0]

# %%
print('reorganising pilus data...')
pilusdata = pilush.reorganise(ptr)
print('done.')
# %%
# reorganise tracking data into time searchable dictionary
# ...

# %% 
# first test just print the first pilus lab axis
pdatalist = iter(pilusdata.items())
pfidx, parb = next(pdatalist)
parb.keys()

# %% 
# plot a pilus length
ax = plt.gca()
ptime = parb['time']
print(ptime[0], ptime[-1])
ax.plot(ptime, parb['pleq'], label='leq')
ax.plot(ptime, parb['plength'], label=r'$|anchor - attach|$')
ax.legend()
plt.show()
# %% 
# need to select out the time data for the body frame
trseg, prowdata = pilush.get_segment(tr, ptr, pfidx, ptime)

# %% 
# plot pilus angle
frame = [pilush._construct_frame(trdata) for trdata in trseg]
labaxes = [pilush._construct_axis(f, row) for f, row in zip(frame, prowdata)]

e_z = np.array([0,0,1])
angle = [np.dot(e_z, labax) for labax in labaxes]
ax = plt.gca()
ax.plot(ptime, angle)
ax.set_ylim([-1.1,0])
ax.set_ylabel(r'$\vec{a} \cdot e_z$')
plt.show()


