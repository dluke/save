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
# load a file, write to binary, check the difference in size
import os, sys
import readtrack
import numpy as np
join = os.path.join

# %% 
notedir = os.getcwd()
datadir = join(notedir, "exampledata/two_parameters/pilivar_0013.00000_k_spawn_00.50000/data/")
bacterium_ex = join(datadir, "bacterium_00000.dat")
event_ex = join(datadir, "event_00000.dat")
filetype = ["bacterium", "event"]
file_ex = [bacterium_ex, event_ex]

# %%
for i, ex in enumerate(file_ex):
    file_t = filetype[i]
    print("loading file", ex)
    data = readtrack.Track(ex)
    os_size = os.path.getsize(ex)
    print("file type {} has size {}".format(file_t, os_size))
    binary_save = join("tmp/", file_t+"_00000.npy")
    print("saving binary data to ", binary_save)
    np.save(binary_save, data)
    os_binary_size = os.path.getsize(binary_save)
    print("binary size is {} which is {:4.1f}%".format(os_binary_size, 100*os_binary_size/os_size))
    print()

# %% [markdown]
# which for these examples reads off as at most 80%
# hence if we want to reduce our data footprint we should look at other options for example cleaning up the 3/4 output files we currently use 
# and removing columns that are not in use
