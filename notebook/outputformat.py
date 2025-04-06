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
# reduce the datasize of the output format
# binary formats.
# i.e.
# http://www.pytables.org/
#
# modern alternatives better/simpler than hdf5?

# %% 
import os, sys
join = lambda *x: os.path.abspath(os.path.join(*x))
import numpy as np
import pili
import shutil

# %% 
example_sim = "~/usb_twitching/run/5bfc8b9/cluster/mc4d/_u_WzGvK2wO/"
notedir, notename = os.getcwd(), __file__
notename, ext = os.path.splitext(notename)
root = pili.root
datadir = join(notedir, notename)
if not os.path.exists(datadir):
    print("making directory: %s" % datadir)
    os.mkdir(datadir)
datadir

# %% 
conf = join(example_sim, "config.txt")
seed = join(example_sim, "random_seeds.txt")
shutil.copy(conf, join(datadir, "config.txt"))
shutil.copy(seed, join(datadir, "random_seeds.txt"))

# %% 


