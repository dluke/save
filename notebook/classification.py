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
# pull together subsets of the fanjin dataset that we can use
# as reference data for calibrating model parameters

# Datasets:
# * candidate (fastest crawling track)
# * whitelist (fast crawling)
# * medianlist (median crawling speed tracks)
# * walking
#  
# for whitelist see velocity_profiles.py
#  
# whitelist and median list will be determined 
# using this notebook (previously velocity_profiles.py)  
#  
# walking list is chosen using annotate_walking.py

# %% 
import os
import json
join = lambda *x: os.path.abspath(os.path.join(*x))
import numpy as np
import matplotlib.pyplot as plt
import collections
import _fj
import plotutils
import command
import twanalyse
import fjanalysis
import twanimation
import pili

import pandas as pd
import seaborn as sns
import pili.publication as pub
sns.set_style()
# %% 
verbose = False
work = False
if work:
    print("Re-animate trajectories")

# %% 
# cache the fanjin summary statistics
# with command.timer("cache summary stats"):
#     fjlocal = fjanalysis.summary()
fjlocal = fjanalysis.load_summary()
short = np.array([fld.get("failed", False) for fld in fjlocal])
print('failed', np.sum(short))

# %% 
# paths
notedir = pili.notedir
notename = "classification"
def make_data_dir(path):
    path = join(notedir, path)
    if not os.path.exists(path):
        os.makedirs(path)
all_idx, ltrs = _fj.slicehelper.load_linearized_trs('all')
_ = _fj.redefine_poles(ltrs)

# %% 
crawling_idx = _fj.slicehelper.load("default_crawling_list")
_low_aspect_idx = _fj.slicehelper.load("default_walking_list")
walking_idx = _fj.slicehelper.load("pure_walking")
whitelist_idx = _fj.slicehelper.load("candidates_whitelist")
print(len(crawling_idx))
print(len(_low_aspect_idx))
3113 - len(_low_aspect_idx)

# %% 
lvel = np.nan_to_num([twanalyse.lvel(ltr) for ltr in ltrs])
clfy = np.full(lvel.size, "other", dtype="object")
clfy[crawling_idx] = "crawling"
clfy[walking_idx] = "walking"
nsteps  = np.array([len(tr.step_idx)-1 for tr in ltrs])

# lveldf = pd.DataFrame([lvel,clfy], columns=["clfy", "lvel"])
lveldf = pd.DataFrame({
    "idx": all_idx, 
    "clfy":clfy, 
    "lvel":lvel,
    "nsteps":nsteps
    })
cr = lveldf.loc[lveldf["clfy"] == "crawling"]
walk = lveldf.iloc[walking_idx]

# %%
# the ranges in velocity to look for tracks for each dataset
nstep_threshold = 50
n = 100
xrd = {}
# top 100
# sorted_lvel = lveldf.sort_values("lvel")
whitelist_lvel = lvel[whitelist_idx]

sort_cr = cr.sort_values("lvel")
sort_lvel = sort_cr["lvel"].to_numpy()
xrd["whitelist"] = (whitelist_lvel.min(), whitelist_lvel.max())
halfspeed = whitelist_lvel.mean()/2

lvel_median = np.median(sort_cr["lvel"])
medidx = np.searchsorted(sort_cr["lvel"], lvel_median)
halfidx = np.searchsorted(sort_cr["lvel"], halfspeed)

mid_cr = sort_cr.iloc[medidx-n//2: medidx+n//2]
half_cr = sort_cr.iloc[halfidx-n//2: halfidx+n//2]
s_mid = mid_cr["lvel"].to_numpy()
s_half  = half_cr["lvel"].to_numpy()
xrd["midlist"] = (s_mid[0], s_mid[-1])
xrd["halflist"] = (s_half[0], s_half[-1])
#
# p = sns.histplot(mid_cr["nsteps"])
# p.set(title="median subset")
#
mid_low_data = mid_cr["nsteps"] < nstep_threshold
half_low_data = half_cr["nsteps"] < nstep_threshold
print("median: {:d}/{:d} tracks have < {} steps".format(
    sum(mid_low_data),mid_low_data.size,nstep_threshold))
print("halfspeed: {:d}/{:d} tracks have < {} steps".format(
    sum(half_low_data),half_low_data.size,nstep_threshold))
# exclude short tracks
mid_candidates = mid_cr["idx"][~mid_low_data]
half_candidates = half_cr["idx"][~half_low_data]

def save_tracks(idx_list, savedir):
    sample = 100
    make_data_dir(savedir)
    rule = join(savedir, "track_{:04d}.mp4")
    for track_id in idx_list:
        plt.clf()
        tr = ltrs[track_id]
        savefile = rule.format(track_id)
        twanimation.outline(plt.gcf(), [tr], sample, 
            camera='follow', savefile=savefile)

candidate_summary = fjlocal[2924]
candidate_lvel = candidate_summary["lvel"]["mean"]

# %%
# unused annotation data
def tokey(idx): 
    # convert to key from idx
    return '{:04d}'.format(int(idx))
def prep_annotation_template(annotate_file, c_idx):
    template = collections.OrderedDict(
        [(tokey(idx),[]) for idx in sorted(c_idx)]
    )
    if not os.path.exists(annotate_file):
        print("new annotation template at ", annotate_file)
        with open(annotate_file, "w") as f:
            json.dump(template, f, indent=1)

# %%
# 
savedir = join(notedir, notename, "median")
if work:
    save_tracks(mid_candidates, savedir)
prep_annotation_template(join(savedir, "annotation.json"),  mid_candidates)
# %%
# 
savedir = join(notedir, notename, "halfspeed")
if work:
    save_tracks(half_candidates, savedir)
prep_annotation_template(join(savedir, "annotation.json"),  half_candidates)

# %% [markdown]
# crawling trajectories at with median lvel.mean appear in animation to have 
# a mixture of trapped and slow moving persistent forward progress. 
# occasionally bacteria appear to travel backwards short distances
# or make nearly in-place rotations
# My best guess is these cells are overpilated rather than underpilated

# crawling trajectories at half of whitelist lvel.mean only rarely show 
# a mixture of trapping and moving, instead most trajectories move 
# persistently with steady speed while a few trajectories appear 
# to change direction or have low peristence. I flagged those trajectories
# in the annotation data 

# %% 
# load annotations and remove flagged data
with open(join(savedir, "annotation.json"), "r") as f:
    annotation = json.load(f)
flag = np.full(half_cr.shape[0], "short", dtype=object)
for i, idx in enumerate(half_cr["idx"]):
    key = tokey(idx)
    if key in annotation:
        if "flag" in annotation[key]:
            flag[i] = "flag"
        else:
            flag[i] = ""
    else:
        pass # value stays as "short"
half_cr.loc[:,"flag"] = pd.Series(flag,index=half_cr.index)
halflist = half_cr.loc[half_cr["flag"] == ""]["idx"].to_numpy()
_fj.slicehelper.save("halflist", halflist)
_fj.slicehelper.save("median", mid_candidates)

# %%
print("distribution of mean velocities in fanjin population")
# TODO xrd should contain limits of the actual subsets after 
# filtering out undesirable trajectories
df = pd.concat([cr,walk])
xlim = (0, np.quantile(walk["lvel"], 0.95))
p = sns.histplot(df, x="lvel", hue="clfy", 
    binrange=xlim, element="step", 
    common_norm=True, stat="density")
# p.set(xlabel="lvel.mean", xlim=xlim)
p.set(xlabel=r"$\langle u \rangle$", xlim=xlim)
p.axes.set_ylabel('')
spanstyle = {"alpha":0.1}
p.axes.axvspan(*xrd["midlist"], color='darkred', **spanstyle)
p.axes.axvspan(*xrd["whitelist"], color='g', **spanstyle)
p.axes.axvspan(*xrd["halflist"], color='m', **spanstyle)
p.axes.axvline(candidate_lvel, c='gray')
p.axes.grid('off')
plt.savefig("/home/dan/usb_twitching/notes/sensitivity")

# %%
# for presentation
fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(df["lvel"], element="step")
ax.set_xlim(0,0.2)
ax.set_ylabel('count')
ax.set_xlabel(r"$\langle u \rangle$")

ax.axvspan(*xrd["midlist"], color='darkred', **spanstyle)
ax.axvspan(*xrd["whitelist"], color='g', **spanstyle)
ax.axvspan(*xrd["halflist"], color='m', **spanstyle)
ax.axvline(candidate_lvel, c='gray')
ax.grid(False)


# %%
subsetdata = _fj.load_subsets()
subsetidx = _fj.load_subset_idx()

# %%
default_walking_idx = _fj.slicehelper.load("default_walking_list")
print("walking candidates ", len(default_walking_idx))
# 2000 seconds is
2000./60
print("crawling candidates ", len(crawling_idx))


# %%
# create pandas table for publication
# dataframe: subset name, median velocity, numpy of tracks, min and max velocity
distrib = {}
subsets = list(subsetdata.keys())
pubnames = ["fastest", "high speed", "half speed", "median speed", "walking"]

for name, ltrs in subsetdata.items():
    _dct = {}
    def wmean(a,w):
        return np.sum(a*w)/np.sum(w)
    lvel_mean = np.array([wmean(ltr.get_step_speed(), ltr.get_step_dt()) for ltr in ltrs])
    _dct["N"] =  len(ltrs)
    # _dct["median_step"] = 0.1 * np.median([ltr.get_nsteps() for ltr in ltrs]) 
    _dct["median_step"] = np.median(np.concatenate([ltr.get_step_dt() for ltr in ltrs])) 
    _dct["min"] =  lvel_mean.min()
    _dct["max"] =  lvel_mean.max()
    _dct["median"] =  np.median(lvel_mean)
    distrib[name] = _dct

colkey = ["N", "median_step", "median", "min", "max"]
pubcol = ["N", "median step (s)", "median ($\mu ms^{-1}$)", "min v", "max v"]
pubkey = {k: v for k, v in zip(colkey, pubcol)}
_cols = {"subset" : pubnames}
_cols.update( {pubkey[key] : [distrib[name][key] for name in subsets] for key in colkey} )

# _vcol = "velocity"
# _tup = [("name", ""), ("n", ""), (_vcol, "median"), (_vcol, "min"), (_vcol, "max")] 
# df = pd.DataFrame(_cols, 
#     index=pubnames,
#     columns=pd.MultiIndex.from_tuples(_tup))
df = pd.DataFrame(_cols)
s = df.style.format().hide_index()
f1 = lambda x: "{:0.1f}".format(x)
f2 = lambda x: "{:0.3f}".format(x)
_form = [str, str, f1, f2, f2, f2]
lkw = {"index":False, "formatters":_form, "escape": False}
pub.save_dflatex(df, "subsets", "classification", to_latex=lkw)

# %%
random_sample = False
random_sample = True
if random_sample:
    for i in np.random.choice(range(len(top)), size=10, replace=False):
        tr = top[i]
        fig, ax = plt.subplots(figsize=(12,4))
        plotutils.plot_lvel(ax, tr)
        ax.set_title('i = {}'.format(i))
        ax.set_ylim((0,2))


# %%



# %%
