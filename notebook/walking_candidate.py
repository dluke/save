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
# load the walking dataset and animate several trajectories 
# then choose some trajectories that we consider typical walking behaviour

# %% 
work = False
verbose = False

# %% 
# 
import sys, os
join =  os.path.join
import _fj
import twanimation
import matplotlib.pyplot as plt
# %% 
# setup a directory for animations
notedir = os.path.normpath(os.path.dirname(__file__))
savedir = join(notedir, "animate/walking/")
if not os.path.exists(savedir):
    os.makedirs(savedir)

# %% 
walking_idx, walking_trs = _fj.slicehelper.load_trs('default_walking_list')
#  2,    6,   25,   26,   33,   34,   35,   37,   43,   44,   45, ...

# %% 
form  = join(savedir,  "animate_{:04d}.mp4")
sample = 100
i = 3
tr = walking_trs[i]
tr_i = walking_idx[i]
twanimation.outline(plt.gcf(), [tr], sample, camera='follow', savefile=form.format(tr_i))


# %% 
# WORK
N = None
sample = 100
if work:
    for tr_i, tr in list(zip(walking_idx, walking_trs))[:None]:
        plt.clf()
        savefile = form.format(tr_i)
        print("save animation to ", savefile)
        twanimation.outline(plt.gcf(), [tr], sample, camera='follow', savefile=savefile)

# %% 

if verbose:
    candidate_walking_idx = 4
    candidate_data_idx = 33
    tr = walking_trs[candidate_walking_idx]
    savefile = join("animate/", "walking_example_{:04d}.mp4".format(candidate_data_idx))
    sample = 10
    plt.clf()
    twanimation.outline(plt.gcf(), [tr], sample, camera='follow', savefile=savefile)


# %% 
all_idx, ltrs = _fj.slicehelper.load_linearized_trs('all')
# %% 
# one candidate for each type of motion
# %% 
types = ["walking", "crawling", "mixed", "static"]
candidates = [1504, 2924, 1279, 43]
# check the durations
for i, idx in enumerate(candidates):
    tr = ltrs[idx]
    score = _fj.flip_score(tr)
    print("flip score", score)
    # force  flip on walking track because it look right to me
    # flip = score < 0 or types[i] == "walking" 
    flip = score < 0 
    if flip:
        print("Flipping candidate ", idx)
        tr.flip_poles()
    print("{} candidate duration {}".format(types[i], tr.get_duration()))

# %% 
sample = 10
fps = 50
for i, idx in enumerate(candidates):
    tr = ltrs[idx]
    plt.clf()
    savefile = join(notedir, "animate/candidates/{}_example_{:04d}.mp4".format(types[i], idx))
    print("save to", savefile)
    twanimation.outline(plt.gcf(), [tr], sample, camera='follow', 
        savefile=savefile, fps=fps)

print("end")


