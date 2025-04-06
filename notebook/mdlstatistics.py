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
# compute statistics from the PWL solution


# %% 
import os
import json
import numpy as np
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
import pandas as pd
import json
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

import pili
import _fj
import mdl
import annealing 
import pwlpartition

import fjanalysis
import pwlstats

# %% 
# operate on data object
def get_array(getter, data):
    return np.array([getter(local) for local in data])

def make_get(localvar):
    keys = localvar.split('.')
    def getter(localdata):
        _dct = localdata
        for k in keys:
            _dct = _dct.get(k, None)
            if _dct is None: 
                return np.nan
        return _dct
    return getter

# # %% 

# root = pwlstats.root
# from pwlstats import load_pwl, load_lptr, load_smoothed, model_summary


# path = join(root, "run/_candidate_pwl/")
# config, solver = load_pwl(path)
# candidate_data = load_lptr(config['trackidx'])
# candidate_model = solver.anneal.get_current_model()
# candidate = load_smoothed(config['trackidx'])


# path = join(root, "run/_top_pwl/")
# config, solver = load_pwl(path)
# top_data = load_lptr(config['trackidx'])
# top_model = solver.anneal.get_current_model()
# top = load_smoothed(config['trackidx'])

# # %% 
# # fig, ax = plt.subplots(figsize=(200,200))
# # mdl.plot_model_on_data(ax, model, data, intermediate=None, config={})


# # %% 
# candidate_dist = candidate_model.get_distance()
# top_dist = top_model.get_distance()
# print('means', np.mean(candidate_dist), np.mean(top_dist))

# color = iter(['r', 'b'])
# _opt = {'stat': 'density', 'element':'bars', 'fill':True, 'alpha': 0.5}

# fig, ax = plt.subplots(figsize=(5,5))
# sns.histplot(candidate_dist, color=next(color), ax=ax, label='candidate', **_opt)
# sns.histplot(top_dist, color=next(color), ax=ax, label='top', **_opt)
# ax.legend()
# # ax.set_ylabel("")
# ax.set_xlabel(r"distance $(\mu m)$")
# # min(dist), max(dist)

# # %% 
# # TODO run on the whole top dataset

# # %% 
# #------------------------------------------------------------------------------ 
# # load the track data on the bottom of the jobs list (lowest velocity) 
# trackidx = 1687
# wavelet = load_smoothed(trackidx).cut_time(0,20)
# path = join(root, 'run/top/_top_1687')
# config, solver = load_pwl(path)
# lptr = load_lptr(config['trackidx'])
# _data = lptr.cut(0,20)
# _model = solver.anneal.get_current_model().cut(0,20)

# fig, ax  = plt.subplots(figsize=(10,10))
# inter = {'is_outlier': solver.anneal.get_outliers()}
# _config = {'h_outlier': True}
# mdl.plot_model_on_data(ax, _model, _data, intermediate=inter, config=_config)

# # for good measure lets draw the smoothed data as well
# # 
# x, y = wavelet.get_head2d().T
# lwave, = ax.plot(x,y, lw=2)

# # append to the legend
# handles = ax.get_legend().legendHandles
# labels = [t.get_text() for t in ax.get_legend().texts]
# handles.append(lwave)
# labels.append('wavelet')
# ax.legend(handles, labels, fontsize=26)

# pub.save_figure("pwl_vs_wavelet_method", "mdlstatistics")


# # %% 
# # load model for track track_1687
# path = join(pili.root, "../sparseml", "run/top/notebook", f"track_1687")
# _model = solver.load_state(path).anneal.get_current_model()
# _wavelet = load_smoothed(1687)
# stat = model_summary(1687, _wavelet, _model)

# stat['pwlm']/stat['speed']

# %% 
# clusterpath = join(pili.root, "../sparseml/run/partition/top/cluster")

clusterpath = join(pwlstats.root, "run/partition/top/cluster")


def list_at(path):
    return [join(path,d) for d in os.listdir(path) if d.startswith('_')]
lst = list_at(clusterpath)


def split_path_trackidx(path):
    return int(os.path.split(path)[-1].split('_')[-1])

def load_at(path):
    _path = join(path, 'config.json')
    with open(_path) as f:
        config = json.load(f)
    trackidx = split_path_trackidx(path)
    wavelet = pwlstats.load_smoothed(trackidx)
    # solver = annealing.Solver.load_state(join(path,'solver'))
    solver = pwlpartition.Solver.load_state(join(path,'solver'))
    return (trackidx, wavelet, solver)

datas = [load_at(l) for l in lst if os.path.exists(join(l, 'solver.pkl'))]

print('found {}/200 tracks'.format(len(datas)))

# %% 

def local_summary(tup):
    trackidx, wavelet_track, solver = tup
    local = {}
    local.update(pwlstats.track_summary(trackidx, wavelet_track))
    local.update(pwlstats.solver_summary(solver))
    local.update(pwlstats.empirical_summary(solver, wavelet_track))
    return local

datalist = [local_summary(tup) for tup in datas] 

# %% 
deviation_std = get_array(lambda data: np.std(data['deviation']), datalist)

with mpl.rc_context({"font.size": 16}):
    fig, ax= plt.subplots(figsize=(6,4))
    sns.histplot(deviation_std, binrange=(0,np.pi), ax=ax)
    ax.set_xlabel(r"$std(\theta_{\mathrm{deviation}})$")
    fig.tight_layout()

pub.save_figure("deviation_angle_standard_deviation_distribution", "mdlstatistics")

# %% 
mean_step = get_array(make_get('mean_step_length'), datalist)

with mpl.rc_context({"font.size": 16}):
    fig, ax= plt.subplots(figsize=(6,4))
    sns.histplot(mean_step, binrange=(0,0.4), ax=ax)
    ax.set_xlabel(r"mean step length $(\mu m)$")
    fig.tight_layout()

pub.save_figure("mean_step_length_distrib", "mdlstatistics")

# %% 
step_freq = get_array(make_get('step_rate'), datalist) / _DT
with mpl.rc_context({"font.size": 16}):
    fig, ax= plt.subplots(figsize=(6,4))
    sns.histplot(step_freq,  ax=ax)
    ax.set_xlabel(r"step frequency $(s^{-1})$")
    fig.tight_layout()

pub.save_figure("step_frequency_distribution", "mdlstatistics")

# %% 
_DT = 0.1
duration =_DT * get_array(make_get("duration"), datalist)
mean_speed = get_array(make_get("contour_length"), datalist) / duration

fj_mean_speed = get_array(make_get("speed"), datalist)

defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']

with mpl.rc_context({"font.size": 20}):
    fig, ax= plt.subplots(figsize=(5,4))
    sns.histplot(mean_speed, binrange=(0, 0.2), ax=ax, color=defcolors[0])
    # sns.histplot(fj_mean_speed, binrange=(0, 0.2), ax=ax, color=defcolors[1])
    ax.set_xlabel(r"mean speed $(\mu m/s)$")
    fig.tight_layout()

# %% 

with mpl.rc_context({"font.size": 20}):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(mean_speed, step_freq, alpha=0.4)
    ax.set_ylabel(r"step frequency $(s^{-1})$")
    ax.set_xlabel("mean speed")
    xlim = 0, 0.2
    ax.set_xlim(xlim)
    ylim = 0, 1.0
    ax.set_ylim(ylim)

# %% 

with mpl.rc_context({"font.size": 20}):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(mean_speed, mean_step, alpha=0.4)
    ax.set_ylabel(r"$\langle l \rangle$")
    ax.set_xlabel("mean speed")
    xlim = 0, 0.2
    ylim = 0, 0.4
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # x, y = np.clip(mean_speed, *xlim), np.clip(mean_step, *ylim)
    # A = np.vstack([x, np.ones(len(x))]).T
    # m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    # xspace = np.linspace(*xlim, num=1000)
    # lfit, = ax.plot(xspace, m*xspace + c, linestyle='--', lw=3)


# x, y = _df["speed"], _df["step_rate"],
# A = np.vstack([x, np.ones(len(x))]).T
# m, c = np.linalg.lstsq(A, y, rcond=None)[0]
# xspace = np.linspace(x.min()-0.02, x.max()+0.02, 1000)
# lfit, = ax.plot(xspace, m*xspace + c, linestyle='--', color=palette[1], lw=3)



# %% 
# ------------------------------------------------------------------------------

# def to_dataframe(summary_data):
#     keys = summary_data[0].keys()
#     data = {}
#     for key in keys:
#         data[key] = [local[key] for local in summary_data]
#     return pd.DataFrame(data)

# summary_data = [model_summary(*tup) for tup in datas]
# local = to_dataframe(summary_data)

# %% 
# steps per second

palette = sns.color_palette()

sns.set_theme(font_scale = 2, style='white')
ax = sns.scatterplot(data=df, x="speed", y="step_rate", hue='whitelist', s=160)
xlim = (0,0.2)
ax.set_xlim(0,0.2)
ax.set_ylim(0,None)

topdf = df.loc[df["whitelist"] == True]
otherdf = df.loc[df["whitelist"] == False]
dfs = [otherdf, topdf]

for i in range(2):
    _df = dfs[i]
    color = palette[i]
    # TODO how to compute the standard error of a counting process  (poisson distribution)
    err = np.sqrt(_df["pwlm"])/_df["duration"]
    ax.errorbar(
        _df["speed"], _df["step_rate"],
        yerr=err, fmt='none', 
        alpha=0.6, color=color
    )

ax.set_xlabel(r"FJ wavelet instantaneous speed $(\mu m/s)$")
ax.set_ylabel(r"steps per second $(s^{-1})$")
# linear regression

x, y = _df["speed"], _df["step_rate"],
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
xspace = np.linspace(x.min()-0.02, x.max()+0.02, 1000)
lfit, = ax.plot(xspace, m*xspace + c, linestyle='--', color=palette[1], lw=3)

handles = ax.get_legend().legendHandles
handles.append(lfit)
labels = ["other", "top", "top_fit"]
ax.legend(handles, labels)

pub.save_figure("preliminary_result_step_rate", "mdlstatistics")

# %%
# median step length

sns.set_theme(font_scale = 2, style='white')
ax = sns.scatterplot(data=df, x="speed", y="median_step_length", hue='whitelist', s=160)
xlim = (0,0.2)
ax.set_xlim(0,0.2)
ax.set_ylim(0,None)
ax.set_xlabel(r"FJ wavelet instantaneous speed $(\mu m/s)$")
ax.set_ylabel(r"median step length $(\mu m)$")

x, y = _df["speed"], _df["median_step_length"],
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
xspace = np.linspace(x.min()-0.02, x.max()+0.02, 1000)
lfit, = ax.plot(xspace, m*xspace + c, linestyle='--', color=palette[1], lw=3)

ax.set_ylim(0,0.4)

handles = ax.get_legend().legendHandles
handles.append(lfit)
labels = ["other", "top", "top_fit"]
ax.legend(handles, labels)


pub.save_figure("preliminary_result_median_step_length", "mdlstatistics")

# %%
# check the first track in the walking set 
# ... OK lets solve it ...

walking_idx = _fj.slicehelper.load("pure_walking")
walking_track = _fj.trackload([walking_idx[0]])[0]

walking_data = mdl.get_lptrack(track)
defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.3, 'color':defcolors[2]}
_data = walking_data.cut(10,20)
plt.plot(_data.x, _data.y, **ptlkw)

sigma = pwlpartition.estimate_error(walking_data.x, walking_data.y)
pwlpartition.estimate_r(sigma)




