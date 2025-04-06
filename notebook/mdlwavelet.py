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
# wavelet transform


# %% 
import os
import json
import numpy as np
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
import pandas as pd
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

import pili
import _fj
import mdl
import annealing 

import fjanalysis
import pwlstats

import pywt
from skimage.restoration import denoise_wavelet
from skimage.restoration import  estimate_sigma

# %% 
# /home/dan/usb_twitching/sparseml/run

# %% 

# trackidx = 1687
trackidx = 2924
track = _fj.trackload_original([trackidx])[0]
# fanjin wavelet track
fjtr = pwlstats.load_smoothed(trackidx)
pwlmodel = pwlstats.load_pwl_model(join(pwlstats.root, "run/_candidate_pwl"))


# %% 
# estimate sigma
xy = track.get_head2d()
x, y = xy.T
def estimate_error(x, y):
    sigma_est_x = estimate_sigma(x)
    sigma_est_y = estimate_sigma(y)
    sigma_est = np.mean([sigma_est_x, sigma_est_y])
    return sigma_est
sigma_est = estimate_error(x, y)
print('estimate sigma', sigma_est)

# %% 
xy = track.get_head2d()
x, y = xy.T
# N = 100
N = 400
x = x[:N]
y = y[:N]
_data = {'x':x, 'y':y}

wave = 'db1'
method = 'VisuShrink'
mode = 'hard'
sigma = 0.03
scikit_config = {"wavelet":'db1', 'method':'VisuShrink', "mode":'soft', "rescale_sigma":False}

def vary_sigma(x, y, sigma, config=scikit_config):
    x_denoise = denoise_wavelet(x, sigma=sigma, **config)
    y_denoise = denoise_wavelet(y, sigma=sigma, **config)
    denoised = np.stack([x_denoise, y_denoise])
    return denoised

denoised = vary_sigma(x, y, sigma)


# %% 
# how do we coarsen the wavelet solution to a piecewise linear curve?
_DT = 0.1

def contract(denoised, sigma):
    threshold = 1e-2 * sigma
    N = denoised[0].size
    xyt = denoised.T
    #  the denoised trajectory has repeated values (VisuShrink) method
    diff = xyt[1:] - xyt[:-1]
    eq = norm(diff, axis=1) < threshold
    seg_idx = np.concatenate([[0], np.argwhere(eq == False).ravel()+1, [N-1]])
    x, y = xyt[seg_idx].T
    dt = np.zeros(x.size, dtype=int)
    dt[1:] = np.diff(seg_idx)
    dt[1] += 1 # the 0th point represents a one timepoint
    return mdl.LPtrack(dt, x, y)

def model_from_denoised(denoised, sigma=0.04, angle_threshold=15):
    contracted = contract(denoised, sigma)

    # check the distances between nodes
    wavemodel = mdl.recursive_coarsen(contracted, 2*sigma)

    def contract_by_angle(wavemodel, angle_threshold=5):
        # threshold in degrees
        theta = wavemodel.get_angle() * 180/np.pi
        list_theta = theta.tolist()[::-1]

        keepidx = [0]
        i = 1
        _curr = 0
        dtlist = [0]
        while list_theta:
            _curr += list_theta.pop()
            if abs(_curr) > angle_threshold:
                _curr = 0
                keepidx.append(i)
            i += 1
        keepidx.append(len(wavemodel)-1)
        n= len(keepidx)
        dt = np.zeros(n, dtype=int)
        dt[1:] = np.diff(np.cumsum(wavemodel.dt)[keepidx])
        model = mdl.LPtrack(dt, wavemodel.x[keepidx], wavemodel.y[keepidx])
        return model

    bmodel = contract_by_angle(wavemodel, angle_threshold)
    return bmodel

# # check the angles afterwards
# new_angles = bmodel.get_angle() *180/np.pi 
# new_angles


# %% 


sig = 0.04
sig = estimate_error(x ,y)
print(f'using sigma = {sig}')
denoised = vary_sigma(x, y, sig)

wavemodel = model_from_denoised(denoised, sigma=sig, angle_threshold=15)

wavemodel.dt < 2

# %%

def plot_denoised_on_data(ax, denoised, data, config={}):
    ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.4}
    ax.plot(data['x'],  data['y'], label="data", **ptlkw)
    xwave, ywave = denoised
    _label = config.get("denoised_label", "my wavelet")
    ax.plot(xwave, ywave, lw=4, marker='D', label=_label)
    ax.plot(xwave, ywave, linestyle='none', marker='D', markersize=8, c='#FC7B33')
    ax.set_aspect("equal")

def plot_wavemodel_on_data(ax, wavemodel, data, config={}):
    match_data = config.get("match_data", False)
    color = itertools.cycle(['#FEC216', '#F85D66', '#75E76E'])
    ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.4}
    

    if match_data:
        # use the color cycle and the time data to plot chunks of points and lines  with matching colors
        time = wavemodel.get_time()
        x, y = data['x'], data['y']
        xwave, ywave = wavemodel.x, wavemodel.y
        for i in range(len(wavemodel)-1):
            _col = next(color)
            ti, tf = time[i], time[i+1]
            ax.plot(x[ti:tf], y[ti:tf], color=_col, **ptlkw) 
            ax.plot([xwave[i], xwave[i+1]], [ywave[i], ywave[i+1]], color=_col, lw=2, alpha=0.6)
        ax.plot(xwave, ywave, linestyle='none', marker='D', alpha=0.6)

    else:
        ax.plot(data['x'],  data['y'], label="data", **ptlkw)
        xwave, ywave = wavemodel.x, wavemodel.y
        _label = config.get("denoised_label", "my wavelet")
        ax.plot(xwave, ywave, lw=4, marker='D', label=_label)
        ax.plot(xwave, ywave, linestyle='none', marker='D', markersize=8, c='#FC7B33')
    ax.set_aspect("equal")


fs = 20
fig, ax = plt.subplots(figsize=(fs,fs))

print('wavemodel has {} nodes'.format(len(wavemodel)))
_config = {'match_data': True}
plot_wavemodel_on_data(ax, wavemodel, _data, config=_config)

# %% 
fs = 20
fig, ax = plt.subplots(figsize=(fs,fs))
ax.set_xlim(lptr.x.min(), lptr.x.max())
ax.set_ylim(lptr.y.min(), lptr.y.max())

_config = {'match_data': True}
plot_wavemodel_on_data(ax, partition.model, _data, config=_config)



# %% 
fjtr_data = fjtr.cut(0,N).get_head2d().T
pwlmodel_part = pwlmodel.cut(0,40)

ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.4}
for sigma in [0.01,0.02,0.03,0.04,0.06]:

    fs = 20
    fig, ax = plt.subplots(figsize=(fs,fs))

    # denoised = contract(vary_sigma(x, y, sigma), sigma)
    denoised = vary_sigma(x, y, sigma)
    plot_denoised_on_data(ax, denoised, _data, 
        config={'denoised_label':f'sigma={sigma}'})
    ax.set_title(f'sigma={sigma}')

    offset=-0.5
    ax.plot(_data['x'], _data['y']-offset, c='#2077B4', **ptlkw)
    fjtrx, fjtry = fjtr_data
    ax.plot(fjtrx, fjtry-offset, lw=4, label="fj wavelet")

    offset=0.5
    wavemodel = model_from_denoised(denoised, sigma)
    ax.plot(_data['x'], _data['y']-offset, c='#2077B4', **ptlkw)
    wmconf = {'lw':4, 'marker':'D', 'markersize':8, 'c':'#FC7B33'}
    ax.plot(wavemodel.x, wavemodel.y-offset, label="processed wavelet", **wmconf)


    offset=1.0
    ax.plot(_data['x'], _data['y']-offset, c='#2077B4', **ptlkw)
    ax.plot(pwlmodel_part.x, pwlmodel_part.y-offset, c='#FEC216', lw=4, alpha=0.4, label="pwl model")

    ax.legend()
    

# %% [markdown]
# sigma = 0.03/0.04 appears to give good results


# %% 
# produce some useful plots of the candidate trajectory with wavelet 
# run on whole trajectories
def plot_wavemodel(track, pwlmodel):
    xy = track.get_head2d().T
    # sigma = 0.04
    sigma = sigma_est
    denoised = vary_sigma(*xy, sigma=sigma)
    x, y = xy
    _data = {'x':x, 'y':y}

    fig, ax = plt.subplots(figsize=(200,200))
    mdl.plot_model_on_data(ax, pwlmodel, _data)

    wavemodel = model_from_denoised(denoised, sigma, angle_threshold=20)
    # ax.plot(_data['x'], _data['y'], c='#2077B4', **ptlkw)
    wmconf = {'lw':4, 'marker':'D', 'markersize':8, 'c':'#FC7B33'}
    ax.plot(wavemodel.x, wavemodel.y, label="processed wavelet", **wmconf)
    ax.set_aspect("equal")

    # plot_denoised_on_data(ax, denoised, _data, config={})
    # ax.plot(pwlmodel.x, pwlmodel.y, c='#FEC216', lw=4, alpha=0.4, label="pwl model")

# %%
plot_wavemodel(track, pwlmodel)

# pub.save_figure("candidate_scikit_wavelet", "mdlwavelet")
path = join(pwlstats.root, "candidate_scikit_wavelet_process_estimate.svg")
print("writing to ", path)
plt.savefig(path)

# %% 
trackidx = 1687
_track = _fj.trackload_original([trackidx])[0]
_pwlmodel = pwlstats.load_pwl_model(join(pwlstats.root, 'run/top/cluster/_top_1687'))
plot_wavemodel(_track, _pwlmodel)

path = join(pwlstats.root, "scikit_process_1687.svg")
print("writing to ", path)
plt.savefig(path)

# %% 
# 0040
def compute_wavemodel(track):
    x, y = track.get_head2d().T
    sigma = estimate_error(x, y)
    print(f'estimate sigma {sigma}')
    denoised = vary_sigma(x, y, sigma=sigma)
    wavemodel = model_from_denoised(denoised, sigma, angle_threshold=20)
    return wavemodel

trackidx = 40
_track = _fj.trackload_original([trackidx])[0]
wavemodel = compute_wavemodel(_track)

fig, ax = plt.subplots(figsize=(200,200))

ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.4}
ax.plot(_track['x'],  _track['y'], label="data", **ptlkw)

wmconf = {'lw':4, 'marker':'D', 'markersize':8, 'c':'#FC7B33'}
ax.plot(wavemodel.x, wavemodel.y, label="processed wavelet", **wmconf)
ax.set_aspect("equal")

# %% 
allow_annealing = False
if allow_annealing:
    r = 0.03

    loss_conf = {}
    loss_conf["contour_term"] = 0.1

    anneal = annealing.Anneal(r)
    anneal.default_loss_conf = loss_conf
    data = mdl.get_lptrack(_track)
    anneal.initialise(wavemodel, data)
    solver = annealing.Solver(anneal)
    solver.cleanup()
    cleanmodel = solver.get_current_model()
    print("finished")

# %% 
if allow_annealing:
    fig, ax = plt.subplots(figsize=(200,200)) 
    is_outlier = anneal.get_outliers()
    mdl.plot_model_on_data(ax, cleanmodel, data, 
        intermediate={'is_outlier':is_outlier}, config={"h_outlier":True})


# %% 
# suggest no benfit to varying the wavelet
"""
pywt.families()
pywt.wavelist()
try_wavelet = ['haar', 'db1', 'sym2', 'dmey', 'bior1.1']
for wave in try_wavelet:
    fs = 20
    fig, ax = plt.subplots(figsize=(fs,fs))

    conf = scikit_config.copy()
    conf['wavelet'] = wave
    print(conf)
    denoised = vary_sigma(0.04, conf)

    plot_denoised_on_data(ax, denoised, _data, 
        config={'denoised_label':f'sigma={sigma}'})
    ax.set_title(f'wavelet={wave}')

    offset=-0.5
    ax.plot(_data['x'], _data['y']-offset, c='b', **ptlkw)
    fjtrx, fjtry = fjtr_data
    ax.plot(fjtrx, fjtry-offset, lw=4, label="fj wavelet")

    offset=0.5
    ax.plot(_data['x'], _data['y']-offset, c='b', **ptlkw)
    ax.plot(pwlmodel_part.x, pwlmodel_part.y-offset, c='#FEC216', lw=4, alpha=0.4, label="pwl model")

    ax.legend()
"""
None



