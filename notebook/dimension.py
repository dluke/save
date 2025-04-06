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
# Playing with dimensionality reduction and machine learning
# %%
verbose = False


# %%
import sys, os
import json
import numpy as np
import _fj
import matplotlib.pyplot as plt
import scipy.signal
import twanalyse
# %%
from os.path import join
notedir = os.path.dirname(__file__)
notename = 'dimension'
metadir = join(notedir, 'velocity_profiles/sort_fastest')

with open(join(metadir,'candidates.list'), 'r') as f:
    _whitelist = list(map(int, f.read().split()))
with open(join(metadir,'meta.json'), 'r') as f:
    meta = json.load(f)
whitelist = [meta[str(_i)]['track_id'] for _i in _whitelist]

candidates = _fj.lintrackload(whitelist)

# %%
fast_crawling_id = 2924
track = _fj.trackload([fast_crawling_id])[0]
ltrack = _fj.lintrackload([fast_crawling_id])[0]
print()
print('candidate track has {} timesteps and {} linearised steps'.format(
    track['time'].size, len(ltrack.step_idx)
))

# %%
print('track columns')
print(track.get_dtype().names)


ax = plt.gca()
def plot_candidate(ax):
    time = track['time']
    velocity = track.get_head_v()
    track_speed = np.linalg.norm(velocity, axis=1)
    ax.plot(time[1:]-time[0], track_speed)
plot_candidate(ax)

# %%

ax = plt.gca()
def plot_linearised_candidate(ax):
    step_velocity = ltrack.get_step_velocity()
    step_speed = np.linalg.norm(step_velocity, axis=1)
    step_dt = ltrack.get_step_dt()
    step_time = np.cumsum(step_dt)

    ax.set_xlabel('time (s)')
    ax.set_ylabel('speed')
    ax.plot(step_time, step_speed)
plot_linearised_candidate(ax)

# %%
step_dt = ltrack.get_step_dt()
step_time = np.cumsum(step_dt)
step_velocity = ltrack.get_step_velocity()
step_speed = np.linalg.norm(step_velocity, axis=1)
ax = plt.gca()
ax.hist(step_speed, bins=20, range=(0,1.0))
ax.set_xlabel('step speed')
plt.show()

# %%
ax = plt.gca()
ax.hist(step_dt, bins=20, range=(0,1.0))
ax.set_xlabel('step time')
plt.show()

# %%
window = 'hamming'
velocity = track.get_head_v()
track_speed = np.linalg.norm(velocity, axis=1)
f, Pxx = scipy.signal.periodogram(track_speed, 10.0,
    window=window, scaling='density')
fig = plt.figure(figsize=(10,5))
ax = fig.add_axes([0,0,1,1]) 
ax.plot(f, Pxx)
ax.set_title('periodogram (hamming)')
plt.show()


# %%
# power spectral density
window = 'boxcar'
window = 'hamming'
window = 'blackmanharris'
f, Pxx = scipy.signal.periodogram(track_speed, 10.0,
    window=window, scaling='density')

descending = sorted(list(enumerate(Pxx)), key= lambda t:t[1], reverse=True)
print(descending[:10])

descidx = [t[0] for t in descending]
cutidx = 20
print(descidx[:cutidx])
peaks = [f[idx] for idx in descidx[:cutidx]]
for pk in peaks:
    print(pk)

# %%
# we want to automatically pull out the first few peaks (be size)
period_threshold = 20.0 # seconds
f_threshold = 1./period_threshold
# descending = sorted(list(enumerate(Pxx)), key= lambda t:t[1], reverse=True)
# descidx = [t[0] for t in descending]

peaks, properties = scipy.signal.find_peaks(Pxx)
peaks_value = [Pxx[pid] for pid in peaks]
descending = sorted(list(zip(peaks, peaks_value)), key= lambda t:t[1], reverse=True)
print(descending[:10])

# %%
  
twanalyse.plot_signal(track, 20.0)
plt.show()

# %%
# lets load some more data 
use_dir = join(notedir,notename,'periodgram/')
rule = join(use_dir,'signal_{:04d}.png')
if not os.path.exists(use_dir):
    os.makedirs()
# %%
if verbose:
    print('output with rule', rule)
    for i, (track_id, tr) in enumerate(zip(whitelist, candidates)):
        fig = twanalyse.plot_signal(tr)
        out = rule.format(i)
        plt.savefig(out)
        print('write periodogram to ', out)
    plt.close()
# %% [markdown]
# for fastest trajectories it's common to see a large peaks around [9,4]
# 
# It's really confusing for me why the periodgram has to so many 
# large values for low frequencies. 

# lets identify by eye some periograms with a similar set of peaks
# as our candidate trajectory and then check their velocity profiles

# %%
_similar_periodgram = [4, 5, 10, 11, 12, 13, 29]
similar_periodgram = [whitelist[i] for i in _similar_periodgram]
n_plots = len(similar_periodgram)+1
fig, axes = plt.subplots(n_plots,figsize=(10,n_plots*3), 
    sharey=True, sharex=True)
axes[0].set_title('candidate track')
plot_linearised_candidate(axes[0])
for enum, i in enumerate(_similar_periodgram):
    ax = axes[enum+1]
    ltr = candidates[i]
    dt = ltr.get_step_dt()
    velocity = ltr.get_step_velocity()
    track_speed = np.linalg.norm(velocity, axis=1)
    time = ltr['time']
    ax.plot(np.cumsum(dt), track_speed)
    ax.set_ylabel('linearised velocity')
    ax.set_xlabel('time')
    ax.set_title('track {:04d}'.format(whitelist[i]))
    ax.set_ylim(0,2.0)
plt.tight_layout(pad=1.0) 

# %%
fig, axes = plt.subplots(n_plots,figsize=(10,n_plots*3),
    sharex=True)
axes[0].set_title('candidate track')
twanalyse._ltr_vel(axes[0], [ltrack])
for enum, i in enumerate(_similar_periodgram):
    ax = axes[enum+1]
    ltr = candidates[i]
    twanalyse._ltr_vel(ax, [ltr], kdestyle={'xlims':(0,2.0)})
    ax.set_xlim(0,2.0)
    ax.xaxis.set_tick_params(which='both', labelbottom=True)


# %% [markdown]
# lets try this on simulated data
# --- we implemented this twanalyse.py
# as usual there are many peaks in the periogram, we should 
# consider a more directed or coarse grained analysis of out 
# the velocity profile, perhaps looking again at G. Wong approach

# %%
# using my imagination to count the picks of what I think might be a major
# frequency in the data
imagine_peaks = 31
duration = 0.1 * time.size
print('duration = {:.3f}s'.format(duration))
# our time series has 2079 datapoints
imagine_period = duration/imagine_peaks
print('suggested period {:.3f}s'.format(imagine_period))
# the maximum period is N or N/2?
imagine_frequency = 1/imagine_period
print('imagined frequency {:.3f}'.format(imagine_frequency))

# %% [markdown]
# this track as some irregular periodicity
# try DFT and then reconstructing the speed profile

# %% 
# odd numbers are awkward
speed = track_speed[:-1]
time = track['time'][:-1]
print('input shape')
print(speed.shape, time.size)

# wait. step_speed are not regular samples so does this even make sense?
dft = np.fft.rfft(speed)
print('dft shape')
print(dft.size)
dftcut = dft[:dft.size//2]
# choosing the first m coefficients
dftcut = dft[:20]
print('after cutting ', dftcut.size)
# dft is ordered with largest frequency first
# reduce dimension ...

recover = np.fft.irfft(dft, n=speed.size)
approx_recover = np.fft.irfft(dftcut, n=speed.size)
print('recovered size', approx_recover.size)

# %% 
if verbose:
    cutlist = [2,5,10,20,50,speed.size//2]
    n = len(cutlist)
    fig, axes = plt.subplots(len(cutlist),1, figsize=(5,n*3))
    labelform = 'first {:d} coefficients'
    for i,ax in enumerate(axes):
        cut = cutlist[i]
        cut_recover = np.fft.irfft(dft[:cut], n=speed.size)
        ax.plot(time[1:]-time[0], cut_recover, label=labelform.format(cut))
        ax.legend()
    plt.show()

# %% 
# 
# def inversedft(dft, lcoef, n):
#     # lcoef is a set of k values 
#     # n is the output size
#     # M = len(lcoef)
#     N = len(dft)
#     idft = np.empty(n)
#     for i in range(n):
#         idft[i] = 1/N * np.sum([dft[k]*np.exp((1j * 2 * np.pi * i * k)/N)
#             for k in lcoef])
#     return idft

# abscoef = np.absolute(dft)
# sortcoefidx = np.argsort(abscoef)[::-1]
# n = 5
# largestk = sortcoefidx[:n]
# print('k', largestk)
# largest = dft[sortcoefidx[:n]]

# l_recover = inversedft(dft, largestk, n=speed.size)


