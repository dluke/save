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
# estimate the instantaneous velocity using the solved pwl model


# %% 
import os
import json
import numpy as np
import scipy.stats
import scipy.optimize
import pickle
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

import pili
from pili import support
import _fj
import mdl
import pwlpartition

import fjanalysis
import pwlstats

from skimage.restoration import denoise_wavelet
from skimage.restoration import  estimate_sigma

# %% 
mplstyle = {"font.size": 20}
notename = "mdlvelocity"
publish = False

# %% 

# target = join(pwlstats.root, "run/partprio/_candidate_pwl/")
# target = join(pwlstats.root, "run/partition/candidate/_candidate_pwl/")
target = join(pwlstats.root, "run/partition/candidate/no_heuristic/_candidate_pwl/")
# target = join(pwlstats.root, "run/partition/candidate/no_heuristic/_top_0040/")

# N = 200
N = 400

solver = pwlpartition.Solver.load_state(join(target, 'solver'))

# !tmp
solver.partition.use_probability_loss = False
solver.partition.inter_term = False
# !~tmp

solver.partition.cache_local_coord = np.empty(solver.partition.N) # hotfix
solver.partition.update_residuals()


# %% 
# for presentation

with mpl.rc_context({'font.size': 16}):
    fig, ax = plt.subplots(figsize=(10,10))
    pwlpartition.simple_model_plot(ax, solver.partition.model.cut(0,N), solver.partition.data.cut_index(0,N))
    ax.set_ylabel('y')
    ax.set_xlabel('x')

# plot_target = join(pwlstats.root, "impress/images")
# target = join(plot_target, "track_0040.png")
# print('saving to ', target)
# plt.tight_layout()
# plt.savefig(target)


# %% 
_DT = 0.1

def load_pkl(at):
    if not at.endswith('.pkl'): at += '.pkl'
    with open(at, 'rb') as f: return pickle.load(f)

initial = load_pkl(join(target, 'initial_guess')) 
print(type(initial))
fig, ax = plt.subplots(figsize=(12,5))
def plot_lvel(ax, initial, N, label='wavelet'):
    initial_wavelet_velocity = initial.get_step_length() / (_DT * initial.dt[1:])
    time = initial.get_time()
    time_index = np.searchsorted(time, N, side='right')
    t = time[:time_index]
    lvel = initial_wavelet_velocity[:time_index]
    style = {'alpha':0.6}
    ax.plot(np.repeat(t,2)[1:], np.repeat(lvel,2)[:-1], label=label, **style)

plot_lvel(ax, initial, N)

# %% 
# expand the wavelet velocity to N
initial_wavelet_velocity = initial.get_step_length() / (_DT * initial.dt[1:])
initial_wavelet_velocity.size
# use my breakpoints or initial break points?


# %% 
# todo allow mapping curve coordinates to adjacent segments?
time = solver.partition.model.get_time()

curve_coord = solver.partition.get_curve_coord()

def plot_curve_coord(ax, curve_coord, time):
    ax.plot(curve_coord[:N])
    time_index = np.searchsorted(time, N, side='right')
    for t in time[:time_index]:
        index = t
        l = ax.scatter(index, curve_coord[index], marker='D', c='green', s=20)
    ax.set_xlabel('index')
    ax.set_ylabel(r'curve coordinate $(\mu m)$')
    ax.legend([l], ['breakpoint'])

sigma = estimate_sigma(curve_coord)
print('estimate sigma = ', sigma)
_sigma =   sigma

wave_config = {"wavelet":'db1', 'method':'BayesShrink', "mode":'soft', "rescale_sigma":False}
wave_curve_coord = denoise_wavelet(curve_coord, sigma=_sigma, **wave_config)

with mpl.rc_context(mplstyle):
    def_figsize = (12, 5)
    fig, ax = plt.subplots(figsize=(12,5))
    plot_curve_coord(ax, curve_coord, time)

    plot_curve_coord(ax, wave_curve_coord, time)
if publish:
    pub.save_figure("curve_coordinate_example", notename)

# now compute velocity
curve_velocity = np.diff(curve_coord, prepend=curve_coord[0]) / _DT
print('curve velocity estimate sigma', estimate_sigma(curve_velocity))
# curve_velocity = np.diff(wave_curve_coord, prepend=wave_curve_coord[0]) / _DT

# wave_curve_velocity = denoise_wavelet(curve_velocity, sigma=0.15, **wave_config)
# wave_curve_velocity = denoise_wavelet(curve_velocity, sigma=0.25, **wave_config)

from scipy.ndimage import gaussian_filter1d
gauss_curve_velocity = gaussian_filter1d(curve_velocity, sigma=1.5)

with mpl.rc_context(mplstyle):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(curve_velocity[:N], alpha=0.4, label="no filter")
    ax.plot(gauss_curve_velocity[:N], alpha=0.6, label="guassian_filter")
    ax.axhline(0, c='k', alpha=0.2)
    ax.set_xlabel('index')
    ax.set_ylabel(r'curve velocity $(\mu m/s)$')
    plot_lvel(ax, initial, N)
    # plot_lvel(ax, solver.partition.model, N, label='solve')

    time_index = np.searchsorted(time, N, side='right')
    for t in time[:time_index]:
        index = t
        ax.axvline(index, alpha=0.3, linestyle='--', color='k', zorder=-1)
        # ax.scatter(index, curve_coord[index], marker='D', c='green', s=20)

    ax.legend(fontsize=16)

if publish:
    pub.save_figure("mapped_velocity_example", notename)

# mean instantaneous velocity
print('instantaneous velocities:') 
print('wavelet ', np.mean(curve_velocity))
print('smoothed', np.mean(gauss_curve_velocity))
# it's unexpectedly higher than FJ wavelet, right?

# %% 

near_r = solver.sigma
near_r = solver.r

def get_next_idx(solver, near_r):
    model = solver.get_model()
    coord = solver.partition.get_curve_coord()
    breakpts = np.insert(np.cumsum(model.get_step_length()),0,0)
    mapping = solver.partition.get_mapping()
    rdistance = np.abs(breakpts[mapping]-coord)
    fdistance = np.abs(breakpts[mapping+1]-coord)
    near_idx = np.logical_or(rdistance < near_r, fdistance < near_r)
    return near_idx

near_idx = get_next_idx(solver, near_r)

# %% 
# for _sig in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]:
#     gauss_curve_velocity = gaussian_filter1d(curve_velocity, sigma=_sig)
#     fig, ax = plt.subplots(figsize=(4,4))
#     sns.histplot(gauss_curve_velocity)
# sns.histplot(np.diff(wave_curve_coord), binrange=(-0.1,0.1))

# try standard averaging
discrete_velocity = np.diff(curve_coord) / _DT
discrete_dx = np.diff(curve_coord) 
pair_near_idx = np.logical_or(near_idx[1:], near_idx[:-1])
# data = discrete_dx[pair_near_idx]
data = discrete_dx
N = data.size

# remove zeros
# data = data[~(data==0)]
# print('remove zeros {}/{}'.format(data.size, N))

def shorten(arr, n):
    return arr[:arr.size//n * n]

shstyle = dict(element="step", fill=False, alpha=0.8)

avg = data

negative_only = True
ng_data = avg[avg<0]
ps_data = avg[avg>=0]
print("N = ", avg.size)

fig, ax = plt.subplots(figsize=(4,4))
sns.histplot(ng_data, ax=ax, stat="density", **shstyle)
sns.histplot(ps_data, ax=ax, stat="density", **shstyle)
_shstyle  = shstyle.copy()
_shstyle["alpha"] = 0.3
sns.histplot(data, ax=ax, **_shstyle)
# ax.set_xlim((-0.8,1.2))
xlim = (-0.08,0.12)
ax.set_xlim(xlim)

ps_data.size, ng_data.size

# TODO check that this shape is not due to edge effects at breakpoints

# loc, scale = scipy.stats.expon.fit(-ng_data)
# print(loc, scale)
# space = np.linspace(0, -xlim[0], 1000)
# # exp_ng = (loc, scale)
# expon = scipy.stats.expon(loc, scale)
# ax.plot(-space, expon.pdf(space), **fitstyle)

# next construct double exponential distribution


# %% 
# now switch to continuous distributions and subtract the laplace (error) distribution 
import statsmodels.api as sm

def get_kde(data, bw_factor=1.0):
    kde = sm.nonparametric.KDEUnivariate(data)
    kde.fit()
    kde.fit(bw=bw_factor * kde.bw)
    return kde

def plot_dx_data(ax, data):
    kde = get_kde(data, bw_factor=0.5)
    sns.histplot(data, ax=ax, stat="density", label=r"bin $\Delta x$", **shstyle)
    plot_dx_kde(ax, kde)

def plot_dx_kde(ax, kde):
    shstyle = dict(element="step", fill=False, alpha=0.8)
    ax.plot(kde.support, kde.density, label=r"smooth $\Delta x$")
    ax.set_xlim(xlim)
    ax.set_xlabel(r"curve $\Delta x$")
    ax.axvline(0, c='k', alpha=0.2)
    ax.legend()

fig, ax = plt.subplots(figsize=(6,4))
plot_dx_data(ax, data)

# %% 
# plot the gradient of
def plot_dx_gradient(ax, data, rescale=None):
    kde = get_kde(data, bw_factor=0.5)
    grad = np.gradient(kde.density, kde.support)
    # ax.axhline(0, c='k', alpha=0.2)
    # ax.axvline(0, c='k', alpha=0.2)
    re = grad.max()/rescale if rescale != None else 1.
    ax.plot(kde.support, grad/re, linestyle='--', alpha=0.6, label="gradient")
    ax.set_xlim(xlim)

fig, ax = plt.subplots(figsize=(6,4))
plot_dx_gradient(ax, data)

# %% 
# put it together

kde = get_kde(data, bw_factor=0.5)

def plot_dx_with_gradient(ax, data):
    plot_dx_data(ax, data)
    plot_dx_gradient(ax, data, rescale=kde.density.max())
    ax.axhline(0, c='k', alpha=0.2)
    # ax.legend(['a'])

fig, ax = plt.subplots(figsize=(6,4))
plot_dx_with_gradient(ax, data)



# %% 
# * we plot standard velocity distributions later but lets do it here for easy comparision
original = _fj.trackload_original([2924])[0]
candidate = _fj.trackload([2924])[0]
mywave = pwlstats.denoise_track(original)
lincandidate = _fj.lintrackload([2924])[0]

origvel = original.get_speed()
vel = candidate.get_speed()
myvel = mywave.get_speed()
# lvel = lincandidate.get_step_speed()
lvel = lincandidate.get_speed()

shstyle = dict(element="step", fill=False, alpha=0.8)

with mpl.rc_context(mplstyle):
    fig, ax= plt.subplots(figsize=(12,8))
    # sns.histplot(lvel, ax=ax, stat="density", **shstyle)

    sns.histplot(_DT * origvel, stat="density", label="original", **shstyle)
    sns.histplot(data, stat="density", label="PWL \Delta x", **shstyle)

    sns.histplot(_DT * vel, stat="density", label="wavelet", **shstyle)
    sns.histplot(_DT * myvel, stat="density", label="my-wavelet", **shstyle)

    ax.legend()

    # _xlim = [0,0.1]
    ax.set_xlim(xlim)
    ax.axvline(0, c='k', alpha=0.2)

# * what I learned from this 
# the same peak (shifted slightly) can be seen in the 
# unprocessed velocity distribution but is harder to identify using FJ wavelet transform 

# * there is a peak using sklearn wavelet transform / sigma estimate
# * sklearn wavelet transform is quite different to FJ

# %% 
# plot these separately for presentation
# just wavelet transform
# with wavelet transform + course graining
# original noisy displacements
# PWL perpendicular displacements

plot_target = join(pwlstats.root, "impress/images")

xlim = (-0.05,0.12)
vstyle = dict(alpha=0.4, c='k', linestyle='--')
figsize = (3,3)

context = {"font.size": 14, "text.usetex":True, "axes.labelsize" : 18}

with mpl.rc_context(context):
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(_DT * vel, stat="density", ax=ax, **shstyle)
    ax.axvline(0, **vstyle)
    ax.set_xlim(xlim)
    ax.set_xlabel(r"wavelet denoised $\Delta x$ $(\mu m)$", fontsize=16)

target = join(plot_target, "candidate_wavelet_denoised_dx.png")
print('saving to ', target)
plt.tight_layout()
plt.savefig(target)

with mpl.rc_context(context):
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(_DT * origvel, stat="density", ax=ax, **shstyle)
    ax.set_xlim(xlim)
    ax.axvline(0, **vstyle)
    ax.set_xlabel(r"original $\Delta x$ $(\mu m)$")

target = join(plot_target, "candidate_original_dx.png")
print('saving to ', target)
plt.tight_layout()
plt.savefig(target)


with mpl.rc_context(context):
    fig, ax = plt.subplots(figsize=figsize)
    entries, bins = np.histogram(_DT * lvel, bins='auto')
    sns.histplot(_DT * lvel, stat="density", bins=bins.size//2, ax=ax, **shstyle)
    ax.axvline(0, **vstyle)
    ax.set_xlabel(r"denoised \& coarse grained ", fontsize=14)
    ax.set_xlim(xlim)

target = join(plot_target, "candidate_wavelet_linearised_dx.png")
print('saving to ', target)
plt.tight_layout()
plt.savefig(target)


with mpl.rc_context(context):
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data, stat="density", ax=ax, **shstyle)
    ax.axvline(0, **vstyle)
    ax.set_xlim(xlim)
    ax.set_xlabel(r"PWL solve $\Delta x^\parallel$ $(\mu m)$")

target = join(plot_target, "candidate_pwl_parallel_dx.png")
print('saving to ', target)
plt.tight_layout()
plt.savefig(target)





# %% 
# * simply to my-wavelet and original/PWL

def plot_wave_dx(track):
    # take unprocessed track as input
    vel = track.get_speed()
    mywave = pwlstats.denoise_track(track)
    myvel = mywave.get_speed()
    sns.histplot(_DT * vel, stat="density", label="original", **shstyle)
    sns.histplot(_DT * myvel, stat="density", label="my-wavelet", **shstyle)

with mpl.rc_context({"font.size":16}):
    fig, ax= plt.subplots(figsize=(8,4))
    plot_wave_dx(original)
    ax.set_xlim(xlim)
    ax.legend()

# %% 
# TODO now rather than kde both sides, just obtain a smooth density for x < 0
# then consider exponential decays at x = 0 convolved with error distribution 

# zero_idx = np.searchsorted(kde.support, 0)

# mirror_data = np.concatenate([ng_data, -ng_data])
# ngkde = sm.nonparametric.KDEUnivariate(mirror_data)
# ng_support = kde.support[:zero_idx]

# ngkde.fit()
# ng_density = 2*ngkde.evaluate(ng_support)

# fig, ax = plt.subplots(figsize=(6,4))
# ax.plot(ng_support, ng_density, label=r"smooth $\Delta x$")
# ax.set_xlim(xlim)

# # * OR use exonential distribution instead of KDE?

# loc, scale = scipy.stats.expon.fit(-ng_data)
# print(loc, scale)
# space = np.linspace(0, -xlim[0], 1000)
# # exp_ng = (loc, scale)
# expon = scipy.stats.expon(loc, scale)
# ax.plot(-space, expon.pdf(space), label="fit exp", **fitstyle)
# ax.legend()

# target_support, target_density = kde.support, kde.density


# exp_part = expon.pdf(target_support[zero_idx:2*zero_idx])

# target_density[:zero_idx] = exp_part[::-1] 
# fig, ax = plt.subplots(figsize=(6,4))

# ax.plot(target_support, target_density)
# ax.set_xlim(xlim)

# target = (target_support, target_density)


# # %% 
# # fit left side of distribution only

# def convolve_fit_error(params, target):
#     target_support, target_density = target
#     loc = 0 
#     lam, sigma = params

#     def expon(x):
#         arr = lam * np.exp( -lam * (x-loc))
#         arr[x<loc] = 0
#         return arr
#     true = expon
        
#     def error(x):
#         return scipy.stats.norm(0, sigma).pdf(x)

#     in1 = true(target_support)

#     width = target_support[0]
#     err_support = target_support[:np.searchsorted(target_support, -width)]
#     in2 = error(err_support)

#     trial_density = scipy.signal.convolve(in1,in2,mode="same")
#     trial_density /= scipy.integrate.simpson(trial_density, target_support)

#     return trial_density

# def lsq_convolve_fit_error(params, target):
#     trial_density = convolve_fit_error(params, target)
#     target_support, target_density = target
#     N = zero_idx

#     # weighed least squares?
#     # weights = np.abs(kde.density)
#     weights = np.sqrt(np.abs(target_density))
#     weights = np.ones(kde.density.size)

#     weights = weights[:N]
#     squares = (target_density[:N] - trial_density[:N])**2

#     # print(weights)

#     value = np.sum( weights  * squares ) / np.sum(weights)
#     return value


# guess = [1/scale, solver.sigma]

# target = (kde.support, kde.density)
# _args =  (target,)

# result = scipy.optimize.minimize(lsq_convolve_fit_error, guess, args=_args, tol=1e-12, method="Nelder-Mead")

# print(guess)
# print(result.x.tolist())

# fig, ax = plt.subplots(figsize=(6,4))
# estimate_density = convolve_fit_error(result.x, target)

# ax.plot(*target)
# ax.plot(target[0], estimate_density)
# ax.set_xlim(xlim)



# %% 
# convolve exponential and normal distribiutions
normal = scipy.stats.norm(0, 0.012/2)
expon = scipy.stats.expon(0, 0.033)
space = np.linspace(-0.1, 0.2, 1000)

in1 = expon.pdf(space)
in1[in1<0] = 0
in2 = normal.pdf(space)

import scipy.signal

with mpl.rc_context(context):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(space, in1)
    ax.plot(space, in2)
    conv = scipy.signal.convolve(in1, in2, mode="same")

    conv = conv/scipy.integrate.simpson(conv, x=space)

    ax.plot(space, conv)
    ax.legend([r"$f = exp(-x)$", r"$g = exp(-x^2/\sigma^2)$", r"$f * g$"])

    ax.set_ylabel("density")
    ax.set_xlabel("x")


target = join(plot_target, "convolution_example.png")
print('saving to ', target)
plt.tight_layout()
plt.savefig(target)


# TODO: check if simulated velocity distribution is exponential
# TODO: assume an error distribution and a velocity distribution, solve the convolution for all the arameters
# TODO: improve by mixing in another distribution
# TODO: try to find two velocity modes following G. Wong ideas as well

# %% 

def renormalize(arr, support):
    print('norm', scipy.integrate.simpson(arr, x=support))
    return arr / scipy.integrate.simpson(arr, x=support)

def make_fast_mode(params, mix='normal'):
    loc, lam, sigma, A, mode_loc, mode_sigma = params

    if mix == "normal":
        def fast_mode(x):
            return A * scipy.stats.norm(mode_loc, mode_sigma).pdf(x)

    elif mix == "truncnorm":
        def fast_mode(x):
            loc, scale = mode_loc, mode_sigma
            clip_a = 0
            clip_b = kde.support[-1]
            a, b = (clip_a - loc) / scale, (clip_b - loc) / scale
            return A * scipy.stats.truncnorm(a, b, loc=loc, scale=scale).pdf(x)

    elif mix == "skewnorm":
        def fast_mode(x):
            def find_skew(a, r=0.01):
                skewcdf = scipy.stats.skewnorm(a, loc=loc, scale=scale).cdf(0) 
                return abs(skewcdf - r)
            result = scipy.optimize.minimize(find_skew, x0=0)
            a = result.x[0]
            print('skew', a)
            return A * scipy.stats.skewnorm(a, loc=loc, scale=scale).pdf(x)


    elif mix == "exponential":
        def fast_mode(x):
            lam = mode_sigma
            _x = x - mode_loc
            arr = A * lam * np.exp( -lam  * _x)
            arr[_x < 0] = 0
            return arr

    elif mix == "point":
        dw = np.diff(kde.support)[0]
        idx = np.searchsorted(kde.support, mode_loc)
        def fast_mode(x):
            arr = np.zeros(kde.support.size)
            # is this stable?
            arr[idx] = A/dw
            return arr

    return fast_mode


def convolve_with_kde(params, kde, mix_fast_mode=False):

    if mix_fast_mode:
        loc, lam, sigma, A, mode_loc, mode_sigma = params
    else:
        loc, lam, sigma = params

    def expon(x):
        arr = lam * np.exp( -lam * (x-loc))
        arr[x<loc] = 0
        return arr

    if mix_fast_mode:
        # mix in another guassian around 0.05
        fast_mode = make_fast_mode(params, mix)

        def true(x):
            return (expon(x) + fast_mode(x))/(1 + A)
    else:
        true = expon

    def error(x):
        return scipy.stats.norm(0, sigma).pdf(x)

    in1 = true(kde.support)

    width = kde.support[0]
    err_support = kde.support[:np.searchsorted(kde.support, -width)]
    in2 = error(err_support)

    trial_density = scipy.signal.convolve(in1,in2,mode="same")
    trial_density /= scipy.integrate.simpson(trial_density, kde.support)

    return trial_density

def lsq_convolve_with_kde(params, kde, mix_fast_mode, fit_error=False):
    trial_density = convolve_with_kde(params, kde, mix_fast_mode)

    if fit_error:
        N = np.searchsorted(kde.support, 0)
    else:
        N = None

    # weighed least squares?
    # weights = kde.density
    weights = np.sqrt(np.abs(kde.density))
    # weights = np.ones(kde.density.size)

    squares = (kde.density - trial_density)**2

    weights = weights[:N]
    squares = squares[:N]
    value = np.sum( weights  * squares ) / np.sum(weights)

    return value

def _lsq_convolve_with_kde(params, *args):
    params = [0, *params]
    return lsq_convolve_with_kde(params, *args)

data = discrete_dx

kde = sm.nonparametric.KDEUnivariate(data)
kde.fit()
kde.fit(bw=kde.bw/2)

mix_fast_mode = True

fit_error = False # fit x < 0 only

mix = "skewnorm"
mix = "normal"
mix = "truncnorm"
mix = "point"

# initial guess
loc, scale = scipy.stats.expon.fit(data[data>0])

bounds = None

if mix_fast_mode:
    if mix == "exponential":
        x0 = [0, 1/scale, solver.sigma, 0.1, 0.05, 1/scale]
    elif mix in ["normal", "skewnorm", "truncnorm"]:
        x0 = [0, 1/scale, solver.sigma, 0.1, 0.05, solver.sigma]
        print("exp loc, exp scale, err sigma, A, fast loc, fast scale")
        # bounds = [(None,None), (None,None), (0.004,0.025), (0,None), (0, None), (None, None)]
    elif mix == "point":
        x0 = [0, 1/scale, solver.sigma, 0.10, 0.05, np.nan]
        err_sigma_bound = (0.012, 0.013)
        bounds = [(None,None), (None,None), err_sigma_bound, (0,None), (0, None), (None, None)]
else:
    # x0 = [0, 1/scale, solver.sigma]
    x0 = [-0.008, 32, 0.8 * solver.sigma]

_args = (kde, mix_fast_mode, fit_error)

# test
# x0[3] = 0.3
print('x0', x0)
initial_density = convolve_with_kde(x0, kde, mix_fast_mode)
# 
# value = lsq_convolve_with_kde(x0, kde, mix_fast_mode)
# value

fig , ax = plt.subplots(figsize=(6,4))
ax.plot(kde.support, kde.density)
ax.plot(kde.support, initial_density)
ax.set_xlim(xlim)

# %%
solver.sigma 

# %%
# make more initial guesses

# _guess = [-0.008, 32, 0.8 * solver.sigma]
# print(_guess)

# initial_density = convolve_with_kde(_guess, kde, mix_fast_mode)

# fig , ax = plt.subplots(figsize=(6,4))
# ax.plot(kde.support, kde.density)
# ax.plot(kde.support, initial_density)
# ax.set_xlim(xlim)

# %%

# all vars
result = scipy.optimize.minimize(lsq_convolve_with_kde, x0, args=_args, bounds=bounds, tol=1e-12, method="Nelder-Mead")
result_x = result.x

print(result)

# fix loc
# result = scipy.optimize.minimize(_lsq_convolve_with_kde, x0[1:], args=_args, tol=1e-12, method="Nelder-Mead")
# result_x = [0, *result.x]

def pretty_float(lst):
    return [float("{0:0.4f}".format(i)) for i in lst]

print("initial guess", pretty_float(x0))
print("solved for ", pretty_float(result_x))


with mpl.rc_context(context):
    fig , ax = plt.subplots(figsize=(6,4))
    ax.plot(kde.support, kde.density)

    fit_density = convolve_with_kde(result_x, kde, mix_fast_mode)

    ax.plot(kde.support, fit_density)
    ax.set_xlim(xlim)
    ax.axvline(0, c='k', alpha=0.4)
    # ax.legend([r"smooth $\Delta x$", r"fit with $A\exp{(-x^2)} + B\exp{(x - \mu)^2/(2\sigma^2)}$"])
    ax.legend([r"smooth $\Delta x$", r"fit with exponential + normal"])

target = join(plot_target, "mix_guassian_model_fit.png")
print('saving to ', target)
plt.tight_layout()
plt.savefig(target)


# %%
# TODO: take a break from curve fitting and look for the peak at \Delta x ~ 0.045 in other trajectories
# TODO: also look for this peak our standard instantaneous velocity (wavelet transform) data



# %%
# check estimated fast mode distribution for x < 0

if mix_fast_mode:
    fast_mode = make_fast_mode(result_x, mix=mix)

    x = kde.support
    zero_idx = np.searchsorted(x, 0)
    cut_area = scipy.integrate.simpson(fast_mode(x)[:zero_idx], x=x[:zero_idx])
    print("cut area",  cut_area)

    plt.plot(kde.support, fast_mode(x))
 


# %% 

# %% 
# * LOAD
# * lets also look at the curve velocity distribution for the other data
select_idx = _fj.load_subset_idx()["top"]
look = [(idx, join(pwlstats.root, f"run/cluster/no_heuristic/top/_top_{idx:04d}"))  for idx in select_idx]
found = [(idx, directory) for idx, directory in look if os.path.exists(join(directory, "solver.pkl"))]
found_idx = [t[0] for t in found]
solverlist= [pwlstats.load_solver_at(t[1]) for t in found]
residual_list  = [solver.partition.get_signed_residuals() for solver in solverlist]
top_curve_coord = [solver.partition.get_curve_coord() for solver in solverlist]
top_curve_vel = [np.diff(coord) / _DT for coord in top_curve_coord]

# %% 
# TODO plot a few curve_velocity distributions for other trajectories
for i in range(5):
    _solver = solverlist[i]
    dx =  np.diff(_solver.partition.get_curve_coord())
    # ! set threshold to remove rare errors
    print("N = ", dx.size)
    
    def clean(dx, dx_threshold=1.0):
        print("cleanup {}/{}".format( np.sum(~(dx < dx_threshold)), dx.size ))
        return dx[dx < dx_threshold]
    data = clean(dx)
    
    fig, ax = plt.subplots(figsize=(6,4))
    # plot_dx_data(ax, data)
    plot_dx_with_gradient(ax, data)
    ax.set_title("Track {}".format(found_idx[i]))

# %% 
# TODO: and all these data together
data = clean( np.concatenate([np.diff(coord) for coord in top_curve_coord][:10]) )
fig, ax = plt.subplots(figsize=(6,4))
plot_dx_data(ax, data)
np.sum(data<0)/data.size

# %% 
# check the shapes of the unprocess instantaneous velocity and the wavelet velocity and the linearised wavelet



# %% 
# ?problem no 2
idx = 2
_curve = solverlist[idx].partition.get_curve_coord()
_data = np.diff(_curve)

_data.size, _data.min(), _data.max()
np.sort(_data)[-3]


# %% 
fjlocal = fjanalysis.load_summary()

# %% 
vstyle = dict(alpha=0.4, c='k', linestyle='--')

for i in range(10):
    idx = found_idx[i]
    print("idx", found_idx[i])
    print("lvel.mean", fjlocal[idx]["lvel"]["mean"])
    data = np.clip(top_curve_vel[i], -10, 10)
    print('data length', len(data))

    with mpl.rc_context(context):
        fig, ax= plt.subplots(figsize=(4,4))

        sns.histplot(data, stat="density", ax=ax, **shstyle)
        ax.axvline(0, **vstyle)
        ax.set_xlim(xlim)
        ax.set_xlabel(r"PWL solve $\Delta x^\parallel$ $(\mu m)$")
        ax.set_title("track index = {:04d}".format(idx))

        ax.set_xlim((-0.8,1.2))
        print()

        target = join(plot_target, "candidate_pwl_parallel_{:04d}.png".format(idx))
        print('saving to ', target)
        plt.tight_layout()
        plt.savefig(target)





# %% 
# * simply to my-wavelet and original/PWL

def plot_wave_dx(track):
    # take unprocessed track as input
    vel = track.get_speed()
    mywave = pwlstats.denoise_track(track)
    myvel = mywave.get_speed()
    sns.histplot(_DT * vel, stat="density", label="original", **shstyle)
    sns.histplot(_DT * myvel, stat="density", label="my-wavelet", **shstyle)

with mpl.rc_context({"font.size":16}):
    fig, ax= plt.subplots(figsize=(8,4))
    plot_wave_dx(original)
    ax.set_xlim(xlim)
    ax.legend()

# %% 
# TODO now rather than kde both sides, just obtain a smooth density for x < 0
# then consider exponential decays at x = 0 convolved with error distribution 

# zero_idx = np.searchsorted(kde.support, 0)

# mirror_data = np.concatenate([ng_data, -ng_data])
# ngkde = sm.nonparametric.KDEUnivariate(mirror_data)
# ng_support = kde.support[:zero_idx]

# ngkde.fit()
# ng_density = 2*ngkde.evaluate(ng_support)

# fig, ax = plt.subplots(figsize=(6,4))
# ax.plot(ng_support, ng_density, label=r"smooth $\Delta x$")
# ax.set_xlim(xlim)

# # * OR use exonential distribution instead of KDE?

# loc, scale = scipy.stats.expon.fit(-ng_data)
# print(loc, scale)
# space = np.linspace(0, -xlim[0], 1000)
# # exp_ng = (loc, scale)
# expon = scipy.stats.expon(loc, scale)
# ax.plot(-space, expon.pdf(space), label="fit exp", **fitstyle)
# ax.legend()

# target_support, target_density = kde.support, kde.density


# exp_part = expon.pdf(target_support[zero_idx:2*zero_idx])

# target_density[:zero_idx] = exp_part[::-1] 
# fig, ax = plt.subplots(figsize=(6,4))

# ax.plot(target_support, target_density)
# ax.set_xlim(xlim)

# target = (target_support, target_density)


# # %% 
# # fit left side of distribution only

# def convolve_fit_error(params, target):
#     target_support, target_density = target
#     loc = 0 
#     lam, sigma = params

#     def expon(x):
#         arr = lam * np.exp( -lam * (x-loc))
#         arr[x<loc] = 0
#         return arr
#     true = expon
        
#     def error(x):
#         return scipy.stats.norm(0, sigma).pdf(x)

#     in1 = true(target_support)

#     width = target_support[0]
#     err_support = target_support[:np.searchsorted(target_support, -width)]
#     in2 = error(err_support)

#     trial_density = scipy.signal.convolve(in1,in2,mode="same")
#     trial_density /= scipy.integrate.simpson(trial_density, target_support)

#     return trial_density

# def lsq_convolve_fit_error(params, target):
#     trial_density = convolve_fit_error(params, target)
#     target_support, target_density = target
#     N = zero_idx

#     # weighed least squares?
#     # weights = np.abs(kde.density)
#     weights = np.sqrt(np.abs(target_density))
#     weights = np.ones(kde.density.size)

#     weights = weights[:N]
#     squares = (target_density[:N] - trial_density[:N])**2

#     # print(weights)

#     value = np.sum( weights  * squares ) / np.sum(weights)
#     return value


# guess = [1/scale, solver.sigma]

# target = (kde.support, kde.density)
# _args =  (target,)

# result = scipy.optimize.minimize(lsq_convolve_fit_error, guess, args=_args, tol=1e-12, method="Nelder-Mead")

# print(guess)
# print(result.x.tolist())

# fig, ax = plt.subplots(figsize=(6,4))
# estimate_density = convolve_fit_error(result.x, target)

# ax.plot(*target)
# ax.plot(target[0], estimate_density)
# ax.set_xlim(xlim)



# %% 
# convolve exponential and normal distribiutions
normal = scipy.stats.norm(0, 0.012/2)
expon = scipy.stats.expon(0, 0.033)
space = np.linspace(-0.1, 0.2, 1000)
fig, ax = plt.subplots(figsize=(6,4))
in1 = expon.pdf(space)
in1[in1<0] = 0
in2 = normal.pdf(space)

ax.plot(space, in1)
ax.plot(space, in2)
import scipy.signal
conv = scipy.signal.convolve(in1, in2, mode="same")

conv = conv/scipy.integrate.simpson(conv, x=space)

ax.plot(space, conv)

# TODO: check if simulated velocity distribution is exponential
# TODO: assume an error distribution and a velocity distribution, solve the convolution for all the arameters
# TODO: improve by mixing in another distribution
# TODO: try to find two velocity modes following G. Wong ideas as well

# %% 

def renormalize(arr, support):
    print('norm', scipy.integrate.simpson(arr, x=support))
    return arr / scipy.integrate.simpson(arr, x=support)

def make_fast_mode(params, mix='normal'):
    loc, lam, sigma, A, mode_loc, mode_sigma = params

    if mix == "normal":
        def fast_mode(x):
            return A * scipy.stats.norm(mode_loc, mode_sigma).pdf(x)

    elif mix == "truncnorm":
        def fast_mode(x):
            loc, scale = mode_loc, mode_sigma
            clip_a = 0
            clip_b = kde.support[-1]
            a, b = (clip_a - loc) / scale, (clip_b - loc) / scale
            return A * scipy.stats.truncnorm(a, b, loc=loc, scale=scale).pdf(x)

    elif mix == "skewnorm":
        def fast_mode(x):
            def find_skew(a, r=0.01):
                skewcdf = scipy.stats.skewnorm(a, loc=loc, scale=scale).cdf(0) 
                return abs(skewcdf - r)
            result = scipy.optimize.minimize(find_skew, x0=0)
            a = result.x[0]
            print('skew', a)
            return A * scipy.stats.skewnorm(a, loc=loc, scale=scale).pdf(x)


    elif mix == "exponential":
        def fast_mode(x):
            lam = mode_sigma
            _x = x - mode_loc
            arr = A * lam * np.exp( -lam  * _x)
            arr[_x < 0] = 0
            return arr

    elif mix == "point":
        dw = np.diff(kde.support)[0]
        idx = np.searchsorted(kde.support, mode_loc)
        def fast_mode(x):
            arr = np.zeros(kde.support.size)
            # is this stable?
            arr[idx] = A/dw
            return arr

    return fast_mode


def convolve_with_kde(params, kde, mix_fast_mode=False):

    if mix_fast_mode:
        loc, lam, sigma, A, mode_loc, mode_sigma = params
    else:
        loc, lam, sigma = params

    def expon(x):
        arr = lam * np.exp( -lam * (x-loc))
        arr[x<loc] = 0
        return arr

    if mix_fast_mode:
        # mix in another guassian around 0.05
        fast_mode = make_fast_mode(params, mix)

        def true(x):
            return (expon(x) + fast_mode(x))/(1 + A)
    else:
        true = expon

    def error(x):
        return scipy.stats.norm(0, sigma).pdf(x)

    in1 = true(kde.support)

    width = kde.support[0]
    err_support = kde.support[:np.searchsorted(kde.support, -width)]
    in2 = error(err_support)

    trial_density = scipy.signal.convolve(in1,in2,mode="same")
    trial_density /= scipy.integrate.simpson(trial_density, kde.support)

    return trial_density

def lsq_convolve_with_kde(params, kde, mix_fast_mode, fit_error=False):
    trial_density = convolve_with_kde(params, kde, mix_fast_mode)

    if fit_error:
        N = np.searchsorted(kde.support, 0)
    else:
        N = None

    # weighed least squares?
    # weights = kde.density
    weights = np.sqrt(np.abs(kde.density))
    # weights = np.ones(kde.density.size)

    squares = (kde.density - trial_density)**2

    weights = weights[:N]
    squares = squares[:N]
    value = np.sum( weights  * squares ) / np.sum(weights)

    return value

def _lsq_convolve_with_kde(params, *args):
    params = [0, *params]
    return lsq_convolve_with_kde(params, *args)

data = discrete_dx

kde = sm.nonparametric.KDEUnivariate(data)
kde.fit()
kde.fit(bw=kde.bw/2)

mix_fast_mode = True

fit_error = False # fit x < 0 only

mix = "skewnorm"
mix = "normal"
mix = "truncnorm"
mix = "point"

# initial guess
loc, scale = scipy.stats.expon.fit(data[data>0])

bounds = None

if mix_fast_mode:
    if mix == "exponential":
        x0 = [0, 1/scale, solver.sigma, 0.1, 0.05, 1/scale]
    elif mix in ["normal", "skewnorm", "truncnorm"]:
        x0 = [0, 1/scale, solver.sigma, 0.1, 0.05, solver.sigma]
        print("exp loc, exp scale, err sigma, A, fast loc, fast scale")
        # bounds = [(None,None), (None,None), (0.004,0.025), (0,None), (0, None), (None, None)]
    elif mix == "point":
        x0 = [0, 1/scale, solver.sigma, 0.10, 0.05, np.nan]
        err_sigma_bound = (0.012, 0.013)
        bounds = [(None,None), (None,None), err_sigma_bound, (0,None), (0, None), (None, None)]
else:
    # x0 = [0, 1/scale, solver.sigma]
    x0 = [-0.008, 32, 0.8 * solver.sigma]

_args = (kde, mix_fast_mode, fit_error)

# test
# x0[3] = 0.3
print('x0', x0)
initial_density = convolve_with_kde(x0, kde, mix_fast_mode)
# 
# value = lsq_convolve_with_kde(x0, kde, mix_fast_mode)
# value

fig , ax = plt.subplots(figsize=(6,4))
ax.plot(kde.support, kde.density)
ax.plot(kde.support, initial_density)
ax.set_xlim(xlim)

# %%
solver.sigma 

# %%
# make more initial guesses

# _guess = [-0.008, 32, 0.8 * solver.sigma]
# print(_guess)

# initial_density = convolve_with_kde(_guess, kde, mix_fast_mode)

# fig , ax = plt.subplots(figsize=(6,4))
# ax.plot(kde.support, kde.density)
# ax.plot(kde.support, initial_density)
# ax.set_xlim(xlim)

# %%

# all vars
result = scipy.optimize.minimize(lsq_convolve_with_kde, x0, args=_args, bounds=bounds, tol=1e-12, method="Nelder-Mead")
result_x = result.x

print(result)

# fix loc
# result = scipy.optimize.minimize(_lsq_convolve_with_kde, x0[1:], args=_args, tol=1e-12, method="Nelder-Mead")
# result_x = [0, *result.x]

def pretty_float(lst):
    return [float("{0:0.4f}".format(i)) for i in lst]

print("initial guess", pretty_float(x0))
print("solved for ", pretty_float(result_x))


fig , ax = plt.subplots(figsize=(6,4))
ax.plot(kde.support, kde.density)

fit_density = convolve_with_kde(result_x, kde, mix_fast_mode)

ax.plot(kde.support, fit_density)
ax.set_xlim(xlim)
ax.axvline(0, c='k', alpha=0.4)


# %%
# TODO: take a break from curve fitting and look for the peak at \Delta x ~ 0.045 in other trajectories
# TODO: also look for this peak our standard instantaneous velocity (wavelet transform) data


# %%
# check estimated fast mode distribution for x < 0

if mix_fast_mode:
    fast_mode = make_fast_mode(result_x, mix=mix)

    x = kde.support
    zero_idx = np.searchsorted(x, 0)
    cut_area = scipy.integrate.simpson(fast_mode(x)[:zero_idx], x=x[:zero_idx])
    print("cut area",  cut_area)

    plt.plot(kde.support, fast_mode(x))
 


# %% 

# %% 
# * LOAD
# * lets also look at the curve velocity distribution for the other data
select_idx = _fj.load_subset_idx()["top"]
look = [(idx, join(pwlstats.root, f"run/cluster/no_heuristic/top/_top_{idx:04d}"))  for idx in select_idx]
found = [(idx, directory) for idx, directory in look if os.path.exists(join(directory, "solver.pkl"))]
found_idx = [t[0] for t in found]
solverlist= [pwlstats.load_solver_at(t[1]) for t in found]
residual_list  = [solver.partition.get_signed_residuals() for solver in solverlist]
top_curve_coord = [solver.partition.get_curve_coord() for solver in solverlist]
top_curve_vel = [np.diff(coord) / _DT for coord in top_curve_coord]

# %% 
# TODO plot a few curve_velocity distributions for other trajectories
for i in range(1):
    _solver = solverlist[i]
    dx =  np.diff(_solver.partition.get_curve_coord())
    # ! set threshold to remove rare errors
    print("N = ", dx.size)
    
    def clean(dx, dx_threshold=1.0):
        print("cleanup {}/{}".format( np.sum(~(dx < dx_threshold)), dx.size ))
        return dx[dx < dx_threshold]
    data = clean(dx)
    
    with mpl.rc_context(context):
        fig, ax = plt.subplots(figsize=(6,4))
        # plot_dx_data(ax, data)
        plot_dx_with_gradient(ax, data)
        ax.set_title("Track {}".format(found_idx[i]))

    # ax.legend(["binned", "smoothed", "derivative"])
    ax.legend()
    
    target = join(plot_target, "peak_finding_with_gradient.png")
    print('saving to ', target)
    plt.tight_layout()
    plt.savefig(target)



# %% 
# TODO: and all these data together
data = clean( np.concatenate([np.diff(coord) for coord in top_curve_coord][:10]) )
fig, ax = plt.subplots(figsize=(6,4))
plot_dx_data(ax, data)
np.sum(data<0)/data.size

# %% 
# check the shapes of the unprocess instantaneous velocity and the wavelet velocity and the linearised wavelet



# %% 
# ?problem no 2
idx = 2
_curve = solverlist[idx].partition.get_curve_coord()
_data = np.diff(_curve)

_data.size, _data.min(), _data.max()
np.sort(_data)[-3]


# %% 
fjlocal = fjanalysis.load_summary()

# %% 
for i in range(10):
    idx = found_idx[i]
    print("idx", found_idx[i])
    print("lvel.mean", fjlocal[idx]["lvel"]["mean"])
    data = np.clip(top_curve_vel[i], -10, 10)
    print('data length', len(data))
    fig, ax= plt.subplots(figsize=(4,4))

    sns.histplot(data)

# %% 
# ok fit all of these distributions
# start by fitting v < 0 to exponential distribution



# %% 
# * compare to fanjin linearised velocity
original = _fj.trackload_original([2924])[0]
candidate = _fj.trackload([2924])[0]
lincandidate = _fj.lintrackload([2924])[0]

lintop = _fj.load_subsets()["top"]
toplvel = np.concatenate([ltr.get_step_speed() for ltr in lintop])

fig, ax = plt.subplots(figsize=(4,4))
speed =  original.get_speed()
sns.histplot(speed, ax=ax, log_scale=False)
ax.set_xlim((-0.1,0.8))
ax.set_xlabel(r"$\mu m/s$")
ax.set_title("raw 0.1s velocity")
# * 

speed.min(), speed.max()
# np.sum(np.isinf(np.log(speed)))

# %%



fig, ax = plt.subplots(figsize=(4,4))
sns.histplot(candidate.get_speed(),ax=ax)
ax.set_xlim((-0.1,0.8))
ax.set_xlabel(r"$\mu m/s$")
ax.set_title("wavelet 0.1s velocity")

fig, ax = plt.subplots(figsize=(4,4))
sns.histplot(lincandidate.get_step_speed(),ax=ax)
ax.set_xlim((-0.1,0.8))
ax.set_xlabel(r"$\mu m/s$")
ax.set_title("linearised velocity")

fig, ax = plt.subplots(figsize=(4,4))
# sns.histplot(candidate.get_speed())
sns.histplot(toplvel,ax=ax)
ax.set_xlim((-0.1,0.8))
ax.set_xlabel(r"$\mu m/s$")
ax.set_title("TOP dataset linearised velocity")

# %% 

# %% 

# * raw velocity distribution for more 
top_idx = _fj.load_subset_idx()["top"]
for idx in top_idx[:5]:
    original = _fj.trackload_original([idx])[0]
    fig, ax = plt.subplots(figsize=(4,4))
    plot_wave_dx(original)

    ax.set_xlabel(r"$\mu m$")
    ax.legend()
    ax.set_title("Track {}".format(idx))
    ax.set_xlim((-0.01,0.08))




# %% 
# * here we estimate the density of the mapped coordinates
# this is the 1d equicalent of scanning the curve with a guassian filter
bw_factor = solver.sigma/curve_coord.std()

kde = scipy.stats.gaussian_kde(curve_coord, bw_method=bw_factor)
x = curve_coord[:N]
basis = np.linspace(x.min(), x.max(), 1000)
density = kde.evaluate(basis)
with mpl.rc_context(mplstyle):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(basis, density)

    steps = solver.partition.model.get_step_length()
    for step in np.cumsum(steps)[:time_index]:
        ax.axvline(step, alpha=0.4, linestyle='--')

    ax.set_xlabel(r"curve coordinate $\mu m$")
    ax.set_ylabel(r"curve coordinate density")

    candidate_threshold = 0.06
    top_threshold = 0.1
    threshold = top_threshold
    ax.axhline(threshold, c='k', alpha=0.4, linestyle='--')
    # can todo proceed by peak finding ...

# %% 
# * we need more robust ways to analyse our segements
# * counting points within r of segment
# we need to relax the idea that data should be mapped to one segment in particular
partition = solver.partition
cts = partition.count_segment_data(r)
lengths = partition.model.get_step_length()
ctdensity = cts/lengths 
ctvelocity = lengths/(_DT * cts)

# sns.histplot(ctdensity)
# sns.histplot(cts)
with mpl.rc_context({'font.size': 12}):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(ctvelocity, ax=ax)
    ax.set_xlabel(r"segment velocity $\mu m/s$", fontsize=16)

if publish:
    pub.save_figure("example_segment_velocity_track_0040",notename)
    # pub.save_figure("example_segment_velocity_candidate",notename)


# %% 
# try simple peak detection using thresholds
def peak_detect(data, threshold=0):
    # use the threshold to split the data
    diff = np.diff(data > threshold)
    ind = np.argwhere(diff == 1).ravel()+1
    edges = np.concatenate([[0], ind])
    
    offset = 0 if data[0] > threshold else 1
    split = np.split(data, ind)
    high = split[offset::2]
    chunk = edges[offset::2]
    print(data[chunk[0]])
    print(high[0])
    peak_index = [chunk_index + np.argmax(h) for chunk_index, h in zip(chunk, high)]

    offset = 1 if data[0] > threshold else 0
    low = split[offset::2]
    chunk = edges[offset::2]
    trough_index = [chunk_index + np.argmin(h) for chunk_index, h in zip(chunk, low)]

    return np.array(peak_index), np.array(trough_index)

# %%
# * try to identify the low velocity dwells
cv_threshold =  0.1
peak_index, trough_index = peak_detect(gauss_curve_velocity, threshold=cv_threshold)
trough_index[:time_index]

# plot curve velocity with curve coordinate as the dependent variable

with mpl.rc_context(mplstyle):
    fig, ax = plt.subplots(figsize=(12,5))
    x = curve_coord
    # print(x.tolist())
    ax.axhline(0, c='k', alpha=0.2)
    ax.axhline(cv_threshold, c='k', alpha=0.4,  linestyle='--')
    # ax.plot(x, curve_velocity[:N], alpha=0.6, label="no filter")
    ydata = gauss_curve_velocity
    ax.plot(x[:N], ydata[:N], alpha=0.6, label="guassian_filter")
    ax.set_xlabel(r"curve coordinate $\mu m$")
    ax.set_ylabel(r"curve velocity $\mu m/s$")

    r = solver.r
    # ruler = mpl.patches.Rectangle((0,0.06), 2*r, 0.002)
    # ax.add_patch(ruler)
    ax.plot([0,2*r], [0.5,0.5], c='b', linewidth=4)

    ax.annotate(r'$2r$', (0,0.51), xycoords='data', fontsize=12)
    

    steps = solver.partition.model.get_step_length()
    for step in np.cumsum(steps)[:time_index]:
        ax.axvline(step, alpha=0.4, linestyle='--')
        
    for index in trough_index:
        if index < N:
            ax.scatter(x[index], ydata[index], marker='D', c='green', s=20)


if publish:
    pub.save_figure("curve_velocity_vs_curve_coord_example", notename)

# %% 
def dwell_is_breakpt(trough_coord, model, r):
    breakcoord = model.get_break_coord()
    bcdistance = np.empty(len(trough_coord))

    for i, coord in enumerate(trough_coord):
        m_index = np.searchsorted(breakcoord, coord)
        adjacent = [min(_index, model.M-1) for _index in [m_index, m_index+1]]
        bcd = min([abs(breakcoord[_index] - coord) for _index in adjacent])
        bcdistance[i] = bcd
        
    is_close = bcdistance < r
    return is_close
        

trough_coord = curve_coord[trough_index]
is_close = dwell_is_breakpt(trough_coord, solver.partition.model, r=solver.r)

ratio = np.sum(is_close)/is_close.size
print('at beakpoint {:d}/{:d} ({:.3f})'.format(np.sum(is_close), is_close.size, ratio))

# and what is the statistical significance of this result?
# we can obtain the proportion from uniformly sampling the curve
# do we still call this bootstrapping?
N_bootstrap = 5000
N_sample = is_close.size
def bootstrap_dwell_is_breakpt(model, r, N_sample, N=N_bootstrap):
    breakcoord = model.get_break_coord()
    def sample():
        ru = np.random.uniform(breakcoord[0], breakcoord[-1], size=N_sample)
        is_close = dwell_is_breakpt(ru, model, r)
        return np.sum(is_close)/N_sample
    return [sample() for _ in range(N_bootstrap)]
    # print('bootstrap close to breakpoint {:d}/{:d}'.format(np.sum(is_close),N))

# %%
# plot the bootstrap distribution
bt_distrib = bootstrap_dwell_is_breakpt(solver.partition.model, solver.r, N_sample)

# %%
with mpl.rc_context(mplstyle):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(bt_distrib, ax=ax)
    ax.bar(ratio, 1, width= 0.01)
    ax.annotate('{:.3f}'.format(ratio), (ratio,0), xytext=(0.7,0.5), textcoords='axes fraction', arrowprops=dict(facecolor='black', width=2, shrink=0.05)) 
    # bootstrap_ratio = bootstrap_dwell_is_breakpt(solver.partition.model, solver.r, N_sample)
    ax.set_xlabel("ratio")
    ax.set_title("sample distribution")

print('ratio', np.mean(bt_distrib), 'std', np.std(bt_distrib))
if publish:
    pub.save_figure("candidate_bootstrap_dwell_on_breakpoint", notename)

# %%
# * instead of looking for "dwells", look for low/high velocity regions of the trajectory
# we will implement a sliding window count of the data

def sliding_window_count(model, data, sampling_dx, r):
    shape = mdl.LPshape.from_lptr(model)
    point_data = data.get_n2()

    lx, ly = 0, shape.get_contour_length()
    sampling = np.arange(lx, ly, sampling_dx)

    def count_at(s, r):
        m_pt = shape(s)[np.newaxis]
        i = shape.get_index(s)
        s1, s2 = shape.get_cumulative_length()[i], shape.get_cumulative_length()[i+1]

        adjacent = []
        if i > 0 and (s - s1) < r:
            adjacent.append(i-1)
        adjacent.append(i)
        if i < model.M-2 and (s2 - s) < r:
            adjacent.append(i+1)
        count = 0
        for _i in adjacent:
            si, sf = _i, _i+1
            p = point_data[time[si]:time[sf]]
            count += int(np.sum(norm(p - m_pt, axis=1) < r))
        return count
  
    cts = [count_at(s, r) for s in sampling]
    return sampling, np.array(cts)


partition = solver.partition
sliding_support, sliding_cts = sliding_window_count(partition.model, partition.data, solver.r/10, solver.r)

# inst velocity can't be defined this way
# inst_velocity = 2*solver.r/(_DT * sliding_cts)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(sliding_support, sliding_cts)


ct_threshold = 8
threshold_velocity = 2*solver.r/(_DT * ct_threshold)
print('threshold \"velocity\"', threshold_velocity)

ax.axhline(ct_threshold, c='k', alpha=0.4, linestyle='--')

# %%

# plot threshold information directly on the trajectory

def arg_split_data(data, threshold):
    high_idx = data > threshold
    ind = np.argwhere(np.diff(high_idx) == 1).ravel()+1
    if ind[0] != 0:
        ind = np.insert(ind, 0, 0)
    if ind[-1] != data.size-1:
        ind = np.append(ind, data.size-1)
    return ind

def arg_split_hchunk(data, threshold):
    ind = arg_split_data(data, threshold)
    offset = 0 if data[0] > threshold else 1
    hchunk = []
    for i in range(offset, ind.size, 2):
        if i+1 < ind.size:
            hchunk.append((ind[i], ind[i+1]-1))
    return hchunk

ind = arg_split_data(sliding_cts, ct_threshold)
# hchunk = arg_split_hchunk(sliding_cts, ct_threshold)

_N = N
short = solver.partition.model.cut(0,_N)
data = solver.partition.data.cut_index(0,_N)

def plot_annotated_line(short, data, sliding_support, split_index, r):
    FAST = 1
    SLOW = 0

    curve_max = np.sum(short.get_step_length())
    fig, ax = plt.subplots(figsize=(12, 12))

    shape = mdl.LPshape.from_lptr(short)
    print('M', shape.M)
    # breakcoord = shape.get_break_coord()

    defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.3, 'color':defcolors[2]}

    ax.plot(data.x, data.y, label='data', **ptlkw)
    plt.draw()


    travel_model_index  = 0
    lpts = [shape.get_point(0)]
    mode = FAST
    modes = []
    breakpts = shape.get_n2()
    for i in range(split_index.size-1):
        f_index, t_index = split_index[i], split_index[i+1]
        lf, lt = sliding_support[f_index], sliding_support[t_index]
        mf_index, mt_index = shape.get_index(lf), shape.get_index(lt)
        if mt_index > travel_model_index:
            mpts = breakpts[mf_index+1:mt_index+1]
            lpts.extend(mpts)
            lpts.append(shape(lt))
            modes.extend([mode for _ in range(len(mpts)+1)])
        else:
            # this chunk of the trajectory does  not pass  over a breakpoint
            lpts.append(shape(lt))
            modes.append(mode)
        mode = SLOW if mode == FAST else FAST

    for i in range(len(lpts)-1):
        mode = modes[i]
        color = defcolors[0] if mode == SLOW else defcolors[1]
        pta, ptb = lpts[i], lpts[i+1]
        _x, _y = zip(pta, ptb)

        ax.plot(_x, _y, color=color)
        if mode == SLOW:
            lw = support.scale_lw(ax, r)
            slowstyle = {"alpha":0.2, "linewidth":lw, "solid_capstyle":'round'}
            ax.plot(_x, _y, color=color, **slowstyle)
    ax.set_aspect("equal")

plot_annotated_line(short, data, sliding_support, ind, solver.r)

# pub.save_figure("top_annotated_modes_example", notename)



# %%

# we want to plot this information directly on trajectory
_N = N
# _N = 220
_time_index = np.searchsorted(time, _N, side='right')
short = solver.partition.model.cut(0,_N)
data = solver.partition.data.cut_index(0,_N)
fig, ax = plt.subplots(figsize=(12,12))
pwlpartition.simple_model_plot(ax, short, data)
shape = mdl.LPshape.from_lptr(short)
for i, index in enumerate(trough_index[:_time_index]):
    coord = curve_coord[index]
    xy = shape(coord)
    circstyle = {"linewidth": 2, "alpha": 0.7}
    color = 'k' if is_close[i] else 'red'
    patch = mpl.patches.Circle(xy, radius=1.1*solver.r, fill=False, color=color, **circstyle)
    ax.add_patch(patch)

# now, are the dwells we identified at the breakpoints or not?
if publish:
    pub.save_figure("candidate_annotated_dwell_example", notename)
# pub.save_figure("top_annotated_dwell_example", notename)

# %% 
# now identify periods of activity/inactivity?
# try standard step detection ideas

filter_len = 10

data = curve_velocity - np.average( curve_velocity ) 
steps = np.hstack([-1 * np.ones(filter_len), np.ones(filter_len)])
data.shape, steps.shape
convolution = np.convolve(data, steps, mode='valid')
gauss_step_convolve = gaussian_filter1d(convolution, sigma=1.5)


peak_index, trough_index = peak_detect(gauss_step_convolve, threshold=0)
# peak_index = trough_index + filter_len - 1

peak_max_index = np.searchsorted(peak_index, N, side='right')

fig, axes = plt.subplots(2, 1, figsize=(12,10))
ax = axes[0]
plot_curve_coord(ax, curve_coord, time)
for index in peak_index[:peak_max_index]:
    ax.scatter(index, curve_coord[index], color='orange', marker='D', s=40)

ax = axes[1]
ax.plot(convolution[:N])
ax.plot(gauss_step_convolve[:N])
for index in peak_index[:peak_max_index]:
    ax.scatter(index, gauss_step_convolve[index], color='orange', marker='D', s=40)
for index in trough_index[:peak_max_index]:
    ax.scatter(index, gauss_step_convolve[index], color='green', marker='D', s=40)

ax.axhline(0, c='k', alpha=0.2)

ax.set_xlabel('index')
ax.set_ylabel('convolution with step')



# %% 
