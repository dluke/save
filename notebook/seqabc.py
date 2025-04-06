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
# approximate bayesian computation

# %% 
import sys, os
import copy
join = lambda *x: os.path.abspath(os.path.join(*x))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import scipy.stats

import pili
import parameters
import _fj
import fjanalysis
import twanalyse
import rtw
import sobol
import abcimplement
from abcimplement import mirror, mirrorpts

# %%

# %% 
verbose = False
style = {}

# %% 
notedir = os.getcwd()
root = pili.root
# candidate to compare against
# simdir = join(root, "../run/5bfc8b9/cluster/mc4d")
simdir = join(root, "../run/825bd8f/cluster/mc4d")


# %% 
# load fanjin data
all_idx, ltrs = _fj.slicehelper.load_linearized_trs("all")
reference_idx = _fj.load_subset_idx()
subsets = list(reference_idx.keys())

# %% 
mc4d = {}
mc4d["simdir"] = simdir
objectives = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
mc4d["objectives"] = objectives
abcimplement.load_problem_simulation(mc4d)
lookup = mc4d["lookup"]
problem = mc4d["problem"]
lduid = mc4d["lduid"]

# %% 
# velocity similarity scores are summary statistics that are 
# already computed against the reference data
# We need to handle them differently
ks_scores = ['fanjin.%s.ks_statistic' % subset for subset in reference_idx.keys()]

print("objectives", objectives)
_data =  sobol.collect_obs(lookup, lduid, objectives, objectives)
missing = sobol.check_missing(lookup, _data)
print(missing)

# %% 
refdf = fjanalysis.compute_reference_data(ltrs, reference_idx, objectives)

# %% 
# construct data frame with parameters and objectives
params = mc4d["data"].paramsdf(objectives)
params
# %% 
# similar for ks_statistic
_parlist = list(zip(problem["names"],  zip(*[lookup[1][_u] for _u in lookup[0]])))
_col = {k:v for k, v in _parlist}
_data =  sobol.collect_obs(lookup, lduid, ks_scores, ks_scores)
_col.update({name:_data[name] for name in ks_scores})
ksparams = pd.DataFrame(_col)

for score_subset in ks_scores:
    print("{} min = {:.4f}".format(score_subset, ksparams[score_subset].min()))

# %% 
# plot the sampling distribution (uniform random)
# any projection will do
if verbose:
    plt.rcParams.update({'text.usetex': False})
    fig, ax = plt.subplots(figsize=(5,5))
    sns.scatterplot(params["pilivar"], params["anchor_angle_smoothing_fraction"], 
        hue=params["ahat.estimate"], ax=ax)

# %% 
# Implement rejection ABC
# we can construct an approximate posterior distribution for 
# 1. simulated reference
# 2. fanjin data subsets
from abcimplement import rejection_abc
parnames = problem["names"]
bounds = problem["bounds"]

# select the "top" subset as the preferred reference data
subset = "top"
reference = refdf.iloc[1]

# %% 
# construct a dictionary of dataframes, one for each summary statistic we will use
accept  = {}
N = 200
for objective in objectives:
    statdf = params[parnames+[objective]]
    accepted = rejection_abc(statdf, [objective], reference, N)
    # max_score = accepted.iloc[N-1]["score"]
    # print("max score is", max_score)
    accept[objective] = accepted

# add the lvel similarity stat
top_ks = "fanjin.top.ks_statistic"
ks_statdf = ksparams[parnames+[top_ks]]
accept[top_ks] = rejection_abc(ks_statdf, [top_ks], reference, N)


# %% 
# show the rejection_abc results in all pairs of dimensions
n = len(problem["names"])
plt.rcParams.update({'text.usetex': False})
sns.set(rc={'figure.figsize':(20,20)})
print("plotting pairplots for ", list(accept.keys()))
for objective, accepted in accept.items():
    # print(objective)
    is_lvel = objective.startswith("fanjin")
    if is_lvel:
        data  = accepted[parnames+[objective]] 
        hue = objective
    else:
        data  = accepted[parnames+["score"]] 
        hue = "score" 
    g = sns.pairplot(data, hue=hue, **style)
    g.fig.suptitle(objective, y=1.08) # y= some height>1
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            _xlim = bounds[j]
            _ylim = bounds[i]
            g.axes[i,j].set_xlim(_xlim)
            g.axes[i,j].set_ylim(_ylim)


# %% 
# pull estimates for anchor parameter for "top" from ks_statistic and activity
projection = ["anchor_angle_smoothing_fraction", "pilivar"]
p1, p2 = projection
xlim = (0,1)
_p1distrib = accept["ahat.estimate"][p1]
fig = plt.figure(figsize=(5,5))
ax= sns.histplot(_p1distrib, binrange=xlim)
ax.axvline(0.0625)
print('anchor estimate 1', np.mean(_p1distrib))

_p1distrib = accept["fanjin.top.ks_statistic"][p1]
fig = plt.figure(figsize=(5,5))
ax= sns.histplot(_p1distrib, binrange=xlim)
ax.axvline(0.0625)
print('anchor estimate 2', np.mean(_p1distrib))

# %% 
# based on sobol, project this dataset onto pilivar/anchor and do 
# rejection ABC to obtain posterior distribution
N = 200
objective = "ahat.estimate"
projection = ["anchor_angle_smoothing_fraction", "pilivar"]
# params[projection+"ahat.estimate"]

statdf = params[parnames+[objective]]
postactive = rejection_abc(statdf, ["ahat.estimate"], reference, N)
# ahat_accepted = postactive[projection+["score"]]
# %% 
# plot the rejection_abc for activity statistic for each subset of the data
def plot_activity(par_accepted, projection, problem):
    parnames = problem["names"]
    fig, axes = plt.subplots(1,2,figsize=(10,5))
    ax1, ax2 = axes
    x, y = projection
    ix, iy = [parnames.index(name) for name in projection]
    sns.set(rc={'figure.figsize':(5,5)})
    ax1 = sns.scatterplot(data=par_accepted, x=x, y=y, hue="score", ax=ax1)
    ax1.set_xlim(bounds[ix])
    ax1.set_ylim(bounds[iy])
    #
    X, Y = par_accepted[x], par_accepted[y]
    ax2 = sns.kdeplot(X,Y, fill=True, ax=ax2)
    ax2.set_xlim(bounds[ix])
    ax2.set_ylim(bounds[iy])
    return fig, axes
    
N = 200
for i, subset in enumerate(subsets):
    _reference = refdf.iloc[i]
    statdf = params[parnames+[objective]]
    post = rejection_abc(statdf, ["ahat.estimate"], _reference, N)
    ahat_accepted = post[projection+["score"]]
    fig, axes = plot_activity(ahat_accepted, projection, problem)
    fig.suptitle(subset)


# %% [markdown]
# The shape of the kernel is not that important but the bandwidth is very important.
# 

# %% 
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# TODO move kernel study to a different note book ... (?)

# # plot the 1d projection of the distribution onto anchor parameter
# # so that we can test sklearn kernel selection
# anchor_parameter = projection[0]
# activitydf = accept["ahat.estimate"]
# activity = activitydf[anchor_parameter].to_numpy()
# sns.histplot(activity, binrange=(0,1.0))

# %% 
# statsmodels is easier to use than sklearn and gives similar results
use_sklearn = False 
if use_sklearn:
    # https://scikit-learn.org/stable/modules/density.html
    # https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    x = activity
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    # https://scikit-learn.org/stable/modules/grid_search.html#grid-search
    import warnings
    # warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.filterwarnings("ignore")
    # some global structure to allow use to both set bandwidths and let them be estimated
    bw_space_res = 40
    bw_settings = {
        "pilivar": {
            "estimate": True,
            "set": 5.0,
            "geom": np.geomspace(0.1,10.0,bw_space_res)
            },
        "anchor_angle_smoothing_fraction": {
            "estimate": True,
            "set": 0.01,
            "geom":np.geomspace(0.001,0.5,bw_space_res)
            },
        }
    # allow the notebook to vary this
    mod_bw_settings = copy.deepcopy(bw_settings)

def bandwidth_xvalidate(x, bw_space=None, bounds=None, mirror_boundaries=True):
    if mirror_boundaries:
        _x = mirror(x, bounds)
    else:
        _x = x
    if bw_space is None:
        bw_space = np.linspace(np.std(x)/2, (np.max(x)-np.min(x))/2, num=20)
    print("searching bw in range ", bw_space[0], bw_space[-1])
    grid = GridSearchCV(KernelDensity(kernel="epanechnikov"),
                        {'bandwidth': bw_space},
                        cv=20) # 20-fold cross-validation
    grid.fit(_x[:, None])
    return grid

def kernel_pdf(par_accepted, xlim, bw_space=None, estimate_bandwidth=True, mirror_boundaries=True):
    x_grid = np.linspace(xlim[0], xlim[1], 200)
    # 
    if estimate_bandwidth:
        grid = bandwidth_xvalidate(par_accepted, bw_space=bw_space, bounds=xlim, 
            mirror_boundaries=mirror_boundaries)
        print("best", grid.best_params_['bandwidth'])
        kde = grid.best_estimator_
    #
    else:
        if mirror_boundaries:
            par_accepted = mirror(par_accepted, xlim)
        bw = bw_space
        print("set bandwidth", bw)
        kde = KernelDensity(kernel='epanechnikov', bandwidth=bw)
        kde.fit(par_accepted[:, None])
    #
    pdf = np.exp(kde.score_samples(x_grid[:, None]))
    if mirror_boundaries:
        pdf *= 3
    return x_grid, pdf, kde.bandwidth

if use_sklearn:
    estimate_bandwidth = True
    mirror_boundaries = True
    for par in projection:
        print("projection axis ", par)
        ix = parnames.index(par)
        par_accepted = accept["ahat.estimate"][par].to_numpy()
        xlim = problem["bounds"][ix]
        bw_setting = bw_settings[par]
        estimate = bw_setting["estimate"]
        bw_space = bw_setting["geom"] if estimate else bw_setting["set"]
        x_grid, pdf, bandwidth = kernel_pdf(par_accepted.copy(), xlim, bw_space=bw_space,
            estimate_bandwidth=bw_setting["estimate"], mirror_boundaries=False)
        x_grid, m_pdf, m_bandwidth = kernel_pdf(par_accepted.copy(), xlim, bw_space=bw_space,
            estimate_bandwidth=bw_setting["estimate"], mirror_boundaries=True)
        mod_bw_settings[par]["set"] = m_bandwidth
        
        fig, ax = plt.subplots()
        ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.4f' % bandwidth)
        ax.plot(x_grid, m_pdf, linewidth=3, alpha=0.5, label='mirror bw=%.4f' % m_bandwidth)
        ax.set_ylim(bottom=0.0)
        ax.legend(loc='lower right')
        ax.set_title(par)
        plt.show()

# %% 
# tryout statsmodels for multivariate KDE
import statsmodels.api as sm
ix, iy = [parnames.index(name) for name in projection]
acc = accept["ahat.estimate"]
data = [acc[par].to_numpy() for par in projection]
bw_estimate = sm.nonparametric.KDEMultivariate(data=data, var_type='cc', 
    bw='cv_ls')
    
# mirror the data in the yaxis at the top and bottom
m_data = mirrorpts(data, bounds[iy])
dens = sm.nonparametric.KDEMultivariate(data=m_data, var_type='cc', 
    bw='normal_reference')
# use bandwidth estimate from un-mirrored data
dens.bw = bw_estimate.bw
print("statsmodels least squares cross validation bandwidth estimate")
print("bw", dens.bw)

# %% 
# setup control parameters and grid 
N = 200
res = 100
ss = ["ahat.estimate"]
projection = ["anchor_angle_smoothing_fraction", "pilivar"]
ix, iy = [parnames.index(par) for par in projection]
xlim, ylim = bounds[ix], bounds[iy]
anchor =  np.linspace(*xlim, num=res)
pilivar =  np.linspace(*ylim, num=res)
pardata = [anchor, pilivar]

# %% 
# accepted = rejection_abc(statdf, [objective], reference, N)
    # statdf = params[parnames+[objective]]
statdf, statref = abcimplement.regularise_stats(params, reference, objectives)
_acc = rejection_abc(statdf, ss, reference, N, ss_component=True)

# %% 
llacc = abcimplement.llregression(_acc, reference, ss, projection)
llacc

# %% 
# scatter side by side
fig, axes = plt.subplots(1,2,figsize=(10,5))
p1, p2 = projection
#
ax = axes[0]
ax = sns.scatterplot(data=_acc, x=p1, y=p2, hue="score", ax=ax)
ax.set_xlim(bounds[ix])
ax.set_ylim(bounds[iy])
ax.set_title("accepted")
#
ax = axes[1]
ax = sns.scatterplot(data=llacc, x=p1, y=p2, hue="score", ax=ax)
ax.set_xlim(bounds[ix])
ax.set_ylim(bounds[iy])
ax.set_title("local linear regression")

# %% 
# we implement epanechnikov kernel but scipy.stats.gaussian_kde does a better job
# we still use epanechnikov kernel for weighting by summary statistic 
from abcimplement import new_epanechnikov

# Now attempting to implement much more complex beaumont regression ABC
# 1. compute standard deviation of summary statistics
# if we use only one summary statistic does this even matter?
def smooth_regression_abc(problem, params, sstat, reference, projection=None,
    N= 200
    ):
# params dataframe contains parameters and normal summary statistics but not lvel similarity statistic
    _names = problem["names"]
    _bounds  = problem["bounds"]
    ix, iy = [_names.index(par) for par in projection]
    xlim, ylim = _bounds[ix], _bounds[iy]

    statdf, statref = abcimplement.regularise_stats(params, reference, sstat)
    _acc = rejection_abc(statdf, sstat, reference, N, ss_component=True)
    _acc = abcimplement.llregression(_acc, reference, sstat, projection)
    weight = _acc["weight"]

    def statsmodels_dens(pardata, res, mirror_yaxis=True):
        # NO WEIGHTS
        data = [_acc[_p].to_numpy() for _p in projection]
        bw_estimate = sm.nonparametric.KDEMultivariate(data=data, var_type='cc', bw='cv_ls')
        if mirror_yaxis:
            data = mirrorpts(data, ylim)
        dens = sm.nonparametric.KDEMultivariate(data=data, var_type='cc', 
            bw='normal_reference')
        dens.bw = bw_estimate.bw
        print("bw", dens.bw)
        X1, X2 = np.meshgrid(anchor, pilivar)
        result = 3*dens.pdf([X1.ravel(), X2.ravel()])
        _grid = result.reshape([100,100])
        return X1, X2, _grid

    # posterior distibution at parameter 
    def gkde_eval_posterior(pardata, res, mirror_yaxis=True):
        # take the accepted parameter data arrays
        # 
        anchor, pilivar = pardata
        X1, X2 = np.meshgrid(anchor, pilivar)
        # the accepted parameters 
        data = [_acc[_p].to_numpy() for _p in projection]
        _weight = weight
        if mirror_yaxis:
            data = mirrorpts(data, ylim)
            _weight = np.concatenate([weight,weight,weight])
        gkde = scipy.stats.gaussian_kde(np.stack(data),weights=_weight)
        pdf = 3*gkde.evaluate(np.stack([X1.ravel(), X2.ravel()]))
        return X1, X2, pdf.reshape((res,res))

    def eval_posterior(pardata, res, mirror_yaxis=True):
        delta_p1, delta_p2 = bw_estimate.bw
        kern1 = new_epanechnikov(delta_p1)
        kern2 = new_epanechnikov(delta_p2)
        anchor, pilivar = pardata
        X1, X2 = np.meshgrid(anchor, pilivar)
        posterior = np.zeros((len(anchor),len(pilivar)))
        par1, par2 = [_acc[_p].to_numpy() for _p in projection]
        # slow double loop
        for index, anchor_v in np.ndenumerate(X1):
            pilivar_v = X2[index]
            _num = np.array([kern1(par1[i] -anchor_v)*kern2(par2[i] - pilivar_v) for i in range(N)])
            posterior[index] = np.sum(_num * weight)/np.sum(weight)
        return X1, X2, posterior

    # return gkde_eval_posterior
    # return eval_posterior
    return statsmodels_dens

eval_posterior = smooth_regression_abc(problem, params, ss, reference, 
    projection=projection, N=N)
#
X1, X2, posterior = eval_posterior(pardata, res=res)
print("computed posterior")

# %%
# palette = sns.color_palette("viridis", as_cmap=True)
pdata = pd.DataFrame(posterior, anchor, pilivar)
fig, axes = plt.subplots(1,2,figsize=(10,5))
ax = axes[1]
sns.heatmap(pdata, ax=ax)
_p1, _p2 = projection
def _heatmap_ax(ax, _bounds):
    xlim, ylim = _bounds
    ax.invert_yaxis()
    # simplify this tick locator business
    ax.set_xticks([0,res//2,res-1])
    ax.set_yticks([0,res//2,res-1])
    ax.set_xticklabels(["{:.3g}".format(x) for x in np.linspace(*xlim, num=3, endpoint=True)])
    ax.set_yticklabels(["{:.3g}".format(x) for x in np.linspace(*ylim, num=3, endpoint=True)])
    ax.set_xlabel(_p1)
    ax.set_ylabel(_p2)
_heatmap_ax(ax, [xlim, ylim])
# scatter
ax = axes[0]
p1, p2 = projection
ax = sns.scatterplot(data=llacc, x=p1, y=p2, hue="score", ax=ax)
ax.set_xlim(bounds[ix])
ax.set_ylim(bounds[iy])

# %%
# Must be after the anchor/pilivar basis is defined
# statsmodels = False
statsmodels = True
if statsmodels:
    X1, X2 = np.meshgrid(anchor, pilivar)
    result = 3*dens.pdf([X1.ravel(), X2.ravel()])
    _grid = result.reshape([100,100])
    ax = sns.heatmap(_grid)
    _heatmap_ax(ax, [xlim, ylim])


# %%
# compare bandwidth settings
# print("me", [mod_bw_settings[par]["set"] for par in projection])
# print("statmodels", dens.bw)
# check normalization 
# https://coderedirect.com/questions/502920/integrating-2d-samples-on-a-rectangular-grid-using-scipy
check = False
if check: # normalization
    from scipy.integrate import simps
    print("pdf integrates to ", simps(simps(posterior, anchor), pilivar))
    line = np.sum(posterior,axis=0)
    plt.plot(anchor, line)
    estimate1d = np.sum(anchor*line)/np.sum(line)
    print("simple anchor parameter estimate {:.2g}".format(estimate1d))
# we used this estimate to skip ahead and do abc with the anchor parameter frozen out
# see ~/usb_twitching/run/5bfc8b9/cluster/mc3d_frozen


# %%
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# sampling arbitrary distributions
# %%
# sample this pdf
# 1. interpolate to a smooth function
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html
# 2. check the interpolation
# 3. sampling
# https://stackoverflow.com/questions/49211126/efficiently-sample-from-arbitrary-multivariate-function
# %%
# pinky library does give any better result than our brute force sampling
if verbose:
    # someone has done the work for us so lets use it
    sys.path.append(join(notedir, "lib/pinky"))
    from pinky import Pinky
    ix, iy = [parnames.index(par) for par in projection]
    xlim, ylim = bounds[ix], bounds[iy]
    extent = np.array([*xlim, *ylim])
    print(extent)
    pink = Pinky(P=posterior, extent=extent)
    x0 = np.array([0.2,10.0])
    samples = pink.sample(1000, r=1)
    x, y = samples.T
    ax = sns.kdeplot(x, y, levels=10, fill=True, bw_adjust=0.8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("resampled")
# %% [markdown]
# pinky is fast but the sampling doesn't seem all that accurate
# the denisty tapers the y-boundary and seems unnecessarily smoothed
# although that could be seaborn.kdeplot

# %%
# for our own sampling implementation we need to interpolate the posterior distribution
from numpy.random import uniform
import scipy.interpolate
# image data is stored in numpy row order arrays 
# hence the 0th axis in the array is the y axis
# we choose to construct interpolated functions which accept (y,x) as arguments rather than (x,y)
bbox = [*ylim, *xlim]
print("bbox", bbox)
# For whatever reason scipy recommends RectBivariateSpline for rectangular samples. Lets trust them.
target_density = scipy.interpolate.RectBivariateSpline(pardata[1],pardata[0],posterior, 
    bbox=bbox)
    
# test the interpolation
show = np.zeros(posterior.shape)
for index, x in np.ndenumerate(X1):
    y = X2[index]
    show[index]  = target_density(y,x)
ax = sns.heatmap(show)
_heatmap_ax(ax, [xlim,ylim])
ax.set_title("recovered from interpolation")
# its good

# %%
# test our brute force sampling
from abcimplement import force_sample_target

# maximum (from sampled pdf)
zlim = (0, np.max(posterior))
xt, yt = force_sample_target(target_density, [xlim, ylim, zlim])
print("nsamples", len(xt))
fig, ax = plt.subplots()
ax  = sns.scatterplot(x=xt,y=yt)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
fig, ax = plt.subplots()
ax = sns.kdeplot(xt,yt, fill=True, ax=ax)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_title("brute force resampled")
# works well

# %%
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# mc3d with fixed anchor parameter = 0.17
# see rejectionabc.py

