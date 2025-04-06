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
# check that for simulated data with many tracks that persistence and 
# activity calculations are robust

# %% 
import os
import numpy as np
import twanalyse
join = os.path.join
import readtrack
import _fj
import rtw
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotutils

from pili import publication as pub

notename = "persistence_pt3"

# %% 
# load simulation with target parameters
crawlingsim = "/home/dan/usb_twitching/run/new/qstable"
cr_trs = readtrack.trackset(ddir=join(crawlingsim,'data/'))
crltrs =  [_fj.linearize(tr) for tr in cr_trs]

# %% 
#  simulated estimate
get_q = rtw.make_get("qhat.estimate")
get_a = rtw.make_get("ahat.estimate")

qasd = twanalyse._qaparams(crltrs)

q = get_q(qasd)
a = get_a(qasd)
print("concatentating trajectories gives", q, a)

# %% 
# compute weighted means
_qasds = [twanalyse._qaparams([tr]) for tr in crltrs]
_qls = np.array([get_q(data) for data in _qasds])
_als = np.array([get_a(data) for data in _qasds])
nsteps = np.array([ltr.get_nsteps() for ltr in crltrs])
# durations = np.array([ltr.get_duration() for ltr in ltrs])

w_q = np.sum(nsteps*_qls)/np.sum(nsteps)
w_a = np.sum(nsteps*_als)/np.sum(nsteps)

print("weighted means gives", w_q, w_a)
# which is nearly identical

# %% 
# do the same for fanjini data (subset = top)
subsets = _fj.load_subsets()
subset = "top"

# %% 
# the same concatentation method
fltrs = subsets["top"]
sd = twanalyse._qaparams(fltrs)
get_q(sd), get_a(sd)
# very low q, and when we try weighted means we get...


# %% 
# weighted means
f_qasds = [twanalyse._qaparams([ltr]) for ltr in fltrs]
f_qls = np.array([get_q(data) for data in f_qasds])
f_als = np.array([get_a(data) for data in f_qasds])
f_nsteps = np.array([ltr.get_nsteps() for ltr in fltrs])
w_q = np.sum(f_nsteps*f_qls)/np.sum(f_nsteps)
w_a = np.sum(f_nsteps*f_als)/np.sum(f_nsteps)

print(len(f_nsteps), np.sum(f_nsteps))
print(f"fanjin weighted mean q = {w_q:0.4f}, a = {w_a:0.4f}")
# complete different values

# %% 
# Check robustness of persistence calculation
norm = np.linalg.norm
# step_vel = [ltr.get_step_velocity()/ltr.get_step_speed()[:,None] for ltr in subsets["top"]]
# step_vel = [ltr.get_step_dx() for ltr in subsets["top"]]
def concatutup(step_vel, threshold=False):
    sim_ut = []
    sim_up = []
    for svel in step_vel:
        sim_ut.append(svel[1:])
        sim_up.append(svel[:-1])
    u_t = np.concatenate(sim_ut, axis=0)
    u_p = np.concatenate(sim_up, axis=0)
    return u_t, u_p

def test_window_prw(u_t, u_p, threshold=False):
    
    # shuffle_idx = list(range(len(u_t)))
    # np.random.shuffle(shuffle_idx)
    # u_t = u_t[shuffle_idx]
    # u_p = u_p[shuffle_idx]

    if threshold:
        th = threshold
        s_t = norm(u_t,axis=1)
        s_p = norm(u_p,axis=1)
        select = np.logical_and(s_t < th, s_p < th)
        u_t = u_t[select]
        u_p = u_p[select]
    #
    _sample = np.array(range(u_t.shape[0]))
    q = []
    a = []
    Ns = []
    ns = []
    
    Ns = [200,500,1000,2000]
    for N in Ns:
        _split_at = np.array(range(N, _sample.shape[0], N))
        #
        qs = []
        _as = []
        for window in np.split(_sample, _split_at)[:-1]:
            sample = np.array(window)
            n = sample.size
            # print(sample)
            qhat = np.sum( (u_t*u_p).sum(axis=1)[sample] )/np.sum( (u_p*u_p).sum(axis=1)[sample])
            upart = (u_t - qhat*u_p)[sample]
            # sum over both dimensions here is ok
            ahat = np.sqrt(np.sum(upart*upart)/(2*(n-1)))
            qs.append(qhat)
            _as.append(ahat)
        _n = len(qs)
        q.append(np.mean(qs))
        a.append(np.mean(_as))
        ns.append(_n)
        print(_n, N, np.mean(qs), np.mean(_as))
    return ns, Ns, q, a
        

step_vel = [ltr.get_step_velocity() for ltr in subsets["top"]]
u_t, u_p = concatutup(step_vel)

print("n N <q> <a>")


# the windowing method confirms this calculation is not robust
# windowing method on simulated data
sim_step_vel = [ltr.get_step_velocity() for ltr in crltrs]
su_t, su_p = concatutup(sim_step_vel)

ns, Ns, q, a = test_window_prw(u_t, u_p)
print()
print("on simulated data?")

sim_ns, Ns, sim_q, sim_a = test_window_prw(su_t, su_p)


mplstyle = plotutils.get_style('ft8')
with plt.style.context(mplstyle):
    fig, ax = plt.subplots(figsize=(10,8))

    defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c1, c2 = defcolors[:2]

    style = {"marker":'X', 'linewidth':3, 'markersize':12}
    ax.plot(Ns, q, color=c1, label=r'$\bar{q}$', **style)
    ax.plot(Ns, a, color=c2, label=r'$\bar{a}$', **style)
    ax.set_ylim(0,0.5)


    simstyle = {"marker":'X', 'linestyle':'--', 'linewidth':3, 'markersize':12}
    ax.plot(Ns, sim_q, color=c1, label=r'simulation $\bar{q}$', **simstyle)
    ax.plot(Ns, sim_a, color=c2, label=r'simulation $\bar{a}$', **simstyle)
    ax.set_ylim(0,0.5)
    
    ax.set_ylabel('statistic')
    ax.set_xlabel('N')
    ax.legend()

pub.save_figure("robustness_check", notename)

# so there is no such discrepancy for simulated data
# %% 

# dig into u_t \cdot u_p to understand why this happening
with mpl.rc_context({'font.size':20,  'xtick.labelsize': 18, 'ytick.labelsize': 18}):
    fig, axes = plt.subplots(2,1, figsize=(12,6))
    ax1, ax2 = axes
    utp = (u_t*u_p).sum(axis=1)
    print('min, max',  utp.min(), utp.max())


    ax1.plot(utp, linestyle='', marker='+', alpha=0.5)
    ax1.set_title("Experiment")
    sutp = (su_t*su_p).sum(axis=1)
    print('min, max',  sutp.min(), sutp.max())

    ax2.set_title("Simulation")
    ax2.plot(sutp, linestyle='', marker='+', alpha=0.5)

    ax1.set_ylim(-10,10)
    ax2.set_ylim(-10,10)
    ax1.set_xlabel(r'$i$', fontsize=18)
    ax2.set_xlabel(r'$i$', fontsize=18)

    ax1.set_ylabel(r'$\mathbf{v}_i \cdot \mathbf{v}_{i-1}$')
    ax2.set_ylabel(r'$\mathbf{v}_i \cdot \mathbf{v}_{i-1}$')


    plt.tight_layout()

pub.save_figure('correlation_outliers', notename)
# pub.save_figure('correlation_outliers_same_yaxis', notename)

# Even though we selected well resolved trajectories there are still issues
# the huge outliers in fanjin velocities are most likely errors

# %% 
# but if we threshold out large velocities, does this fix the problem?
th = 1.0
print("fanjin")
test_window_prw(u_t, u_p, threshold=th)
print()
print("sim")
test_window_prw(su_t, su_p, threshold=th)
# fanjins data becomes consitent with this threshold

# %% 
# now recover information on what was removed
# th = 1.0
s_t = norm(u_t,axis=1)
s_p = norm(u_p,axis=1)
# quantiles are more convenient than fixed thresholds
th = np.quantile(s_t, 0.99)
print(th)
select = np.logical_and(s_t < th, s_p < th)
# cutu_t = u_t[~select]
# cutu_p = u_p[~select]
print("{}/{}".format(np.sum(~select), select.size))
ax = sns.histplot(s_t[~select])
ax.set_xlabel("value")

# %% 
# compute the reference data
import fjanalysis
all_idx, ltrs = _fj.slicehelper.load_linearized_trs("all")
reference_idx = _fj.load_subset_idx()
# %% 
objectives = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
df = fjanalysis.compute_reference_data(ltrs, reference_idx, objectives)
df

# %% 


# %% 
# fltrs = subsets["top"]
# qa_each = [twanalyse._qaparams([ltr]) for ltr in fltrs]
# qeach = np.array([get_q(sd)  for sd in qa_each])
# qposidx = np.argwhere(qeach > 0.).ravel()
# _mod_subset = [fltrs[i] for i in qposidx]
# twanalyse._qaparams(_mod_subset)

# %%
