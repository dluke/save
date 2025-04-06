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
# load two_parameter data, compute metrics and return the optimal parameters
# we may want to refer to sobolnote.py where these objectives are also used

# %% 
import sys, os
import numpy as np
join = lambda *x: os.path.abspath(os.path.join(*x))
import readtrack
import _fj
import fjanalysis
import rtw

# %% 
# paths
notedir, notename = os.path.split(__file__)
rundir = join(notedir, "../../run")
datapath = join(rundir, "c450a32/two_parameters_koch")

# %% 
dc = rtw.DataCube(target=datapath)

objectives = ['lvel.mean', 'deviation.var', 'qhat.estimate']
labels = ['mean velocity', r'variance of body deviation', 'persistence']
getters = [rtw.make_get(name) for name in objectives]
lldata = dc.load_local()

# %% 
candidate_track = _fj.lintrackload([2924])[0]
print()
ld = fjanalysis.lsummary([candidate_track])
Ycandidate = {name : getters[i](ld) for i,name in enumerate(objectives)}
Ycandidate

# %% 
# first sort by each objective individually, then by a combined objective
plt.rcParams.update({'text.usetex': False})

for i, name in enumerate(objectives):
    print(name)
    fig, ax = plt.subplots()
    def _linpar(ax,  dc, param_idx, ldata, label='', linekw=None):
        basis = dc.slice_basis[param_idx]
        getter = rtw.make_get(name)
        lvel = [getter(ld) for ld in ldata]
        print(label)
        ax.semilogx(basis, lvel, label=label, marker='D')
        ax.set_ylabel(labels[i])
    dc.plotmethod(ax, _linpar)
    plt.tight_layout()
    

# %% 
# find optimal point for linear objective function
Ydata = {}
for i, name in enumerate(objectives):
    metric_cube = dc.new_cube()
    for index in dc.ibasis:
        metric_cube[index] = getters[i](lldata[index])
    Ydata[name] = metric_cube
Ydata["lvel.mean"]
Yobjective = np.stack([Ydata[name] for name in objectives], axis=2) 

# %% 
# linear combination function
# Ycost = 
candidate_pt = np.array([Ycandidate[name] for name in objectives])
cost = np.abs(Yobjective - candidate_pt.reshape(1,1,3))
# linear combination
lcost = np.sum(cost,axis=2)
bestidx = np.argsort(lcost.ravel())[:10]
for i, idx in enumerate(bestidx):
    p0, p1 = dc.ibasis[idx]
    bestparam = dc.basis[0][p0], dc.basis[0][p1]
    print(bestparam, lcost.ravel()[idx])
# pilivar = 13.0, k_spawn = 7.0
