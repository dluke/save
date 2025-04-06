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
# some examples to understand sobol analysis 

# %% 
import sys, os
join = os.path.join
import numpy as np
import scipy.integrate
import SALib as sal
#
import sobol 
import matplotlib.pyplot as plt
import txtdata
import parameters

# %% 
from SALib.analyze import fast

# %% 
problem = {
	'num_vars': 6,
	'names': ['x', 'y', 'z', 'a', 'b', 'c'],
	'bounds': [
		[-1.0,1.0],
		[-1.0,1.0],
		[-1.0,1.0],
		[-1.0,1.0],
		[-1.0,1.0],
		[-1.0,1.0],
		]
}

def model(x,y,z,a,b,c):
	return x + y + z + a + b + 0.01*c

values = sal.sample.saltelli.sample(problem, 128)

# %% 

Y = np.zeros([values.shape[0]])

for i, X in enumerate(values):
	Y[i] = model(*X)

Y.shape

# %% 
Si = sal.analyze.sobol.analyze(problem, Y)

print(["{:.2f}".format(a) for a in Si['S1']])
# print(Si['S2'])
print(["{:.2f}".format(a) for a in Si['ST']])

# %% [markdown]
# since sobol analysis seems to work perfectly for this toy example
# the next step is two reduce the dimensions of our real sobol problem so we can analyse the issues
# more conveniently

# %% 
# paths for the 2d problem
notedir = os.path.abspath(os.path.dirname(__file__))
rundir = os.path.abspath(join(notedir, '../../run/'))
# simdir = join(rundir, "b2392cf/cluster/sobol_2d")
simdir = join(rundir, "b2392cf/cluster/sobol_2d")

# %% 

objectives = ['nbound.mean', 'lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
# scrambled
Y, _ = sobol.collect(objectives, targetdir=simdir, alldata=True)
#
samples = np.loadtxt(join(simdir, "problem_samples.npy"))
# load all directories in lookup and read their args
lookup = sobol.read_lookup(simdir)
# parl = []
# for dirname in lookup[0]:
# 	args = parameters.thisread(directory=join(simdir, dirname))
# 	params = np.array([args.ACell.k_spawn, args.Pili.k_resample])
# 	parl.append((dirname, params))

# def match(pars, parl):
# 	tol = 1e-5
# 	matching = []
# 	for dirname, params in parl:
# 		m = np.all(np.abs(pars-params) < tol)
# 		if m:
# 			matching.append(dirname)
# 	print(matching)	
# 	assert(len(matching) == 1)
# 	return matching[0]

# for sample in samples:
# 	# find the directory by searching for a floating point match ...
# 	# what if there are duplicate samples ...
# 	m = match(sample, parl)
# 	print(m)
# 	break

# %% 
problem = sobol.read_problem(simdir)
print(problem)
lookup = sobol.read_lookup(simdir)
samples = np.array(list(lookup[1].values()))
Xs, Ys = samples.T
ax = plt.gca()
color = Y['nbound.mean']
ax.scatter(Xs, Ys, c=color)
ax.set_aspect("equal")
xlabel, ylabel = [txtdata.longnames.get(x) for x in problem["names"]]
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
plt.show()

# %% 
fast.analyze(problem, Y['nbound.mean'])

# %% 

# check for missing data
print("check for missing data")
for name in Y.keys():
	print('{:<30} {}/{}'.format(name, np.count_nonzero(np.isnan(Y[name])), Y[name].size))

def factor_cut(arr, n):
	x = arr.size // n
	N = x * n
	return arr[:N]

Si = {}
d = problem['num_vars']
Yfiltered = {}
data_indices = {}
for name in Y.keys():
	arr = Y[name]
	data_idx = ~np.isnan(arr)
	data_indices[name] = data_idx
	filtered = arr[data_idx]
	filtered = factor_cut(filtered, 2*d+2)
	print(arr.size, filtered.size)
	Yfiltered[name] = filtered
	Si[name] = sal.analyze.sobol.analyze(problem, filtered)

for name in Y.keys():
	S = Si[name]
	print(name)
	print('S1', S['S1'])
	print('ST', S['ST'])


# %% 
# integrate in both directions in parameter space to plot the change in nbound
name = 'nbound.mean'
scipy.integrate.simps()
Y[name]


# %% 
# constrain nbound to see if that helps (it doesn't)
nbound = Y["nbound.mean"]
fig, ax = plt.subplots(figsize=(6,2))
ax = plt.gca()
ax.hist(nbound, rwidth=0.9)
ax.set_xlabel("nbound")
ax.show()

constraints = [
	('nbound', (0.5, 3.0))
]

cdata = constraints[0]
cname, lims = cdata
left, right = lims

# nan values not included
loweridx = nbound < left 
upperidx = nbound > right
mididx = np.logical_and(nbound < right, nbound > left)

# %% 

# for the other objectives
for name in Y.keys():
	mid_Y = Y[name][mididx]
	print(name)
	d = problem['num_vars']
	mid_Y = factor_cut(mid_Y, 2*d+2)
	Smid = sal.analyze.sobol.analyze(problem, mid_Y)
	print('S1', Smid['S1'])
	print('ST', Smid['ST'])



# %% [markdown]
# so there is still something fundamentally wrong with our approach here
# but now that we have 2d data, surely we can figure out what it is

# %% 
# scatter plot showing the selected range in nbound
def widen(pair, c=0.1):
	left, right = pair
	d = right - left
	return left - c*d, right + c*d

samples = np.array(list(lookup[1].values()))
X, Y = samples[mididx].T
ax = plt.gca()
ax.scatter(X, Y)
xb, yb = problem["bounds"]
ax.set_xlim(widen(xb))
ax.set_ylim(widen(yb))
ax.set_aspect("equal")
xlabel, ylabel = [txtdata.longnames.get(x) for x in problem["names"]]
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
plt.show()

# showing that nbound depends almost exclusively on the the spawn rate as expected
# so what do we need to do for the sensitivity to reflect this
