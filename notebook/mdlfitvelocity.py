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
# mdlvelocity.py has expanded too far so we continue fitting mixture models here


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

import statsmodels.api as sm

# %% 
mplstyle = {"font.size": 20}
notename = "mdlfitvelocity"
work = False

notedir = join(pili.root, "notebook/")

# %% 
# load the pwl mapped velocity from the PWL solver
# target = join(pwlstats.root, "run/partition/candidate/no_heuristic/_candidate_pwl/")

# TEST
target = "/home/dan/usb_twitching/sparseml/run/cluster/no_heuristic/top/_top_2368/"

solver = pwlstats.load_solver_at(target)

solver.partition.cache_local_coord = np.empty(solver.partition.N) # hotfix
solver.partition.update_residuals()

curve_coord = solver.partition.get_curve_coord()
udata = np.diff(curve_coord) 

# %% 
# * For latex notes. Plot the original and pwl mapped 0.1s velocities

shstyle = dict(element="step", fill=False, alpha=0.8)
# xlim for plotting
xlim = (-0.08, 0.16)
# xlim for preprocess 
pxlim = (-0.16, 0.40)

def preprocess(data, xlim=pxlim):
	#! zeros are over represented because of the way we measure distances around segment ends
	n = data.size
	data = data[data!=0]

	# ! In order to get a sensible initial guess, we need to set a maximum curve velocity
	# quick fix
	data = data[ np.logical_and(data > xlim[0], data < xlim[-1]) ]

	print('preprocess: keep {} / {} ({:.2f}%)'.format(data.size, n, 100*float(data.size)/n))
	return data


# fig, ax = plt.subplots(1,2, figsize = (8,4))
with mpl.rc_context({"font.size": 18}):
	fig, ax = plt.subplots(figsize = (4,4))
	original_vel = solver.partition.data.get_distance()

	sns.histplot(original_vel, ax=ax, **shstyle)
	ax.set_xlim(xlim)
	ax.set_xlabel('displacement ($\mu m$)')
	ax.set_ylim(0,300)

pub.save_figure("candidate_displacement_distribution", notename)

with mpl.rc_context({"font.size": 18}):
	fig, ax = plt.subplots(figsize = (4,4))

	sns.histplot(preprocess(udata), ax=ax, **shstyle)
	ax.set_xlim(xlim)
	ax.set_xlabel('displacement ($\mu m$)')
	ax.set_ylim(0,300)

pub.save_figure("candidate_pwl_mapped_displacement_distribution", notename)




# %% 
# plot distribution

fig, ax = plt.subplots(figsize=(4,4))
sns.histplot(udata, ax=ax, **shstyle)
ax.set_xlim(xlim)
ax.set_title('original')

data = preprocess(udata)

fig, ax = plt.subplots(figsize=(4,4))
sns.histplot(data, ax=ax, **shstyle)
ax.set_xlim(xlim)
ax.set_title('preprocessed')


# %% 
# get a smoothed distribution

def get_kde(data, bw_factor=1.0):
	kde = sm.nonparametric.KDEUnivariate(data)
	kde.fit()
	kde.fit(bw=bw_factor * kde.bw)
	return kde

kde = get_kde(data, bw_factor=0.5)
print('support', [kde.support.min(), kde.support.max()], kde.support.size)

fig, ax = plt.subplots(figsize=(4,4))
ax.plot(kde.support, kde.density)


# %% 
# initial guess for exponential part
loc, scale = scipy.stats.expon.fit(data[data>0])
loc, scale, solver.sigma

# %% 
# plot this exponential fit 
fig, ax = plt.subplots(figsize=(4,4))

l = 1/scale
def expon(x):
	return l * np.exp(-l * x)

idx = np.searchsorted(kde.support, 0)

supp = kde.support[idx:]
ax.plot(supp, expon(supp))
ax.plot(kde.support, kde.density)

# %% 
# a mixture model is a function of several variables, 
# we need initial guesses for these variables

# use a dictionary to keep track of parameters?
initial = {'lambda': 1/scale, 'sigma': solver.sigma}
print('initial', initial)

class MixtureConstructor(object):

	def __init__(self, support, density, initial):
		self.initial = initial 
		self.support = support
		self.density = density
		# 
		self.mix = []
		self.x0 = []
		self.remap = []
		self.bounds = []
		self.count = 0

		# default err distribution
		sigma = self.initial.get('sigma')
		self.error = self.make_error(sigma)

		width = support[0]
		self.err_support = support[:np.searchsorted(support, -width)]

	def add_expon(self, lam):
		def var(lam, loc=0):
			def expon(x):
				arr = lam * np.exp( -lam * (x-loc))
				arr[x<loc] = 0
				return arr
			return expon
		self.mix.append(var)
		self.x0.append(lam)
		self.remap.append( (self.count,) )
		self.count += 1
		self.bounds.append([0,None])

	def add_normal(self, loc, sigma):
		def var(loc, sigma):
			def normal(x):
				return scipy.stats.norm(loc, sigma).pdf(x)
			return normal
		self.mix.append(var)
		self.x0.extend([loc, sigma])
		self.remap.append( (self.count, self.count+1) )
		self.count += 2
		self.bounds.append([0,None])
		self.bounds.append([0,None])

	def add_impulse(self, loc, bound=[0,None]):
		# ! discrete problems in optimising this location?
		self.initial["impulse"] = loc
		def var(loc):
			def impulse(x):
				idx = np.searchsorted(x, loc)
				imp = np.zeros_like(x)
				delta = x[1] - x[0]
				imp[idx] = 1/delta
				return imp
			return impulse
		self.mix.append(var)
		self.x0.append(loc)
		self.remap.append( (self.count,) )
		self.count += 1
		self.bounds.append(self.fix_to_support(bound))

	def make_error(self, sigma):
		def error(x):
			return scipy.stats.norm(0, sigma).pdf(x)
		return error

	def convolve(self, in1):
		in2 = self.error(self.err_support)
		trial_density = scipy.signal.convolve(in1,in2,mode="same")
		trial_density /= scipy.integrate.simpson(trial_density, self.support)
		return trial_density

	def linear_combination_variables(self):
		# initial guesses for linear combination variables
		linvar = [1.0 for _i in range(len(self.remap)-1)]
		self.x0.extend(linvar)
		self.bounds.extend([ [0,None] for _ in range(len(linvar)) ])
		# self.fix_bounds_to_support()
	
	def fix_to_support(self, bound):
		mn, mx = bound
		if mn == None:
			mn = self.support.min()
		if mx == None:
			mx = self.support.max()
		return [mn, mx]

	def get_components(self, varlist):
		x = self.support
		component = []
		for i, item in enumerate(self.remap):
			v = self.mix[i](*[varlist[i] for i in item])(x)
			component.append(v)
		return component

	def get_mixture(self, varlist):
		n = len(self.remap)
		component = self.get_components(varlist)
		return self.compute_linear_combination(component, varlist[-(n-1):])

	@staticmethod
	def compute_linear_combination(component, coef=[]):
		# linear combination
		n = len(component)
		if n == 1:
			return 1 * component[0]
		elif n > 1:
			# we put the linear coefficients at the end
			return 1 * component[0] + sum([coef[i] * component[i+1] for i in range(len(coef))])

	def get_functional(self):

		n = len(self.remap)

		def functional(varlist):
			def compute_components(x):
				# compute components
				component = []
				for i, item in enumerate(self.remap):
					v = self.mix[i](*[varlist[i] for i in item])(x)
					component.append(v)
				return component

			component = compute_components(self.support)
			in1 = self.compute_linear_combination(component, varlist[-(n-1):])
			
			trial_density = self.convolve(in1)

			# weighted LSQ
			weights = np.sqrt(np.abs(self.density))
			squares = (self.density - trial_density)**2
			value = np.sum( weights  * squares ) / np.sum(weights)

			return value
		return functional

	def optimize(self):
		functional = self.get_functional()
		x0 = self.x0
		print("optimize with initial guess", x0)
		result = scipy.optimize.minimize(functional, x0, tol=1e-12, method="Nelder-Mead", bounds=self.bounds)
		self.result = result
		return result

construct = MixtureConstructor(kde.support, kde.density, initial)

lam = initial.get('lambda')
construct.add_expon(lam)
functional = construct.get_functional()

result = construct.optimize()

# %% 
construct = MixtureConstructor(kde.support, kde.density, initial)
construct.add_expon(lam)
construct.add_impulse(0.0)
construct.add_normal(0.03, 0.01)
construct.linear_combination_variables()
print(construct.x0)
print(construct.remap)

result = construct.optimize()
print("result.x", ['{:.3f}'.format(x) for x in result.x])
print("weighted lsq value", result.fun)

# %% 

def model1(kde, par):
	lam = par.get('lambda')
	construct = MixtureConstructor(kde.support, kde.density, par)
	construct.add_expon(lam)
	construct.linear_combination_variables()
	return construct

def model2(kde, par):
	lam = par.get('lambda')
	construct = MixtureConstructor(kde.support, kde.density, par)
	construct.add_expon(lam)
	construct.add_impulse(0.0)
	construct.linear_combination_variables()
	return construct

def model3(kde, par):
	lam = par.get('lambda')
	construct = MixtureConstructor(kde.support, kde.density, par)
	construct.add_expon(lam)
	construct.add_impulse(0.03)
	construct.linear_combination_variables()
	return construct

def model4(kde, par):
	lam = par.get('lambda')
	construct = MixtureConstructor(kde.support, kde.density, par)
	construct.add_expon(lam)
	construct.add_impulse(0.00)
	construct.add_impulse(0.03)
	construct.linear_combination_variables()
	return construct

def model5(kde, par):
	lam = par.get('lambda')
	construct = MixtureConstructor(kde.support, kde.density, par)
	construct.add_expon(lam)
	construct.add_impulse(0.00)
	construct.add_normal(0.03, 0.01)
	construct.linear_combination_variables()
	return construct

def model6(kde, par):
	construct = MixtureConstructor(kde.support, kde.density, par)
	construct.add_impulse(0.00)
	construct.add_impulse(0.03)
	construct.linear_combination_variables()
	return construct

def model7(kde, par):
	construct = MixtureConstructor(kde.support, kde.density, par)
	construct.add_impulse(0.00)
	construct.add_impulse(0.03)
	construct.add_impulse(-0.03, bound=[kde.support.min(),0])
	construct.linear_combination_variables()
	return construct

def model8(kde, par):
	lam = par.get('lambda')
	construct = MixtureConstructor(kde.support, kde.density, par)
	construct.add_expon(lam)
	construct.add_impulse(0.00)
	construct.add_impulse(0.03)
	construct.add_impulse(-0.03, bound=[kde.support.min(),0])
	construct.linear_combination_variables()
	return construct



constructors = [model1, model2, model3, model4, model5, model6, model7, model8]

names = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7', 'model8']

par = {"lambda": 1/scale, "sigma": solver.sigma}
models = [model(kde, par) for model in constructors]


import pandas as pd

dct = {'name' : [], 'chi' : [], 'par' : []}

for i, model in enumerate(models):
	result = model.optimize()
	# print("model ", result.x.tolist())
	# print("weighted lsq value", result.fun)
	dct['name'].append(names[i])
	dct['chi'].append(result.fun)
	dct['par'].append(result.x.round(3))
	

df = pd.DataFrame(dct)
df


# %% 
# plotting

def plot_construct(construct):
	support, density = construct.support, construct.density
	xlim = (-0.08, 0.20)

	in1 = construct.get_mixture(construct.result.x)
	in2 = construct.convolve(in1)

	fig, ax = plt.subplots(figsize=(4,4))
	ax.plot(support, density, label='data', alpha=0.7)
	ax.plot(support, in2, label='mixture', linestyle='--')
	ax.set_xlim(xlim)
	ax.legend()

	ax.axvline(0, c='k', alpha=0.4, linestyle='--')

plot_construct(models[3])
# plot_construct(models[0])
plot_construct(models[6])

# %% 
# * LOAD MORE DATA
from glob import glob
look = glob( join(pwlstats.root, f"run/cluster/no_heuristic/top/_top_*") )
found = [directory for directory in look if os.path.exists(join(directory, "solver.pkl"))]
solverlist = [pwlstats.load_solver_at(directory) for directory in found]

def get_data(solver):
	# solver.partition.cache_local_coord = np.empty(solver.partition.N) # hotfix
	solver.partition.update_residuals()
	curve_coord = solver.partition.get_curve_coord()
	udata = np.diff(curve_coord) 
	data = preprocess(udata)
	return data

datalist = [get_data(solver) for solver in solverlist]

# %%
index = 2
print('size', datalist[index].size)
print('sigma', solverlist[index].sigma)
print('directory', found[index])

fig, ax = plt.subplots(figsize=(4,4))
sns.histplot(datalist[index], ax=ax, stat='density', **shstyle)
ax.axvline(0, c='k', alpha=0.4, linestyle='--')
ax.set_xlim(xlim)

_kde = get_kde(datalist[index], bw_factor=0.5)
ax.plot(_kde.support, _kde.density)

# %% 

sigma = solverlist[index].sigma

def solve_mixture(data, sigma=sigma):

	kde = get_kde(data, bw_factor=0.5)
	print('support', [kde.support.min(), kde.support.max()], kde.support.size)
	loc, scale = scipy.stats.expon.fit(data[data>0])
	par = {"lambda" : 1/scale, "sigma": sigma}
	print('par', par)
	models = [model(kde, par) for model in constructors]

	dct = {'name' : [], 'chi' : [], 'par' : []}

	for i, model in enumerate(models):
		result = model.optimize()
		dct['name'].append(names[i])
		dct['chi'].append(result.fun)
		dct['par'].append(result.x.round(3))
		
	df = pd.DataFrame(dct)
	return df, models

tdf, models = solve_mixture(datalist[index], sigma)
tdf

# %%
plot_construct(models[4])
plot_construct(models[-1])

# %% 
# * ANALYSE ALL 
work = True
if work:
	dflist = []
	for index in range(len(datalist)):
		print('analysing data {}/{}'.format( index, len(datalist)))
		sigma = solverlist[index].sigma
		tdf, models = solve_mixture(datalist[index], sigma)
		dflist.append(tdf)

# %% 
# * CACHE 
out = join(notedir, notename, "dflist.pkl")
print('writing to ', out)
with open(out, 'wb') as f:
	pickle.dump(dflist, f)

# %% 

with open(out, 'rb') as f:
	dflist = pickle.load(f)


# %% 

# for the best idea is to normalize chi 
def norm_score(tdf):
	score = tdf['chi']
	bestchi = score.min()
	normed = score/bestchi
	return normed.to_numpy()

scoresT = [norm_score(tdf) for tdf in dflist]
scores = list(zip(*scoresT))

# %% 
# plot a scatter/boxplot for each model similar to this link
# https://nbviewer.org/gist/fonnesbeck/5850463

fig, ax = plt.subplots(figsize=(8,4))
ax.set_yscale('log')
mstyle = dict(alpha=0.1)
for i in range(len(models)):
	m_scores = scores[i]
	N = len(m_scores)
	x = np.full(N, i+1)
	dx = np.random.normal(0, 0.04, size=N)
	ax.scatter(x+dx, m_scores, **mstyle)

# ax.set_ylim(0,5)
ax.set_xlabel('model')

# %% 
# do a violin plot too 
lfig, ax = plt.subplots(figsize=(8,4))
ax.set_yscale('log')
ax.violinplot(scores)

# %% 
# * Continue by analysing models to see what the parameter values are?
# lets look at model 7
# fig, ax = plt.subplots(figsize=(6,4))


m7parT = [tdf['par'][6] for tdf in dflist]
m7par = list(zip(*m7parT))
x0, x1, x2, c1, c2 = m7par
N = len(x0)

fig, ax = plt.subplots(figsize=(6,4))

defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color = iter(defcolors)

vstyle = dict(alpha=0.2,c='black',linestyle='--')
ax.axvline(-0.03, **vstyle)
ax.axvline(0.03, **vstyle)
ax.axvline(0, **vstyle)
	
columns = [(x0, np.full(N, 1.0)), (x1, c1), (x2, c2)]
for data in columns:
	x, y = data
	c = next(color)
	ax.scatter(x, y, alpha=0.1, color=c)

# ax.set_ylim(0,4.0)
ax.set_ylim(0,1.5)




# %% 

# %% 
# ---------------------------------------------------------------------------------
# old testing
construct = MixtureConstructor(kde.support, kde.density, deau)
construct.add_impulse(0.1)
imp = construct.mix[0](0.1)(kde.support)
# plt.plot(support, imp)
density = construct.convolve(imp)
plt.plot(support, density)



# %% 
# plot exponential
xlim = (-0.05,0.12)

lam_prime = result.x[0]
support = construct.support
fig, ax = plt.subplots(figsize=(4,4))

ax.set_xlim(xlim)
ax.axvline(0, c='k', alpha=0.4, linestyle='--')
part = construct.mix[0](lam_prime)(support)
# ax.plot(support, part, label='true', linestyle='--', alpha=0.8)

ax.plot(construct.err_support, construct.error(construct.err_support), label='error', alpha=0.4)
ax.plot(support, construct.convolve(part), label='trial', linestyle='--')
ax.plot(support, construct.density, label='data')

ax.legend()




# %%
