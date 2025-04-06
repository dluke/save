# %% [markdown]
# Analyse myosin VI stepping data

# %% 
import os
import random
import numpy as np
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
pi = np.pi

import scipy.stats

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from skimage.restoration import denoise_wavelet
from skimage.restoration import estimate_sigma

import sklearn.mixture

import pili
import pili.support as support
import mdl
import pwltree
import pwcshuffle
import pwcsearch

# %% 
# setup
data_dir = join(pili.root, '../sparseml/data/')


# load
ldata = np.loadtxt(join(data_dir, "fig4_left.csv"), skiprows=1)
rdata = np.loadtxt(join(data_dir, "fig4_right.csv"), skiprows=1)
print("loaded data with shapes", ldata.shape, rdata.shape)

# convert to nm
ldata[:,1] /= 10
rdata[:,1] /= 10

# %% 
import thesis.publication as thesis

# plot data
defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
ptlkw = {"linestyle":'--', 'lw':2, "marker":"o", "alpha":0.5, 'markerfacecolor': 'none', 'markeredgewidth':1.5}

with mpl.rc_context(thesis.basestyle):
	fig, ax = plt.subplots(figsize=(12,4))
ltime, ldisp = ldata.T
ax.plot(ltime, ldisp, color=defcolor[0], **ptlkw)

# fig, ax = plt.subplots(figsize=(6,4))
rtime, rdisp = rdata.T
# ax.plot(rtime, rdisp, color=defcolor[1], **ptlkw)

ax.set_ylabel("displacement (nm)")
ax.set_xlabel("time (s)")

# estimate the noise
rsigma, lsigma = estimate_sigma(rdisp), estimate_sigma(ldisp)
print('noise estimates (right, left)', rsigma, lsigma)
sigma = np.mean([rsigma, lsigma])

# draw the error on the plot for reference?

annotation = True
if annotation:
	y_ref = np.mean(ldisp[:40])
	refstyle = dict(linestyle = '--', color='k', alpha=0.5)
	# ax.axhline(y_ref + lsigma, **refstyle)
	# ax.axhline(y_ref - lsigma, **refstyle)

	# ax.axvline(ltime[219], **refstyle)
	ax.axvspan(ltime[0], ltime[219], facecolor='g', alpha=0.2, lw=2, edgecolor='g')


# %% 
# use pwltree as a first approximation

lparam = {
	'sigma': lsigma
	}

rparam = {
	'sigma': rsigma
	}


lpartition, lt_data, ly_data = pwcsearch.solve(lparam, ltime, ldisp)
print("solved left")
rpartition, rt_data, ry_data = pwcsearch.solve(rparam, rtime, rdisp)
print("solved right")

# %% 
# pull the long segment
lpartition
select_left = 219
select_right = 552

# linear fit
_x = ltime[select_left:select_right]
_y = ldisp[select_left:select_right]

# m*x + c
m, c = np.polyfit(_x, _y, 1)
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(ltime, ldisp, **ptlkw)
ax.plot(_x, m*_x + c, linewidth=3)

print("gradient {:.03f}".format(m))
trial_gradient = m

# what is the error distribution?

residual = (m*_x + c) - _y
mu, one_feature_sigma = scipy.stats.norm.fit(residual)
one_feature_sigma
# plt.figure()
# sns.histplot(residual)

# now draw this gradient on to the other side
# we could do this by fixing a gradient in the piecewise model

# %% 
# similar thing on the right 
rpartition
select_left = 234
select_right = 353
select_right = 333

# linear fit
_x = rtime[select_left:select_right]
_y = rdisp[select_left:select_right]

# m*x + c
m, c = np.polyfit(_x, _y, 1)
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(rtime, rdisp, **ptlkw)
ax.plot(_x, m*_x + c, linewidth=3)

print("gradient {:.03f}".format(m))



# %% 
# check the noise estimate
select_data = ldisp[:219]
fig, ax = plt.subplots(figsize=(6,4))

sns.histplot(select_data, bins=25, shrink=0.8, stat="density", ax=ax)
gmm = sklearn.mixture.GaussianMixture(n_components=4, covariance_type='tied')
gmm.fit(select_data.reshape(-1,1))
print(gmm.weights_)
print(gmm.means_)
print(gmm.covariances_)

def plot_normal(ax, mu, weight, variance, style={'lw':3, 'linestyle':'--'}):
	scale = np.sqrt(variance)
	space = np.linspace(mu - 3*scale, mu + 3*scale, 1000)
	curve = weight * scipy.stats.norm(mu, scale).pdf(space)
	ax.plot(space, curve, **style)

# plotting
variance = gmm.covariances_[0][0]
for mu, weight in zip(gmm.means_, gmm.weights_):  
	plot_normal(ax, mu, weight, variance)


gmm_lsigma = np.sqrt(gmm.covariances_[0][0])
gmm_lsigma

# %% 
solve_style = {'linestyle':'--', 'lw':3.0}

fig, axes = plt.subplots(2,1,figsize=(10,6), sharex=True, sharey=True)
ax = axes[0]
ax.plot(ltime, ldisp, **ptlkw)
pwcsearch.plot_pwc(ax, lt_data, ly_data, style=solve_style)
ax.set_title("left (red)")

ax = axes[1]
ax.plot(rtime, rdisp, **ptlkw)
pwcsearch.plot_pwc(ax, rt_data, ry_data, style=solve_style)
ax.set_title("right (yellow)")

figure_dir = "/home/dan/notes/sparseml"
support.save_figure("initial_solve", target_dir=figure_dir)


# %% 

def sample_probabilities(partition, sample_data, y_data, sigma):
	# get the probability associated with each block after random generation
	result = []
	for i in range(len(partition)-1):
		samples = sample_data[partition[i]:partition[i+1]]
		residual = np.abs(samples - y_data[i])
		p = pwcshuffle.residual_probability(residual, sigma)
		result.append(p)
	return result

from IPython.display import display

print('probabilities using wavelet heuristic')
display( sample_probabilities(lpartition, ldisp, ly_data, lsigma) )

print('probabilities using gmm estimate')
sample_probabilities(lpartition, ldisp, ly_data, gmm_lsigma)

# * GMM estimate is clearly more reasonable

# %% 
# same for right data now
display( sample_probabilities(rpartition, rdisp, ry_data, gmm_lsigma) )

# %% 
_ylim = (-909.965, -882.135)
_xlim = (12.09, 34.309999999999995)

# now draw just the solutions 

fig, ax = plt.subplots(figsize=(12,4))
pwcsearch.plot_pwc(ax, lt_data, ly_data, style={'color':'red', **solve_style})
pwcsearch.plot_pwc(ax, rt_data, ry_data, style={'color':'orange', **solve_style})


# %% 
# setup shuffle solver
from pwltree import PWCLoss, PWCMinimizeLoss
from pwcshuffle import probability_threshold

#! note that the use of sigma estimated from the left data
_param = {'sigma': gmm_lsigma}
_partition = rpartition

data = (rtime, rdisp)
shuffle = pwcshuffle.ShuffleSolver(data, rpartition,
	Loss = PWCLoss,
	evaluate_loss= probability_threshold,
	MinimizeLoss=PWCMinimizeLoss,
	param=_param
	)

shuffle.init()
probabilities = shuffle.get_probabilities()
probabilities

# %% 

threshold = 0.90
print('threshold', threshold)

N = 5000
shuffle.annealing(probability_threshold, threshold , N=N) 
print('finished annealing')
N = 5000
shuffle.annealing(probability_threshold, threshold , N=N, start_t=0.0) 
print('finished')


p, t, y = shuffle.get_model()
prob = shuffle.get_probabilities()
for pair in zip(p[:-1], prob):
	print(pair)	

# %%
ml = shuffle.evaluate_join(4)
probability_threshold(ml, _param)

# %%

fig, ax = plt.subplots(figsize=(12,4))
# ax.plot(ltime, ldisp, **ptlkw)
ax.plot(rtime, rdisp, **ptlkw)
p, t, y = shuffle.get_model()
print('partition', p)
print('t', t)
print('y', y)
pwcsearch.plot_pwc(ax, t, y, 
	{**solve_style, **dict(color=defcolor[1])}
	)

support.save_figure("shuffle_solve_yellow", target_dir=figure_dir)

# %%
prob = shuffle.get_probabilities()
for pair in zip(p[:-1], prob):
	print(pair)	


# %% 
# RED
# setup shuffle solver 

#! note that the use of sigma estimated from the left data
_param = {'sigma': gmm_lsigma}
_partition = lpartition

data = (ltime, ldisp)
shuffle = pwcshuffle.ShuffleSolver(data, lpartition,
	Loss = PWCLoss,
	evaluate_loss= probability_threshold,
	MinimizeLoss=PWCMinimizeLoss,
	param=_param
	)

shuffle.init()
probabilities = shuffle.get_probabilities()
probabilities

# %% 

threshold = 0.90
print('threshold', threshold)

N = 5000
shuffle.annealing(probability_threshold, threshold , N=N) 
print('finished annealing')
N = 5000
shuffle.annealing(probability_threshold, threshold , N=N, start_t=0.0) 
print('finished')

prob = shuffle.get_probabilities()
for pair in zip(p[:-1], prob):
	print(pair)	

# %%

fig, ax = plt.subplots(figsize=(12,4))
# ax.plot(ltime, ldisp, **ptlkw)
ax.plot(ltime, ldisp, **ptlkw)
p, t, y = shuffle.get_model()
print('partition', p)
print('t', t)
print('y', y)
pwcsearch.plot_pwc(ax, t, y, 
	{**solve_style, **dict(color=defcolor[1])}
	)

support.save_figure("shuffle_solve_red", target_dir=figure_dir)

# %%
prob = shuffle.get_probabilities()
for pair in zip(p[:-1], prob):
	print(pair)	

# %%
# setup treesolver with a slight gradient

from pwltree import PWCLoss, mean_squared_error, PWCMinimizeLoss
# * we actually need the time data here so it will have (N,2)
# this class takes data as argument so the tile value needs to be hard coded
# -- factory pattern --


def make_loss(m):
	'''
	Constructs the loss function with a fixed tilt angle
		Parameters:
	'''

	class TiltPWCLoss(PWCLoss):

		def __init__(self, data, jac=True):
			super().__init__(data, jac)

		def initial_guess(self):
			x0, y0 = self.data[0]
			# y = m*x0 + c
			c = y0 - m*x0
			return c

		def get_default(self):
			return np.array([self.initial_guess()])

		def __call__(self, c):
			# interpret y as the starting height of the segment
			X, Y = self.data.T
			y = m*X + c
			# after this it is the same as PWCLoss
			yd = Y-y
			self.sqresidual = yd**2
			# MEAN SQUARED ERROR
			return np.mean(self.sqresidual), -2*np.mean(yd)
	
	return TiltPWCLoss

TiltPWCLoss = make_loss(trial_gradient)

tilted = pwltree.TreeSolver(
	ldata,
	overlap=False,
	Loss = TiltPWCLoss,
	evaluate_loss=mean_squared_error,
	MinimizeLoss=PWCMinimizeLoss
	)

def solve(tilted, param, data):
	sigma = param.get('sigma')
	tilted.build_max_tree()
	tilted.build_priority()
	tilted.solve(pwltree.stop_at(2 * sigma**2))
	# TODO: get pwc tilted model
	solution = pwcsearch.get_pwc_model(tilted)
	index, c_data = zip(*solution)
	space = data[:,0]
	t_data = space[np.array(index)]
	return index, t_data, c_data

index, t_data, c_data = solve(tilted, lparam, ldata)
l_tilt_partition = index

# %%
# need new plotting function too

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(ltime, ldisp, **ptlkw)

def plot_tilted(ax, t_data, c_data, trial_gradient=trial_gradient):
	for i in range(len(t_data)-1):
		m = trial_gradient
		c = c_data[i]
		t1, t2 = t_data[i], t_data[i+1]
		_t = [t1, t2]
		_y = [m*t1+c, m*t2+c]
		style = {'linestyle':'--', 'lw':3.0, 'color':defcolor[1]}
		ax.plot(_t, _y, **style)

plot_tilted(ax, t_data, c_data)

# y_data = trial_gradient*t_data + c_data
# plot_pwc(ax, t_data, y_data)

# %%
# now run MC solver

#! note that the use of sigma estimated from the left data

data = (ltime, ldata)
shuffle = pwcshuffle.ShuffleSolver(data, l_tilt_partition,
	Loss = TiltPWCLoss,
	evaluate_loss= probability_threshold,
	MinimizeLoss=PWCMinimizeLoss,
	param={'sigma' : gmm_lsigma}
	)

shuffle.init()
probabilities = shuffle.get_probabilities()
probabilities

# %% 

threshold = 0.80
print('threshold', threshold)

N = 10000
shuffle.annealing(probability_threshold, threshold , N=N) 
print('finished annealing')
N = 10000
shuffle.annealing(probability_threshold, threshold , N=N, start_t=0.0) 
print('finished')


p, t, y = shuffle.get_model()
prob = shuffle.get_probabilities()
for pair in zip(p[:-1], prob):
	print(pair)	

# %%

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(ltime, ldisp, **ptlkw)
plot_tilted(ax, t, y)

# %%
# compute the standard error in the position measurements and plot path with errorbars

