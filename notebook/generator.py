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
# boltzmann generator for TFP configurations

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import scipy.integrate
# %%
def make_pd(ka,a,Beta):
    def pd(E):
        if E == 0.:
            return 0
        return 1/np.sqrt(E) * np.sin(np.sqrt(a*E/ka)) * np.exp(-Beta*E)
    return pd
a = 1.0
a = 0.1
T = 273 + 30
kB = scipy.constants.Boltzmann
# This is kbT is pN and micrometres 
kbT = (kB * T) * 10**18. 
Beta = 1/kbT
Lp = 5. # persistence length
ka = Lp * kbT

pd = make_pd(ka,a,Beta)
# %%

N = 10000
E_max = ka * (np.pi**2)/a
basis = np.linspace(0,E_max,N,True)
print(E_max)
print(pd(0))
print(pd(E_max))

# %%
ax = plt.gca()
ax.plot(basis, [pd(x) for x in basis])
ax.set_xlim(0,0.025)
plt.show()
# %%
def make_ge(ka,a):
    def ge(E):
        if E == 0:
            return 0.
        return 1/np.sqrt(E) * np.sin(np.sqrt(a*E/ka)) 
    return ge
ge = make_ge(ka,a)

# %%
phibasis = np.linspace(0,E_max,int(10e5),True)
dens = [ge(x) for x in phibasis]
# %%
ax = plt.gca()
# ax.set_xlim(0-10e-4,0.025)
ax.plot(phibasis, dens)
plt.show()


# %%
# using instead E = (ka/a) cos(\phi) 

def make_kp_pd(ka,a,Beta):
    def pd(E):
        return np.exp(-Beta*E)
    return pd
def make_kp_ge(ka,a):
    def pd(E):
        pass
    return pd

print(ka,a,Beta)
pd = make_kp_pd(ka,a,Beta)
ge = make_kp_ge(ka,a)


# %%

N = 10000
anglemin = 0
anglemax = np.pi/2
E_min = -ka * np.cos(anglemin)/a
E_max = -ka * np.cos(anglemax)/a
print(E_min,E_max)
basis = np.linspace(E_min,E_max,N,True)
# %%
# cumulative distribution function
Z = (np.exp(Beta*ka/a) - 1)/Beta
def make_Cx(ka,a,Beta):
    def C(x):
        return (np.exp(Beta*ka/a) - np.exp(-Beta*x))/(Z*Beta)
    def X(c):
        return -1 * np.log(np.exp(Beta*ka/a) - Z*Beta*c)/Beta
    return C, X
C, X = make_Cx(ka,a,Beta)

print(C(E_min))
print(C(E_max))

print(E_min, X(C(E_min)))
print(E_max, X(C(E_max)))
''
# %%
fig, axes = plt.subplots(1,2,figsize=(10,5))
ax, ax2 = axes
pdf = [pd(x)/Z for x in basis]
# cdf = scipy.integrate.cumulative_trapezoid(pdf, basis, initial=0)
cdf = [C(x) for x in basis]
ax.plot(basis, pdf)
ax.set_ylabel('P(x)')
ax2.set_ylabel('CDF(x)')
ax2.plot(basis, cdf)
plt.show()

# %%

# implement inverse transform sampling
Nsample = int(10e5)
uniform = np.random.random(N)
Esample = X(uniform)
plt.hist(Esample,bins=100)
plt.show()

# %%
def phi(E):
    return np.arccos(-a*E/ka)
phisample = phi(Esample)
print(phisample)
plt.hist(phisample, bins=30, range=(0,np.pi/2))
plt.show()

# %%
# if we want to check persistence length we need to reconstruct chain in
# 3d and compute the angles between segments
thetasample = 2*np.pi * np.random.random(Nsample)
# %%

