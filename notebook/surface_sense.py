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
# We implement surface sensing by varying k_ret_on
# to create a delayed retraction after contacting the surface

# %% 
import os, sys
join = os.path.join
import pili
import numpy as np
import pandas as pd
import seaborn as sns
import stats
import json
import matplotlib.pyplot as plt
from tabulate import tabulate
import SALib.analyze as sal
#
import rtw
import _fj
import txtdata
import sobol
import twanalyse
import twutils
import pili.publication as pub

from sobol import collect_obs, compute_sobol, format_sobol
# %% 
# start with sobol at 
notedir, notename = os.path.split(os.getcwd())
simdir = "/home/dan/usb_twitching/run/b2392cf/cluster/sobol_ssense"
lookup = sobol.read_lookup(simdir)
problem = sobol.read_problem(simdir)
twutils.print_dict(problem)

# %% 
# first job is to check that the simulation and analysis actually ran
stats_mask = np.array([os.path.exists(join(simdir, udir, "local.json")) for udir in lookup[0]])
num, denom = np.count_nonzero(stats_mask), len(lookup[0])
print('ran samples {}/{} ({:.1f}%)'.format(num, denom, 100*float(num)/denom))

# %% 
# load summary statistics 
objectives = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate', 
    'nbound.mean', 'ntaut.mean']
Y, lduid = sobol.collect(objectives, targetdir=simdir, alldata=True)
# %% 
# check for missing data
missing = sobol.check_missing(lookup, Y)
if missing:
    print(tabulate(missing, headers=["objective", "nan data"]))
else:
    print("No nan data")

# %% 

# %% 

second_order = False
Si = sobol.compute_sobol(problem, Y, second_order=second_order)
Si["lvel.mean"]["ST"]

dftable1, dftableT = sobol.format_sobol(problem, Si)
dftableT

# %% 
from IPython.display import display, HTML
# display(HTML(dftable1.to_html()))
display(HTML(dftableT.to_html()))

with open("/home/dan/usb_twitching/notes/sensitivity/tex/ssense_table.tex", "w") as f:
    f.write(dftableT.to_latex())


# %% [markdown]

# We see that the retraction delay (k_ret_on)
# has almost no effect on the dynamics. Despite influencing nbound.mean
# it does not change ntaut.mean. We do not allow the extension 
# motor to force pili into bent configurations but we do allow 
# "bent" (l_ideal < l_contour) configurations due to attachment
# and changing body position. 
# We expect pili which are not retracting to rarely be involvd in the dynamics
# This may not be the case if the bacteria has low or negative persistence
# e.g. In walking state. 
# TODO redo this simulation in walking state

# %% 
# cm = sns.light_palette("green", as_cmap=True)
cm = pub.get_modgreen_cm()

# %% 
sty = dftableT.style.hide_index()
sty.to_latex()
sty.background_gradient(cmap=cm)
sty

# %% 
# need this table in latex 
cfl = ["s", ".5g", ".5g", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f"]
body = pub.construct_template(cfl, 6)

# publication.write_tex(template, "all_sobol_part", notename)
head = r"{h0} & {h1} & {h2} & {h3} & {h4} & {h5} & {h6} & {h7} & {h8} \\"

tt = pub.load_template("sobolsense.template")
template = tt.format(head=head, body=body)
pub.save_template("sobolsense", template)

# %% 

header = ["parameter", 'min', 'max', 
    r'$\mean{u}$', r'$Var({\theta_d})$', r'$ \hat{q} $', r'$ \hat{a} $', 
    r"$\langle N_{\mathrm{bound}} \rangle$", r"$\langle N_{\mathrm{taut}} \rangle$"
    ]

lpar = ['\\keoff', '\\tdwell', '\\kappa', '\\alpha', '\\kspawn', '\\kron']

# %% 
df = dftableT
df
# %% 

_hdct, _pdct, _cdct, _tdct, _vdct = pub.construct_dicts(df, 
    problem, objectives, cm, header=header, lpar=lpar)

# %%
template = pub.load_template("sobolsense")
# need to cut first and last lines
_lines = template.strip().split('\n')
template = r'\n'.join(_lines[1:-1])
filled = template.format(**_hdct, **_pdct, **_cdct, **_tdct, **_vdct)
filled = _lines[0]+r'\n'+filled+r'\n'+_lines[-1]
pub.write_tex(filled, "sobolsense_gen", notename)
# filled


