
# this script is a JOB
# compute the PWL model of a whole experimental trajectory 

# ----------------------------------------------------------------

import os, sys
import matplotlib.pyplot as plt
import _fj
from pili import support
import mdl
import annealing
import pickle
join = os.path.join


# certain debugging must be turned off
annealing._debug = False

# ----------------------------------------------------------------
# setup paths

# setup output directory
name, py = os.path.splitext(__file__)
outputdir = '_' + name
if os.path.exists(outputdir):
    if not input("directory {} already exists. continue (y,N)\n".format(outputdir)) == 'y':
        sys.exit()
else:
    os.mkdir(outputdir)

annealing.set_logging_directory(outputdir)
annealing.set_logging_level(annealing.FLog.WARNING)

# ----------------------------------------------------------------
# setup

# half pixel size of experimental microscope
r = 0.03

trackidx = int(_fj.load_subset_idx()["top"][1])
print(f"loading track {trackidx}")
# trackidx =  2924 # candidate
track = _fj.trackload_original([trackidx])[0]
lptr = mdl.get_lptrack(track)
duration = track.get_duration()

# record the data
import json
path = join(outputdir, 'config.json')
config = {'trackidx' : trackidx}
print(f'writing config {config} to {path}')
with open(path, 'w') as f:
    json.dump(config, f)


# DEBUG
duration = 20
lptr = lptr.cut(0, duration)

path = join(outputdir, "data.pkl")
with open(path, 'wb') as f:
    pickle.dump(lptr, f)

# guess M
# very simple guess is half as many segments as second of trajectory
M = int(duration)//2 + 1
print("initial guess with M = ", M)

def save_initial_guess(data, M):
    _guess = mdl.recursive_coarsen(data, M, parameter='M')
    fig, ax = plt.subplots(figsize=(20,20))
    mdl.plot_model_on_data(ax, _guess, data)
    path = join(outputdir, "initial_guess.pkl")
    print("writing to ", path)
    with open(path, 'wb') as f:
        pickle.dump(_guess, f)
    path = join(outputdir, "initial_guess.svg")
    print("writing to ", path)
    plt.savefig(path)
save_initial_guess(lptr, M)


# ----------------------------------------------------------------
# solve

# TODO: push this to cluster so i don't cook my laptop
# TODO: output useful message of (DL, M) at each step
# TODO: output incomplete solutions at regular intervals
loss_conf = {}
loss_conf["contour_term"] = 0.01

with support.PerfTimer() as timer:
    solver = annealing.solve_exp(lptr, M, r, default_loss_conf=loss_conf)
print("execution time ", timer.time)

# dumps seperate pkl files for the model / Anneal object / random
path = join(outputdir, "solver")
solver.dump_state(path=path)

def save_final_state():
    fig, ax = plt.subplots(figsize=(20,20))
    annealing.model_plot(ax, solver.anneal, lptr)
    path = join(outputdir, "final_state.svg")
    print("writing to ", path)
    plt.savefig(path)
    path = join(outputdir, "final_state.png")
    print("writing to ", path)
    plt.savefig(path)
save_final_state()





