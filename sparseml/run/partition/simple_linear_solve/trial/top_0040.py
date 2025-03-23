
# this script is a JOB
# compute the PWL model of a whole experimental trajectory 

# ----------------------------------------------------------------

import os, sys
import matplotlib.pyplot as plt
import _fj
from pili import support
import mdl
import pwlpartition
import pickle
join = os.path.join


# certain debugging must be turned off
# annealing._debug = False

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


# ----------------------------------------------------------------
# setup
debug = False
# redirect_stdout = open(join(outputdir, "pwlpartition.out"), 'w')
# pwlpartition.local_f = redirect_stdout

# half pixel size of experimental microscope
r = 0.03

# trackidx = _fj.load_subset_idx()["top"][0]
trackidx =  40 # candidate
track = _fj.trackload_original([trackidx])[0]

# initial guess
if debug:
    track = track.cut(0,200)

wavemodel, lptr, meta = pwlpartition.wavelet_guess(track)

path = join(outputdir, "data.pkl")
with open(path, 'wb') as f:
    pickle.dump(lptr, f)


def save_initial_guess(wavemodel, data):
    fig, ax = plt.subplots(figsize=(20,20))
    mdl.plot_model_on_data(ax, wavemodel, data)
    path = join(outputdir, "initial_guess.pkl")
    print("writing to ", path)
    with open(path, 'wb') as f:
        pickle.dump(wavemodel, f)
    path = join(outputdir, "initial_guess.svg")
    print("writing to ", path)
    plt.savefig(path)

save_initial_guess(wavemodel, lptr)


# ----------------------------------------------------------------
# solve

# TODO: output useful message of (DL, M) at each step
# TODO: output incomplete solutions at regular intervals
loss_conf = {}
loss_conf["contour_term"] = 0.01

with support.PerfTimer() as timer:
    solver = pwlpartition.percolate_cleanup(wavemodel, lptr, loss_conf=loss_conf)
print("execution time ", timer.time)

# dumps seperate pkl files for the model / Anneal object / random
path = join(outputdir, "solver")
solver.dump_state(path=path)

def save_final_state():
    pwlpartition.model_plot(solver, lptr, fs=20)
    path = join(outputdir, "final_state.svg")
    print("writing to ", path)
    plt.savefig(path)
    path = join(outputdir, "final_state.png")
    print("writing to ", path)
    plt.savefig(path)
save_final_state()





