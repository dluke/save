
# this script is a JOB
# compute the PWL model of a whole experimental trajectory 

# ----------------------------------------------------------------

import os, sys
join = os.path.join
import json
import matplotlib.pyplot as plt
import _fj
from pili import support
import mdl
import annealing
import pickle

import time
import multiprocessing


# certain debugging must be turned off
annealing._debug = False

# ----------------------------------------------------------------
# cluster config
maxcores = 64
# if True no computation, just check the multiprocessing works
dry_run = False
# if True only run on the first track
single_debug_run = True
# if True cut the first 20s of data
debug = True
# if True don't ask for input on commmand line
script = True

debug_duration = 100
redirect_printing = False

# ----------------------------------------------------------------
def main():
    # get the indices of the tracks we will analyze
    # idx = list(map(int, _fj.slicehelper.load('top_200_crawling')))
    idx = [40]

    jobs = list(enumerate(idx))
    if single_debug_run:
        jobs = [jobs[-1]]
    parallel_run(jobs, maxcores)


def pwl_solve(trackidx, debug=False): 

    # ----------------------------------------------------------------
    # setup paths

    # setup output directory
    name = "top_{:04d}".format(trackidx)
    outputdir = '_' + name
    if os.path.exists(outputdir):
        if script:
            print(f"directory {outputdir} already exists. overwriting")
        else:
            if not input("directory {} already exists. continue (y,N)\n".format(outputdir)) == 'y':
                sys.exit()
    else:
        os.mkdir(outputdir)

    if redirect_printing:
        redirect_stdout = open(join(outputdir, "annealing.out"), 'w')
        annealing.local_f = redirect_stdout


    # ----------------------------------------------------------------
    # configure

    # half pixel size of experimental microscope
    r = 0.03

    print(f"loading track {trackidx}")
    track = _fj.trackload_original([trackidx])[0]
    lptr = mdl.get_lptrack(track)
    duration = track.get_duration()

    # record the configuration
    path = join(outputdir, 'config.json')
    config = {'trackidx' : trackidx}

    loss_conf = {}
    loss_conf["contour_term"] = 0.01

    config['loss'] = loss_conf
    
    print(f'writing config {config} to {path}')
    with open(path, 'w') as f:
        json.dump(config, f)

    # DEBUG
    if debug:
        lptr = lptr.cut(0, debug_duration)

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

    def solve_exp(_data, M, r, default_loss_conf={}):
        def _print_break(message):
            print("----------------------------------------------------------", file=local_f)
            print(message, file=local_f)

        config = {"use_ordering": True}
        rngstate = np.random.RandomState(0)
        _guess = mdl.recursive_coarsen(_data, M, parameter='M')
        anneal = Anneal(r)
        anneal.default_loss_conf = default_loss_conf
        anneal.initialise(_guess, _data)
        solver = Solver(anneal, rng=rngstate, config=config)

        _print_break("initial solve")
        solver.multiple_linear_solve(n=1, n_local=2)

        _print_break("cleanup")
        solver.cleanup()

        _print_break("priority solve")
        pqsolve = {
            'greedy' : False, 
            'Tc' : 0.01, 
            'fc' : 0.5
            }
        solver.priority_solve(control=pqsolve)

        _print_break("greedy solve")
        greedy = {
            'greedy' : True, 
            'Tc' : 0.01, 
            'fc' : 0.2
        }
        solver.priority_solve(control=greedy)

        # _print_break("remapping")
        # solver.remapping()

        _print_break("cleanup")
        solver.cleanup()
        return solver

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

    # summary
    path = join(outputdir, 'summary.json')
    local = {'execution_time' : timer.time}
    print(f'writing summary {local} to {path}')
    with open(path, 'w') as f:
        json.dump(config, f)

    if redirect_printing:
        redirect_stdout.close()

#
def parallel_run(jobs, maxcores):
    running = []
    finished = []

    def check_finished(running, finished):
        still_running = []
        for p in running:
            if p.is_alive():
                still_running.append(p)
            else:
                finished.append((i, p))
        running = still_running
        return running, finished
    
    while(running or jobs):
        while(len(running) >= maxcores):
            running, finished = check_finished(running, finished)
            time.sleep(0.01)
        if jobs:
            i, trackidx = jobs.pop(0)
            print('starting job number {} on track {}'.format(i, trackidx))
            _args = [trackidx, debug]
            if dry_run:
                p = multiprocessing.Process(target=lambda *x:print(i, *x), args=_args)
            else:
                p = multiprocessing.Process(target=pwl_solve, args=_args)
            p.start()
            running.append(p)
            if not jobs:
                print('waiting for the remaining {} jobs to finish...'.format(len(running)))
        else:
            running, finished = check_finished(running, finished)
            time.sleep(0.01)

    print('finished parallel jobs')



if __name__ == '__main__':
    main()
