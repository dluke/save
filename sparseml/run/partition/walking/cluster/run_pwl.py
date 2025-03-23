
# this script is a JOB
# compute the PWL model of a whole experimental trajectory 

# ----------------------------------------------------------------

import os, sys
import numpy as np
join = os.path.join
import json
import matplotlib.pyplot as plt
import pickle
import time
import multiprocessing

import _fj
from pili import support
import mdl
import pwlpartition 




# ----------------------------------------------------------------
# cluster config
maxcores = 64
# if True no computation, just check the multiprocessing works
dry_run = False
# if True only run on the first track
single_debug_run = True
# if True cut the first 20s of data
debug = False
# if True don't ask for input on commmand line
script = True


# ----------------------------------------------------------------
def save_local_config(target_dir, config):
    target = join(target_dir, 'config.json')
    with open(target, 'w') as f:
        print('writing to ', target)
        json.dump(config, f)

def save_local_data(target_dir, data):
    target = join(target_dir, 'data.pkl')
    with open(target, 'wb') as f:
        print('writing to ', target)
        pickle.dump(data, f)

def save_local_summary(target_dir, summary):
    target = join(target_dir, 'summary.json')
    print(f'writing summary {summary} to {target}')
    with open(target, 'w') as f:
        json.dump(summary, f)


# ----------------------------------------------------------------
def main():
    # get the indices of the tracks we will analyze
    idx = list(map(int, _fj.slicehelper.load('pure_walking')))

    jobs = list(enumerate(idx))
    if single_debug_run:
        jobs = [jobs[10]]
    parallel_run(jobs, maxcores)


def pwl_solve(trackidx, debug=False): 

    # ----------------------------------------------------------------
    # setup paths

    # setup output directory
    name = "top_{:04d}".format(trackidx)
    target_dir = '_' + name
    if os.path.exists(target_dir):
        if script:
            print(f"directory {target_dir} already exists. overwriting")
        else:
            if not input("directory {} already exists. continue (y,N)\n".format(target_dir)) == 'y':
                sys.exit()
    else:
        os.mkdir(target_dir)

    redirect_stdout = open(join(target_dir, "pwlpartition.out"), 'w')
    pwlpartition.local_f = redirect_stdout


    # ----------------------------------------------------------------
    # configure

    track = _fj.trackload_original([trackidx])[0]
    lptr = mdl.get_lptrack(track)
    N = len(lptr)
    print(f"loading track {trackidx} with {N} data points")


    # DEBUG
    if debug:
        duration = 10
        lptr = lptr.cut(0, duration)

    save_local_data(target_dir, lptr)

    local_config = {'trackidx':trackidx}

    # generate initial guess
    wavemodel, lptr, meta = pwlpartition.initial_guess(lptr.x, lptr.y)
    local_config['meta'] = meta

    # loss_conf = {"contour_term": None, "continuity_term": 1}
    loss_conf = {}
    loss_conf = {"contour_term": 0.05, "continuity_term": 1}
    local_config['loss_conf'] = loss_conf
    
    # construct solver
    partition = pwlpartition.PartAnneal(wavemodel, lptr, loss_conf=loss_conf)
    rng = np.random.RandomState()
    solver = pwlpartition.Solver(partition, rng=rng, r=meta["r"], sigma=meta["sigma"])

    # save the initial guess
    solver.dump_state(join(target_dir, 'initial_guess'))
    fig, ax = plt.subplots(figsize=(20,20))
    pwlpartition.simple_model_plot(ax, wavemodel, lptr)
    fig.savefig(join(target_dir, 'initial_guess.png'))

    # priority_solve configuration
    control = {'maxiter': 5000, 't_end': 0., 'tolerance': 1e-6}

    local_config['control'] = control
    save_local_config(target_dir, local_config)


    # ----------------------------------------------------------------
    # solve

    output = pwlpartition.Output(target_dir, allow_write=True, index=trackidx)
    with support.PerfTimer() as timer:
        solver.priority_solve(control, output=output)

    print("execution time ", timer.time)

    solver.dump_state(join(target_dir, 'solver'))
    # fig, ax = plt.subplots(figsize=(20,20))
    # pwlpartition.simple_model_plot(ax, solver.partition.model, lptr)
    fig, ax = pwlpartition.model_plot(solver, lptr, config = {"shaded":False})
    fig.savefig(join(target_dir, 'final_state.png'))

    # summary
    exec_time = timer.get_time()
    summary = {'execution_time' : exec_time, 'solver_iter' : solver.count_iter}
    save_local_summary(target_dir, summary)

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
