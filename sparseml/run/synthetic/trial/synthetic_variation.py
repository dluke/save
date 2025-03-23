

import sys, os 
join = os.path.join
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import multiprocessing

from pili import support
import pwlpartition
from synthetic import *
import synthetic 

# config
debug = False

# synthetic generator used global rng
seed = 0
np.random.seed(seed)

# generate 
N = 40
params = {"l": 1.0, "sigma": 0.03, "N": 40, "dx": 0.1, "p_threshold": 0.98, "angle_sigma": pi/8}




def generate_pwl(params):
    N = params.get("N")
    l = params.get("l")
    #
    length = Uniform(l, l)
    mirror_angle = AlternatingMirrorNormal(loc=pi/4, scale=pi/16)
    pwl = synthetic.new_static_process(length, mirror_angle, N)
    return pwl

def new_synthetic_data(pwl, params):
    dx =  params.get("dx")
    sigma = params.get("sigma")
    tracer = Constant(dx)
    error = Normal(scale=sigma)
    synthdata = sample_pwl(pwl, tracer, error)
    return synthdata

# first generate the pwl true model
pwl = generate_pwl(params)

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

def save_local_true_model(target_dir, pwl):
    target = join(target_dir, 'truth.pkl')
    with open(target, 'wb') as f:
        print('writing to ', target)
        pickle.dump(pwl, f)


def save_local_summary(target_dir, summary):
    target = join(target_dir, 'summary.json')
    print(f'writing summary {summary} to {target}')
    with open(target, 'w') as f:
        json.dump(summary, f)

# 
base_local_config = {}

default_control = {'maxiter': 5000, 't_end': 0., 'tolerance': 1e-6}

def run_variation(job_index, target_dir, pwl, params, update, control={}):

    # redirect_stdout = open(join(target_dir, "pwlpartition.out"), 'w')
    # pwlpartition.local_f = redirect_stdout

    save_local_true_model(target_dir, pwl)

    local_params = params.copy()
    local_params.update(update)

    local_config = base_local_config.copy()
    local_config['update'] = update
    local_config['params'] = local_params


    synthdata = new_synthetic_data(pwl, local_params)
    save_local_data(target_dir, synthdata)

    # generate initial guess
    wavemodel, lptr, meta = pwlpartition.initial_guess(synthdata.x, synthdata.y)
    local_config['meta'] = meta
    r = meta['r']

    # loss_conf = {"contour_term": None, "continuity_term": 1}
    loss_conf = {}
    local_config['loss_conf'] = loss_conf
    
    # construct solver
    partition = pwlpartition.PartAnneal(wavemodel, lptr, loss_conf=loss_conf, inter_term=True)
    rng = np.random.RandomState(seed)
    solver = pwlpartition.Solver(partition, rng=rng, r=r, sigma=meta["sigma"], min_constraint=1)

    # save the initial guess
    solver.dump_state(join(target_dir, 'initial_guess'))
    truth_style = {"linestyle": '--', 'lw':4, 'alpha':0.5, 'label':'truth'}
    fig, ax = pwlpartition.model_plot(solver, synthdata, fs=20)
    ax.plot(pwl.x, pwl.y, **truth_style)
    fig.savefig(join(target_dir, 'initial_guess.png'))

    # priority_solve configuration
    _control = default_control.copy()
    _control.update(control)

    local_config['control'] = _control
    print(local_config)
    save_local_config(target_dir, local_config)

    if debug:
        return 

    output = pwlpartition.Output(target_dir, allow_write=False, index=job_index)
    with support.PerfTimer() as timer:
        solver.priority_solve(_control, output=output)

    solver.dump_state(join(target_dir, 'solver'))
    fig, ax = pwlpartition.model_plot(solver, synthdata, fs=20)
    ax.plot(pwl.x, pwl.y, **truth_style)
    fig.savefig(join(target_dir, 'final_state.png'))

    # summary
    exec_time = timer.get_time()
    summary = {'execution_time' : exec_time, 'solver_iter' : solver.count_iter}
    save_local_summary(target_dir, summary)

    # redirect_stdout.close()

    return solver


variant = 'sigma'
output_form = "{}_{:07.4f}"
# variant_basis = [0.03,0.05,0.10, 0.12, 0.15, 0.20,0.30,0.40]
variant_basis = [0.10]
update_list = [{variant: sigma} for sigma in variant_basis]

data_directories = [output_form.format(variant, update[variant]) for update in update_list]

# the global config
script_config = {"variant": variant, "basis" : variant_basis, "data_directories" : data_directories}
save_local_config('.', script_config)


for job_id, update in enumerate(update_list):
    print("update", update)

    # target_dir = output_form.format(variant, update[variant])
    target_dir = data_directories[job_id]
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    p = multiprocessing.Process(target=run_variation, args=[job_id, target_dir, pwl, params, update])
    p.start()
