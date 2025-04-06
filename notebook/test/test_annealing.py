
import unittest

import sys, os
import numpy as np
import matplotlib.pyplot as plt 

sqrt = np.sqrt 
norm = np.linalg.norm


import mdl
import synthetic
import annealing

def _setup_simple():
    #  setup
    x = np.array([0., 1., 2.])
    y = np.array([0., 1., 1.])
    dt = np.array([0.,0.1,0.2])
    pwl = synthetic.LPshape(x, y)
    truemodel = mdl.LPtrack(dt, x, y)
    ex = np.array([1.,0.])
    ey = np.array([0.,1.])
    en = (-ex + ey)/norm(ex + ey)
    ptlist = [
        pwl(0.3) + 0.2 * en,
        pwl(0.7) - 0.2 * en,
        pwl(sqrt(2.)) + 0.1 * ey,
        pwl(sqrt(2.)+0.5) - 0.4 * ey
    ] 
    x, y = np.array(ptlist).T
    _DT = 0.1
    dt = np.arange(0, _DT*len(x), _DT)
    data = mdl.LPtrack(dt, x, y)
    #
    anneal = annealing.Anneal()
    anneal.initialise(truemodel, data)
    return data, truemodel, anneal

class TestSolver(unittest.TestCase):

    def test_load_dump(self):
        rngstate = np.random.RandomState(1234)
        data, truemodel, anneal = _setup_simple()
        solver = annealing.Solver(anneal, rng=rngstate)
        tmppath = "tmp/solver"
        solver.dump_state(tmppath)
        new_solver = annealing.Solver.load_state(tmppath)
        # print(new_solver.anneal.llmodel)

class TestAnneal(unittest.TestCase):

    def test_load_dump(self):
        data, truemodel, anneal = _setup_simple()
        tmppath = "tmp/current"
        anneal.dump_state(tmppath)
        new_anneal = annealing.Anneal.load_state(tmppath)
        self.assertEqual(anneal.get_current_model(), new_anneal.get_current_model())

    def test_anneal_model(self):
        data, truemodel, anneal = _setup_simple()
        lptr = anneal.get_current_model()
        _anneal =  annealing.Anneal()
        _anneal.initialise(lptr, data)
        newlptr = _anneal.get_current_model()
        self.assertEqual(lptr, newlptr)

    def test_pwl_coord(self):
        data, truemodel, anneal = _setup_simple()
        pwl, mapped =  anneal.get_pwl_coord(get_mapped=True)
        # print(pwl)
        # print(mapped)
        fig, ax = plt.subplots(figsize=(10,10))
        mdl.plot_mapped(ax, mapped, data, truemodel)
        ax.set_aspect('equal')
        fig.tight_layout()
        plt.savefig("local.png")

    def test_clone(self):
        data, truemodel, anneal = _setup_simple()
        _anneal = anneal.clone()
        _anneal.llmodel.nodeat(0).value.x = -1.0

if __name__ == "__main__":
    unittest.main()
        