
import unittest

import sys, os
import numpy as np
import matplotlib.pyplot as plt 
import pili.support as support

sqrt = np.sqrt 
norm = np.linalg.norm


class TestGeometry(unittest.TestCase):
    
    def test_linseg_dist(self):
        # p = np.array([[1,1],[2,1],[2,2]])
        p = np.array([2,1])
        a = np.array([1,1]).reshape(1,2)
        b = np.array([2,2]).reshape(1,2)
        dist = support.lineseg_dists(p, a, b)[0]
        self.assertAlmostEqual(dist, sqrt(2.)/2)

    def test_line_coord_dist(self):
        # p = np.array([[1,1],[2,1],[2,2]])
        p = np.array([[0,0], [2,1], [2,1.1], [3,3]])
        a = np.array([1,1]).reshape(1,2)
        b = np.array([2,2]).reshape(1,2)
        _s, d = support.line_coord_dist(p, a, b)
        self.assertAlmostEqual(d[0], sqrt(2.))
        self.assertAlmostEqual(_s[1], sqrt(2.)/2) 
        self.assertAlmostEqual(d[1], sqrt(2.)/2)
        self.assertTrue(d[1] > d[2])
        self.assertAlmostEqual(d[3], sqrt(2.))
        #
        delta = 1.0
        p = np.array([[delta,0], [delta,1.0]])
        a = np.array([0,0]).reshape(1,2)
        b = np.array([2,1]).reshape(1,2)
        _s, d = support.line_coord_dist(p, a, b)
        # mapped points
        length = np.linalg.norm(b-a)
        mapped = _s.reshape(-1,1) * (b-a)/length
        # check perpendicular
        self.assertAlmostEqual(np.dot(b - a, mapped[0] - p[0])[0], 0.)
        self.assertAlmostEqual(np.dot(b - a, mapped[1] - p[1])[0], 0.)

        
        
    def test_line_coord_dist_01(self):
        m1 = np.array([-0.0019042, -0.0018468])
        m2 = np.array([0.14588244, 0.00631537])
        a = m1.reshape(1,2)
        b = m2.reshape(1,2)
        delta = np.array([0.01,0.01])
        p = np.array([m1-delta, m1+delta, m2+delta])
        _s, d = support.line_coord_dist(p, a, b)
        # print('s', _s)
        # print('d', d)
        length = np.linalg.norm(m2-m1)
        self.assertAlmostEqual(_s[0], 0.0)
        self.assertTrue(_s[1] > 0.0 and _s[1] < length)
        self.assertAlmostEqual(_s[2], length)


if __name__ == "__main__":
    unittest.main()
