"""
Example calculation: Lorenz 96
"""

import numpy as np
from l96 import l96, l96jac
from pyLE import computeLE

D = 5
x0 = 20.0*(np.random.rand(D) - 0.5)
p = 8.17# * np.ones(D)

ttrans = np.arange(0.0, 100.0, 1.0)
t = np.arange(0.0, 1000.0, 1.0)

LE = computeLE(l96, l96jac, x0, t, p, ttrans)
print LE[-1]
