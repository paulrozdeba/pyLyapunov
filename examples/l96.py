"""
Lorenz 96 function definitions for scipy.integrate.ode
"""

import numpy as np

#def l96(t, x, p):
#    D = len(x)
#    dxdt = np.empty(D)
#    for i in range(D):
#        dxdt[i] = x[(i-1)%D] * (x[(i+1)%D] - x[(i-2)%D]) - x[i] + p
#    return dxdt

def l96(t, x, p):
    return np.roll(x,1) * (np.roll(x,-1) - np.roll(x,2)) - x + p

def l96jac(t, x, p):
    D = len(x)
    J = np.zeros((D,D), dtype='float')
    for i in range(D):
        J[i,(i-1)%D] = x[(i+1)%D] - x[(i-2)%D]
        J[i,(i+1)%D] = x[(i-1)%D]
        J[i,(i-2)%D] = -x[(i-1)%D]
        J[i,i] = -1.0
    return J
