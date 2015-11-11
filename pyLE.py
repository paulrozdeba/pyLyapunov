"""
Calculate the Lyapunov exponents for a set of ODEs using the method described 
in Sandri (1996), through the use of the variational matrix.
"""

import numpy as np
from scipy.integrate import ode

def computeLE(f, fjac, x0, t, p=(), ttrans=None, method='dop853'):
    """
    Computes the global Lyapunov exponents for a set of ODEs.
    f - ODE function. Must take arguments like f(t, x, p) where x and t are 
        the state and time *now*, and p is a tuple of parameters. If there are 
        no model paramters, p should be set to the empty tuple.
    x0 - Initial position for calculation. Integration of transients will begin 
         from this point.
    t - Array of times over which to calculate LE.
    p - (optional) Tuple of model parameters for f.
    fjac - Jacobian of f.
    ttrans - (optional) Times over which to integrate transient behavior.
             If not specified, assumes trajectory is on the attractor.
    method - (optional) Integration method to be used by scipy.integrate.ode.
    """
    D = len(x0)
    N = len(t)
    Ntrans = len(ttrans)
    dt = t[1] - t[0]

    def dPhi_dt(t, Phi, x):
        """ The variational equation """
        rPhi = np.reshape(Phi, (D,D))
        rdPhi = np.dot(fjac(t, x, p), rPhi)
        return rdPhi.flatten()

    def dSdt(t, S):
        """
        Differential equations for combined state/variational matrix
        propagation. This combined state is called S.
        """
        x = S[:D]
        Phi = S[D:]
        return np.append(f(t,x,p), dPhi_dt(t,Phi,x))

    # set up integrator
    itg = ode(dSdt)
    itg.set_integrator(method)

    # integrate transient behavior
    Phi0 = np.eye(D, dtype='float').flatten()
    S0 = np.append(x0, Phi0)
    if ttrans is not None:
        itg.set_initial_value(S0, ttrans[0])
        Strans = np.zeros((Ntrans,D*(D+1)), dtype='float')
        Strans[0] = S0
        for i,tnext in enumerate(ttrans[1:]):
            itg.integrate(tnext)
            Strans[i+1] = itg.y
            # perform QR decomposition on Phi
            rPhi = np.reshape(Strans[i,D:], (D,D))
            Q,R = np.linalg.qr(rPhi)
            itg._y = np.append(Strans[i,:D],Q.flatten())
        S0 = np.append(Strans[-1,:D], Phi0)

    # start LE calculation
    LE = np.zeros((N-1,D), dtype='float')
    itg.set_initial_value(S0, t[0])
    Ssol = np.zeros((N,D*(D+1)), dtype='float')
    Ssol[0] = S0
    for i,tnext in enumerate(t[1:]):
        itg.integrate(tnext)
        Ssol[i+1] = itg.y
        # perform QR decomposition on Phi
        rPhi = np.reshape(Ssol[i+1,D:], (D,D))
        Q,R = np.linalg.qr(rPhi)
        itg.set_initial_value(np.append(Ssol[i+1,:D],Q.flatten()), tnext)
        LE[i] = np.abs(np.diag(R))

    # compute LEs
    LE = np.cumsum(np.log(LE),axis=0) / np.tile(t[1:],(D,1)).T
    return LE
