"""
Example calculation: Lorenz 96
"""

import numpy as np
import matplotlib.pyplot as plt

from l96 import l96, l96jac
from pyLE import computeLE

plotspec = False

D = 20
np.random.seed(46895045)
x0 = 20.0*(np.random.rand(D) - 0.5)
p = 8.17 * np.ones(D)

ttrans = np.arange(0.0, 100.0, 0.001)
#ttrans = None
t = np.arange(0.0, 100.0, 0.001)

LE = computeLE(l96, l96jac, x0, t, p, ttrans)
print LE[-1]

if plotspec:
    # Plot Nplot largest Lyapunov exponents, over time
    Nplot = 3
    fig,ax = plt.subplots(1,1,sharex=True)
    fig.set_tight_layout(True)
    #colors = ['orangered', 'limegreen', 'dodgerblue', 'mediumorchid']
    colors = ['dodgerblue']*Nplot
    for i in range(Nplot):
        ax.plot(t[1:], LE[:,i], color=colors[i%4], label=r'$\lambda_{%d}$'%(i+1,), lw=1.2)
    ax.set_ylabel(r'$\lambda$', size=14)
    ax.set_xlabel(r'$t$', size=14)
    ax.set_title('%d largest Lyapunov exponents'%(Nplot,))

    # Plot ALL Lyapunov exponents, over time
    Nplot = LE.shape[1]
    fig,ax = plt.subplots(1,1,sharex=True)
    fig.set_tight_layout(True)
    #colors = ['orangered', 'limegreen', 'dodgerblue', 'mediumorchid']
    #colors = ['dodgerblue']*Nplot
    for i in range(Nplot):
        ax.plot(t[1:], LE[:,i], color="dodgerblue", label=r'$\lambda_{%d}$'%(i+1,), lw=1.2)
    ax.set_ylabel(r'$\lambda$', size=14)
    ax.set_xlabel(r'$t$', size=14)
    ax.set_title('All Lyapunov exponents')

    # Plot entire spectrum at final time
    fig,ax = plt.subplots(1, 1)
    fig.set_tight_layout(True)
    for l in LE[-1, :]:
        ax.axvline(l, lw=2, color="dodgerblue")
    #ind = np.arange(D) + 1
    #width = 0.6
    #ax.bar(ind, LE[-1, :], width=width, align='center', color='dodgerblue')
    #ax.set_xlim(ind[0] - width/2.0, ind[-1] + width/2.0)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(r"$\lambda$")
    ax.set_title("LE Spectrum")

    plt.show()

# Save the spectrum
np.savetxt("D20_p8p17_spectrum.dat", np.sort(LE[-1, :])[::-1])
