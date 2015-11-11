"""
Example calculation: Lorenz 96
"""

import numpy as np
import matplotlib.pyplot as plt

from l96 import l96, l96jac
from pyLE import computeLE

D = 5
x0 = 20.0*(np.random.rand(D) - 0.5)
p = 8.17 * np.ones(D)

ttrans = np.arange(0.0, 300.0, 1.0)
t = np.arange(0.0, 300.0, 1.0)

LE = computeLE(l96, l96jac, x0, t, p, ttrans)
print LE[-1]

# Plot Nplot largest Lyapunov exponents, over time
Nplot = 4
fig,ax = plt.subplots(1,1,sharex=True)
fig.set_tight_layout(True)
colors = ['orangered', 'limegreen', 'dodgerblue', 'mediumorchid']
for i in range(Nplot):
    ax.plot(t[1:], LE[:,i], color=colors[i%4], label=r'$\lambda_{%d}$'%(i+1,), lw=1.2)
ax.set_ylabel(r'$\lambda$', size=14)
ax.set_xlabel(r'$t$', size=14)
ax.set_title('%d largest Lyapunov exponents'%(Nplot,))
plt.show()
