pyLE contains just one function so far, computeLE.  It computes the Lyapunov 
exponents for a set of ODEs.

See the example files for guidance on how to run the calculation.  Importantly, 
you need to define the ODEs (f) and their Jacobian (fjac) in a separate 
Python module.  f and fjac should be in "scipy.integrate.ode style", which
means that they take their arguments in the *very* specific order:
    t, x, *p
where t and x are the time and position *now*, and any extra parameters get 
passed in after t and x (not in a tuple, but as extra individual arguments).
However, the parameters should be passed to computeLE in a tuple!

Once you have this, define an array of times t over which to perform the 
calculation; additionally, if you want to integrate out transient behavior 
first, then define an array of times ttrans and pass it to computeLE.
Finally, decide on an initial position for the calculation (or the initial 
position for the integration of the transient behavior, if ttrans is specified).

computeLE returns an array of exponents *at all times t*.  The last one should, 
in some sense, be the most accurate, since at each time the exponents are 
averaged up to that time point.