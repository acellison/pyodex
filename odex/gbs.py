import numpy as np
from .dtype import dtype


class GBS(object):

    def __init__(self, n):
        """Initialize the GBS time stepper
           :param n: number of subintervals.  must be even
        """
        if n%2 != 0:
            raise ValueError('n must be even!')
        self._n = n
        self._state = None

    def resize(self, state):
        shape = (3,)+np.shape(state)
        self._state = np.zeros(shape,dtype=dtype(state))

    def step(self, system, state, t, dt, fval0=None):
        """Step the system forward one time step.
           :param system: callable ODE to time step, where y\'=system(t,state)
           :param state: state of the system
           :param t: time of the evaluation
           :param dt: time step size
           :param fval0: (optional) first function evaluation
        """
        n  = self._n
        t  = float(t)
        dt = float(dt)/n

        s = self._state

        # Initial state
        s[0] = state

        # Forward Euler step
        if fval0 is None:
            s[1] = s[0]+dt*system(t,s[0])
        else:
            s[1] = s[0]+dt*fval0

        # Leap Frog iteration
        indices = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
        cur = 2
        for ii in range(n):
            cur = cur+1 if cur < 2 else 0
            t   = t+dt

            inds = indices[cur]
            s[inds[2]] = s[inds[0]] + 2*dt*system(t,s[inds[1]])

        # Smoothing step
        return .25*(s[inds[0]]+2*s[inds[1]]+s[inds[2]])

