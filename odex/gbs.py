import numpy as np


class GBS(object):
    def __init__(self, n):
        """Initialize the GBS time stepper
           :param n: number of subintervals.  must be even
        """
        if n%2 != 0:
            raise ValueError('n must be even!')
        self._n = n

    def step(self, system, state, t, dt):
        """Step the system forward one time step.
           :param system: callable ODE to time step, where y\'=system(t,state)
           :param state: state of the system
           :param t: time of the evaluation
           :param dt: time step size
        """
        n = self._n
        dt = dt/n

        # Initial state
        s1 = state

        # Forward Euler step
        s2 = state+dt*system(t,state)

        # Leap Frog iteration
        for ii in range(n):
            s0 = s1
            s1 = s2
            s2 = s0 + 2*dt*system(t,s1)

        # Smoothing step
        return .25*(s0+2*s1+s2)

