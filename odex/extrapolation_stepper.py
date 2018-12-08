import numpy as np
from .process_pool import ProcessPool
from .state import State


class ExtrapolationStepper(object):
    def __init__(self, steppers, steps, weights, system, state, num_cores=1):
        """Initialize the ExtrapolationStepper
           :param steppers: list of underlying time steppers
           :param steps: step counts for each stepper in the extrapolation scheme
           :param weights: weights for each stepper in the scheme
           :param system: ODE system to time step
           :param state: state type returned by the time stepper
           :param num_cores: number of cores on which to evaluate the scheme
        """
        if len(steppers) != len(steps) or len(steppers) != len(weights):
            raise ValueError('number of steppers, step counts, and weights must all match!')

        # Sort the time steppers by step counts
        indices = np.argsort(steps)
        self._steppers = [steppers[index] for index in indices]
        self._steps = np.array(steps)[indices]
        self._weights = np.array(weights)[indices]

        self._num_cores = num_cores
        if num_cores > 1:
            self._evalfn = self._evaluate_parallel
            self._initialize_threads(num_cores, system, state)
        else:
            self._evalfn = self._evaluate_serial
            self._pool = None

    def __del__(self):
        self.join()

    def join(self):
        """Block while waiting for all threads in the pool to join.
        """
        if self._pool:
            self._pool.join()

    def step(self, system, state, t, dt, n):
        """Time step the extrapolation scheme n times, returning output from each time point.
           :param system: callable ODE to time step, where y\'=system(t,state)
           :param state: state of the system
           :param t: time of the evaluation
           :param dt: time step size
           :param n: number of time steps
        """
        evalfn  = self._evalfn
        weights = self._weights
        output  = np.empty((n, *np.shape(state)))
        for ii in range(n):
            results    = evalfn(system, state, t, dt)
            state      = np.dot(weights, results)
            output[ii] = state
            t          = t+dt
        return output

    def _evaluate_serial(self, system, state, t, dt):
        """Evaluate the time steppers in the current thread."""
        return [stepper.step(system, state, t, dt) for stepper in self._steppers]

    def _evaluate_parallel(self, system, state, t, dt):
        """Evaluate the time steppers in parallel across the pool."""
        # Set the arguments to the stepper calls
        self._pool.set_state(state)
        self._pool.set_args('all', (t, dt))

        # Notify the threads to process
        self._pool.notify()

        # Access the pool data, blocking until synchronized
        self._pool.synchronize()

        # Merge the thread worker results into a single array
        return [output.value for output in self._outputs]

    def _initialize_threads(self, num_cores, system, state):
        """Initialize the thread pool, balancing the load across each thread."""
        fns = [stepper.step for stepper in self._steppers]
        num_steppers = len(self._steppers)

        self._outputs = [State(state) for ii in range(num_steppers)]

        partitions = self._partition(self._steps, num_cores)
        print(partitions)

        def make_worker_target_fn(ii):
            partition = partitions[ii]
            steps = list(self._steps)
            inds = [steps.index(p) for p in partition]
            def eval(*args):
                results = [self._steppers[ind].step(*args) for ind in inds]
                for jj in range(len(results)):
                    self._outputs[inds[jj]].value = results[jj]
            return eval

        fns = [make_worker_target_fn(ii) for ii in range(num_cores)]
        self._pool = ProcessPool(fns, system, state)

    def _partition(self, a, k):
        """Pseudo-Minimax partition of the thread step counts for load balance."""
        a = sorted(a)
        n = len(a)
        nmodk = n%k
        b = a[nmodk:]

        if len(b)//(2*k) == 1:
            partitions = [[bi] for bi in b]
        else:
            partitions = []
            for ii in range(k):
                m  = len(b)//(2*k)
                i1 =  ii   *m
                i2 = (ii+1)*m
                i3 = len(b)-(ii+1)*m
                i4 = len(b)- ii   *m
                inds = list(range(i1,i2))+list(range(i3,i4))
                partitions.append([b[ind] for ind in inds])

        for ii in range(nmodk):
            partitions[ii].insert(0,a[nmodk-1-ii])

        return partitions

