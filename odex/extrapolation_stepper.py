import numpy as np
from .process_pool import ProcessPool
from .shared_state import SharedState
from .partition import partition

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
        for stepper in self._steppers:
            stepper.resize(state)
        self._steps = np.array(steps)[indices]
        self._weights = np.array(weights)[indices]
        self._system = system

        self._num_cores = num_cores
        if num_cores > 1:
            self._evalfn = self._evaluate_parallel
            self._initialize_threads(num_cores, system, state)
        else:
            self._evalfn = self._evaluate_serial
            self._pool = None

    def join(self):
        """Block while waiting for all threads in the pool to join.
        """
        if self._pool is not None:
            self._pool.join()

    def step(self, state, t, dt, n, dense_output=True, observer=None):
        """Time step the extrapolation scheme n times, returning output from each time point.
           :param system: callable ODE to time step, where y\'=system(t,state)
           :param state: state of the system
           :param t: time of the evaluation
           :param dt: time step size
           :param n: number of time steps
           :param dense_output: if true, return the output after each time step.  Otherwise,
                                just return the final output after n time steps.
           :param observer: Callable with signature observer(t,state).  Can be used to store
                            output after each time step, or as a movie plotting utility.
        """
        evalfn  = self._evalfn
        weights = self._weights
        system  = self._system
        if dense_output:
            output  = np.empty((n, *np.shape(state)))
        state = np.copy(state)

        for ii in range(n):
            # Compute the output at time t+dt across all threads
            results = evalfn(system, state, t, dt)

            # Extrapolate the outputs
            if np.ndim(state) >= 1:
                np.dot(np.moveaxis(results,0,-1), weights, out=state)
            else:
                state = np.dot(weights, results)

            # If dense output is requested, store it
            if dense_output: output[ii] = state

            # If we have an observer, call it
            if observer: observer(t, state)

            # Increment the time index
            t = t+dt
        if not dense_output:
            output = state
        return output

    def _evaluate_serial(self, system, state, t, dt):
        """Evaluate the time steppers in the current thread."""
        fval0 = system(t,state)
        return [stepper.step(system, state, t, dt, fval0) for stepper in self._steppers]

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

        self._outputs = [SharedState(state) for ii in range(num_steppers)]

        partitions = partition(self._steps, num_cores)

        def make_worker_target_fn(ii):
            part = partitions[ii]
            steps = list(self._steps)
            inds = [steps.index(p) for p in part]
            def eval(system, state, t, dt):
                fval0 = system(t,state)
                for jj in range(len(inds)):
                    ind = inds[jj]
                    stepper = self._steppers[ind]
                    self._outputs[ind].value = stepper.step(system, state, t, dt, fval0)
            return eval

        fns = [make_worker_target_fn(ii) for ii in range(num_cores)]
        self._pool = ProcessPool(fns, system, state)

