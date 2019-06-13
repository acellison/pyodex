import numpy as np
from .process_pool import ProcessPool
from .shared_state import SharedState
from .partition import equipartition
from .dtype import dtype


class ExtrapolationStepper(object):
    def __init__(self, steppers, steps, weights, system, state, parallel=True, isbn=None):
        """Initialize the ExtrapolationStepper
           :param steppers: list of underlying time steppers
           :param steps: step counts for each stepper in the extrapolation scheme
           :param weights: weights for each stepper in the scheme
           :param system: ODE system to time step
           :param state: value of type returned by the time stepper
           :param isbn: normalized imaginary stability boundary
           :param parallel: run the algorithm distributed across cores
        """
        if len(steppers) != len(steps) or len(steppers) != len(weights):
            raise ValueError('number of steppers, step counts, and weights must all match!')

        self._isbn = isbn

        # Sort the time steppers by step counts
        indices = np.argsort(steps)
        self._steppers = [steppers[index] for index in indices]
        for stepper in self._steppers:
            stepper.resize(state)
        self._steps = np.array(steps)[indices]
        self._weights = np.array(weights)[indices]
        self._system = system

        # Set up the evaluation context
        if parallel:
            self._evalfn = self._evaluate_parallel
            self._initialize_threads(system, state)
        else:
            self._evalfn = self._evaluate_serial
            self._pool = None

    def join(self):
        """Block while waiting for all threads in the pool to join.
        """
        if self._pool is not None:
            self._pool.join()

    def isbn(self):
        """Return the Normalized Imaginary Stability Boundary of the scheme
           :returns: float, normalized to [0,1]
        """
        return self._isbn

    def stepcounts(self):
        """Return the step count sequence for the extrapolation scheme
        """
        return self._steps

    def weights(self):
        """Return the weights corresponding to the scheme's step count sequence
        """
        return self._weights

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
        system  = self._system
        if dense_output:
            output  = np.empty((n, *np.shape(state)),dtype=dtype(state))
        state = np.copy(state)

        for ii in range(n):
            # Compute the output at time t+dt across all threads
            state = evalfn(system, state, t, dt)

            # Extrapolate the outputs
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
        # Compute the first function value to share across all steppers
        fval0 = system(t,state)

        # Time step each stepper
        results = [stepper.step(system, state, t, dt, fval0) for stepper in self._steppers]

        # Extrapolate the results
        weights = self._weights
        if np.ndim(state) >= 1:
            np.dot(np.moveaxis(results,0,-1), weights, out=state)
        else:
            state = np.dot(weights, results)
        return state

    def _evaluate_parallel(self, system, state, t, dt):
        """Evaluate the time steppers in parallel across the pool."""
        # Set the arguments to the stepper calls
        self._pool.set_state(state)
        self._pool.set_args('all', (t, dt))

        # Notify the threads to process
        self._pool.notify()

        # Access the pool data, blocking until synchronized
        self._pool.synchronize()

        # Combine the partially extrapolated thread worker results
        return sum([output.value for output in self._outputs])

    def _initialize_threads(self, system, state):
        """Initialize the thread pool, balancing the load across each thread."""
        # Compute the optimal partitioning to balance load
        partitions = equipartition(self._steps)

        # Each core shares its output with the main thread
        num_cores = len(partitions)
        self._outputs = [SharedState(state) for ii in range(num_cores)]

        def make_worker_target_fn(ii):
            # Each core will run the function returned from here.
            # Cores run the time steppers specified by the equipartition,
            # and partially extrapolate the data to minimize data sharing
            # across cores.
            part = partitions[ii]
            steps = list(self._steps)
            inds = [steps.index(p) for p in part]

            def eval(system, state, t, dt):
                # Compute the first function value
                fval0 = system(t,state)

                # Zero out the output
                result = np.zeros_like(state)

                # Run the steppers on this core in series
                for jj in range(len(inds)):
                    # Grab the appropriate stepper and extrapolation weight
                    ind = inds[jj]
                    weight = self._weights[ind]
                    stepper = self._steppers[ind]

                    # Evaluate the stepper and apply extrapolation weight
                    result += weight*stepper.step(system, state, t, dt, fval0)

                # Send back the data
                self._outputs[ii].value = result

            return eval

        # Create the worker functions
        fns = [make_worker_target_fn(ii) for ii in range(num_cores)]

        # Instantiate the process pool to manage the workers
        self._pool = ProcessPool(fns, system, state)

