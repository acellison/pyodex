import dill as pickle
import os
import numpy as np
import multiprocessing as mp
import cProfile
from .shared_state import SharedState


def worker_dispatch(worker, *args):
    return worker.run(*args)

class WorkerProcess(object):
    worker_count = 0
    def __init__(self, target, system, state, callback=None, profile=False):
        """Construct the worker process with a target function and optional
           arguments and callback.  When notified, the target function is
           evaluated, then the callback is called with the result.  This enables
           responding to function completion.
           :param target: Function to be called when notified.
           :param system: Time stepper system.
           :param callback: Callback function called as callback(data) with
                            data the returned value of the target function.
        """
        self._event = mp.Event()
        self._fn = target
        self._exit = mp.Value('i',0)
        if callback is None:
            def callback(data): pass
        self._queue = mp.Queue()
        self._state = state
        system = pickle.dumps(system)

        if profile:
            def target(*args):
                cProfile.runctx('worker_dispatch(*args)', globals(), locals(), 'worker_{}.prof'.format(WorkerProcess.worker_count))
            WorkerProcess.worker_count += 1
            args=(self, self._queue, callback, system)
        else:
            target = self.run
            args=(self._queue, callback, system)
        self._process = mp.Process(target=target, args=args)
        self._process.start()

    def set_args(self, args):
        """Set the arguments to be passed to the worker's target function"""
        self._queue.put(args)

    def join(self):
        """Join the worker thread to the current thread."""
        self._exit.value = 1
        self.notify()
        self._process.join()

    def notify(self):
        """Notify the worker thread to begin processing."""
        self._event.set()

    def run(self, queue, callback, system):
        """Main run loop for the worker thread."""
        # Unpickle the ODE system
        system = pickle.loads(system)

        while True:
            # Wait for notification to process
            self._event.wait()
            self._event.clear()
            if self._exit.value:
                break

            # Get the state values
            state = self.state()

            # Get the extra arguments off the queue
            args = queue.get()

            # Evaluate the function
            callback(self._fn(system, state, *args))

    def state(self):
        return self._state.value


class ProcessPool(object):
    def __init__(self, fns, system, state, callback=None):
        """Initialize the thread pool with functions to be evaluated in parallel
           :param fns: Target functions for the thread pool.  One thread per 
                       function is created, and all are trigger when notify()
                       is called on the pool.
           :param system: Time stepper system.
           :param state: Time stepper output type.
           :param callback: Method to pass to the worker thread when processing
                            completes.
        """
        self._fns = fns
        self._event = mp.Event()
        self._lock = mp.Lock()
        self._counter = mp.RawValue('i',0)

        if callback is None:
            def callback(data): pass

        def make_callback(ii):
            def cb(data):
                callback(data)
                with self._lock:
                    self._counter.value -= 1
                    if self._counter.value == 0:
                        self._event.set()
            return cb

        self._state = SharedState(state)
        self._workers = [WorkerProcess(fns[ii], system, self._state, callback=make_callback(ii)) for ii in range(len(fns))]

    def set_state(self, state):
        """Set the state to be passed on to the worker."""
        self._state.value = state

    def set_args(self, index, args):
        """Set the argument tuple to be called by the target function at index
           :param index: Index of the worker thread to pass on the arguments.
                         If 'all', args are broadcast to each worker.
           :param args: Tuple of arguments to pass to the worker
        """
        if index == 'all':
            for ii in range(len(self._workers)):
                self._workers[ii].set_args(args)
        else:
            self._workers[index].set_args(args)

    def join(self):
        """Join all worker threads to the current thread."""
        for worker in self._workers:
            worker.join()

    def notify(self):
        """Notify the workers that processing should begin."""
        with self._lock:
            self._counter.value = len(self._workers)
        for worker in self._workers:
            worker.notify()

    def synchronize(self):
        """Blocks until the workers have all completed their proecessing.
        """
        # Wait for the workers to finish
        self._event.wait()
        self._event.clear()

