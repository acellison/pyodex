import threading
from .worker import Worker

class ThreadPool(object):
    def __init__(self, fns):
        """Initialize the thread pool with functions to be evaluated in parallel
           :param fns: Target functions for the thread pool.  One thread per 
                       function is created, and all are trigger when notify()
                       is called on the pool.
        """
        self._fns = fns
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._counter = 0
        def callback(data):
            with self._lock:
                self._counter -= 1
                if self._counter == 0:
                    self._event.set()
        self._workers = [Worker(fn, callback=callback) for fn in fns]

    def __del__(self):
        self.join()

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
            if self._counter != 0:
                raise ValueError('Workers still working!')
            self._counter += len(self._fns)
        for worker in self._workers:
            worker.notify()

    def data(self):
        """Access the worker thread data.  Blocks until the workers have all
           completed their proecessing.
        """
        self._event.wait()
        self._event.clear()
        return [worker.data() for worker in self._workers]

