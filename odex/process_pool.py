import dill as pickle
import multiprocessing as mp
import threading
from .worker import Worker


class WorkerProcess(object):
    def __init__(self, target, args=(), callback=None):
        """Construct the worker thread with a target function and optional
           arguments and callback.  When notified, the target function is
           evaluated, with the result accessible by the data() member function.
           Utilize the callback argument to receive the target function result
           or otherwise respond to function evaluation completion.
           :param target: Function to be called when notified.
           :param args: Tuple of arguments to pass to the target function
           :param callback: Callback function called as callback(data) with
                            data the returned value of the target function.
        """
        self._event = mp.Event()
        self._fn = target
        self._exit = mp.Value('i',0)
        if callback is None:
            def callback(data): pass
        self._queue = mp.Queue()
        self._process = mp.Process(target=self.run, args=(self._queue, callback,))
        self._process.start()

    def __del__(self):
        self.join()

    def set_args(self, args):
        """Set the arguments to be passed to the worker's target function"""
        # FIXME: use shared array for state, don't pass constants around
        s = pickle.dumps(args[0])
        args = (s, *args[1:])
        self._queue.put(args)

    def join(self):
        """Join the worker thread to the current thread."""
        self._exit.value = 1
        self.notify()
        self._process.join()

    def notify(self):
        """Notify the worker thread to begin processing."""
        self._event.set()

    def run(self, queue, callback):
        """Main run loop for the worker thread."""
        while True:
            self._event.wait()               # Wait for notification
            self._event.clear()              # Clear the notification
            if self._exit.value: break       # Break on exit flag

            # FIXME: expects first arg to (exist and) be pickled
            args = queue.get()
            system = pickle.loads(args[0])
            callback(self._fn(system, *args[1:]))  # Evaluate the function


class ProcessPool(object):
    def __init__(self, fns):
        """Initialize the thread pool with functions to be evaluated in parallel
           :param fns: Target functions for the thread pool.  One thread per 
                       function is created, and all are trigger when notify()
                       is called on the pool.
        """
        self._fns = fns
        self._event = threading.Event()
        self._queue = mp.Queue()

        def make_callback(ii):
            def callback(data):
                self._queue.put((ii, data))
            return callback

        self._workers = [WorkerProcess(fns[ii], callback=make_callback(ii)) for ii in range(len(fns))]

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
        for worker in self._workers:
            worker.notify()

    def data(self):
        """Access the worker thread data.  Blocks until the workers have all
           completed their proecessing.
        """
        num_processes = len(self._fns)

        # Wait for the workers to finish
        results = [None]*num_processes
        for ii in range(num_processes):
            results[ii] = self._queue.get()

        # Sort the resulting list by worker index
        output = [None]*num_processes
        for ii in range(len(self._fns)):
            output[results[ii][0]] = results[ii][1]
        return output

