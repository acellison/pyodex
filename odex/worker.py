import threading


class Worker(object):
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
        self._event = threading.Event()
        self._fn = target
        self._args = args
        self._data = None
        self._exit = False
        self._thread = threading.Thread(target=self.run)
        self._thread.start()
        if callback is None:
            def cb(data): pass
            self._callback = cb
        else:
            self._callback = callback

    def __del__(self):
        self.join()

    def set_args(self, args):
        """Set the arguments to be passed to the worker's target function"""
        self._args = args

    def join(self):
        """Join the worker thread to the current thread."""
        self._exit = True
        self.notify()
        self._thread.join()

    def notify(self):
        """Notify the worker thread to begin processing."""
        self._event.set()

    def run(self):
        """Main run loop for the worker thread."""
        while True:
            self._event.wait()                  # Wait for notification
            self._event.clear()                 # Clear the notification
            if self._exit: break                # Break on exit flag
            self._data = self._fn(*self._args)  # Evaluate the function
            self._callback(self._data)          # Call the callback function

    def data(self):
        """Access the data returned by a call to the target function."""
        return self._data

