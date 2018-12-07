import numpy as np
import tempfile


class State(object):
    class Value(object):
        def __init__(self, value):
            self._file = tempfile.TemporaryFile()
            self._value = np.memmap(self._file, dtype=float, mode='r+', shape=np.shape([value]))

        @property
        def value(self):
            return self._value[0]

        @value.setter
        def value(self, v):
            self._value[0] = v

    class Array(object):
        def __init__(self, value):
            self._file = tempfile.TemporaryFile()
            self._value = np.memmap(self._file, dtype=float, mode='r+', shape=np.shape(value))

        @property
        def value(self):
            return self._value

        @value.setter
        def value(self, v):
            self._value[:] = v

    def __init__(self, value):
        if hasattr(value, '__len__'):
            self.__class__ = State.Array
        else:
            self.__class__ = State.Value
        self.__init__(value)

