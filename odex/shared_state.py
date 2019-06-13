import multiprocessing as mp
import numpy as np
import tempfile
from .dtype import dtype


class SharedState_Memmap(object):
    class Value(object):
        def __init__(self, value):
            self._file = tempfile.TemporaryFile()
            self._value = np.memmap(self._file, dtype=dtype(value), mode='r+', shape=np.shape([value]))

        @property
        def value(self):
            return self._value[0]

        @value.setter
        def value(self, v):
            self._value[0] = v

    class Array(object):
        def __init__(self, value):
            self._file = tempfile.TemporaryFile()
            self._value = np.memmap(self._file, dtype=dtype(value), mode='r+', shape=np.shape(value))

        @property
        def value(self):
            return self._value

        @value.setter
        def value(self, v):
            self._value[:] = v

    def __init__(self, value):
        if hasattr(value, '__len__'):
            self.__class__ = SharedState.Array
        else:
            self.__class__ = SharedState.Value
        self.__init__(value)


class SharedState_RawValue(object):
    class Value(object):
        # Fixme:  make data-type aware
        def __init__(self, value):
            shape = np.shape(value)
            self._rawvalue = mp.RawArray('d', 1)
            self._value = np.array(self._rawvalue, copy=False)

        @property
        def value(self):
            return self._value[0]

        @value.setter
        def value(self, v):
            self._value[0] = v

    class Array(object):
        # Fixme:  make data-type aware
        def __init__(self, value):
            shape = np.shape(value)
            self._rawvalue = mp.RawArray('d', np.reshape(value,np.prod(shape)))
            self._value = np.array(self._rawvalue, copy=False).reshape(shape)

        @property
        def value(self):
            return self._value

        @value.setter
        def value(self, v):
            self._value[:] = v

    def __init__(self, value):
        if hasattr(value, '__len__'):
            self.__class__ = SharedState.Array
        else:
            self.__class__ = SharedState.Value
        self.__init__(value)


SharedState = SharedState_Memmap

# Using mp.RawValue as the underlying process data transfer scheme seems to be
# more performant the the numpy.memmap.  Since it is not yet data-type aware
# we disable it for now.
#SharedState = SharedState_RawValue

