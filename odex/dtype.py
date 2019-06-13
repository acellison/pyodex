import numpy as np

def dtype(x):
    if hasattr(x,'dtype'):
        return x.dtype
    else:
        return np.dtype(type(x))

