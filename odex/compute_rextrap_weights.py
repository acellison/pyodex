import sympy

def _vander(x,n):
    V = sympy.ones(n,len(x))
    for ii in range(1,n):
        for jj in range(len(x)):
            V[ii,jj] = x[jj]*V[ii-1,jj]
    return V

def compute_rextrap_weights(ndep, nfree=None, cfree=None):
    """Compute the Richardson extrapolation weights corresponding to the dependent
       step count sequence, given weights for the free step count sequence.
       :param ndep: dependent step count sequence
       :param nfree: free step count sequence
       :param cfree: free extrapolation weights
    """
    b = sympy.zeros(len(ndep),1)
    b[0] = 1

    ndep = sympy.Array(ndep)
    Vdep = _vander(ndep.applyfunc(lambda x:1/x**2),len(ndep))

    if nfree is None and cfree is None:
        rhs = b
    else:
        if len(cfree) != len(nfree):
            raise ValueError('Free step count sequence length must match free weight length')

        nfree = sympy.Array(nfree)
        Vfree = _vander(nfree.applyfunc(lambda x:1/x**2),len(ndep))
        cfree = sympy.Matrix(sympy.sympify(cfree))
        rhs = b-Vfree*cfree

    return sympy.flatten(Vdep.inv()*rhs)

