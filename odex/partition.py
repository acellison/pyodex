
def _try_partition(a, k, maxheight):
    """Partition an array into k bins
       :param a: array of values to partition.  must be sorted descending
       :param k: number of bins to partition the values
       :param maxheight: maximum bin height
    """
    n = len(a)
    bins = [[] for ii in range(k)]
    sums = [0]*k
    used = [0]*n
    
    for ii in range(k):
        for jj in range(n):
            if not used[jj]:
                tmp = sums[ii]+a[jj]
                if tmp <= maxheight:
                    used[jj] = 1
                    sums[ii] = tmp
                    bins[ii].append(a[jj])
    return all(used), bins

def partition(a, k, maxheight=None):
    """Partition an array into k bins
       :param a: array of values to partition
       :param k: number of bins to partition the values
       :param maxheight: maximum bin height
    """
    a = sorted(a, reverse=True)
    if maxheight is None:
        maxheight = a[0]

    fits, bins = _try_partition(a, k, maxheight)
    if not fits:
        raise ValueError('Cannot partition into k bins without exceeding max bin height')

    return bins

def equipartition(a):
    """Partition an array into the fewest bins such that the
       sum of each bin does not exceed the max element of the array
       :param a: array of values to partition
    """
    a = sorted(a, reverse=True)
    maxheight = a[0]

    def ceildiv(a, b):
        return -(-a // b)
    first = ceildiv(sum(a), maxheight)
    for k in range(first,len(a)+1):
        fits, bins = _try_partition(a, k, maxheight)
        if fits:
            return bins

