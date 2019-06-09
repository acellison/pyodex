
def partition(a, k, maxheight=None):
    """Partition an array into k bins
       :param a: array of values to partition
       :param k: number of bins to partition the values
       :param maxheight: maximum bin height
    """
    a = sorted(a,reverse=True)
    n = len(a)
    if maxheight is None:
        maxheight = a[0]

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

    if not all(used):
        raise ValueError('Cannot partition into k bins without exceeding max bin height')

    return bins

