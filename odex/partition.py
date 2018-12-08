
def partition(a, k):
    """Partition an array in a pseudo-minimax way.
       FIXME: This is not yet optimal 
    """
    a = sorted(a)
    n = len(a)

    nmodk = n%(2*k)
    if nmodk == 0:
        partitions = []
        m = n//(2*k)
        for ii in range(k):
            ind1 =    ii   *m
            ind2 =   (ii+1)*m
            ind3 = n-(ii+1)*m
            ind4 = n- ii   *m
            inds = list(range(ind1,ind2))+list(range(ind3,ind4)) 
            partitions.append([a[ind] for ind in inds])
    else:
        partitions = partition(a[nmodk:],k)
        b = a[:nmodk]
        if len(b)<k:
            for ii in range(len(b)):
                partitions[ii].insert(0,b[nmodk-1-ii])
        else:
            for ii in range(k):
                partitions[ii].insert(0,b[nmodk-1-ii])
            for ii in range(nmodk-k):
                partitions[k-1-ii].insert(0,b[ii])
    
    return partitions


