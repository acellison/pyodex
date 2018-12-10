from odex import partition
import numpy as np
   
def print_best_partition(a, k):
    print('Partitioning {0} into {1} partitions'.format(a, k))
    p = partition.partition(a, k)

    found = np.zeros(len(a))
    for sub in p:
        for val in sub:
            found[a.index(val)] += 1
    if not np.all(found):
        for ii in range(len(found)):
            if not found[ii]:
                print(a[ii])
    assert np.all(found)
    assert int(np.sum(found)) == len(a)
    print('The best partitioning is {}\n    With heights {}\n'.format(p, list(map(sum, p))))


def main():
    a = list(range(2,34,2))
    for k in range(2,len(a)+1):
        print_best_partition(a, k)


if __name__=='__main__':
    main()

