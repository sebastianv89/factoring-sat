# count variables, clauses and average clause size per instance

import sys
import numpy as np

for line in sys.stdin:
    if line.startswith('#'):
        continue
    n, p, q, pq = map(int, line.split())
    instance = '{:03}_{}_{}.dimacs'.format(n, p, q)
    v, c, cl = None, None, 0
    try:
        cls = []
        with open(instance) as f:
            for iline in f:
                if iline.startswith('c'):
                    continue
                elif iline.startswith('p'):
                    _, _, v, c = iline.split()
                else:
                    cls.append(len(iline.split())-1)
        cl = np.mean(cls)
    except Exception as e:
        print(e, file=sys.stderr)
        continue
    if v is None or c is None:
        raise Exception('Invalid {}'.format(instance))
    print('{} {} {} {} {} {} {}'.format(n, p, q, pq, v, c, cl))
