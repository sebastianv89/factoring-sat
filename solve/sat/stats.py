import numpy as np
import sys

stat = 'median'
if len(sys.argv) > 1:
    stat = sys.argv[1]

for line in sys.stdin:
    if line.startswith('#'):
        print(line[:-1] + ' ' + stat)
        continue
    words = line.split()
    m, p, q, pq = map(int, words[:4])
    times = list(map(float, words[4:]))
    if stat == 'median':
        y = np.median(times)
    elif stat == 'std':
        y = np.std(times)
    elif stat == 'min':
        y = min(times)
    else:
        raise Exception('Unknown statistic {}'.format(stat))
    print('{} {} {} {} {}'.format(m, p, q, pq, y))
