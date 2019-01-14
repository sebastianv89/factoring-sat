import sys
from collections import defaultdict
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from scipy import stats
from sympy.ntheory.factor_ import smoothness
import gmpy2

def rsquared(y, fit):
    y_mean = np.mean(y)
    ss_tot = np.sum((y-y_mean)**2)
    ss_res = np.sum((y-fit)**2)
    return 1.0 - ss_res/ss_tot

encoding = 'long'
if len(sys.argv) > 1:
    encoding = sys.argv[1]

data = defaultdict(list)
for line in sys.stdin:
    if line.startswith('#'):
        continue
    n, p, q, pq, t = line.split()
    n, p, q, pq = map(int, [n, p, q, pq])
    t = float(t)
    data[n].append(((p,q), t))
    
colors = ['b', 'g', 'r', 'k', 'm', 'y', 'c',]

functions = [(lambda p, q: abs(p-q), '$|p-q|$', 'diff'),
             (lambda p, q: np.log2(p*q), '$\log_2 N$', 'size'),
             (lambda p, q: gmpy2.popcount(p*q), 'Hamming weight $N$', 'hwN'),
             (lambda p, q: gmpy2.popcount(p), 'Hamming weight $p$', 'hwp'),
             (lambda p, q: gmpy2.popcount(q), 'Hamming weight $q$', 'hwq'),
             (lambda p, q: gmpy2.hamdist(p, q), 'Hamming weight $p \oplus q$', 'hdpq'),
             (lambda p, q: gmpy2.hamdist(p, q), 'Hamming weight $p \oplus q$', 'hdpq'),
             (lambda p, q: smoothness(p-1)[0], 'Smoothness of $p-1$', 'smp'),
             (lambda p, q: smoothness(q-1)[0], 'Smoothness of $q-1$', 'smq'),
             (lambda p, q: smoothness(p-1)[0] + smoothness(q-1)[0], 'Smoothness of $p-1$ and $q-1$', 'smpq'),
          ]

for f, desc, fout in functions:
    plt.clf()
    for n, d in data.items():
        if n < 35 or n >= 40:
            continue
        pqs, ts = zip(*d)
        xs = [f(p,q) for p, q in pqs]
        tcoefs = poly.polyfit(xs, ts, 1)
        tfit = poly.Polynomial(tcoefs)
        trsquared = rsquared(ts, tfit(np.array(xs)))
        plt.plot(xs, ts, '.', label='$n={}; r^2={:.2f}$'.format(n, trsquared), color=colors[n%len(colors)])
        linx = np.linspace(min(xs), max(xs), 100)
        plt.plot(linx, tfit(linx), color=colors[n%len(colors)])
    plt.xlabel(desc)
    plt.ylabel('Time (seconds)')
    plt.legend(loc='best')
    plt.savefig(encoding + '_' + fout + '.png', bbox_inches='tight')
