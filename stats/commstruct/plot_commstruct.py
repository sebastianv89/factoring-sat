import sys
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy import stats

mpl.style.use('classic')

def rsquared(y, fit):
    y_mean = np.mean(y)
    ss_tot = np.sum((y-y_mean)**2)
    ss_res = np.sum((y-fit)**2)
    return 1.0 - ss_res/ss_tot

data = defaultdict(list)
for line in sys.stdin:
    if line.startswith('#'):
        continue
    n, p, q, pq, mod, t = line.split()
    n, p, q, pq = map(int, [n, p, q, pq])
    mod, t = map(float, [mod, t])
    data[n].append((mod, t))

colors = ['b', 'g', 'r', 'k', 'm', 'y', 'c']
for n, d in data.items():
    if n < 35 or n >= 40:
        continue
    xs, ys = zip(*d)
    ycoefs = poly.polyfit(xs, ys, 1)
    yfit = poly.Polynomial(ycoefs)
    yrsquared = rsquared(ys, yfit(np.array(xs)))
    plt.plot(xs, ys, '.', label='$n={}; r^2={:.2f}$'.format(n, yrsquared), color=colors[n%len(colors)])
    linx = np.linspace(min(xs), max(xs), 100)
    plt.plot(linx, yfit(linx), color=colors[n%len(colors)])

plt.xlabel('Instance modularity (Q)')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig(sys.stdout.buffer, bbox_inches='tight', format='pdf')
