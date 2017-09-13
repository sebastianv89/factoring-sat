import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import sys
from collections import defaultdict

def rsquared(y, fit):
    y_mean = np.mean(y)
    ss_tot = np.sum((y-y_mean)**2)
    ss_res = np.sum((y-fit)**2)
    return 1.0 - ss_res/ss_tot

solver, encoding = 'maplecomsps', 'long'
if len(sys.argv) > 1:
    solver = sys.argv[1]
if len(sys.argv) > 2:
    encoding = sys.argv[2]

times = defaultdict(list)
for line in sys.stdin:
    if line.startswith('#'):
        continue
    n, p, q, pq, t = line.split()
    n = int(n)
    t = float(t)
    times[n].append(t)

times = list(sorted(times.items()))
xs = [x for x, y in times]
ys = [y for x, y in times]

# determined from inspection of data
fit_lower_bound = 20
fit_upper_bound = 50
xx = [x for x, y in times if fit_lower_bound <= x and x <= fit_upper_bound]
yy = [np.median(y) for x, y in times if fit_lower_bound <= x and x <= fit_upper_bound]

ycoefs = poly.polyfit(xx, np.log2(yy), 1)
yfit = poly.Polynomial(ycoefs)
yrsquared = rsquared(np.log2(yy), yfit(xx))
print('{} {}'.format(solver, encoding), file=sys.stderr)
print('yfit: {} (r^2: {})'.format(yfit, yrsquared), file=sys.stderr)

plt.plot(xx, np.exp2(yfit(xx)), 'g')
plt.xlim([xs[0], xs[-1]])
plt.boxplot(ys, positions=xs)
ax = plt.gca()
ax.set_xticks([xs[0]] + list(range(10, xs[-1], 10)) + [xs[-1]])
ax.set_xticklabels([xs[0]] + list(range(10, xs[-1], 10)) + [xs[-1]])
plt.yscale('log')
plt.xlabel('Semiprime length (bits)')
plt.ylabel('Time (seconds)')
plt.savefig(sys.stdout.buffer, bbox_inches='tight')
