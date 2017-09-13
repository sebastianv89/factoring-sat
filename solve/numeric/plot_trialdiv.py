from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import sys

def rsquared(y, fit):
    y_mean = np.mean(y)
    ss_tot = np.sum((y-y_mean)**2)
    ss_res = np.sum((y-fit)**2)
    return 1.0 - ss_res/ss_tot

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
ts = [y for x, y in times]

xfit = [x for x, y in times if x >= 25]
ys = [np.median(y) for x, y in times if x >= 25]

ycoefs = poly.polyfit(xfit, np.log2(ys), 1)
yfit = poly.Polynomial(ycoefs)
yrsquared = rsquared(np.log2(ys), yfit(xfit))
print('yfit: {} (r^2: {})'.format(yfit, yrsquared), file=sys.stderr)

plt.plot(xfit, np.exp2(yfit(xfit)), 'g')
plt.xlim([xs[0], xs[-1]])
plt.boxplot(ts, positions=xs)
ax = plt.gca()
ax.set_xticks([xs[0]] + list(range(10, 81, 10)) + [xs[-1]])
ax.set_xticklabels([xs[0]] + list(range(10, 81, 10)) + [xs[-1]])
plt.yscale('log')
plt.xlabel('Semiprime length (bits)')
plt.ylabel('Time (seconds)')
plt.savefig(sys.stdout.buffer, bbox_inches='tight')
