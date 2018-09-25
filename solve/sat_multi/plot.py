import sys
from collections import defaultdict
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

def rsquared(y, fit):
    y_mean = np.mean(y)
    ss_tot = np.sum((y-y_mean)**2)
    ss_res = np.sum((y-fit)**2)
    return 1.0 - ss_res/ss_tot

times = defaultdict(list)
for line in sys.stdin:
    if line.startswith('#'):
        continue
    words = line.split()
    n, seed = int(words[0]), int(words[1])
    time = float(words[2])
    times[n].append(time)

sorted_times = list(sorted(times.items()))
xs = [x for x, _ in sorted_times]
ys = [y for _, y in sorted_times]

# determined from inspection of data
fit_lower_bound = 20
fit_upper_bound = 50
xx = [x for x, y in sorted_times if fit_lower_bound <= x]
yy = [np.median(y) for x, y in sorted_times if fit_lower_bound <= x]

# compute the fit
ycoefs = poly.polyfit(xx, np.log2(yy), 1)
yfit = poly.Polynomial(ycoefs)
yrsquared = rsquared(np.log2(yy), yfit(xx))
print('yfit: {} (r^2: {})'.format(yfit, yrsquared), file=sys.stderr)

# plot the data
plt.boxplot(ys, positions=xs)
plt.plot(xx, np.exp2(yfit(xx)), 'g')

# label data etc.
plt.yscale('log')
ax = plt.gca()
ax.set_xticks([xs[0]] + list(range(10, xs[-1], 10)) + [xs[-1]])
ax.set_xticklabels([xs[0]] + list(range(10, xs[-1], 10)) + [xs[-1]])
plt.xlabel('Semiprime length (bits)')
plt.ylabel('Time (seconds)')

plt.savefig(sys.stdout.buffer, bbox_inches='tight')
