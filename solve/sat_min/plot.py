#!/usr/bin/env python3

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

times = defaultdict(list);
current = 0;

for line in sys.stdin:
    if line.startswith('#'):
        continue
    words = line.split()
    n, p, q, pq, s = map(int, words[:5])
    t = float(words[5])
    times[n].append(t)

times = sorted(times.items())
xs = [x for x, _ in times]
ys = [y for _, y in times]

# compute the fit
med_ys = list(map(np.median, ys))
ycoefs = poly.polyfit(xs, np.log2(med_ys), 1)
yfit = poly.Polynomial(ycoefs)
yrsquared = rsquared(np.log2(med_ys), yfit(xs))
print('yfit: {} (r^2: {})'.format(yfit, yrsquared), file=sys.stderr)

# compute the X-percentile: set initial bound for solving timeout here so that
# X% of the trials succeed on the first pass (not succeeding at once is costly)
pct_ys = [np.percentile(y, 90) for y in ys]
pct_yc = poly.polyfit(xs, np.log2(pct_ys), 1)
pct_yf = poly.Polynomial(pct_yc)
pct_yr2 = rsquared(np.log2(pct_ys), pct_yf(xs))
print('ypct: {} (r^2: {})'.format(pct_yf, pct_yr2))

# plot the data
plt.boxplot(ys, positions=xs)

# plot the fits
plt.plot(xs, np.exp2(yfit(xs)), 'g')
plt.plot(xs, np.exp2(pct_yf(xs)), 'c')

# label data etc.
plt.yscale('log')
ax = plt.gca()
ax.set_xticks(xs)
ax.set_xticklabels(xs)
plt.xlabel('Semiprime length (bits)')
plt.ylabel('Time (seconds)')
plt.show()
