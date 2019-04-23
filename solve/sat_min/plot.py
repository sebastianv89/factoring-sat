#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy.polynomial.polynomial as poly
import sys
from collections import defaultdict

mpl.style.use('classic')

def rsquared(y, fit):
    y_mean = np.mean(y)
    ss_tot = np.sum((y-y_mean)**2)
    ss_res = np.sum((y-fit)**2)
    return 1.0 - ss_res/ss_tot

times = defaultdict(list)
current = 0

for line in sys.stdin:
    if line.startswith('#'):
        continue
    words = line.split()
    n, p, q, pq, s = map(int, words[:5])
    if n > 50:
        continue # not enough samples for meaningful interpretation
    t = float(words[5])
    times[n].append(t)

sorted_times = sorted(times.items())
xs = np.array([x for x, _ in sorted_times])
ys = np.array([y for _, y in sorted_times])

# not enough samples for meaningful interpretation outside this ranges
fit_lower_bound = 30
fit_upper_bound = 50
xx = np.array([x for x, _  in sorted_times if fit_lower_bound <= x and x <= fit_upper_bound])
yy = np.array([y for x, y  in sorted_times if fit_lower_bound <= x and x <= fit_upper_bound])

# compute the fit
med_ys = list(map(np.median, yy))
ycoefs = poly.polyfit(xx, np.log2(med_ys), 1)
yfit = poly.Polynomial(ycoefs)
yrsquared = rsquared(np.log2(med_ys), yfit(xx))
label_med = '$2^{{ {:.3g}n {:.3g} }} (r^2 = {:.3g})$'.format(yfit.coef[1], yfit.coef[0], yrsquared)
print('yfit: {} (r^2: {})'.format(yfit, yrsquared), file=sys.stderr)

# compute the X-percentile: set initial bound for solving timeout here so that
# X% of the trials succeed on the first pass (not succeeding at once is costly)
# (this only useful for tweaking time_bound in solve.py)
pct_ys = [np.percentile(y, 90) for y in yy]
pct_yc = poly.polyfit(xx, np.log2(pct_ys), 1)
pct_yf = poly.Polynomial(pct_yc)
pct_yr2 = rsquared(np.log2(pct_ys), pct_yf(xx))
print('ypct: {} (r^2: {})'.format(pct_yf, pct_yr2), file=sys.stderr)

# compute the minimum
min_ys = [min(y) for y in yy]
min_yc = poly.polyfit(xx, np.log2(min_ys), 1)
min_yf = poly.Polynomial(min_yc)
min_yr2 = rsquared(np.log2(min_ys), min_yf(xx))
label_min = '$2^{{ {:.3g}n {:.3g} }} (r^2 = {:.3g})$'.format(min_yf.coef[1], min_yf.coef[0], min_yr2)
print('ymin: {} (r^2: {})'.format(min_yf, min_yr2), file=sys.stderr)

# plot the data
plt.boxplot(ys, positions=xs)

# plot the fits
plt.plot(xx, np.exp2(yfit(xx)), 'g', label=label_med)
#plt.plot(xx, np.exp2(pct_yf(xx)), 'c')
plt.plot(xx, np.exp2(min_yf(xx)), 'r', label=label_min)

# label data etc.
s = 5 # step size for labels
xlabels = [xs[0]] + list(range(s * ((xs[0]+s-1) // s), s*((xs[-1]+s-1) // s), s)) + [xs[-1]]
plt.yscale('log')
ax = plt.gca()
ax.set_xticks(xlabels)
ax.set_xticklabels(xlabels)
#plt.title('Solver time (minimum of 100 seeds)')
plt.xlabel('$n$: Semiprime length (bits)')
plt.ylabel('$T(n)$: Time (seconds)')
plt.legend(loc='upper left')
if len(sys.argv) < 2:
    plt.savefig(sys.stdout.buffer, bbox_inches='tight', format='pdf')
else:
    # focus on one particular bitsize n
    _, bins, _ = plt.hist(times[int(sys.argv[1])])
    plt.clf()
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.hist(times[48], bins=logbins)
    plt.xscale('log')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (%)')
    plt.show()

