import sys
from collections import defaultdict
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')

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
    if n > 50:
        continue
    time = float(words[2])
    times[n].append(time)

sorted_times = list(sorted(times.items()))
xs = [x for x, _ in sorted_times]
ys = [y for _, y in sorted_times]

# determined from inspection of data
fit_lower_bound = 20
fit_upper_bound = 50
xx = np.array([x for x, y in sorted_times if fit_lower_bound <= x and x <= fit_upper_bound])
ymed = np.array([np.median(y) for x, y in sorted_times if fit_lower_bound <= x and x <= fit_upper_bound])
ymin = np.array([np.amin(y) for x, y in sorted_times if fit_lower_bound <= x and x <= fit_upper_bound])

# compute the fit
ymed_coefs = poly.polyfit(xx, np.log2(ymed), 1)
ymed_fit = poly.Polynomial(ymed_coefs)
ymed_rsquared = rsquared(np.log2(ymed), ymed_fit(xx))
label_med = '$2^{{ {:.3g}n {:.3g} }} (r^2 = {:.3g})$'.format(ymed_fit.coef[1], ymed_fit.coef[0], ymed_rsquared)
print('yfit: {} (r^2: {})'.format(ymed_fit, ymed_rsquared), file=sys.stderr)

ymin_coefs = poly.polyfit(xx, np.log2(ymin), 1)
ymin_fit = poly.Polynomial(ymin_coefs)
ymin_rsquared = rsquared(np.log2(ymin), ymin_fit(xx))
label_min = '$2^{{ {:.3g}n {:.3g} }} (r^2 = {:.3g})$'.format(ymin_fit.coef[1], ymin_fit.coef[0], ymin_rsquared)
print('yfit: {} (r^2: {})'.format(ymin_fit, ymin_rsquared), file=sys.stderr)

# plot the data
plt.boxplot(ys, positions=xs)
plt.plot(xx, np.exp2(ymed_fit(xx)), 'g', label=label_med)
plt.plot(xx, np.exp2(ymin_fit(xx)), 'r', label=label_min)

# label data etc.
plt.yscale('log')
ax = plt.gca()
ax.set_xticks([xs[0]] + list(range(10, xs[-1], 10)) + [xs[-1]])
ax.set_xticklabels([xs[0]] + list(range(10, xs[-1], 10)) + [xs[-1]])
#plt.title('Solver time (100 semi-primes per instance)')
plt.xlabel('$n$: Semiprime length (bits)')
plt.ylabel('$T(n)$: Time (seconds)')
plt.legend(loc='upper left')

plt.savefig(sys.stdout.buffer, bbox_inches='tight', format='pdf')
