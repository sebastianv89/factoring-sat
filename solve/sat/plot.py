import matplotlib as mpl
import matplotlib.pyplot as plt
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

solver, encoding = 'maplecomsps', 'long'
if len(sys.argv) > 1:
    solver = sys.argv[1]
if len(sys.argv) > 2:
    encoding = sys.argv[2]
if len(sys.argv) > 3:
    stat = sys.argv[3]

times = defaultdict(list)
for line in sys.stdin:
    if line.startswith('#'):
        continue
    n, p, q, pq, t = line.split()
    n = int(n)
    if solver == 'maplecomsps' and n > 50:
        continue
    elif solver == 'cryptominisat5' and n > 44:
        continue
    t = float(t)
    times[n].append(t)

times = list(sorted(times.items()))
xs = [x for x, y in times]
ys = [y for x, y in times]

# determined from inspection of data
fit_lower_bound = 20 if solver == 'maplecomsps' else 25
fit_upper_bound = 50 if solver == 'maplecomsps' else 44
xx = np.array([x for x, y in times if fit_lower_bound <= x and x <= fit_upper_bound])
ymed = np.array([np.median(y) for x, y in times if fit_lower_bound <= x and x <= fit_upper_bound])
ymin = np.array([np.amin(y) for x, y in times if fit_lower_bound <= x and x <= fit_upper_bound])

ymed_coefs = poly.polyfit(xx, np.log2(ymed), 1)
ymed_fit = poly.Polynomial(ymed_coefs)
ymed_rsquared = rsquared(np.log2(ymed), ymed_fit(xx))
ymin_coefs = poly.polyfit(xx, np.log2(ymin), 1)
ymin_fit = poly.Polynomial(ymin_coefs)
ymin_rsquared = rsquared(np.log2(ymin), ymin_fit(xx))
print('{} {}'.format(solver, encoding), file=sys.stderr)
print('fit (median): {} (r^2: {})'.format(ymed_fit, ymed_rsquared), file=sys.stderr)
print('fit (minimum): {} (r^2: {})'.format(ymin_fit, ymin_rsquared), file=sys.stderr)

label_med = '$2^{{ {:.3g}n{:.3g} }} (r^2 = {:.3g})$'.format(ymed_fit.coef[1], ymed_fit.coef[0], ymed_rsquared)
label_min = '$2^{{ {:.3g}n{:.3g} }} (r^2 = {:.3g})$'.format(ymin_fit.coef[1], ymin_fit.coef[0], ymin_rsquared)
plt.plot(xx, np.exp2(ymed_fit(xx)), 'g', label=label_med)
# min is not relevant with so few samples
#plt.plot(xx, np.exp2(ymin_fit(xx)), 'r', label=label_min)
plt.xlim([xs[0], xs[-1]])
plt.ylim([10.0**-3.5, 10.0**4])
plt.boxplot(ys, positions=xs)
ax = plt.gca()
ax.set_xticks([xs[0]] + list(range(10, xs[-1], 10)) + [xs[-1]])
ax.set_xticklabels([xs[0]] + list(range(10, xs[-1], 10)) + [xs[-1]])
plt.yscale('log')
plt.xlabel('$n$: Semiprime length (bits)')
plt.ylabel('$T(n)$: Time (seconds)')
plt.legend(loc='upper left')
plt.savefig(sys.stdout.buffer, bbox_inches='tight', format='eps')
